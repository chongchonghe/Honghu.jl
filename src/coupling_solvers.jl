# The two one-step coupling solvers. Both take one backward-Euler step of the
# same equations; they differ in what they call the unknown, and that difference
# is the whole point of keeping them both.
#
# `BracketedCoupling` is the default. It takes the *gas energy alone* as its
# unknown: the group block of the step is linear in `E_g` at fixed gas
# temperature, so `group_energies` eliminates it in closed form and
# `energy_balance` is what is left. One scalar root-find, inside a bracket the
# physics guarantees, with no matrix anywhere in it. Both of those functions
# live in `coupling.jl`, built from `coupling_coefficients`, so this solver has
# no physics of its own to keep in step.
#
# `QuokkaCoupling` is a separate formulation, kept separate on purpose. It takes
# the *integrated transfer* `R_g` as its unknown, which does not exist until a
# step length has been chosen, so its Jacobian cannot be generated from a
# Jacobian of the right-hand side; see the derivation in the middle of its Newton
# loop. It shares `coupling_rhs` (its opening guess is one explicit step of it)
# and nothing else. It is kept because Quokka runs it, and structural fidelity to
# Quokka is a design goal of this code -- but it is no longer the default, and
# `docs/coupling-findings.md` says why: over 240 cells spanning three opacity
# laws it fails 75 where the bracketed solver fails none.
#
# The two agreeing is worth more than either alone. They share the equations and
# nothing else -- not the unknown, not the residual, not the opening iterate, not
# the convergence test, and not the linear algebra, of which one of them has
# none. `test_coupling_adaptive.jl` holds them to each other and to the
# error-controlled solve.
#
# `QuokkaCoupling` also carries the two globalisations that are properties of the
# *iteration* and not of the equations -- the optically-thin shortcut and the
# equilibrium-overshoot guard -- and both solvers finish with
# `apply_energy_floors`. The note at the top of `coupling.jl` says why those
# three live here rather than in the right-hand side. `BracketedCoupling` needs
# no equivalent of either: its bracket is the globalisation, and a transparent
# cell is one where `group_energies` simply returns the old energies.
#
# The linear solve of one Newton step -- `AbstractJacobianSolve`,
# `solve_jacobian` -- is in `newton.jl`, and is `QuokkaCoupling`'s alone.

# --- the two solvers ----------------------------------------------------------

"""
    QuokkaCoupling(; tolerance = 1e-9, max_iterations = 100,
                     linsolve = ArrowheadSolve())

Backward Euler in Quokka's *replaced* base variables `(E_gas, R_g)`, where `R_g`
is the energy handed from gas to radiation group `g` over the step. The default.

Why not the obvious variables: with `(E_gas, E_g)` the residual for a group is a
difference of two large, nearly-cancelling terms once `tau = chat rho kappa dt`
is big, and its Jacobian entry scales like `tau`, so the linear solve is
ill-conditioned exactly where the physics is stiffest. Writing the unknown as
the *transfer* `R_g` moves `tau` from the residual into the definition, and the
condition number stops tracking it. That is what lets one step cross a coupling
time a hundred million times shorter than `dt`.

The Jacobian is an arrowhead -- diagonal in the groups, plus one dense row and
one dense column for the gas -- so `linsolve` defaults to
[`ArrowheadSolve`](@ref) and inverts it in `O(NG)` with no matrix ever formed.
The diagonal is the same physical assumption [`group_energies`](@ref) makes when
it eliminates the group block, and [`DenseSolve`](@ref) is the escape hatch and
the cross-check; see [`AbstractJacobianSolve`](@ref).

It is also hand-derived rather than taken from [`coupling_jacobian`](@ref), and
has to be: `R_g` is not a state variable, so there is no `dR_g/dt` to
differentiate. The derivation is written out in the Newton loop.
"""
struct QuokkaCoupling{T<:AbstractFloat,L<:AbstractJacobianSolve} <: AbstractCoupling
    tolerance::T
    max_iterations::Int
    linsolve::L
end

QuokkaCoupling(; tolerance=1e-9, max_iterations::Integer=100,
               linsolve::AbstractJacobianSolve=ArrowheadSolve()) =
    QuokkaCoupling(float(tolerance), Int(max_iterations), linsolve)

# The group kernels below take every value they need as an *argument*. Written
# as closures inside the Newton loop they would capture `Rvec` and `Erad`, which
# the loop assigns, and Julia would box them: one heap allocation per group per
# iteration per cell. That is the trap CLAUDE.md records from `source_terms.jl`,
# and it costs 24 kB per cell here when it bites. Passing arguments keeps the
# unrolled `ntuple` free of captured state.

# Group energies implied by a transfer vector R -- the closed-form inversion the
# replaced base variables buy.
@inline _erad_from_R(tau::SVec{NG,T}, kPoverE, B, Rvec, work, Erad0, src,
                     ::Val{NG}) where {NG,T} =
    SVec{NG,T}(ntuple(g -> tau[g] <= 0 ? Erad0[g] + src[g] + Rvec[g] :
                           kPoverE[g] * (B[g] - (Rvec[g] - work[g]) / tau[g]), Val(NG))...)

# 1 where the group is too thin to balance its residual, 0 otherwise.
@inline _thin_mask(tau::SVec{NG,T}, B, scale, tol, ::Val{NG}) where {NG,T} =
    SVec{NG,T}(ntuple(g -> tau[g] * max(B[g], scale) < tol * scale ? one(T) : zero(T),
                      Val(NG))...)

@inline _blend(mask::SVec{NG,T}, thinval, thickval, ::Val{NG}) where {NG,T} =
    SVec{NG,T}(ntuple(g -> mask[g] > 0 ? thinval[g] : thickval[g], Val(NG))...)

@inline _residual_g(mask::SVec{NG,T}, Erad, Erad0, Rvec, src, ::Val{NG}) where {NG,T} =
    SVec{NG,T}(ntuple(g -> mask[g] > 0 ? zero(T) :
                           (Erad[g] - Erad0[g]) - (Rvec[g] + src[g]), Val(NG))...)

@inline _jgg_quokka(tau::SVec{NG,T}, kPoverE, big, ::Val{NG}) where {NG,T} =
    SVec{NG,T}(ntuple(g -> tau[g] <= 0 ? big : -kPoverE[g] / tau[g] - one(T), Val(NG))...)

@inline one_svec(::Val{NG}, ::Type{T}) where {NG,T} =
    SVec{NG,T}(ntuple(_ -> one(T), Val(NG))...)

@inline _ratio(a::SVec{NG,T}, b::SVec{NG,T}, ::Val{NG}) where {NG,T} =
    SVec{NG,T}(ntuple(g -> iszero(b[g]) ? one(T) : a[g] / b[g], Val(NG))...)

# The opacity's own temperature dependence, as it enters QuokkaCoupling's
# `dE_g/dE_gas`: `(dkappaP B - dkappaE E) / kappaE`, and zero for a group with no
# energy-mean opacity, which cannot absorb and so has no such term.
@inline _dkappa_term(dkP::SVec{NG,T}, dkE, kE, B, Erad, ::Val{NG}) where {NG,T} =
    SVec{NG,T}(ntuple(g -> iszero(kE[g]) ? zero(T) :
                           (dkP[g] * B[g] - dkE[g] * Erad[g]) / kE[g], Val(NG))...)

# --- Quokka's replaced base variables ----------------------------------------

function solve_coupling(m::QuokkaCoupling, s::CouplingState{NG,M,T}) where {NG,M,T}
    cs = gas_scale(s)
    kdt = (s.dt * s.chat) * s.rho

    Egas = s.Egas
    Erad = s.Erad
    Etot0 = s.Egas + dot(cs, s.Erad + s.src)

    # Open from the *linearised* transfer at the old state, not from R = 0.
    # `R_g = dt rate_g + work_g` evaluated at the old state is exactly the shared
    # right-hand side taken explicitly -- one forward Euler transfer -- so this
    # opening guess is the physics of `coupling_rhs` and not a second copy of it.
    #
    # Starting at zero instead makes the first implied group energy
    # kappaPoverE (B + work/tau), which goes negative whenever the lagged v.F
    # work term is negative and outweighs the cell's thermal emission. The
    # equilibrium guard below then snaps the gas to its floor and the iteration
    # never recovers, emptying the cell. The grey driver in `source_terms.jl`
    # has always opened this way; the multigroup path had not, and every
    # negative-velocity cell of a beta_order = 1 run drained to the floor.
    Rvec = s.dt * last(coupling_rhs(s, Egas, s.Erad)) + s.work

    n = 0
    converged = false
    while n < m.max_iterations
        n += 1
        Tgas = gas_temperature_of(s, Egas)
        fourPiBoverC = thermal_emission(s, Tgas)
        dB_dT = thermal_emission_dT(s, Tgas)
        kappaP, kappaE = coupling_opacity(s, Tgas)
        dkappaP, dkappaE = coupling_opacity_dT(s, Tgas)

        # tau_g is the optical depth across chat*dt: the number of absorption
        # times the step spans, and the scaling that makes this formulation
        # work. Recomputed here rather than frozen at the old temperature,
        # because the opacity is a function of the iterate -- Quokka's
        # `tau = dt * rho * kappaP * chat` sits inside its loop for the same
        # reason.
        tau = mul(kdt, kappaP)
        kPoverE = _ratio(kappaP, kappaE, Val(NG))

        # Given R, the group energies follow in closed form -- this is the
        # inversion that the replacement buys, and why no group energy ever
        # appears as an independent unknown. The opening pass keeps the old
        # energies, which is what makes the linearised R above self-consistent.
        Erad = n == 1 ? s.Erad :
               _erad_from_R(tau, kPoverE, fourPiBoverC, Rvec, s.work, s.Erad, s.src, Val(NG))

        scale = max(abs(Etot0), eps(T))

        # Optically thin groups. When tau is negligible the coupling term cannot
        # balance the group's residual however many iterations we spend, so the
        # group takes the transport-only answer and drops out of the test.
        # Without this the iteration burns all `max_iterations` in every
        # transparent cell -- and a multigroup run is *mostly* transparent
        # groups, since the Wien tail of any spectrum is optically thin.
        thin = _thin_mask(tau, fourPiBoverC, scale, m.tolerance, Val(NG))
        Erad = _blend(thin, s.Erad + s.src, Erad, Val(NG))

        F0 = Egas - s.Egas + dot(cs, Rvec)
        Fg = _residual_g(thin, Erad, s.Erad, Rvec, s.src, Val(NG))

        # Convergence on the residual, scaled by the total energy in the cell so
        # the test means the same thing in code units and in CGS.
        if max(abs(F0), maxabs(Fg)) <= m.tolerance * scale
            converged = true
            break
        end

        # --- the hand-derived Jacobian, and why it has to be hand-derived -----
        #
        # `R_g` is not a state variable, so there is no `dR_g/dt` and
        # `coupling_jacobian` cannot produce these entries. `R_g` is the
        # *integrated* transfer over the step and carries `dt` inside `tau`;
        # this is a different discretisation of the same ODE, not a change of
        # variables applied to the natural ones. (A change of variables would
        # give an invertible map between unknowns at fixed `dt`; here the second
        # unknown does not exist until a step length is chosen.)
        #
        # Chain rule through the closed-form `Erad(R)`:
        #
        #   dF_g/dR_g = (dF_g/dE_g)(dE_g/dR_g) + explicit
        #             = 1 * (-kappaPoverE/tau)  + (-1)
        #   dF_g/dE_gas = dE_g/dE_gas
        #             = [dkappaP_g B_g + kappaP_g dB_g/dT - dkappaE_g E_g]
        #               / (kappaE_g cv)
        #
        # The second is the shared `rad_gas` block divided by `chat rho kappaE`,
        # which is what differentiating `R_g = dt rate_g + work_g` at fixed `R`
        # gives; written out here because the division is singular at
        # `kappaE = 0` and the branch belongs with the rest of this formulation.
        J00 = one(T)
        J0g = cs
        Jg0 = (mul(kPoverE, dB_dT) +
               _dkappa_term(dkappaP, dkappaE, kappaE, fourPiBoverC, Erad,
                            Val(NG))) / s.cv
        # tau <= 0 means the group cannot exchange: a large finite number rather
        # than -Inf, because Inf arithmetic in the elimination makes NaN.
        Jgg = _jgg_quokka(tau, kPoverE, -one(T) / eps(T), Val(NG))

        # Arrowhead by default, and by the same assumption the scalar reduction
        # makes: `Jgg` is a vector because a group's transfer sees no other
        # group's. `DenseSolve` factorises the same system without assuming it,
        # which is what the two agreeing is a test of.
        dEgas, dR = solve_jacobian(m.linsolve, J00, J0g, Jg0, Jgg, -F0, -Fg)

        # Guard against a Newton step that overshoots past the radiation
        # temperature: snap to radiative equilibrium, which is where the
        # iteration is heading anyway. The total radiation energy sets that
        # temperature, since the Planck fractions sum to one.
        Trad = sqrt(sqrt(max(sumvec(Erad), zero(T)) / s.a_rad))
        if dEgas / s.cv > max(Tgas, Trad)
            Egas = s.cv * Trad
        else
            Egas += dEgas
            Rvec += dR
        end

        # Keep the gas on the admissible side during the iteration; a Newton
        # step that overshoots to negative energy would evaluate T^4 on garbage.
        Egas = max(Egas, s.cv * s.Tfloor)
    end

    Egas, Erad = apply_energy_floors(s, Egas, Erad)
    return CouplingResult{NG,T}(Egas, Erad, n, converged)
end

# --- the natural variables, reduced to one scalar unknown ---------------------

"""
    BracketedCoupling(; tolerance = 1e-9, max_iterations = 100)

Backward Euler in the natural variables `(E_gas, E_g)`, reduced to **one scalar
unknown** and solved inside a bracket the physics guarantees. The default.

The group block is linear in `E_g` at fixed gas temperature, so
[`group_energies`](@ref) solves it in closed form and what is left is the single
equation [`energy_balance`](@ref): find the gas energy at which the cell's books
balance. There is no `1 + NG` system, no Jacobian to assemble, no linear solve
and no arrowhead elimination -- `O(NG)` per iteration and zero bytes per cell.

**Why it is the default: it cannot walk away from the root.** `G` is non-positive
at the temperature floor and non-negative once the gas is handed the whole
cell's energy, so a sign change always exists on `[cv Tfloor, E_tot]`; a Newton
step that would leave the bracket is replaced by a bisection step. The
unsafeguarded `1 + NG` iteration has no such guard and does not merely overshoot
-- with a steeply temperature-dependent opacity its Newton *direction* can be
wrong, because it linearises about a state where the group block is not
satisfied and `E_g` is far from equilibrium with `T`. Measured over 240 cells
spanning three opacity laws and `tau_cell` from 1e-4 to 1e12, this solver failed
none and [`QuokkaCoupling`](@ref) failed 75; on a multigroup Marshak wave with
`kappa ~ T^2` every other one-step solver leaves the temperature field 7 per
cent wrong and this one 0.25 per cent. `docs/coupling-findings.md` derives all
of it and `docs/coupling-experiments/coupling_study.jl` reproduces the numbers.

Two further consequences of the reduction, both free:

  - **Its convergence test is unconditional.** `G` is a difference of energies
    with no `tau (B - E)` left in it to cancel -- that cancellation was done
    algebraically -- so nothing amplifies round-off by `tau_cell` and the
    tolerance means what it says at any optical depth. That is the failure the
    replaced base variables of [`QuokkaCoupling`](@ref) exist to avoid, avoided
    a different way.
  - **`dG/dE_gas` is computed without cancellation.** It equals the Schur
    complement of the full Newton matrix exactly, but the Schur complement is
    formed by subtracting two terms of order `tau` to get one of order 1.

It costs 9 per cent more than [`QuokkaCoupling`](@ref) at one group and 40 per
cent at four to sixteen, for the bracket's extra evaluations of `G`.

The reduction inherits one assumption from [`group_energies`](@ref): groups
exchange energy with the gas and never directly with each other. That is the same
assumption [`ArrowheadSolve`](@ref) makes; a term coupling groups directly would
make the group block a matrix and there would be no closed form to eliminate.
"""
struct BracketedCoupling{T<:AbstractFloat} <: AbstractCoupling
    tolerance::T
    max_iterations::Int
end

BracketedCoupling(; tolerance=1e-9, max_iterations::Integer=100) =
    BracketedCoupling(float(tolerance), Int(max_iterations))

function solve_coupling(m::BracketedCoupling, s::CouplingState{NG,M,T}) where {NG,M,T}
    Etot = abs(s.Egas + dot(gas_scale(s), s.Erad + s.src))
    atol = m.tolerance * Etot

    # The bracket. `G` is non-positive at the floor because every group's
    # emission vanishes there, and non-negative at `Etot` because putting the
    # whole cell's energy in the gas leaves the first and last terms of `G`
    # cancelling and only the group energies behind. Both bounds assume
    # `rad0_g >= 0`, which a lagged work term more negative than a group's own
    # energy can break, so the upper end is opened generously and then widened
    # rather than trusted -- and if there is still no sign change the cell is
    # reported unconverged instead of guessed at.
    lo = s.cv * s.Tfloor
    hi = max(4 * Etot, 4 * s.Egas)
    Erad, _ = group_energies(s, lo)
    Glo = energy_balance(s, lo, Erad)
    Erad, _ = group_energies(s, hi)
    Ghi = energy_balance(s, hi, Erad)
    k = 0
    while Ghi < 0 && k < 60
        hi *= 4
        Erad, _ = group_energies(s, hi)
        Ghi = energy_balance(s, hi, Erad)
        k += 1
    end
    if Glo > 0 || Ghi < 0
        Egas, Erad = apply_energy_floors(s, s.Egas, s.Erad + s.src)
        return CouplingResult{NG,T}(Egas, Erad, 0, false)
    end

    x = clamp(s.Egas, lo, hi)
    n = 0
    converged = false
    while n < m.max_iterations
        n += 1
        Erad, dErad = group_energies(s, x)
        G = energy_balance(s, x, Erad)
        # Every evaluation tightens the bracket, whether or not its step is used.
        G < 0 ? (lo = x) : (hi = x)
        if abs(G) <= atol
            converged = true
            break
        end
        dG = one(T) + dot(gas_scale(s), dErad)
        xn = iszero(dG) ? T(Inf) : x - G / dG
        # ... and a Newton step that would leave the bracket is replaced by the
        # bisection of it, which is the whole of the globalisation.
        (xn <= lo || xn >= hi) && (xn = (lo + hi) / 2)
        if abs(xn - x) <= eps(T) * abs(x)
            x = xn
            converged = true
            break
        end
        x = xn
    end

    Erad, _ = group_energies(s, x)
    Egas, Erad = apply_energy_floors(s, x, Erad)
    return CouplingResult{NG,T}(Egas, Erad, n, converged)
end
