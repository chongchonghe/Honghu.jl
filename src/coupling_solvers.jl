# The two in-tree coupling solvers. Both take one backward-Euler step of the ODE
# defined by `coupling_rhs`; they differ in what they call the second unknown,
# and that difference is the whole point of keeping them both.
#
# `DirectCoupling` is generated: its residual is `u - u0 - dt f(u)` and its
# Jacobian `I - dt J`, both straight from `coupling_rhs` / `coupling_jacobian`,
# so it has no physics of its own to keep in step. How it solves that matrix is
# a choice -- `AbstractJacobianSolve`, below -- because the `O(NG)` elimination
# both solvers use is a physical assumption in the shape of an algorithm.
#
# `QuokkaCoupling` is a separate formulation, kept separate on purpose. It takes
# the *integrated transfer* `R_g` as its unknown, which does not exist until a
# step length has been chosen, so its Jacobian cannot be generated from a
# Jacobian of the right-hand side; see the derivation in the middle of its Newton
# loop. It shares the right-hand side (its opening guess is one explicit step of
# it) and nothing else.
#
# `QuokkaCoupling` also carries the two globalisations that are properties of the
# *iteration* and not of the equations -- the optically-thin shortcut and the
# equilibrium-overshoot guard -- and both solvers finish with
# `apply_energy_floors`. The note at the top of `coupling.jl` says why those
# three live here rather than in the right-hand side.

"""
    QuokkaCoupling(; tolerance = 1e-9, max_iterations = 100)

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
one dense column for the gas -- so it inverts in `O(NG)` by elimination rather
than `O(NG^3)`, with no matrix ever formed. See [`solve_arrowhead`](@ref).

It is also hand-derived rather than taken from [`coupling_jacobian`](@ref), and
has to be: `R_g` is not a state variable, so there is no `dR_g/dt` to
differentiate. The derivation is written out in the Newton loop.
"""
struct QuokkaCoupling{T<:AbstractFloat} <: AbstractCoupling
    tolerance::T
    max_iterations::Int
end

QuokkaCoupling(; tolerance=1e-9, max_iterations::Integer=100) =
    QuokkaCoupling(float(tolerance), Int(max_iterations))

# --- the Newton step's linear solve -------------------------------------------
#
# One `(1+NG)x(1+NG)` system per Newton iteration per cell, and two ways to solve
# it. Which one is a choice `DirectCoupling` hands to its caller, for a reason
# spelled out under `AbstractJacobianSolve`: the fast one is only correct while
# the group block of the Jacobian is diagonal, and that is a property of the
# terms currently in `coupling_rhs` rather than of radiation.

"""
    solve_arrowhead(J00, J0g, Jg0, Jgg, F0, Fg) -> (dx0, dxg)

Solve the bordered ("arrowhead") linear system

    [ J00   J0g^T ] [dx0]   [F0]
    [ Jg0   diag  ] [dxg] = [Fg]

for the Newton update, by eliminating the group block into the gas row. Exact,
`O(NG)`, and forms no matrix -- which matters because this runs per cell, per
Newton step, per timestep.

A group with `Jgg[g] == 0` is decoupled from the gas (zero optical depth), and
contributes nothing to the elimination; its update is zero.
"""
@inline function solve_arrowhead(J00::T, J0g::SVec{NG,T}, Jg0::SVec{NG,T},
                                 Jgg::SVec{NG,T}, F0::T, Fg::SVec{NG,T}) where {NG,T}
    # dxg = (Fg - Jg0 dx0) / Jgg, substituted into J00 dx0 + sum(J0g dxg) = F0
    inv_d = SVec{NG,T}(ntuple(g -> iszero(Jgg[g]) ? zero(T) : one(T) / Jgg[g], Val(NG))...)
    a = J00 - sumvec(mul(J0g, mul(Jg0, inv_d)))
    b = F0 - sumvec(mul(J0g, mul(Fg, inv_d)))
    dx0 = iszero(a) ? zero(T) : b / a
    dxg = mul(Fg - dx0 * Jg0, inv_d)
    return (dx0, dxg)
end

"""
    AbstractJacobianSolve

How [`DirectCoupling`](@ref) solves the linear system of one Newton step. Two
are provided, and the **default is the slower one on purpose**:

  - [`DenseSolve`](@ref)     -- assemble `I - dt J` and factorise it. `O(NG^3)`,
    allocates, assumes nothing. The default.
  - [`ArrowheadSolve`](@ref) -- eliminate the group block into the gas row.
    `O(NG)`, forms no matrix, allocates nothing.

The arrowhead is exact for the coupling as it stands, and it is not a small
saving: at four groups it is the difference between roughly 0.9 kB a call and
none, and between 25 and 70 per cent of the wall time of the whole cell solve
over `NG = 4` to `16`. But it is exact only because of a *physical* assumption
-- groups exchange energy with the gas and never directly with each other, so
the group block of [`CouplingJacobian`](@ref) is diagonal. That is a property of
the terms in [`coupling_rhs`](@ref) today, not a truth about radiation. Add
Compton redistribution between groups, or overlapping lines, and it acquires
off-diagonal entries which the elimination has nowhere to put and would silently
drop, while the dense assembly would only need to write a block where it now
writes a diagonal.

Defaulting to the general solve is the opposite of the usual instinct, which is
to ship the fast path and offer the safe one as an option, so it is worth saying
why. The failure it protects against is quiet. Newton stops on the residual of
the true equations, so a damaged matrix cannot converge to the wrong answer --
it can only fail to converge, and a cell that exhausts `max_iterations` returns
its last iterate and increments a counter rather than stopping the run. What you
would see is a slower, slightly wrong simulation with a nonzero
`newton_failures`, which is exactly the kind of symptom that gets blamed on the
physics. The `O(NG)` shortcut is worth having; it is not worth having by
default, on behalf of a user who has not been told what it assumes.
"""
abstract type AbstractJacobianSolve end

"""
    DenseSolve()

Assemble the Newton matrix `I - dt J` as an ordinary `(1+NG)x(1+NG)` matrix and
solve it by LU factorisation (LinearAlgebra's `lu!`/`ldiv!`, in place on the
freshly built matrix and right-hand side). The default for
[`DirectCoupling`](@ref).

It makes no use of the arrowhead structure, which is precisely why it is the
default -- see [`AbstractJacobianSolve`](@ref) for the argument.

**It allocates, and it must**: `NG` is a compile-time constant but a `Matrix` is
heap storage, so every Newton iteration of every cell buys a matrix, a
right-hand side and the factorisation's pivot vector. Measured: 272 bytes per
linear solve at one group, 464 at four, 976 at eight, 3.0 kB at sixteen, which
makes a converged three-iteration cell cost 0.54, 0.93, 1.9 and 5.9 kB
respectively. Wall time is 1.3 to 1.7 times [`ArrowheadSolve`](@ref) at four to
sixteen groups, and 4.7 times at one, where the linear solve is most of what a
grey cell does. `ArrowheadSolve()` restores exactly zero bytes per cell.

Nothing in the package's default configuration is on this path --
[`RadiationPhysics`](@ref) couples with [`QuokkaCoupling`](@ref), which is
unaffected -- so this is a cost paid by a run that asks for `DirectCoupling`,
which is the solver kept for verification. A run that wants it in production
should pass `ArrowheadSolve()` and know why.
"""
struct DenseSolve <: AbstractJacobianSolve end

"""
    ArrowheadSolve()

Solve the Newton step with [`solve_arrowhead`](@ref): eliminate the diagonal
group block into the gas row, `O(NG)`, no matrix formed, nothing allocated.

Opt in when the linear solve is measurably worth it *and* the coupling is still
the one in [`coupling_rhs`](@ref), where groups meet only through the gas. It is
what [`QuokkaCoupling`](@ref) uses unconditionally, and what `DirectCoupling`
used unconditionally before the choice existed.
"""
struct ArrowheadSolve <: AbstractJacobianSolve end

"""
    solve_jacobian(linsolve, J00, J0g, Jg0, Jgg, F0, Fg) -> (dx0, dxg)

Solve one Newton step's system, given the four blocks of the Newton matrix and
the two blocks of the right-hand side, in the layout [`solve_arrowhead`](@ref)
draws. Dispatches on an [`AbstractJacobianSolve`](@ref).

The two paths differ once the matrix is singular: the arrowhead treats a zero
group entry as a decoupled group and returns a zero update for it, while an LU
factorisation of the same matrix throws. Neither case arises from
[`DirectCoupling`](@ref), whose group entries are `1 + dt chat rho kappa_E >= 1`.
"""
@inline solve_jacobian(::ArrowheadSolve, J00::T, J0g::SVec{NG,T}, Jg0::SVec{NG,T},
                       Jgg::SVec{NG,T}, F0::T, Fg::SVec{NG,T}) where {NG,T} =
    solve_arrowhead(J00, J0g, Jg0, Jgg, F0, Fg)

function solve_jacobian(::DenseSolve, J00::T, J0g::SVec{NG,T}, Jg0::SVec{NG,T},
                        Jgg::SVec{NG,T}, F0::T, Fg::SVec{NG,T}) where {NG,T}
    A = zeros(T, NG + 1, NG + 1)
    b = Vector{T}(undef, NG + 1)
    @inbounds A[1, 1] = J00
    @inbounds b[1] = F0
    @inbounds for g in 1:NG
        A[1, 1+g] = J0g[g]
        A[1+g, 1] = Jg0[g]
        A[1+g, 1+g] = Jgg[g]      # the block the arrowhead assumes is diagonal
        b[1+g] = Fg[g]
    end
    # In place on both, so no third copy of the matrix is made; `lu!` still buys
    # its pivot vector, and that is counted in the figures quoted for
    # `DenseSolve`. `b` is never rebound, only written through, so the `ntuple`
    # below captures a value rather than a box.
    ldiv!(lu!(A), b)
    return (@inbounds(b[1]), SVec{NG,T}(ntuple(g -> @inbounds(b[1+g]), Val(NG))...))
end

"""
    DirectCoupling(; tolerance = 1e-9, max_iterations = 100, linsolve = DenseSolve())

Backward Euler in the natural variables `(E_gas, E_g)`, with the same Newton
iteration as [`QuokkaCoupling`](@ref) and a choice of linear solve.

Nothing about it is hand-derived: the residual is `u - u0 - dt f(u)` with `f` =
[`coupling_rhs`](@ref), and the Newton matrix is `I - dt * J` with `J` =
[`coupling_jacobian`](@ref).

`linsolve` is an [`AbstractJacobianSolve`](@ref) and defaults to the general
[`DenseSolve`](@ref) rather than the `O(NG)` [`ArrowheadSolve`](@ref), which is
the reverse of the usual defaulting instinct: the arrowhead is fast because it
assumes the Jacobian's group block is diagonal, and that assumption belongs to
the physics currently in `coupling_rhs` rather than to the solver. Its
docstrings argue the case; the short version is that the dense path costs a
kilobyte a cell in a solver kept for verification, and the arrowhead's failure
mode -- when a future term couples groups directly -- would be silent.

This is the textbook formulation, and it exists here to be compared against
[`QuokkaCoupling`](@ref): the two solve the same equations to the same
tolerance, so they must agree wherever both converge, and where the direct form
struggles is a measurement of what the base-variable replacement buys.
"""
struct DirectCoupling{T<:AbstractFloat,L<:AbstractJacobianSolve} <: AbstractCoupling
    tolerance::T
    max_iterations::Int
    linsolve::L
end

DirectCoupling(; tolerance=1e-9, max_iterations::Integer=100,
               linsolve::AbstractJacobianSolve=DenseSolve()) =
    DirectCoupling(float(tolerance), Int(max_iterations), linsolve)

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
    cscale = s.c / s.chat
    kdt = s.dt * s.chat * s.rho

    Egas = s.Egas
    Erad = s.Erad
    Etot0 = s.Egas + cscale * (sumvec(s.Erad) + sumvec(s.src))

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
        tau = kdt * kappaP
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

        F0 = Egas - s.Egas + cscale * sumvec(Rvec)
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
        # variables applied to `DirectCoupling`'s. (A change of variables would
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
        J0g = cscale * one_svec(Val(NG), T)
        Jg0 = (mul(kPoverE, dB_dT) +
               _dkappa_term(dkappaP, dkappaE, kappaE, fourPiBoverC, Erad,
                            Val(NG))) / s.cv
        # tau <= 0 means the group cannot exchange: a large finite number rather
        # than -Inf, because Inf arithmetic in the elimination makes NaN.
        Jgg = _jgg_quokka(tau, kPoverE, -one(T) / eps(T), Val(NG))

        dEgas, dR = solve_arrowhead(J00, J0g, Jg0, Jgg, -F0, -Fg)

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

# --- the natural variables ----------------------------------------------------

function solve_coupling(m::DirectCoupling, s::CouplingState{NG,M,T}) where {NG,M,T}
    cscale = s.c / s.chat
    kdt = (s.dt * s.chat * s.rho)
    Etot0 = s.Egas + cscale * (sumvec(s.Erad) + sumvec(s.src))

    # The residual of this formulation cannot be *evaluated* below a floor set by
    # the optical depth. Its group term is tau (B - E), and at the solution B and
    # E agree to round-off, so cancellation gives B - E an absolute error of
    # eps*B and the residual an error of eps*tau*B. Asking for 1e-9 at tau = 1e9
    # asks for a number smaller than the noise in computing it, and the iteration
    # then runs to `max_iterations` while sitting on the right answer.
    #
    # So the achievable tolerance degrades in proportion to tau. That is not a
    # deficiency of this implementation; it is the conditioning that Quokka's
    # replaced base variables are designed to avoid, and stating it here is what
    # makes the comparison against `QuokkaCoupling` an honest one.
    tau_max = maxabs(kdt * first(coupling_opacity(s, gas_temperature_of(s, s.Egas))))
    tol = max(m.tolerance, 4 * eps(T) * max(tau_max, one(T)))

    Egas = s.Egas
    Erad = s.Erad + s.src

    n = 0
    converged = false
    while n < m.max_iterations
        n += 1

        # Backward Euler, written out as `u - u0 - dt f(u)` and nothing else: the
        # constant increments `src` and `work` are the whole-step contributions
        # of terms the right-hand side carries as `(src + work)/dt`, so they
        # integrate to themselves and appear undivided here.
        fgas, frad = coupling_rhs(s, Egas, Erad)
        F0 = (Egas - s.Egas) - s.dt * fgas + cscale * sumvec(s.work)
        Fg = (Erad - s.Erad) - s.dt * frad - s.src - s.work

        scale = max(abs(Etot0), eps(T))
        if max(abs(F0), maxabs(Fg)) <= tol * scale
            converged = true
            break
        end

        # ... and its Jacobian is `I - dt J` with nothing hand-derived, which is
        # the point: this solver has no formulation of its own to keep in step
        # with the physics.
        J = coupling_jacobian(s, Egas, Erad)
        J00 = one(T) - s.dt * J.gas_gas
        J0g = -s.dt * J.gas_rad
        Jg0 = -s.dt * J.rad_gas
        Jgg = one(T) - s.dt * J.rad_rad

        # How that system is solved is the caller's choice, and defaults to the
        # general one. `Jgg` is a vector rather than a matrix only because `J`'s
        # group block is diagonal today (see `CouplingJacobian`); the arrowhead
        # elimination is that fact turned into an algorithm, and `m.linsolve` is
        # where a user says they have checked it still holds.
        dEgas, dErad = solve_jacobian(m.linsolve, J00, J0g, Jg0, Jgg, -F0, -Fg)
        Egas = max(Egas + dEgas, s.cv * s.Tfloor)
        Erad += dErad
    end

    Egas, Erad = apply_energy_floors(s, Egas, Erad)
    return CouplingResult{NG,T}(Egas, Erad, n, converged)
end
