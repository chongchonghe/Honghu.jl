# The two in-tree coupling solvers. Both take one backward-Euler step of the ODE
# defined by `coupling_rhs`; they differ in what they call the second unknown,
# and that difference is the whole point of keeping them both.
#
# `DirectCoupling` is generated: its residual is `u - u0 - dt f(u)` and its
# Jacobian `I - dt J`, both straight from `coupling_rhs` / `coupling_jacobian`,
# so it has no physics of its own to keep in step. How either solver inverts that
# matrix is a `linsolve` choice -- `AbstractJacobianSolve`, below -- because the
# `O(NG)` elimination they both default to is a physical assumption in the shape
# of an algorithm, and the other option is what checks it.
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

# --- the Newton step's linear solve -------------------------------------------
#
# One `(1+NG)x(1+NG)` system per Newton iteration per cell, and two ways to solve
# it. Both solvers default to the fast one and take the other as a `linsolve`
# argument, for the reason spelled out under `AbstractJacobianSolve`: the fast
# one is correct only while the group block of the Jacobian is diagonal, that is
# a property of the terms currently in `coupling_rhs` rather than of radiation,
# and the general solve is what tests the difference.

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

How [`QuokkaCoupling`](@ref) and [`DirectCoupling`](@ref) solve the linear system
of one Newton step. Two are provided:

  - [`ArrowheadSolve`](@ref) -- eliminate the group block into the gas row.
    `O(NG)`, forms no matrix, allocates nothing. The default for both.
  - [`DenseSolve`](@ref)     -- assemble the Newton matrix and factorise it.
    `O(NG^3)`, allocates, assumes nothing. The reference.

The arrowhead is exact for the coupling as it stands, and it is not a small
saving: at four groups it is the difference between roughly 0.9 kB a call and
none, and between 25 and 70 per cent of the wall time of the whole cell solve
over `NG = 4` to `16`. That is why it is the default.

It is exact because of a *physical assumption*: groups exchange energy with the
gas and never directly with each other, so the group block of
[`CouplingJacobian`](@ref) is diagonal. That is a property of the terms in
[`coupling_rhs`](@ref) today, not a truth about radiation. Compton
redistribution between groups, or overlapping lines, would give that block
off-diagonal entries, and the elimination has nowhere to put them.

What makes it acceptable as a default is that the assumption is now *checked*.
`DenseSolve` solves the same system without it, and the suite asserts that the
two agree -- block by block on a matrix built from the real Jacobian, and end to
end through a converged cell. A default may encode an assumption when a checked
alternative exists; it may not when the assumption is silent, and until
`DenseSolve` existed this one was silent: both in-tree solvers called the
elimination, so they could agree with each other and both be wrong.

The cross-check has to exist because the failure would be quiet. Newton stops on
the residual of the true equations, so a damaged matrix cannot converge to a
wrong answer -- it can only fail to converge, and a cell that exhausts
`max_iterations` returns its last iterate and increments a counter rather than
stopping the run. What a broken elimination looks like from outside is a slower,
slightly wrong simulation with a nonzero `newton_failures`, which is exactly the
kind of symptom that gets blamed on the physics. Nothing about a run reports it;
a test has to.

`DenseSolve` stays the reference implementation, and is what to switch to the day
a term is added that couples groups directly. That day is not free there either
-- `rad_rad` is an `SVec` today, so the dense assembly writes a diagonal where it
would have to write a block -- but that is one line of assembly against an
elimination that has no place for the entries at all.
"""
abstract type AbstractJacobianSolve end

"""
    DenseSolve()

Assemble the Newton matrix `I - dt J` as an ordinary `(1+NG)x(1+NG)` matrix and
solve it by LU factorisation (LinearAlgebra's `lu!`/`ldiv!`, in place on the
freshly built matrix and right-hand side).

It makes no use of the arrowhead structure, which is what makes it the
reference: it solves the system that [`ArrowheadSolve`](@ref) assumes something
about, so the two agreeing is a direct test of the assumption rather than of two
copies of it. Pass `linsolve = DenseSolve()` to either solver to run that
comparison, and switch to it for good if a term is added that couples groups
directly -- see [`AbstractJacobianSolve`](@ref).

**It allocates, and it must**: `NG` is a compile-time constant but a `Matrix` is
heap storage, so every Newton iteration of every cell buys a matrix, a
right-hand side and the factorisation's pivot vector. Measured: 272 bytes per
linear solve at one group, 464 at four, 976 at eight, 3.0 kB at sixteen, which
makes a converged three-iteration cell cost 0.54, 0.93, 1.9 and 5.9 kB
respectively. Wall time is 1.3 to 1.7 times [`ArrowheadSolve`](@ref) at four to
sixteen groups, and 4.7 times at one, where the linear solve is most of what a
grey cell does.

Nothing is on this path unless it is asked for: both in-tree solvers default to
`ArrowheadSolve()`, which is exactly zero bytes per cell.
"""
struct DenseSolve <: AbstractJacobianSolve end

"""
    ArrowheadSolve()

Solve the Newton step with [`solve_arrowhead`](@ref): eliminate the diagonal
group block into the gas row, `O(NG)`, no matrix formed, nothing allocated. The
default for [`QuokkaCoupling`](@ref) and [`DirectCoupling`](@ref).

Exact while the coupling is the one in [`coupling_rhs`](@ref), where groups meet
only through the gas. [`AbstractJacobianSolve`](@ref) says what that assumption
is, what checks that it still holds, and what to do the day it stops.
"""
struct ArrowheadSolve <: AbstractJacobianSolve end

"""
    solve_jacobian(linsolve, J00, J0g, Jg0, Jgg, F0, Fg) -> (dx0, dxg)

Solve one Newton step's system, given the four blocks of the Newton matrix and
the two blocks of the right-hand side, in the layout [`solve_arrowhead`](@ref)
draws. Dispatches on an [`AbstractJacobianSolve`](@ref).

The two paths differ once the matrix is singular: the arrowhead treats a zero
group entry as a decoupled group and returns a zero update for it, while an LU
factorisation of the same matrix throws. Neither solver produces one.
[`DirectCoupling`](@ref)'s group entries are `1 + dt chat rho kappa_E >= 1`, and
[`QuokkaCoupling`](@ref) writes a large finite sentinel rather than `-Inf` for a
group of zero optical depth, so its matrix factorises too.
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
The diagonal is the same physical assumption `DirectCoupling` makes, and
[`DenseSolve`](@ref) is the same escape hatch and the same cross-check; see
[`AbstractJacobianSolve`](@ref).

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

"""
    DirectCoupling(; tolerance = 1e-9, max_iterations = 100,
                     linsolve = ArrowheadSolve())

Backward Euler in the natural variables `(E_gas, E_g)`, with the same Newton
iteration as [`QuokkaCoupling`](@ref) and a choice of linear solve.

Nothing about it is hand-derived: the residual is `u - u0 - dt f(u)` with `f` =
[`coupling_rhs`](@ref), and the Newton matrix is `I - dt * J` with `J` =
[`coupling_jacobian`](@ref).

`linsolve` is an [`AbstractJacobianSolve`](@ref) and defaults to the `O(NG)`
[`ArrowheadSolve`](@ref), which is exact for the coupling as it stands and
allocates nothing. [`DenseSolve`](@ref) solves the same system without the
structural assumption behind that speed; it is the reference the default is
checked against, and what to switch to if a term is added that couples groups
directly.

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
               linsolve::AbstractJacobianSolve=ArrowheadSolve()) =
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

        # Arrowhead by default, and by the same assumption `DirectCoupling`
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
        # `O(NG)` elimination. `Jgg` is a vector rather than a matrix only
        # because `J`'s group block is diagonal today (see `CouplingJacobian`);
        # the arrowhead is that fact turned into an algorithm, and `DenseSolve`
        # is the same solve without it, which is what keeps the fact honest.
        dEgas, dErad = solve_jacobian(m.linsolve, J00, J0g, Jg0, Jgg, -F0, -Fg)
        Egas = max(Egas + dEgas, s.cv * s.Tfloor)
        Erad += dErad
    end

    Egas, Erad = apply_energy_floors(s, Egas, Erad)
    return CouplingResult{NG,T}(Egas, Erad, n, converged)
end
