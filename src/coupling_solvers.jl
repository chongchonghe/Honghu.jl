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
is the energy handed from gas to radiation group `g` over the step. The
formulation Quokka runs; the default is [`BracketedCoupling`](@ref).

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

function solve_coupling(m::QuokkaCoupling, s::CouplingState{NG,M,T,O,NoDust}) where {NG,M,T,O}
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
    BracketedCoupling(; tolerance = 1e-9, max_iterations = 100, root_finder = Brent())

Backward Euler in the natural variables `(E_gas, E_g)`, reduced to **one scalar
equation** and handed to a bracketing root-finder. The default.

The solve is two separate statements, and this is the whole of it:

  - **The equation.** The group block is linear in `E_g` at fixed gas
    temperature, so [`group_energy_values`](@ref) solves it in closed form and
    what is left is [`energy_balance`](@ref): the gas energy at which the cell's
    books balance. One unknown, no `1 + NG` system, no Jacobian, no linear solve.
  - **The algorithm.** Any bracketing method written against SciMLBase's
    `IntervalNonlinearProblem` interface, as `root_finder`. The default is
    BracketingNonlinearSolve's `Brent`: Brent's method as Brent (1973) wrote it,
    with the minimum step and one evaluation per iteration, both since 1.12.8
    (SciML/NonlinearSolve.jl#1340). Over the 240-cell sweep of
    `docs/coupling-new-method.md` it needs 7.7 evaluations of the equation on
    average and 12 at worst, with no failures. The package's other methods plug
    in unchanged -- `ModAB()` is close behind, `Bisection()` is the reference,
    slow and impossible to get wrong on a valid bracket. Before 1.12.8 its
    `Brent` had neither property and took 20.1 evaluations on average and 70 at
    worst, and an in-tree copy of the method, `ZBrent`, stood in for it.

Derivative-free methods are the right fit here even though the derivative is
known in closed form: each evaluation of `G` costs a Planck integral, and its
derivative costs another (`planck_fractions_dT`). A derivative-free method that
needs as few evaluations as a Newton iteration therefore wins outright, and
`Brent`'s 7.7 are fewer than the 9 value-and-derivative evaluations of the
hand-written Newton this replaced.

**Why it is the default: it cannot walk away from the root.** A bracket always
exists (`G <= 0` at the temperature floor, `G >= 0` once the gas holds the whole
cell's energy), and a bracketing method never leaves it. The unsafeguarded
`1 + NG` iteration has no such guard and does not merely overshoot -- with a
steeply temperature-dependent opacity its Newton *direction* can be wrong.
Measured over 240 cells spanning three opacity laws and `tau_cell` from 1e-4 to
1e12, this solver fails none and [`QuokkaCoupling`](@ref) fails 75.

Three things the solve does that a generic root-finder does not, each for a
reason:

  - **The bracket is built by marching outward from the old gas energy**, by
    factors of two, not taken as `[cv Tfloor, E_tot]`. Where the opacity makes
    `G` non-monotone the step can have several roots, and a bracketing method
    returns *one* of them without saying which. On the sweep's one such cell
    (`kappa ~ T^2`, gas at 0.1 and radiation at 3, three roots at 0.110, 1.17
    and 2.52) the error-controlled ODE ends at 0.109, and which root a method
    returned from the wide bracket depended on the method. Marching from the old
    state isolates the root continuously connected to where the cell started,
    unless two roots lie within a factor of two of each other.
  - **The tolerance is converted.** Every method in the package stops when the
    bracket is narrower than `abstol` in `E_gas`; this solver promises `|G| <=
    tolerance * E_tot`, and `G` is exactly the cell's energy-conservation error.
    The two differ by `dG/dE_gas`, which reaches `1 + (c/chat) 4 a T^3 / cv` --
    thousands, in a radiation-dominated cell. So after the first solve the
    residual is checked, and if it is too large the solve is repeated on the
    library's own final bracket with `abstol` divided by `dG/dE_gas` there, which
    is effectively constant across so narrow a bracket. Convergence is judged on
    the residual, not on the library's return code.
  - **`iterations` counts evaluations this solver makes itself** -- the bracket
    march -- plus one per call to the root-finder. The library does not report its
    own evaluation count, and counting it through the equation would allocate 16
    bytes a cell, which this solver does not.

The reduction inherits one assumption from [`group_energies`](@ref): groups
exchange energy with the gas and never directly with each other. That is the same
assumption [`ArrowheadSolve`](@ref) makes; a term coupling groups directly would
make the group block a matrix and there would be no closed form to eliminate.
"""
struct BracketedCoupling{T<:AbstractFloat,A} <: AbstractCoupling
    tolerance::T
    max_iterations::Int
    root_finder::A
end

BracketedCoupling(; tolerance=1e-9, max_iterations::Integer=100, root_finder=Brent()) =
    BracketedCoupling(float(tolerance), Int(max_iterations), root_finder)

function solve_coupling(m::BracketedCoupling, s::CouplingState{NG,M,T,O,NoDust}) where {NG,M,T,O}
    Etot = abs(s.Egas + dot(gas_scale(s), s.Erad + s.src))
    atol = m.tolerance * Etot

    # The equation. The state rides along as the root-finder's parameter rather
    # than being captured, so this closes over nothing and cannot be boxed.
    G = (Egas, s) -> energy_balance(s, Egas, group_energy_values(s, Egas))

    Emin = s.cv * s.Tfloor
    lo, hi, nbracket, found = _energy_bracket(G, s, max(s.Egas, Emin), Emin)
    if !found
        Egas, Erad = apply_energy_floors(s, s.Egas, s.Erad + s.src)
        return CouplingResult{NG,T}(Egas, Erad, nbracket, false)
    end

    # The algorithm.
    sol = solve(IntervalNonlinearProblem{false}(G, (lo, hi), s), m.root_finder;
                abstol=atol, maxiters=m.max_iterations)
    Egas = sol.u
    Erad, dErad = group_energies(s, Egas)
    residual = energy_balance(s, Egas, Erad)
    nsolve = 1

    # The library stopped on bracket width in `E_gas`; the promise is on `G`.
    if abs(residual) > atol
        dG = max(abs(one(T) + dot(gas_scale(s), dErad)), one(T))
        sol = solve(IntervalNonlinearProblem{false}(G, (sol.left, sol.right), s),
                    m.root_finder; abstol=atol / dG, maxiters=m.max_iterations)
        Egas = sol.u
        Erad = group_energy_values(s, Egas)
        residual = energy_balance(s, Egas, Erad)
        nsolve = 2
    end

    converged = abs(residual) <= atol
    Egas, Erad = apply_energy_floors(s, Egas, Erad)
    return CouplingResult{NG,T}(Egas, Erad, nbracket + nsolve, converged)
end

# --- with dust: one equation in the dust temperature ---------------------------

# With `BateKetoDust` the radiation couples to the dust, and the unknown is the
# dust temperature: `dust_energy_balance` gives the group energies and the gas
# energy from it in closed form, and leaves one equation. Same bracket march,
# same root-finder, same tolerance -- but a different convergence test, because
# that equation's residual amplifies round-off by the collisional coupling
# strength and the state across the final bracket does not. The derivation, the
# bracket and the measurements are in `docs/coupling-new-method-dust.md`.
function solve_coupling(m::BracketedCoupling,
                        s::CouplingState{NG,M,T,O,D}) where {NG,M,T,O,D<:BateKetoDust}
    Etot = abs(s.Egas + dot(gas_scale(s), s.Erad + s.src))
    atol = m.tolerance * Etot

    H = (Td, s) -> first(dust_energy_balance(s, Td))

    # From the old gas temperature, as Quokka starts its dust-temperature solve.
    # Where the dust balance has several roots this selects the one connected to
    # dust as warm as the gas; storing `T_d` between steps would do better.
    lo, hi, nbracket, found = _energy_bracket(H, s, gas_temperature_of(s, s.Egas), s.Tfloor)
    if !found
        Egas, Erad = apply_energy_floors(s, s.Egas, s.Erad + s.src)
        return CouplingResult{NG,T}(Egas, Erad, nbracket, false, gas_temperature_of(s, s.Egas))
    end

    # The root-finder stops on bracket width in `T_d`. Open with a relative width
    # in temperature, then test what the promise is about: the gas energy -- and,
    # by conservation, the radiation energy -- across the final bracket.
    sol = solve(IntervalNonlinearProblem{false}(H, (lo, hi), s), m.root_finder;
                abstol=m.tolerance * hi / 2, maxiters=m.max_iterations)
    width = _dust_state_width(s, sol.left, sol.right)
    nsolve = 1
    # Convert by the slope across the current bracket, with a factor 4 margin:
    # a factor 2 because the methods stop on the *half*-width, and another for
    # the slope changing across a wide first bracket.
    while width > atol && nsolve < 4
        slope = width / max(sol.right - sol.left, eps(sol.u))
        sol = solve(IntervalNonlinearProblem{false}(H, (sol.left, sol.right), s),
                    m.root_finder; abstol=atol / (4 * slope), maxiters=m.max_iterations)
        width = _dust_state_width(s, sol.left, sol.right)
        nsolve += 1
    end

    Td = sol.u
    _, Egas, Erad = dust_energy_balance(s, Td)
    Egas, Erad = apply_energy_floors(s, Egas, Erad)
    # `iterations`: the march, plus per solve one for the call and two for the
    # width test.
    return CouplingResult{NG,T}(Egas, Erad, nbracket + 3 * nsolve, width <= atol, Td)
end

# How far apart the gas energies at the two ends of a bracket in `T_d` are: a
# bound on the error of the returned state, since the root lies inside.
@inline _dust_state_width(s, lo, hi) =
    abs(dust_energy_balance(s, lo)[2] - dust_energy_balance(s, hi)[2])

"""
    _energy_bracket(G, state, x0, xmin) -> (lo, hi, evaluations, found)

A bracket of an increasing `G` around the root continuously connected to `x0`:
start there, and step by factors of two in the direction the sign of `G` says the
root lies until the sign changes, never below `xmin`.

For the dust-free equation `x0` is the old gas energy and `xmin = cv Tfloor`.
Downwards the walk stops at the temperature floor, where `G <= 0` whenever the
state is admissible; upwards it stops once the gas holds more than the cell's
total energy, where `G >= 0` whenever no group has been handed a negative
starting energy by the lagged work term. For the dust equation `x0` is the old
gas temperature and `xmin = Tfloor`; see [`dust_energy_balance`](@ref). Neither
bound is trusted blindly: if the walk runs out without a sign change the cell is
reported unconverged rather than guessed at.

`G::F` rather than `G`, so the walk specialises on the equation and allocates
nothing -- a function argument that is only passed through is not specialised on
by default, and the measured cost of that was 2.8 kB a cell.
"""
@inline function _energy_bracket(G::F, s::CouplingState{NG,M,T}, x0, Emin) where {F,NG,M,T}
    x = x0
    g = G(x, s)
    n = 1
    iszero(g) && return (x, x, n, true)
    if g < 0
        while n < 200
            y = 2x
            n += 1
            G(y, s) >= 0 && return (x, y, n, true)
            x = y
        end
    else
        while n < 200
            y = max(x / 2, Emin)
            n += 1
            G(y, s) <= 0 && return (y, x, n, true)
            y == Emin && return (y, x, n, false)
            x = y
        end
    end
    return (x, x, n, false)
end
