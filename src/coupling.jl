# How the matter-radiation energy exchange is solved, per cell.
#
# The physics is one stiff ODE system per cell, in the gas internal energy and
# the NG group radiation energies:
#
#   dE_gas/dt = -chat rho sum_g (kappa_P,g 4 pi B_g / c - kappa_E,g E_g)
#   dE_g  /dt = +chat rho (kappa_P,g 4 pi B_g / c - kappa_E,g E_g) + S_g
#
# with 4 pi B_g / c = a_rad T^4 f_g(T), where f_g is the fraction of a Planck
# spectrum in group g. It is stiff because rho kappa chat dt reaches 10^8 in an
# optically thick cell, and its equilibrium is a T^4 balance rather than a fixed
# point of the linearisation, so an explicit step is hopeless.
#
# This file defines the *interface*: a per-cell `CouplingState` carrying
# everything a solver needs, and `solve_coupling` dispatching on an
# `AbstractCoupling`. The two hand-written solvers live in `coupling_solvers.jl`,
# and the two backed by OrdinaryDiffEq -- `DirectCoupling`'s default backward-Euler
# stepper and the error-controlled reference solver -- live in `coupling_ode.jl`.
#
# Keeping the state explicit costs one small immutable struct per cell and buys
# the thing this code exists for: a new integrator is a new subtype and a new
# method, not an edit to the middle of a triple loop.
#
# --- what is physics and what is a solver's own business ----------------------
#
# `coupling_rhs` and `coupling_jacobian` below are the single source of truth for
# the equations: every solver's residual and every solver's linearisation come
# from these two, so a change to the physics is a change in one place. The
# Jacobian is *exact*, including the opacity's own temperature dependence, which
# is what a Newton iteration needs to converge quadratically -- and which the
# right-hand side must therefore also see, so the opacity law travels on the
# state rather than a frozen pair of numbers.
#
# Three things are deliberately *not* in the right-hand side, because they are
# not terms of the ODE at all. They are properties of a discretisation, and each
# has one home:
#
#   optically-thin shortcut     `QuokkaCoupling`, in `coupling_solvers.jl`.
#                               `tau -> 0` makes the group's residual unbalanceable
#                               by any transfer, so the solver stops asking. The
#                               ODE has no such problem: its trajectory is simply
#                               flat, and the adaptive solver integrates it.
#   equilibrium-overshoot guard `QuokkaCoupling`, likewise. A globalisation of
#                               Newton -- a step-length rule, not a term.
#   `apply_energy_floors`       here, applied by every solver to its *answer*.
#                               Projection onto the admissible set, done by moving
#                               energy between gas and radiation so the total is
#                               untouched. It is not integrated; it is applied once.
#
# Putting any of them in `coupling_rhs` would make the three solvers integrate
# different problems, which is precisely what the interface exists to prevent.

"""
    AbstractCoupling

Supertype for matter-radiation coupling solvers. A subtype implements

    solve_coupling(method, state) -> CouplingResult

taking a [`CouplingState`](@ref) for one cell and returning the updated gas
internal energy and group radiation energies.

Three are provided:

  - [`QuokkaCoupling`](@ref)  -- backward Euler in Quokka's replaced base
    variables `(E_gas, R_g)`, the default;
  - [`DirectCoupling`](@ref)  -- backward Euler in the natural variables
    `(E_gas, E_g)`, by either of two [`AbstractBackwardEuler`](@ref) steppers;
  - [`AdaptiveCoupling`](@ref) -- an error-controlled ODE solve.

They solve the same equations and should agree to their tolerances; that they
do is a test, and the reason the interface exists at all.
"""
abstract type AbstractCoupling end

"""
    CouplingState{NG,T}

Everything the coupling of one cell needs: the old state, the material
properties, and the step.

  - `Egas`      : gas internal energy density at the start of the step
  - `Erad`      : group radiation energy densities at the start of the step
  - `src`       : external radiation energy added to each group over the step
  - `work`      : the lagged `v . F` work term for each group, already times `dt`
  - `rho`, `cv` : density and *volumetric* heat capacity `d(eint)/dT`, which is
    what [`heat_capacity`](@ref) returns, so `T = E_gas / cv` directly
  - `opacity`   : the opacity law, evaluated at whatever temperature the solver
    is currently at (see [`coupling_opacity`](@ref))
  - `dt`, `c`, `a_rad`, `k_B`
  - `chat`      : the reduced speed of light *of each group*. One number per
    group rather than one per cell because a [`AbstractChatLimiter`](@ref) sets
    it from the group's own cell optical depth; with the default
    [`ConstantChat`](@ref) every entry is the same `phys.chat`. The gas side of
    the exchange is never scaled by it -- see [`coupling_rhs`](@ref).
  - `groups`    : the frequency group structure, for the Planck fractions
  - `Tfloor`, `Efloor` : admissibility floors on gas temperature and group energy

The *law* is carried, not a frozen pair of numbers, because the coupling is
solved by Newton's method and the exact Jacobian of an emission term whose
opacity moves with temperature contains `dkappa/dT`. A frozen coefficient would
make that term meaningless -- no finite difference of the residual would show
it -- and the two would then be inconsistent, which is worse for a Rosenbrock
method than no Jacobian at all. Quokka re-evaluates the opacity in every Newton
iteration for the same reason, and so does the grey driver in `source_terms.jl`.
"""
struct CouplingState{NG,M,T<:AbstractFloat,O<:AbstractOpacity}
    Egas::T
    Erad::SVec{NG,T}
    src::SVec{NG,T}
    work::SVec{NG,T}
    rho::T
    cv::T
    opacity::O
    dt::T
    c::T
    chat::SVec{NG,T}
    a_rad::T
    k_B::T
    groups::RadiationGroups{NG,M,T}
    Tfloor::T
    Efloor::T
end

"""
    CouplingResult{NG,T}

What a coupling solver returns: the new gas internal energy, the new group
radiation energies, how many iterations it took, and whether it converged.

`converged = false` is reported rather than thrown so that one stubborn cell in
a large grid shows up in the run's diagnostics instead of killing the run.
"""
struct CouplingResult{NG,T<:AbstractFloat}
    Egas::T
    Erad::SVec{NG,T}
    iterations::Int
    converged::Bool
end

"""
    solve_coupling(method, state) -> CouplingResult

Advance one cell's matter-radiation coupling over `state.dt`.
"""
function solve_coupling end

# --- shared pieces ------------------------------------------------------------

"""
    gas_temperature_of(state, Egas) -> T

Gas temperature from internal energy, floored. `cv` is volumetric, so `E_int = cv T`.
"""
@inline gas_temperature_of(s::CouplingState, Egas) = max(Egas / s.cv, s.Tfloor)

"""
    gas_scale(state) -> SVec{NG}

`c / chat_g`: what one unit of group `g`'s radiation energy costs the gas.

The reduced speed of light scales the radiation side of the exchange and not the
matter side, so every quantity that crosses from one to the other picks this up.
It is a vector because each group may run at its own reduced speed; with the
default [`ConstantChat`](@ref) every entry is the same number and `dot(gas_scale(s), x)`
is `(c/chat) * sum(x)`.
"""
@inline gas_scale(s::CouplingState{NG,M,T}) where {NG,M,T} =
    SVec{NG,T}(ntuple(g -> s.c / s.chat[g], Val(NG))...)

"""
    thermal_emission(state, T) -> SVec{NG}

`4 pi B_g / c = a_rad T^4 f_g(T)`, the equilibrium radiation energy density in
each group at temperature `T`. Sums to `a_rad T^4` exactly, whatever the group
edges, because the Planck fractions do.
"""
@inline function thermal_emission(s::CouplingState{NG,M,T}, Tgas) where {NG,M,T}
    aT4 = s.a_rad * Tgas^4
    return aT4 * planck_fractions(s.groups, Tgas, s.k_B)
end

"""
    coupling_opacity(state, T) -> (kappaP, kappaE)

The Planck- and energy-mean opacity of each group at temperature `T`.

The two coincide, because the multigroup opacity here is piecewise constant
across a group and averaging a constant against any weight returns it (see
`opacity.jl`). They are returned as a pair anyway: this is the one place a model
that separates them would change, and every solver reads the pair rather than
the model.
"""
@inline function coupling_opacity(s::CouplingState{NG,M,T,O}, Tgas) where {NG,M,T,O}
    kap = group_opacities(s.opacity, s.groups, s.rho, Tgas)
    return (kap, kap)
end

"""
    coupling_opacity_dT(state, T) -> (dkappaP_dT, dkappaE_dT)

`d kappa/dT` of each group at temperature `T`: the derivative of the law
[`coupling_opacity`](@ref) evaluates, and what makes
[`coupling_jacobian`](@ref) exact. Zero for an opacity that does not supply one
(see [`dkappa_dT`](@ref)).
"""
@inline function coupling_opacity_dT(s::CouplingState{NG,M,T,O}, Tgas) where {NG,M,T,O}
    dkap = group_opacities_dT(s.opacity, s.groups, s.rho, Tgas)
    return (dkap, dkap)
end

"""
    thermal_emission_dT(state, T) -> SVec{NG}

`d(4 pi B_g / c)/dT = 4 a T^3 f_g + a T^4 f_g'`.

The second term is what moves energy between groups as the gas heats. A grey
problem has `f = 1` and `f' = 0` and this reduces to `4 a T^3`.
"""
@inline function thermal_emission_dT(s::CouplingState{NG,M,T}, Tgas) where {NG,M,T}
    aT4 = s.a_rad * Tgas^4
    f = planck_fractions(s.groups, Tgas, s.k_B)
    df = planck_fractions_dT(s.groups, Tgas, s.k_B)
    return (4 * s.a_rad * Tgas^3) * f + aT4 * df
end

"""
    coupling_coefficients(state, T) -> (emission, absorption)

The two numbers per group that every formulation of the coupling is built from:

  - `emission_g   = rho kappa_P,g(T) 4 pi B_g(T) / c`, the rate at which the gas
    puts energy into group `g`;
  - `absorption_g = rho kappa_E,g(T)`, the coefficient multiplying `E_g` in the
    rate at which the group gives it back.

so that the unscaled exchange rate of group `g` is `emission_g - absorption_g E_g`,
linear in `E_g` and with every temperature dependence in the two coefficients.

**This is the single source of truth for the coupling physics**, and it is stated
as this pair rather than as a right-hand side because the three solvers need two
*different* things from it and must not each assemble their own:

  - [`AdaptiveCoupling`](@ref) integrates the rate, `dE_g/dt = chat_g (emission_g
    - absorption_g E_g)`, which is [`coupling_rhs`](@ref) below.
  - [`BracketedCoupling`](@ref) never forms a rate at all. It uses the *same* two
    coefficients to solve the backward-Euler group block in closed form -- see
    [`group_energies`](@ref) -- because that block is linear in `E_g`, which is
    exactly the property this pair makes visible.

Writing them as a product rather than as `(kappa_P, B)` separately is also the
more general statement: an opacity model that defines `kappa_P` by an integral
against `B` (Quokka's `PPL_opacity_full_spectrum`) has an emission rate but no
meaningful `kappa_P` on its own. [`thermal_emission`](@ref) stays available for
[`QuokkaCoupling`](@ref), whose Jacobian needs `kappa_P/kappa_E` and so cannot
work with the product.
"""
@inline function coupling_coefficients(s::CouplingState{NG,M,T}, Tgas) where {NG,M,T}
    fourPiBoverC = thermal_emission(s, Tgas)
    kappaP, kappaE = coupling_opacity(s, Tgas)
    return (s.rho * mul(kappaP, fourPiBoverC), s.rho * kappaE)
end

"""
    coupling_coefficients_dT(state, T) -> (d emission/dT, d absorption/dT)

The temperature derivatives of [`coupling_coefficients`](@ref), and the reason
[`coupling_jacobian`](@ref) is exact rather than merely linear in `E`:

    d emission_g   / dT = rho (dkappa_P,g/dT B_g + kappa_P,g dB_g/dT)
    d absorption_g / dT = rho dkappa_E,g/dT

Both `dkappa/dT` terms are zero for an opacity that does not supply one (see
[`dkappa_dT`](@ref)), and the emission derivative then reduces to the familiar
`rho kappa_P 4 a T^3` plus the Planck fractions' own drift between groups.
"""
@inline function coupling_coefficients_dT(s::CouplingState{NG,M,T}, Tgas) where {NG,M,T}
    fourPiBoverC = thermal_emission(s, Tgas)
    dB_dT = thermal_emission_dT(s, Tgas)
    kappaP, kappaE = coupling_opacity(s, Tgas)
    dkappaP, dkappaE = coupling_opacity_dT(s, Tgas)
    return (s.rho * (mul(dkappaP, fourPiBoverC) + mul(kappaP, dB_dT)), s.rho * dkappaE)
end

"""
    coupling_rhs(state, Egas, Erad) -> (dEgas, dErad)

The right-hand side of the coupling ODE, without the external source: the rate
at which gas internal energy is lost and each group's radiation energy gained.

Shared by every solver, so all three integrate demonstrably the same equations.

The unscaled exchange rate `rho (kappa_P B_g - kappa_E E_g)` is formed first and
then scaled two different ways: the *radiation* side by that group's `chat`, and
the *gas* side by the full `c`. That asymmetry is the reduced speed of light
approximation itself -- and writing it this way is what lets each group carry its
own `chat` without the gas equation acquiring a group index it has no business
having.

The opacity is evaluated *here*, at `Egas`'s own temperature, rather than
frozen by the caller. That is what makes [`coupling_jacobian`](@ref) the true
derivative of this function when `kappa` depends on temperature.
"""
@inline function coupling_rhs(s::CouplingState{NG,M,T}, Egas, Erad::SVec{NG,T}) where {NG,M,T}
    Tgas = gas_temperature_of(s, Egas)
    emis, absorp = coupling_coefficients(s, Tgas)
    exch = emis - mul(absorp, Erad)
    return (-s.c * sumvec(exch), mul(s.chat, exch))
end

"""
    group_energies(state, Egas) -> (Erad, dErad_dEgas)

**Part 1 of the reduction.** The group radiation energies that solve the
backward-Euler group block exactly at this gas energy, and their derivative with
respect to it.

The group block is

    (E_g - E_g^0 - S_g - W_g) = dt chat_g (emission_g(T) - absorption_g(T) E_g) ,

which is *linear* in `E_g` at fixed `T`, and in which no group appears in
another group's equation -- they meet only through the gas temperature. So it is
not solved by iteration at all:

    E_g(T) = (rad0_g + dt chat_g emission_g) / (1 + dt chat_g absorption_g),
    rad0_g = E_g^0 + S_g + W_g .

With `kappa_P = kappa_E` that is a convex combination of where the group started
and the Planck value at the gas temperature, weighted by the optical depth of
the step: `dt chat rho kappa -> 0` leaves the group alone, `-> infinity` puts it
in equilibrium with the gas. All of the stiffness is in that one weight, and it
is handled in closed form rather than by a Newton iteration, which is why
[`BracketedCoupling`](@ref) has no linear algebra in it.

The derivative follows by differentiating the same expression,

    dE_g/dT = dt chat_g (d emission_g/dT - d absorption_g/dT E_g)
              / (1 + dt chat_g absorption_g) ,

and `dE_g/dE_gas` is that over `cv`. It is returned alongside rather than as a
separate function because every caller that wants one wants both, and the two
share the coefficient evaluation.

This is a property of the *backward-Euler step*, not of the coupling ODE: the
group equations are linear in `E_g` with time-dependent coefficients, so along a
trajectory they integrate to a convolution over the gas history and not to any
algebraic `E_g(T)`. [`AdaptiveCoupling`](@ref) therefore cannot use this and
integrates the full `1 + NG` system.
"""
@inline function group_energies(s::CouplingState{NG,M,T}, Egas) where {NG,M,T}
    Tgas = gas_temperature_of(s, Egas)
    emis, absorp = coupling_coefficients(s, Tgas)
    demis, dabsorp = coupling_coefficients_dT(s, Tgas)
    cdt = s.dt * s.chat
    rad0 = s.Erad + s.src + s.work
    den = SVec{NG,T}(ntuple(g -> one(T) + cdt[g] * absorp[g], Val(NG))...)
    Erad = SVec{NG,T}(ntuple(g -> (rad0[g] + cdt[g] * emis[g]) / den[g], Val(NG))...)
    dErad = SVec{NG,T}(ntuple(g -> cdt[g] * (demis[g] - dabsorp[g] * Erad[g]) /
                                  (s.cv * den[g]), Val(NG))...)
    return (Erad, dErad)
end

"""
    energy_balance(state, Egas, Erad) -> G

**Part 2 of the reduction.** The cell's total-energy residual,

    G = (E_gas - gas0) + sum_g (c/chat_g) (E_g - rad0_g) ,
    gas0 = E_gas^0 - sum_g (c/chat_g) W_g ,   rad0_g = E_g^0 + S_g + W_g ,

which expands to `E_gas + sum_g (c/chat_g) E_g` minus the same combination at
the start of the step -- new total energy minus old. The lagged work term
cancels out of it completely and survives only inside `rad0_g`.

It contains **no physics**: no opacity, no Planck function, nothing but the
`c/chat_g` weighting that the reduced speed of light imposes on what one unit of
group energy costs the gas. It is linear in its arguments. That is what makes it
shareable, and it is shared exactly -- [`QuokkaCoupling`](@ref)'s gas residual
`(E_gas - E_gas^0) + (c/chat) sum_g R_g` reduces to this identically once its
group residual vanishes, work term included.

**It is the residual of the system only where the group block is solved.**
Writing `F_0` for the gas residual and `F_g` for the group ones,

    F_0 = G - sum_g (c/chat_g) F_g ,

so `G` and `F_0` agree exactly when every `F_g = 0` and not otherwise. Hand this
an arbitrary `Erad` and it returns a number that vanishes in the wrong place;
hand it [`group_energies`](@ref)'s output and it is the scalar equation the
whole backward-Euler step reduces to. That pairing is the contract between the
two parts, and [`BracketedCoupling`](@ref) is the only thing that relies on it.
"""
@inline function energy_balance(s::CouplingState{NG,M,T}, Egas,
                                Erad::SVec{NG,T}) where {NG,M,T}
    cs = gas_scale(s)
    return (Egas - (s.Egas - dot(cs, s.work))) + dot(cs, Erad - (s.Erad + s.src + s.work))
end

"""
    CouplingJacobian{NG,T}

The exact Jacobian of [`coupling_rhs`](@ref), in four blocks:

  - `gas_gas` : `d(dE_gas/dt)/dE_gas`
  - `gas_rad` : `d(dE_gas/dt)/dE_g`, one entry per group
  - `rad_gas` : `d(dE_g/dt)/dE_gas`, one entry per group
  - `rad_rad` : `d(dE_g/dt)/dE_g`, *diagonal* -- group `g`'s rate sees no other
    group's energy, only the gas temperature they all share

That structure is what the arrowhead elimination in `coupling_solvers.jl`
exploits to cost `O(NG)` rather than `O(NG^3)`. It is a property of the physics
rather than of any solver, which is why it is stated here -- and why it is an
assumption rather than a guarantee: a term that moved energy between groups
directly (Compton redistribution, overlapping lines) would make `rad_rad` a
matrix, and every solver that assumed otherwise would be quietly wrong. See
[`AbstractJacobianSolve`](@ref), which is where that choice is made explicit.
"""
struct CouplingJacobian{NG,T<:AbstractFloat}
    gas_gas::T
    gas_rad::SVec{NG,T}
    rad_gas::SVec{NG,T}
    rad_rad::SVec{NG,T}
end

"""
    coupling_jacobian(state, Egas, Erad) -> CouplingJacobian

The exact Jacobian of [`coupling_rhs`](@ref) at `(Egas, Erad)`.

With `exch_g = rho (kappa_P,g B_g(T) - kappa_E,g E_g)` and `T = E_gas / cv`,

    d exch_g / d E_gas = rho (dkappaP_g B_g + kappaP_g dB_g/dT
                              - dkappaE_g E_g) / cv
    d exch_g / d E_g   = -rho kappaE_g

The radiation rows are these times that group's `chat`, and the gas rows are
`-c` times their sum, exactly as in [`coupling_rhs`](@ref).

The `dkappa/dT` terms are what make this *exact* rather than merely
linear-in-`E`; they come from the opacity law through
[`coupling_opacity_dT`](@ref) and are zero for any law that does not depend on
temperature. Including them cuts `DirectCoupling`'s and `QuokkaCoupling`'s
Newton iterations by 25 to 40 per cent wherever the coupling is neither
transparent nor already in equilibrium, and by nothing at all where it is: the
correction is proportional to `B_g - E_g`.

Shared by every solver: [`NewtonStepper`](@ref)'s Newton matrix is `I - dt * J`
verbatim, and the two OrdinaryDiffEq-backed solvers hand `J` straight to
`ODEFunction` rather than paying `1 + NG` extra right-hand side evaluations to
difference one -- which they could not do anyway, since an `SVec`'s element type
is fixed and no dual number can flow through [`coupling_rhs`](@ref).
"""
@inline function coupling_jacobian(s::CouplingState{NG,M,T}, Egas,
                                   Erad::SVec{NG,T}) where {NG,M,T}
    Tgas = gas_temperature_of(s, Egas)
    _, absorp = coupling_coefficients(s, Tgas)
    demis_dT, dabsorp_dT = coupling_coefficients_dT(s, Tgas)

    # dT/dE_gas = 1/cv, the chain rule's only step: everything below is a
    # derivative with respect to T.
    dexch_dgas = (demis_dT - mul(dabsorp_dT, Erad)) / s.cv
    dexch_drad = -absorp

    return CouplingJacobian{NG,T}(-s.c * sumvec(dexch_dgas), -s.c * dexch_drad,
                                  mul(s.chat, dexch_dgas), mul(s.chat, dexch_drad))
end

"""
    apply_energy_floors(state, Egas, Erad) -> (Egas, Erad)

Project the solved state onto the admissible set by moving energy between the
gas and the radiation rather than creating it, so the total is unchanged.
"""
@inline function apply_energy_floors(s::CouplingState{NG,M,T}, Egas,
                                     Erad::SVec{NG,T}) where {NG,M,T}
    Emin = s.cv * s.Tfloor
    ef = s.Efloor

    # Lift every group to the floor, and charge the total lifted to the gas, so
    # the repair moves energy rather than creating it. Every group is converted
    # to gas energy at its *own* `c/chat`. Both `ntuple` closures below capture
    # only values they do not assign; a running total updated inside one would
    # be boxed and would allocate on every cell.
    lifted = SVec{NG,T}(ntuple(g -> max(Erad[g], ef), Val(NG))...)
    paid = s.c * sumvec(divide(lifted - Erad, s.chat))
    eg = Egas - paid

    eg >= Emin && return (eg, lifted)

    # The gas cannot pay. Take the shortfall back from the radiation, drawn from
    # every group in proportion to what it holds above its own floor, so the
    # spectrum keeps its shape instead of one group being singled out. Drawing
    # proportionally also avoids an argmax, whose runtime index into the group
    # tuple was worth 16 bytes a cell. `headroom` is measured on the gas side,
    # which is the side the shortfall is on.
    headroom = s.c * sumvec(divide(lifted - ef, s.chat))
    frac = headroom > 0 ? min((Emin - eg) / headroom, one(T)) : zero(T)
    back = SVec{NG,T}(ntuple(g -> lifted[g] - frac * (lifted[g] - ef), Val(NG))...)
    return (max(eg + frac * headroom, Emin), back)
end
