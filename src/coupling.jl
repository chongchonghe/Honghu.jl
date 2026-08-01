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
# `AbstractCoupling`. The solvers live in `coupling_solvers.jl`, and a third one
# backed by an adaptive ODE integrator lives in the OrdinaryDiffEq extension,
# because a general ODE stack is not something this package will depend on.
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
    `(E_gas, E_g)`;
  - `AdaptiveCoupling`        -- an error-controlled ODE solve, available when
    OrdinaryDiffEq (or DifferentialEquations) is loaded.

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
  - `dt`, `c`, `chat`, `a_rad`, `k_B`
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
    chat::T
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
    coupling_rhs(state, Egas, Erad) -> (dEgas, dErad)

The right-hand side of the coupling ODE, without the external source: the rate
at which gas internal energy is lost and each group's radiation energy gained.

Shared by every solver, so all three integrate demonstrably the same equations.
`c/chat` appears on the gas side because the reduced speed of light rescales the
radiation energy but not the matter's response to it.

The opacity is evaluated *here*, at `Egas`'s own temperature, rather than
frozen by the caller. That is what makes [`coupling_jacobian`](@ref) the true
derivative of this function when `kappa` depends on temperature.
"""
@inline function coupling_rhs(s::CouplingState{NG,M,T}, Egas, Erad::SVec{NG,T}) where {NG,M,T}
    Tgas = gas_temperature_of(s, Egas)
    fourPiBoverC = thermal_emission(s, Tgas)
    kappaP, kappaE = coupling_opacity(s, Tgas)
    rate = (s.chat * s.rho) * (mul(kappaP, fourPiBoverC) - mul(kappaE, Erad))
    return (-(s.c / s.chat) * sumvec(rate), rate)
end

"""
    CouplingJacobian{NG,T}

The exact Jacobian of [`coupling_rhs`](@ref), in four blocks:

  - `gas_gas` : `d(dE_gas/dt)/dE_gas`
  - `gas_rad` : `d(dE_gas/dt)/dE_g`, one entry per group
  - `rad_gas` : `d(dE_g/dt)/dE_gas`, one entry per group
  - `rad_rad` : `d(dE_g/dt)/dE_g`, *diagonal* -- group `g`'s rate sees no other
    group's energy, only the gas temperature they all share

That structure is why the linear solves in `coupling_solvers.jl` are arrowheads
and cost `O(NG)` rather than `O(NG^3)`, and it is a property of the physics
rather than of any solver, which is why it is stated here.
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

With `rate_g = chat rho (kappa_P,g B_g(T) - kappa_E,g E_g)` and `T = E_gas / cv`,

    d rate_g / d E_gas = chat rho (dkappaP_g B_g + kappaP_g dB_g/dT
                                   - dkappaE_g E_g) / cv
    d rate_g / d E_g   = -chat rho kappaE_g

and the gas rows follow from `dE_gas/dt = -(c/chat) sum_g rate_g`.

The `dkappa/dT` terms are what make this *exact* rather than merely
linear-in-`E`; they come from the opacity law through
[`coupling_opacity_dT`](@ref) and are zero for any law that does not depend on
temperature. Including them cuts `DirectCoupling`'s and `QuokkaCoupling`'s
Newton iterations by 25 to 40 per cent wherever the coupling is neither
transparent nor already in equilibrium, and by nothing at all where it is: the
correction is proportional to `B_g - E_g`.

Shared by every solver: `DirectCoupling`'s Newton matrix is `I - dt * J`
verbatim, and the adaptive solver hands `J` straight to `ODEFunction` rather
than paying `1 + NG` extra right-hand side evaluations to difference one.
"""
@inline function coupling_jacobian(s::CouplingState{NG,M,T}, Egas,
                                   Erad::SVec{NG,T}) where {NG,M,T}
    Tgas = gas_temperature_of(s, Egas)
    fourPiBoverC = thermal_emission(s, Tgas)
    dB_dT = thermal_emission_dT(s, Tgas)
    kappaP, kappaE = coupling_opacity(s, Tgas)
    dkappaP, dkappaE = coupling_opacity_dT(s, Tgas)

    # dT/dE_gas = 1/cv, the chain rule's only step: everything below is a
    # derivative with respect to T.
    drate_dgas = (s.chat * s.rho / s.cv) *
                 (mul(dkappaP, fourPiBoverC) + mul(kappaP, dB_dT) - mul(dkappaE, Erad))
    drate_drad = -(s.chat * s.rho) * kappaE

    cscale = s.c / s.chat
    return CouplingJacobian{NG,T}(-cscale * sumvec(drate_dgas), -cscale * drate_drad,
                                  drate_dgas, drate_drad)
end

"""
    apply_energy_floors(state, Egas, Erad) -> (Egas, Erad)

Project the solved state onto the admissible set by moving energy between the
gas and the radiation rather than creating it, so the total is unchanged.
"""
@inline function apply_energy_floors(s::CouplingState{NG,M,T}, Egas,
                                     Erad::SVec{NG,T}) where {NG,M,T}
    cscale = s.c / s.chat
    Emin = s.cv * s.Tfloor
    ef = s.Efloor

    # Lift every group to the floor, and charge the total lifted to the gas, so
    # the repair moves energy rather than creating it. Both `ntuple` closures
    # below capture only values they do not assign; a running total updated
    # inside one would be boxed and would allocate on every cell.
    lifted = SVec{NG,T}(ntuple(g -> max(Erad[g], ef), Val(NG))...)
    paid = cscale * (sumvec(lifted) - sumvec(Erad))
    eg = Egas - paid

    eg >= Emin && return (eg, lifted)

    # The gas cannot pay. Take the shortfall back from the radiation, drawn from
    # every group in proportion to what it holds above its own floor, so the
    # spectrum keeps its shape instead of one group being singled out. Drawing
    # proportionally also avoids an argmax, whose runtime index into the group
    # tuple was worth 16 bytes a cell.
    headroom = sumvec(lifted) - NG * ef
    need = (Emin - eg) / cscale
    frac = headroom > 0 ? min(need / headroom, one(T)) : zero(T)
    back = SVec{NG,T}(ntuple(g -> lifted[g] - frac * (lifted[g] - ef), Val(NG))...)
    return (max(eg + cscale * frac * headroom, Emin), back)
end
