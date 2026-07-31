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
  - `kappaP`, `kappaE` : Planck- and energy-mean opacity per group
  - `dt`, `c`, `chat`, `a_rad`, `k_B`
  - `groups`    : the frequency group structure, for the Planck fractions
  - `Tfloor`, `Efloor` : admissibility floors on gas temperature and group energy

`kappaP` and `kappaE` are held fixed across the solve at their values from the
old temperature. Quokka does the same; it changes how fast the iteration
converges, not what it converges to.
"""
struct CouplingState{NG,M,T<:AbstractFloat}
    Egas::T
    Erad::SVec{NG,T}
    src::SVec{NG,T}
    work::SVec{NG,T}
    rho::T
    cv::T
    kappaP::SVec{NG,T}
    kappaE::SVec{NG,T}
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
"""
@inline function coupling_rhs(s::CouplingState{NG,M,T}, Egas, Erad::SVec{NG,T}) where {NG,M,T}
    Tgas = gas_temperature_of(s, Egas)
    fourPiBoverC = thermal_emission(s, Tgas)
    rate = (s.chat * s.rho) *
           (mul(s.kappaP, fourPiBoverC) - mul(s.kappaE, Erad))
    return (-(s.c / s.chat) * sumvec(rate), rate)
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
