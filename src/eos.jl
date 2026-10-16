# Equation of state.
#
# All energies here are *volumetric* (erg cm^-3), matching the conserved
# variables, except `cv` which is specific (erg g^-1 K^-1).

"""
    AbstractEOS

Supertype for equations of state.

A concrete subtype `E` must implement

  - [`pressure`](@ref)`(eos, rho, eint)`
  - [`internal_energy`](@ref)`(eos, rho, p)`
  - [`soundspeed`](@ref)`(eos, rho, p)`
  - [`adiabatic_index`](@ref)`(eos)`
  - [`temperature`](@ref)`(eos, rho, eint)`
  - [`eint_from_temperature`](@ref)`(eos, rho, T)`
  - [`heat_capacity`](@ref)`(eos, rho, T)`

Only the first four are needed for pure hydrodynamics; the rest are used by the
radiation-matter coupling. `test/test_extensions.jl` pins this list: it defines
an EOS implementing exactly these methods and runs a simulation with it.

!!! warning "The built-in HLLC solver assumes a gamma-law gas"
    [`HLLC`](@ref) uses `adiabatic_index(eos)` as a scalar in the Roe-averaged
    sound speed and in Rider's wave-speed correction, so it is *structurally* a
    gamma-law solver. Defining `adiabatic_index` on a genuinely non-ideal EOS
    silences the `MethodError` and then silently uses an effective constant
    gamma, which is worse than failing. Such an EOS needs its own
    [`AbstractRiemannSolver`](@ref); [`LLF`](@ref) is a good starting point,
    since it needs only the sound speed.
"""
abstract type AbstractEOS end

"""
    IdealGas(; gamma = 5/3, cv = 1.0)

Ideal gas with constant adiabatic index `gamma` and constant specific heat at
constant volume `cv` (per unit mass).

`gamma` and `cv` are independent inputs so that the code stays unit-agnostic:
in code units the usual choice is `cv = 1`, while in CGS you want
`cv = `[`cv_from_mu`](@ref)`(mu, gamma)`.

`cv` only enters through the gas temperature, i.e. only when radiation is
enabled; pure hydrodynamics is unaffected by it.
"""
struct IdealGas{T<:AbstractFloat} <: AbstractEOS
    gamma::T
    cv::T
end

IdealGas(; gamma=5 / 3, cv=1.0) = IdealGas(promote(float(gamma), float(cv))...)

"""
    cv_from_mu(mu, gamma; k_B = CGS.k_B, m_u = CGS.m_u)

Specific heat at constant volume of an ideal gas of mean molecular weight `mu`,
`cv = k_B / (mu m_u (gamma - 1))`. Defaults give CGS units (erg g⁻¹ K⁻¹).
"""
cv_from_mu(mu, gamma; k_B=CGS.k_B, m_u=CGS.m_u) = k_B / (mu * m_u * (gamma - 1))

"""
    pressure(eos, rho, eint)

Thermal pressure from mass density `rho` and internal energy *density* `eint`.
"""
@inline pressure(eos::IdealGas, rho, eint) = (eos.gamma - 1) * eint

"""
    internal_energy(eos, rho, p)

Internal energy *density* from mass density and pressure.
"""
@inline internal_energy(eos::IdealGas, rho, p) = p / (eos.gamma - 1)

"""
    soundspeed(eos, rho, p)

Adiabatic sound speed.
"""
@inline soundspeed(eos::IdealGas, rho, p) = sqrt(eos.gamma * p / rho)

"""
    temperature(eos, rho, eint)

Gas temperature from mass density and internal energy density.
"""
@inline temperature(eos::IdealGas, rho, eint) = eint / (rho * eos.cv)

"""
    eint_from_temperature(eos, rho, T)

Internal energy density from mass density and temperature.
"""
@inline eint_from_temperature(eos::IdealGas, rho, T) = rho * eos.cv * T

"""
    heat_capacity(eos, rho, T)

Volumetric heat capacity `∂(eint)/∂T` at fixed density.
"""
@inline heat_capacity(eos::IdealGas, rho, T) = rho * eos.cv

"""
    adiabatic_index(eos)

Adiabatic index `gamma`.
"""
@inline adiabatic_index(eos::IdealGas) = eos.gamma
