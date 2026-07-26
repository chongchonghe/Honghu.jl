# Opacities for the grey (single-group) radiation module.
#
# Three means are distinguished, following Quokka:
#   kappa_P  Planck mean      -- weights thermal emission (4 pi B / c)
#   kappa_E  energy mean      -- weights absorption of the radiation energy
#   kappa_F  flux mean        -- weights the radiation force / flux damping
# For a grey problem all three coincide, which is the default.

"""
    AbstractOpacity

Supertype for opacity laws. A concrete subtype must implement

  - [`planck_opacity`](@ref)`(op, rho, T)`
  - [`energy_mean_opacity`](@ref)`(op, rho, T)`
  - [`flux_mean_opacity`](@ref)`(op, rho, T)`

each returning a *specific* opacity (cm² g⁻¹ in CGS), so that the absorption
coefficient is `rho * kappa`.
"""
abstract type AbstractOpacity end

"""
    ConstantOpacity(kappa)
    ConstantOpacity(; planck, energy = planck, flux = planck)

Opacity independent of density and temperature.
"""
struct ConstantOpacity{T<:AbstractFloat} <: AbstractOpacity
    planck::T
    energy::T
    flux::T
end

ConstantOpacity(kappa::Real) = ConstantOpacity(promote(float(kappa), float(kappa), float(kappa))...)
function ConstantOpacity(; planck, energy=planck, flux=planck)
    ConstantOpacity(promote(float(planck), float(energy), float(flux))...)
end

@inline planck_opacity(op::ConstantOpacity, rho, T) = op.planck
@inline energy_mean_opacity(op::ConstantOpacity, rho, T) = op.energy
@inline flux_mean_opacity(op::ConstantOpacity, rho, T) = op.flux

"""
    PowerLawOpacity(; kappa0, rho0 = 1.0, T0 = 1.0, m = 0.0, n = 0.0)

Opacity of the form `kappa = kappa0 (rho/rho0)^m (T/T0)^n`, applied to all three
means. Kramers-like laws are `m = 1, n = -3.5`; the classic Marshak-wave test
uses `n = -3`.
"""
struct PowerLawOpacity{T<:AbstractFloat} <: AbstractOpacity
    kappa0::T
    rho0::T
    T0::T
    m::T
    n::T
end

function PowerLawOpacity(; kappa0, rho0=1.0, T0=1.0, m=0.0, n=0.0)
    PowerLawOpacity(promote(float(kappa0), float(rho0), float(T0), float(m), float(n))...)
end

@inline function _powerlaw(op::PowerLawOpacity, rho, T)
    op.kappa0 * (rho / op.rho0)^op.m * (max(T, zero(T)) / op.T0)^op.n
end

@inline planck_opacity(op::PowerLawOpacity, rho, T) = _powerlaw(op, rho, T)
@inline energy_mean_opacity(op::PowerLawOpacity, rho, T) = _powerlaw(op, rho, T)
@inline flux_mean_opacity(op::PowerLawOpacity, rho, T) = _powerlaw(op, rho, T)

"""
    UserOpacity(f)
    UserOpacity(; planck, energy = planck, flux = planck)

Opacity defined by user callables `f(rho, T) -> kappa`.
"""
struct UserOpacity{P,E,F} <: AbstractOpacity
    planck::P
    energy::E
    flux::F
end

UserOpacity(f) = UserOpacity(f, f, f)
UserOpacity(; planck, energy=planck, flux=planck) = UserOpacity(planck, energy, flux)

@inline planck_opacity(op::UserOpacity, rho, T) = op.planck(rho, T)
@inline energy_mean_opacity(op::UserOpacity, rho, T) = op.energy(rho, T)
@inline flux_mean_opacity(op::UserOpacity, rho, T) = op.flux(rho, T)
