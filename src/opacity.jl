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

# --- multigroup ---------------------------------------------------------------
#
# Multigroup here is *piecewise constant*: the opacity is taken to be uniform
# inside each frequency group, evaluated at the group's centre. That is Quokka's
# `OpacityModel::piecewise_constant_opacity`, and it collapses the three means
# into one -- averaging a constant against the Planck function, against the
# radiation energy density, or against the flux all return the same constant.
# Quokka's other models fit a power law across each group and so must keep the
# means apart; those are deliberately not implemented here.

"""
    group_opacity(op, nu, rho, T) -> Real

The opacity of a frequency group centred on `nu`.

Defaults to the grey Planck mean, so any existing opacity law used with several
groups gives the same `kappa` in every group -- a genuine configuration, and the
one Vaytet et al. (2011) call the constant-opacity model. Override it for a law
that actually varies with frequency, or use [`FrequencyOpacity`](@ref).
"""
@inline group_opacity(op::AbstractOpacity, nu, rho, T) = planck_opacity(op, rho, T)

"""
    FrequencyOpacity(f; grey = nothing)

Opacity from a callable `f(nu, rho, T) -> kappa`, evaluated at each group centre.

```julia
# Vaytet's nu^-2 law: kappa = 2000 (nu / 4e13)^-2 cm^2/g
FrequencyOpacity((nu, rho, T) -> 2000.0 * (nu / 4.0e13)^-2)
```

A frequency-dependent law has no single grey value, so the grey entry points
throw unless `grey` gives one explicitly. That is deliberate: silently
collapsing `kappa(nu)` to some arbitrary frequency is how a multigroup opacity
ends up quietly driving a single-group run.
"""
struct FrequencyOpacity{F,G} <: AbstractOpacity
    kappa::F
    grey::G
end

FrequencyOpacity(f; grey=nothing) = FrequencyOpacity(f, grey)

@inline group_opacity(op::FrequencyOpacity, nu, rho, T) = op.kappa(nu, rho, T)

function _grey(op::FrequencyOpacity, rho, T)
    op.grey === nothing && throw(ArgumentError(
        "FrequencyOpacity depends on frequency and has no grey value; pass " *
        "`grey = kappa` if you mean to use it with a single group"))
    return op.grey(rho, T)
end

@inline planck_opacity(op::FrequencyOpacity, rho, T) = _grey(op, rho, T)
@inline energy_mean_opacity(op::FrequencyOpacity, rho, T) = _grey(op, rho, T)
@inline flux_mean_opacity(op::FrequencyOpacity, rho, T) = _grey(op, rho, T)

"""
    group_opacities(op, groups, rho, T) -> SVec{NG}

The opacity in each group, evaluated at the group centres.

With one group this is the grey Planck mean, so a single-group run goes down
exactly the path it did before multigroup existed.
"""
@inline function group_opacities(op::AbstractOpacity, groups::RadiationGroups{NG,M,T},
                                 rho, Tgas) where {NG,M,T}
    NG == 1 && return SVec{1,T}(T(planck_opacity(op, rho, Tgas)))
    nu = group_centers(groups)
    return SVec{NG,T}(ntuple(g -> T(group_opacity(op, nu[g], rho, Tgas)), Val(NG))...)
end
