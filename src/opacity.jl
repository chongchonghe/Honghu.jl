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

Optionally it may also implement [`dkappa_dT`](@ref)`(op, rho, T)`, which makes
the matter-radiation coupling's Newton Jacobian exact. The default is zero,
which is exact for an opacity that does not depend on temperature and merely
slower for one that does.
"""
abstract type AbstractOpacity end

"""
    planck_opacity(op, rho, T) -> kappa_P

The Planck-mean specific opacity of `op` at density `rho` and gas temperature
`T`. It weights thermal emission, the `rho kappa_P a_rad T^4` term of the
matter-radiation coupling. Part of the [`AbstractOpacity`](@ref) interface.
"""
function planck_opacity end

"""
    energy_mean_opacity(op, rho, T) -> kappa_E

The energy-mean specific opacity of `op` at density `rho` and gas temperature
`T`. It weights the absorption of the radiation energy, the `rho kappa_E E_rad`
term of the coupling. Part of the [`AbstractOpacity`](@ref) interface.
"""
function energy_mean_opacity end

"""
    flux_mean_opacity(op, rho, T) -> kappa_F

The flux-mean specific opacity of `op` at density `rho` and gas temperature
`T`. It weights the damping of the radiation flux, `rho kappa_F F`, and so the
radiation force on the gas. Part of the [`AbstractOpacity`](@ref) interface.
"""
function flux_mean_opacity end

"""
    dkappa_dT(op, rho, T) -> (dkappaP_dT, dkappaE_dT)

Temperature derivatives of the Planck- and energy-mean opacities.

The coupling solves `d E_g/dt = chat rho (kappa_P,g B_g(T) - kappa_E,g E_g)` by
Newton's method, so the exact Jacobian carries `dkappa_P/dT * B_g` and
`-dkappa_E/dT * E_g` alongside `kappa_P dB_g/dT`. Supplying these derivatives is
what makes it exact.

**The default is zero.** A zero derivative does not change where the iteration
converges -- the residual it is driven to zero is still the residual of the true
equations -- only how fast it gets there. So it is safe everywhere, and exact
wherever `kappa` is independent of `T`.
"""
@inline function dkappa_dT(::AbstractOpacity, rho, Tgas)
    z = zero(float(typeof(Tgas)))
    return (z, z)
end

# A user-supplied derivative that was not supplied. Dispatch rather than an
# `=== nothing` test, so the zero is a compile-time constant.
@inline _dkappa_call(::Nothing, rho, Tgas) = zero(float(typeof(Tgas)))
@inline _dkappa_call(f, rho, Tgas) = f(rho, Tgas)

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
# `dkappa_dT` falls through to the `AbstractOpacity` default, which for a
# constant is not a fallback but the exact answer.

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

# A power law differentiates to itself: `dkappa/dT = n kappa / T`, exactly, and
# with no second evaluation of the law. At `T <= 0` the law itself is clamped,
# so its derivative there is zero rather than the `n kappa / 0` of the formula.
@inline function dkappa_dT(op::PowerLawOpacity, rho, Tgas)
    k = _powerlaw(op, rho, Tgas)
    d = Tgas > 0 ? op.n * k / Tgas : zero(k)
    return (d, d)
end

"""
    UserOpacity(f)
    UserOpacity(; planck, energy = planck, flux = planck,
                  dplanck_dT = nothing, denergy_dT = dplanck_dT)

Opacity defined by user callables `f(rho, T) -> kappa`.

`dplanck_dT` and `denergy_dT` are optional callables of the same signature
giving `dkappa/dT`. Supplying them makes the coupling's Newton Jacobian exact;
leaving them out substitutes zero, which costs iterations but not the root (see
[`dkappa_dT`](@ref)).
"""
struct UserOpacity{P,E,F,DP,DE} <: AbstractOpacity
    planck::P
    energy::E
    flux::F
    dplanck_dT::DP
    denergy_dT::DE
end

UserOpacity(f) = UserOpacity(f, f, f)
UserOpacity(planck, energy, flux) = UserOpacity(planck, energy, flux, nothing, nothing)
UserOpacity(; planck, energy=planck, flux=planck,
            dplanck_dT=nothing, denergy_dT=dplanck_dT) =
    UserOpacity(planck, energy, flux, dplanck_dT, denergy_dT)

@inline planck_opacity(op::UserOpacity, rho, T) = op.planck(rho, T)
@inline energy_mean_opacity(op::UserOpacity, rho, T) = op.energy(rho, T)
@inline flux_mean_opacity(op::UserOpacity, rho, T) = op.flux(rho, T)

@inline dkappa_dT(op::UserOpacity, rho, Tgas) =
    (_dkappa_call(op.dplanck_dT, rho, Tgas), _dkappa_call(op.denergy_dT, rho, Tgas))

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
    group_opacity_dT(op, nu, rho, T) -> Real

`dkappa/dT` of the group centred on `nu`: the companion of
[`group_opacity`](@ref), and what the coupling's exact Jacobian needs.

Defaults to the grey Planck-mean derivative, which for the constant-opacity
model is the right answer in every group. Zero unless the law supplies one.
"""
@inline group_opacity_dT(op::AbstractOpacity, nu, rho, Tgas) =
    first(dkappa_dT(op, rho, Tgas))

"""
    FrequencyOpacity(f; grey = nothing, dkappa_dT = nothing, dgrey_dT = nothing)

Opacity from a callable `f(nu, rho, T) -> kappa`, evaluated at each group centre.

```julia
# Vaytet's nu^-2 law: kappa = 2000 (nu / 4e13)^-2 cm^2/g
FrequencyOpacity((nu, rho, T) -> 2000.0 * (nu / 4.0e13)^-2)
```

A frequency-dependent law has no single grey value, so the grey entry points
throw unless `grey` gives one explicitly. That is deliberate: silently
collapsing `kappa(nu)` to some arbitrary frequency is how a multigroup opacity
ends up quietly driving a single-group run.

`dkappa_dT(nu, rho, T)` and `dgrey_dT(rho, T)` are the optional temperature
derivatives, per group and grey respectively. They default to zero, which leaves
the coupling's Jacobian inexact -- costing Newton convergence rate, not the root
it converges to (see [`dkappa_dT`](@ref)).
"""
struct FrequencyOpacity{F,G,D,DG} <: AbstractOpacity
    kappa::F
    grey::G
    dkappa_dT::D
    dgrey_dT::DG
end

FrequencyOpacity(f; grey=nothing, dkappa_dT=nothing, dgrey_dT=nothing) =
    FrequencyOpacity(f, grey, dkappa_dT, dgrey_dT)

@inline group_opacity(op::FrequencyOpacity, nu, rho, T) = op.kappa(nu, rho, T)

@inline _dkappa_call(::Nothing, nu, rho, Tgas) = zero(float(typeof(Tgas)))
@inline _dkappa_call(f, nu, rho, Tgas) = f(nu, rho, Tgas)

@inline group_opacity_dT(op::FrequencyOpacity, nu, rho, Tgas) =
    _dkappa_call(op.dkappa_dT, nu, rho, Tgas)

function _grey(op::FrequencyOpacity, rho, T)
    op.grey === nothing && throw(ArgumentError(
        "FrequencyOpacity depends on frequency and has no grey value; pass " *
        "`grey = kappa` if you mean to use it with a single group"))
    return op.grey(rho, T)
end

@inline planck_opacity(op::FrequencyOpacity, rho, T) = _grey(op, rho, T)
@inline energy_mean_opacity(op::FrequencyOpacity, rho, T) = _grey(op, rho, T)
@inline flux_mean_opacity(op::FrequencyOpacity, rho, T) = _grey(op, rho, T)

@inline function dkappa_dT(op::FrequencyOpacity, rho, Tgas)
    d = _dkappa_call(op.dgrey_dT, rho, Tgas)
    return (d, d)
end

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

"""
    group_opacities_dT(op, groups, rho, T) -> SVec{NG}

`dkappa/dT` in each group: [`group_opacities`](@ref) differentiated with respect
to temperature, and the piece [`coupling_jacobian`](@ref) needs to be exact.

Zero unless the opacity law supplies a derivative, which is exact for every
temperature-independent law and an approximation for the rest.
"""
@inline function group_opacities_dT(op::AbstractOpacity, groups::RadiationGroups{NG,M,T},
                                    rho, Tgas) where {NG,M,T}
    NG == 1 && return SVec{1,T}(T(first(dkappa_dT(op, rho, Tgas))))
    nu = group_centers(groups)
    return SVec{NG,T}(ntuple(g -> T(group_opacity_dT(op, nu[g], rho, Tgas)), Val(NG))...)
end
