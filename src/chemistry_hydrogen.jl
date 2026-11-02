# The hydrogen network of Quokka's photoionization module, on the interface of
# `chemistry.jl`: H0, H+ and electrons, one band of ionizing photons, on the spot.
#
# The unknowns are `u = [n_H0, n_H+, e, N]`. Every rate is written once, as a
# coefficient function that returns the value and its temperature derivative
# together, so `chem_rhs!` and `chem_jacobian!` cannot disagree about a formula;
# the finite-difference test in `test_chemistry.jl` pins the rest. The physics
# follows `photoionization_H_caseB_handwritten/actual_rhs.H` in Quokka, whose
# rates are the reference whenever a number here is in doubt.
#
# Where this network differs from Quokka's, on purpose:
#   * `e` is the volumetric internal energy, not per gram, so no `1 / rho`;
#   * the gas-side rate uses `c` and the photon equation `chat`, because Honghu
#     stores `N = phi N_Q` (spec, section 4.3); Quokka's `N_Q` takes `chat` in both;
#   * the electron density is `n_H+`, not an unknown, so a recombination is
#     `alpha n_H+^2` and its derivative has a factor of two;
#   * no neutral-ISM (KI) heating and cooling, and no `get_cle_term` in free-free:
#     they are metal and molecular physics that a hydrogen-only run must not see.

# The ionization energy of hydrogen, 13.6 eV. Internal: it is the energy a
# collisional ionization costs and what the default heating energy subtracts from the
# photon energy.
const CHI_H = 13.6 * CGS.eV

# --- the network ----------------------------------------------------------------

"""
    HydrogenOTSA(; group = 1, sigma, photon_energy, heating_energy = photon_energy - 13.6 eV,
                 alpha0 = 2.63e-13, alpha_exp = -0.7, collisional_ionization = true,
                 cooling = true, isothermal = nothing, Tmin = 1.0)

The photoionization network of pure hydrogen in the on-the-spot approximation, after
Aubert & Teyssier (2008) and Quokka's `photoionization_H_caseB_handwritten`: the species
`(H0, H+)`, both of mass `CGS.m_H`, and one band of ionizing photons. It is
written against [`AbstractNetwork`](@ref), so [`RosenbrockChem`](@ref) and
[`BackwardEulerChem`](@ref) integrate it alike. CGS units only.

  - `group`          : the radiation group the band reads, which must have zero thermal opacity.
  - `sigma`          : the photoionization cross-section of the band, cm^2.
  - `photon_energy`  : the mean photon energy of the band, erg; it is `band_photon_energy`.
  - `heating_energy` : the energy each photoionization gives the gas, erg. The default,
    `photon_energy - 13.6 eV`, is Quokka's. It **overheats** the ionized gas compared with
    the optically thin value for a blackbody, which is `<eps> sigma_E / sigma_N - 13.6 eV`
    with the energy-weighted cross-section `sigma_E` (about 6.3 eV for a `1e5` K blackbody
    against 16 eV): a single band cannot be right for both the photon count and the energy.
  - `alpha0`, `alpha_exp` : the case B recombination coefficient `alpha0 (T / 1e4 K)^alpha_exp`
    in cm^3 s^-1; pass `alpha_exp = 0` for a constant one.
  - `collisional_ionization` : switch for collisional ionization and its energy cost.
  - `cooling`        : switch for the three cooling terms below.
  - `isothermal`     : a temperature in K that replaces the energy equation. `e` is then
    not evolved, and [`chem_internal_energy`](@ref) gives the energy that follows from the
    number densities and that temperature.
  - `Tmin`           : the floor of the temperature the rates see, K.

With `n_e = n_H+`, `T = e / (3/2 k_B (n_H0 + 2 n_H+))` and `Gamma = c sigma N`, the
right-hand side is

    ion = Gamma n_H0 + beta n_e n_H0 ,        rec = alpha_B n_e n_H+
    d n_H0 / dt = rec - ion ,                 d n_H+ / dt = ion - rec
    d N / dt    = - chat sigma n_H0 N
    d e / dt    = heating_energy Gamma n_H0 - 13.6 eV beta n_e n_H0
                  - (L_rec + L_ff) n_e n_H+ - L_ce n_e n_H0

where `c` is `cell.c` on the gas side and `chat` is `cell.chat` in the photon equation: the
photon density is `N = phi N_Q` with `phi = chat / c`, so the budget `n_H+ + N / phi` is
conserved and the equilibrium does not depend on `chat`. A negative `n_H0`, `n_H+` or `N`
enters every rate as zero, as Quokka clamps `N`, and its Jacobian column is zero. Each
rate and its source:

  - `alpha_B` : Quokka's `2.63e-13 (T / 1e4)^-0.7` cm^3 s^-1.
  - `beta`    : `5.84e-11 sqrt(T) exp(-2.18e-11 erg / k_B T)` cm^3 s^-1, Krumholz et al. (2007),
    as Quokka's `get_collisional_ionization_coefficient`. The `13.6 eV beta` term is the
    energy that ionization costs, so it belongs to this switch and not to `cooling`.
  - `L_rec`   : recombination cooling `6.1e-10 k_B T^0.11` erg cm^3 s^-1 for `T >= 100` K and
    zero below, Krumholz et al. (2007), as Quokka's `get_recombination_cooling_coefficient`.
  - `L_ff`    : free-free `1.3 * 1.427e-27 sqrt(T)` erg cm^3 s^-1, the first term of
    Quokka's `get_ion_ff_cooling_coefficient`; its `get_cle_term` is metal-ion line cooling and
    is left out of a hydrogen-only network.
  - `L_ce`    : Lyman alpha collisional excitation `7.5e-19 exp(-118348 K / T) / (1 + sqrt(T /
    1e5 K))` erg cm^3 s^-1, Cen (1992) in the form of Katz, Weinberg & Hernquist (1996),
    Table 1. Quokka has no separate term for it, and its neutral-ISM (KI) terms are not used.

With the three switches `alpha0 = 0`, `collisional_ionization = false` and
`cooling = false`, and the default `heating_energy`, the step conserves exactly
`n_H0 + n_H+`, `n_H+ + N / phi` and `e + 13.6 eV n_H+ + photon_energy N / phi`.

[`initial_mass_fractions`](@ref) reads the key `x_HII` (default 0), the ionization fraction.

```julia
net = HydrogenOTSA(sigma = 1.63e-18, photon_energy = 29.65 * Honghu.CGS.eV)
```
"""
struct HydrogenOTSA{T<:AbstractFloat} <: AbstractNetwork
    group::Int
    sigma::T
    photon_energy::T
    heating_energy::T
    alpha0::T
    alpha_exp::T
    collisional_ionization::Bool
    cooling::Bool
    isothermal::Union{Nothing,T}
    Tmin::T
end

function HydrogenOTSA(; group=1, sigma, photon_energy, heating_energy=photon_energy - CHI_H,
                      alpha0=2.63e-13, alpha_exp=-0.7, collisional_ionization=true,
                      cooling=true, isothermal=nothing, Tmin=1.0)
    sg, eps, eh, a0, p, Tm = promote(float(sigma), float(photon_energy), float(heating_energy),
                                     float(alpha0), float(alpha_exp), float(Tmin))
    T = typeof(sg)
    return HydrogenOTSA{T}(Int(group), sg, eps, eh, a0, p, Bool(collisional_ionization),
                           Bool(cooling), isothermal === nothing ? nothing : T(isothermal), Tm)
end

# --- rate coefficients: (value, d value / dT) --------------------------------

# Case B recombination, `alpha_0 (T / 1e4 K)^p` (Quokka's `get_recombination_coefficient`,
# which has p = -0.7).
function _hydrogen_alpha(net::HydrogenOTSA, T)
    a = net.alpha0 * (T / 1e4)^net.alpha_exp
    return a, net.alpha_exp * a / T
end

# Collisional ionization, Krumholz et al. (2007), as in Quokka's
# `get_collisional_ionization_coefficient`; 2.18e-11 erg is 13.6 eV.
function _hydrogen_beta(net::HydrogenOTSA, T)
    net.collisional_ionization || return zero(T), zero(T)
    b = 5.84e-11 * sqrt(T) * exp(-2.18e-11 / (CGS.k_B * T))
    return b, b * (0.5 / T + 2.18e-11 / (CGS.k_B * T^2))
end

# Recombination cooling per `n_e n_H+`, Krumholz et al. (2007): `6.1e-10 k_B T^0.11`,
# zero below 100 K as in Quokka's `get_recombination_cooling_coefficient`.
function _hydrogen_cool_recombination(net::HydrogenOTSA, T)
    (net.cooling && T >= 100) || return zero(T), zero(T)
    L = 6.1e-10 * CGS.k_B * T^0.11
    return L, 0.11 * L / T
end

# Free-free cooling per `n_e n_H+`, `1.3 * 1.427e-27 sqrt(T)`: the first term of Quokka's
# `get_ion_ff_cooling_coefficient`, without its `get_cle_term` (Frazer & Heitsch 2019).
function _hydrogen_cool_freefree(net::HydrogenOTSA, T)
    net.cooling || return zero(T), zero(T)
    L = 1.3 * 1.427e-27 * sqrt(T)
    return L, 0.5 * L / T
end

# Collisional-excitation (Lyman alpha) cooling per `n_e n_H0`,
# `7.5e-19 exp(-118348 K / T) / (1 + sqrt(T / 1e5 K))`, Cen (1992), in the form of
# Katz, Weinberg & Hernquist (1996), Table 1. Quokka has no such term.
function _hydrogen_cool_excitation(net::HydrogenOTSA, T)
    net.cooling || return zero(T), zero(T)
    s = sqrt(T / 1e5)
    L = 7.5e-19 * exp(-118348 / T) / (1 + s)
    return L, L * (118348 / T^2 - s / (2T * (1 + s)))
end

# The temperature of the cell, and whether it still depends on the state. It does not in
# isothermal mode or where the `Tmin` clamp is active, and the Jacobian then has no
# `dT/du` column to add.
@inline function _hydrogen_T(net::HydrogenOTSA, a0, a1, e)
    iso = net.isothermal          # a local, so that the compiler can narrow the Union
    iso === nothing || return iso, false
    T = e / (1.5 * CGS.k_B * (a0 + 2a1))
    return T > net.Tmin ? (T, true) : (net.Tmin, false)
end

nspecies(::HydrogenOTSA) = 2
nbands(::HydrogenOTSA) = 1
species_mass(::HydrogenOTSA, s) = CGS.m_H
band_group(net::HydrogenOTSA, b) = net.group
band_photon_energy(net::HydrogenOTSA, b) = net.photon_energy

function chem_rhs!(du, u, net::HydrogenOTSA, cell)
    z = zero(eltype(u))
    a0, a1, Nc = max(u[1], z), max(u[2], z), max(u[4], z)
    T = _hydrogen_T(net, a0, a1, u[3])[1]
    Gamma = cell.c * net.sigma * Nc
    beta = _hydrogen_beta(net, T)[1]
    ion = (Gamma + beta * a1) * a0
    rec = _hydrogen_alpha(net, T)[1] * a1 * a1
    du[1] = rec - ion
    du[2] = ion - rec
    if net.isothermal === nothing
        L_ii = _hydrogen_cool_recombination(net, T)[1] + _hydrogen_cool_freefree(net, T)[1]
        du[3] = net.heating_energy * Gamma * a0 - CHI_H * beta * a1 * a0 - L_ii * a1 * a1 -
                _hydrogen_cool_excitation(net, T)[1] * a1 * a0
    else
        du[3] = z
    end
    du[4] = -cell.chat * net.sigma * a0 * Nc
    return nothing
end

function chem_jacobian!(J, u, net::HydrogenOTSA, cell)
    z = zero(eltype(J))
    a0, a1, Nc = max(u[1], z), max(u[2], z), max(u[4], z)
    # `max(a, 0)` has slope one for a >= 0 and zero below it; the columns of a clamped
    # variable vanish, here and through the temperature.
    g0 = ifelse(u[1] >= 0, one(z), z)
    g1 = ifelse(u[2] >= 0, one(z), z)
    gN = ifelse(u[4] >= 0, one(z), z)
    T, active = _hydrogen_T(net, a0, a1, u[3])
    ntot = a0 + 2a1
    dT1 = active ? -g0 * T / ntot : z
    dT2 = active ? -2g1 * T / ntot : z
    dT3 = active ? 1 / (1.5 * CGS.k_B * ntot) : z
    c, sigma, chat = cell.c, net.sigma, cell.chat
    Gamma = c * sigma * Nc
    alpha, dalpha = _hydrogen_alpha(net, T)
    beta, dbeta = _hydrogen_beta(net, T)
    Lr, dLr = _hydrogen_cool_recombination(net, T)
    Lf, dLf = _hydrogen_cool_freefree(net, T)
    Lc, dLc = _hydrogen_cool_excitation(net, T)
    fill!(J, z)

    # d(rec - ion)/dT, then the row; the n_H+ row is its negative
    df1 = dalpha * a1 * a1 - dbeta * a1 * a0
    J[1, 1] = -(Gamma + beta * a1) * g0 + df1 * dT1
    J[1, 2] = (2alpha * a1 - beta * a0) * g1 + df1 * dT2
    J[1, 3] = df1 * dT3
    J[1, 4] = -c * sigma * a0 * gN
    for j in 1:4
        J[2, j] = -J[1, j]
    end

    if net.isothermal === nothing
        eh = net.heating_energy
        df3 = -CHI_H * dbeta * a1 * a0 - (dLr + dLf) * a1 * a1 - dLc * a1 * a0
        J[3, 1] = (eh * Gamma - CHI_H * beta * a1 - Lc * a1) * g0 + df3 * dT1
        J[3, 2] = (-CHI_H * beta * a0 - 2 * (Lr + Lf) * a1 - Lc * a0) * g1 + df3 * dT2
        J[3, 3] = df3 * dT3
        J[3, 4] = eh * c * sigma * a0 * gN
    end

    J[4, 1] = -chat * sigma * Nc * g0
    J[4, 4] = -chat * sigma * a0 * gN
    return nothing
end

# The scales the integrators measure tolerance against: both species against the hydrogen
# density, the energy against itself or against the energy of the cell at `Tmin` (so a cold
# cell does not demand a vanishing absolute error), and the photons against `phi n_H`, the
# photon density whose budget `N / phi` equals one photon per hydrogen nucleus.
function chem_scales!(scale, u, net::HydrogenOTSA, cell)
    nH = u[1] + u[2]
    scale[1] = nH
    scale[2] = nH
    scale[3] = max(abs(u[3]), 1.5 * CGS.k_B * nH * net.Tmin)
    scale[4] = (cell.chat / cell.c) * nH
    return nothing
end

# Electrons are `n_H+`, so a gas of `n_H0` and `n_H+` carries `n_H0 + 2 n_H+` particles.
network_temperature(::HydrogenOTSA, n::AbstractVector, eint) = eint / (1.5 * CGS.k_B * (n[1] + 2n[2]))
network_eint(::HydrogenOTSA, n::AbstractVector, T) = 1.5 * CGS.k_B * (n[1] + 2n[2]) * T

chem_internal_energy(net::HydrogenOTSA, u::AbstractVector) =
    net.isothermal === nothing ? u[3] : network_eint(net, u, net.isothermal)

function initial_mass_fractions(::HydrogenOTSA, nt::NamedTuple)
    x = get(nt, :x_HII, 0.0)
    return (1 - x, x)
end
