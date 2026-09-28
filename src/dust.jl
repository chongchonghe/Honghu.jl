# The dust-gas thermal coupling model: what the radiation couples to when there
# is dust.
#
# With a dust model the radiation exchanges energy with the *dust* alone, at the
# dust temperature, and the gas exchanges energy with the dust by collisions.
# The dust holds no heat, so at every instant what it gains from the gas it loses
# to the radiation; its temperature is fixed by that balance, not evolved. The
# opacity law then describes the dust (per unit mass of gas) and is evaluated at
# the dust temperature. This is the model of Quokka's
# `enable_dust_gas_thermal_coupling_model` and of Bate & Keto (2015).
#
# This file holds the model only. How one backward-Euler step of it is solved --
# reduced to one scalar equation in the dust temperature -- is
# `dust_energy_balance` in `coupling.jl` and the `BateKetoDust` method of
# `solve_coupling(::BracketedCoupling, …)`; `docs/coupling-new-method-dust.md`
# derives it.

"""
    AbstractDustModel

Supertype for the dust-gas thermal coupling model carried by
[`RadiationPhysics`](@ref). [`NoDust`](@ref), the default, couples the radiation
to the gas directly; [`BateKetoDust`](@ref) couples it to dust in instantaneous
equilibrium, and the dust to the gas by collisions.
"""
abstract type AbstractDustModel end

"""
    NoDust()

No dust: the radiation exchanges energy with the gas directly, at the gas
temperature. The default.
"""
struct NoDust <: AbstractDustModel end

"""
    BateKetoDust(; coefficient, particle_mass = 1)

Dust in instantaneous thermal equilibrium, coupled to the radiation at its own
temperature `T_d` and to the gas by the collisional rate of Keto & Field (2005),

    Lambda_gd = coefficient * n^2 * sqrt(T) * (T - T_d),   n = rho / particle_mass ,

the energy per unit volume and time the gas hands to the dust. The dust holds no
energy, so `T_d` is whatever makes `Lambda_gd` equal to what the dust radiates
net, and the total `E_gas + sum_g (c/chat_g) E_g` is conserved.

Use it for the cold ISM, where the gas has negligible opacity and cools or heats
only through the dust. With it, the opacity law of [`RadiationPhysics`](@ref) is
the *dust* opacity per unit mass of gas, evaluated at `T_d`, and the gas has no
opacity of its own. The parameters are Quokka's: `coefficient` is its
`radiation.dust_gas_interaction_coeff` (`2.5e-34` erg cm³ s⁻¹ K⁻³ᐟ² in CGS, from
Keto & Field) and `particle_mass` its `mean_molecular_mass`.

Supported by [`BracketedCoupling`](@ref) only, which reduces the step to one
equation in `T_d` for every value of `coefficient`, zero included: the gas is
then isolated and the dust is in radiative equilibrium with the field.
"""
struct BateKetoDust{T<:AbstractFloat} <: AbstractDustModel
    coefficient::T
    particle_mass::T
    function BateKetoDust{T}(coefficient, particle_mass) where {T}
        coefficient >= 0 ||
            throw(ArgumentError("the dust-gas coefficient must be >= 0, got $coefficient"))
        particle_mass > 0 ||
            throw(ArgumentError("particle_mass must be positive, got $particle_mass"))
        new{T}(coefficient, particle_mass)
    end
end

BateKetoDust(coefficient::T, particle_mass::T) where {T<:AbstractFloat} =
    BateKetoDust{T}(coefficient, particle_mass)
BateKetoDust(; coefficient, particle_mass=1.0) =
    BateKetoDust(promote(float(coefficient), float(particle_mass))...)

"""
    dust_gas_coefficient(dust, rho) -> K

`K = coefficient * (rho / particle_mass)^2`, so that the collisional rate is
`K sqrt(T) (T - T_d)`.
"""
@inline dust_gas_coefficient(d::BateKetoDust, rho) = d.coefficient * (rho / d.particle_mass)^2

# The model must be the simulation's float type, for the same reason the groups
# are: its numbers enter the inner loop.
_convert_dust(d::NoDust, ::Type{T}) where {T} = d
_convert_dust(d::BateKetoDust, ::Type{T}) where {T} = BateKetoDust{T}(T(d.coefficient),
                                                                      T(d.particle_mass))
