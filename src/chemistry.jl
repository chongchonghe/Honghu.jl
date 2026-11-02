# How a chemical network is evolved over one step, per cell.
#
# The physics is one stiff ODE system per cell, in the species number densities,
# the gas internal energy and the photon number density of each radiation band
# the network reads:
#
#   u = [n_1 ... n_NS, e, N_1 ... N_NB]
#
# with `n` in cm^-3, `e` the internal energy density and `N_b = E_b / eps_b` the
# photon number density of band `b`. It is stiff because ionization and
# recombination run on timescales many orders of magnitude shorter than the
# hydrodynamic step, and because the photon loss and the ionization rate share
# the same product `n_H0 N`, so neither can be integrated explicitly without
# one of them going negative.
#
# This file defines the *interface*, and `chemistry_solvers.jl` the two
# integrators written against it. It is the pattern of `coupling.jl`: the physics
# is in the network, the numerics are in the solver, and the only thing they
# share is the right-hand side and its exact Jacobian. A new integrator is a new
# subtype and a new `solve_chemistry!` method; a new network is a new subtype and
# the eight methods listed under `AbstractNetwork`. Neither edits the other.
#
# Three things are deliberately *not* terms of the ODE. Each is a property of how
# the step is embedded in the radiation-hydrodynamics scheme, so each has one
# home, which is the grid driver and not a network or a solver:
#
#   emission              adds `phi q dt` to `N` before the burn, with `phi =
#                         chat / c`. A source is not a rate of the cell's own
#                         state, and the reduced speed of light scales it.
#   flux rescaling        absorption acts on `N` and on the flux with the same
#                         linear rate, so the flux of a band is multiplied by
#                         `N_end / N_start` afterwards rather than carried as three
#                         more unknowns. This is Quokka's `SKIP_PHOTOCHEMFLUX`.
#   positivity clamping   the state written back to the grid is clamped at zero.
#                         A solver's answer may be a rounding error below zero,
#                         and the grid, not the solver, owns what is admissible.
#
# Putting any of them in `chem_rhs!` would make the two integrators integrate
# different problems, which is what the interface exists to prevent.

"""
    AbstractNetwork

Supertype for chemical networks: what is evolved in one cell over one step, and
nothing about how. A subtype implements

  - [`nspecies`](@ref)`(net)` and [`nbands`](@ref)`(net)`, the numbers of species and of
    radiation bands;
  - [`species_mass`](@ref)`(net, s)`, the mass of one particle of species `s` in g;
  - [`band_group`](@ref)`(net, b)` and [`band_photon_energy`](@ref)`(net, b)`, the
    radiation group band `b` reads and its mean photon energy in erg;
  - [`chem_rhs!`](@ref)`(du, u, net, cell)`, the right-hand side of the ODE;
  - [`chem_jacobian!`](@ref)`(J, u, net, cell)`, its **exact** dense Jacobian;
  - [`chem_scales!`](@ref)`(scale, u, net, cell)`, one positive absolute scale
    per unknown, which a solver measures its tolerance against.

The unknowns are `u = [n_1 ... n_NS, e, N_1 ... N_NB]`: number densities in
cm^-3, the internal energy density `e`, and the photon number density `N` of
each band. [`nunknowns`](@ref), [`ienergy`](@ref) and [`iband`](@ref) give the layout.
Electrons are not stored; a network that has them takes them from charge
neutrality.

Four more methods have defaults a network may override:
[`network_temperature`](@ref), [`network_eint`](@ref), [`chem_internal_energy`](@ref)
and [`initial_mass_fractions`](@ref).

The integrators in `chemistry_solvers.jl` are [`RosenbrockChem`](@ref) and
[`BackwardEulerChem`](@ref); both read the right-hand side and the Jacobian from
the network and nothing else, so they integrate the same problem.
"""
abstract type AbstractNetwork end

"""
    nspecies(net)

The number of species of the network, which is also the number of mass scalars a
simulation with this network carries. They are the first `nspecies(net)`
unknowns, as number densities in cm^-3.
"""
function nspecies end

"""
    nbands(net)

The number of radiation bands the network reads. Each is one unknown, the photon
number density, after the internal energy.
"""
function nbands end

"""
    species_mass(net, s)

The mass of one particle of species `s`, in g. It converts the mass scalar
`rho X_s` of the grid to the number density `n_s` the network works in.
"""
function species_mass end

"""
    band_group(net, b)::Int

The index of the radiation group whose energy band `b` of the network reads and
writes. That group must have zero thermal opacity, so the matter-radiation
coupling leaves it alone and only the network changes its energy.
"""
function band_group end

"""
    band_photon_energy(net, b)

The mean energy `eps_b` of a photon of band `b`, in erg. It converts the
group's radiation energy density to the photon number density,
`N_b = E_b / eps_b`.
"""
function band_photon_energy end

"""
    chem_rhs!(du, u, net, cell)

Write the time derivative of the unknowns `u` of one cell into `du`, in
cm^-3 s^-1 for the species and the photons and in erg cm^-3 s^-1
for the energy. `cell` is the [`ChemCell`](@ref) of constants the step needs. This
and [`chem_jacobian!`](@ref) are the single source of truth for the network's
physics: every integrator's residual and linearisation come from them.
"""
function chem_rhs! end

"""
    chem_jacobian!(J, u, net, cell)

Write the exact Jacobian `df_i/du_j` of [`chem_rhs!`](@ref) into the
dense `nunknowns x nunknowns` matrix `J`, overwriting every entry. It must be exact,
not a finite difference: a Newton iteration converges quadratically only if it is,
and a Rosenbrock method is handed it so that it never differentiates the
right-hand side numerically.
"""
function chem_jacobian! end

"""
    chem_scales!(scale, u, net, cell)

Write one positive absolute scale per unknown into `scale`: the size of that
unknown in this cell, at the state `u`. A solver measures its tolerance as a
fraction of it, so a tolerance means the same thing in any unit system and a
species holding a negligible share of the cell does not drive the step size to
zero on its own relative error. Compare the energy scale of
[`AdaptiveCoupling`](@ref).
"""
function chem_scales! end

"""
    nunknowns(net)

The length of the unknown vector `u` of the network, `nspecies + 1 + nbands`.
"""
nunknowns(net::AbstractNetwork) = nspecies(net) + 1 + nbands(net)

"""
    ienergy(net)

The index of the internal energy density in the unknown vector,
`nspecies(net) + 1`.
"""
ienergy(net::AbstractNetwork) = nspecies(net) + 1

"""
    iband(net, b)

The index of the photon number density of band `b` in the unknown vector,
`nspecies(net) + 1 + b`.
"""
iband(net::AbstractNetwork, b) = nspecies(net) + 1 + b

"""
    network_temperature(net, n::AbstractVector, eint)

The gas temperature of a cell with species number densities `n[1:nspecies(net)]`
and internal energy density `eint`. The default is a monatomic ideal gas of the
stored species in CGS,

    T = eint / (1.5 k_B sum_s n_s) ,

which counts no electrons; a network with ionization overrides it to count them.
It is what `gas_temperature` reports in place of the constant-`mu` equation of
state, which is wrong in partially ionized gas.
"""
function network_temperature(net::AbstractNetwork, n::AbstractVector, eint)
    ntot = zero(eltype(n))
    for s in 1:nspecies(net)
        ntot += n[s]
    end
    return eint / (1.5 * CGS.k_B * ntot)
end

"""
    network_eint(net, n::AbstractVector, T)

The internal energy density of a cell with species number densities
`n[1:nspecies(net)]` at temperature `T`, the inverse of
[`network_temperature`](@ref). The default is a monatomic ideal gas of the stored
species. `set_initial_conditions!` uses it to turn a `T` key into an energy when a
network is present.
"""
function network_eint(net::AbstractNetwork, n::AbstractVector, T)
    ntot = zero(eltype(n))
    for s in 1:nspecies(net)
        ntot += n[s]
    end
    return 1.5 * CGS.k_B * ntot * T
end

"""
    chem_internal_energy(net, u)

The internal energy density to write back to the grid from the unknown vector
`u`. The default is `u[ienergy(net)]`. A network with an isothermal mode, which
drops the energy equation, overrides it to return the energy that follows from
the number densities and the fixed temperature.
"""
chem_internal_energy(net::AbstractNetwork, u::AbstractVector) = u[ienergy(net)]

"""
    initial_mass_fractions(net, nt::NamedTuple)

The mass fractions `X_s` of the network's species for an initial condition `nt`
that gave no `X` key. The default throws an `ArgumentError`, because a generic
network cannot know what the user's keys mean; a network overrides it to read its
own, for instance an ionization fraction.
"""
initial_mass_fractions(net::AbstractNetwork, nt::NamedTuple) =
    throw(ArgumentError("a $(nameof(typeof(net))) needs the species mass fractions as " *
                        "the key `X`, or its own initial_mass_fractions method"))

"""
    ChemCell{T}(rho, c, chat, dt)

The constants of one cell and one step that a network's rates may read: the gas
density `rho`, the speed of light `c`, the reduced speed of light `chat` of the
band's group, and the step length `dt`. Both speeds appear because the gas-side
photoionization rate uses `c` while the radiation-side photon loss uses
`chat`. An immutable bit-type, so building one per cell allocates nothing.
"""
struct ChemCell{T<:AbstractFloat}
    rho::T
    c::T
    chat::T
    dt::T
end

"""
    AbstractChemSolver

Supertype for chemistry integrators. A subtype implements

    chem_workspace(solver, net, ::Type{T}) -> work
    solve_chemistry!(work, solver, net, cell, u) -> ChemStats

so that a new integrator is a new subtype and two methods, not an edit to the
grid driver. Two are provided: [`RosenbrockChem`](@ref), an adaptive stiff
integrator from OrdinaryDiffEq and the default, and [`BackwardEulerChem`](@ref),
one backward-Euler step with a Newton iteration, which is the comparison.
"""
abstract type AbstractChemSolver end

"""
    ChemStats(steps, converged)

What a chemistry solve reports for one cell: `steps`, the number of internal
steps an adaptive integrator took or the number of Newton iterations a
backward-Euler step took, and whether it `converged`. An unconverged cell is
reported, never hidden: the driver counts it in `Diagnostics`.
"""
struct ChemStats
    steps::Int
    converged::Bool
end

"""
    chem_workspace(solver, net, ::Type{T})

Allocate the scratch space for solving one cell at a time with `solver` on `net`
in precision `T`. One workspace belongs to one user at a time: the grid driver
makes one per row of cells and never shares it between threads, so no workspace is
ever keyed on `Threads.threadid()`.
"""
function chem_workspace end

"""
    solve_chemistry!(work, solver, net, cell, u::Vector{T}) -> ChemStats

Evolve the cell's unknowns `u` over `cell.dt`: on entry `u` is the initial state,
on exit the answer. `work` comes from [`chem_workspace`](@ref) for the same
`solver`, `net` and `T`. Returns the [`ChemStats`](@ref) of the solve; when it did not
converge, `u` holds the solver's last state, which the caller should not trust.
"""
function solve_chemistry! end
