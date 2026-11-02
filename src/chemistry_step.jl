# The chemistry step inside a simulation: the grid driver of `chemistry.jl`.
#
# `chemistry_step!` runs once per radiation substep, after the IMEX substep, and for
# every interior cell it
#
#   1. packs the cell into the network's unknown vector `u`: the species number
#      densities from the mass scalars, the internal energy of the gas, and the
#      photon number density `N = E / eps` of each band;
#   2. adds the emission to the first band, `phi q dt`, with `phi = chat / c` -- the
#      reduced speed of light scales the source, and it is the only place it does;
#   3. hands `u` to the solver for the whole substep;
#   4. writes the answer back: the scalars, the gas energy by the change the network
#      made to it, and the band energy `E = N eps` with the band's flux scaled by
#      `N_end / N_start`.
#
# The three things `chemistry.jl` keeps out of the ODE -- emission, the flux rescaling
# and the clamp at zero -- are exactly steps 2 and 4, so the two integrators see one
# problem. Gas momentum and density are not touched, so the kinetic energy is what the
# cell had and the gas energy moves by the change of the internal energy alone.
#
# The loop over rows is the one of `radiation_source_step!`: each `(j, k)` row is a
# task of `threaded_jk` and owns one workspace, so nothing is shared and nothing is
# keyed on `threadid()`. The per-cell work is in `_chem_row!`, an ordinary function,
# and not in the `threaded_jk` closure, so that no variable the closure captures is
# ever reassigned (the boxing trap of CLAUDE.md).

"""
    Photochemistry(network; solver = RosenbrockChem(), emission = nothing)

Photochemistry for a [`Simulation`](@ref): the [`AbstractNetwork`](@ref) to evolve, the
[`AbstractChemSolver`](@ref) that integrates it in every cell, and the source of ionizing
photons. Pass it as `Simulation(...; chemistry = Photochemistry(net))`; the simulation then
carries `nspecies(network)` mass scalars, one per species, as `rho X_s`.

  - `network`  : what is evolved in one cell, for instance [`HydrogenOTSA`](@ref).
  - `solver`   : the integrator, [`RosenbrockChem`](@ref) by default or
    [`BackwardEulerChem`](@ref).
  - `emission` : `emission(x, y, z, t)` gives the photons emitted per cm^3 and per second
    into the first band of the network at the cell centre, or `nothing` for no source.
    It is evaluated once per cell and per radiation substep, with `t` the *start* of the
    substep. A point source on the grid is a narrow Gaussian normalised to the photon rate.

The chemistry step runs after each IMEX radiation substep, over the substep's length,
and does four things in every cell: it adds `phi q dt` to the photon density `N` of the
first band, with `phi = chat / c` of the band's group, which is how the reduced speed of
light enters the source; it integrates the network over the substep; it writes the species
and the gas internal energy back; and it sets the band's energy to `N eps` and scales its
flux by `N_end / N_start`, the photon density after and before the integration. The
network's gas-side rates use `c` and its photon loss uses `chat`, so the budget
`n_H+ + N / phi` is conserved and the ionization equilibrium does not depend on `chat`.

A simulation with chemistry has these requirements, and refuses to be built without them
(an `ArgumentError`): radiation is enabled, the chat limiter is [`ConstantChat`](@ref),
`beta_order = 0`, the group of each band has zero thermal opacity (so the matter-radiation
coupling leaves it alone and only the network changes its energy), and `nscalars` is 0 or
`nspecies(network)`. The network's rates are in CGS, so the run must be in CGS units.
Momentum deposited by the absorbed photons is not modelled.

```julia
net = HydrogenOTSA(sigma = 1.63e-18, photon_energy = 29.65 * Honghu.CGS.eV)
chem = Photochemistry(net; emission = (x, y, z, t) -> 1e-17)
```
"""
struct Photochemistry{N<:AbstractNetwork,S<:AbstractChemSolver,E}
    network::N
    solver::S
    emission::E
end

Photochemistry(network::AbstractNetwork; solver::AbstractChemSolver=RosenbrockChem(),
               emission=nothing) = Photochemistry(network, solver, emission)

"""
    ChemStepStats(cells, steps, max_steps, failures)

What one [`chemistry_step!`](@ref) did: the number of `cells` solved, the `steps` of the
solver summed over them (see [`ChemStats`](@ref)), the most steps any one cell took
(`max_steps`), and the number of cells whose solve did not converge (`failures`). Two add:
the counts sum and the maximum is taken, which is how `advance_radiation!` accumulates the
substeps into [`Diagnostics`](@ref).
"""
struct ChemStepStats
    cells::Int
    steps::Int
    max_steps::Int
    failures::Int
end

ChemStepStats() = ChemStepStats(0, 0, 0, 0)

Base.:+(a::ChemStepStats, b::ChemStepStats) = ChemStepStats(a.cells + b.cells, a.steps + b.steps,
                                                            max(a.max_steps, b.max_steps),
                                                            a.failures + b.failures)

# The scratch of one row of cells: the solver's own workspace, the unknown vector the
# solve works on in place, and a copy of the state it started from (the photon density
# before the solve is what the band's flux is scaled against).
struct ChemRowWork{W,T<:AbstractFloat}
    w::W
    u::Vector{T}
    u0::Vector{T}
end

# One workspace per interior `(j, k)` row, built once when the simulation is. A row is a
# single task of `threaded_jk`, so each workspace has one user at a time.
function chem_row_workspaces(ch::Photochemistry, ::Type{T}, nrows::Integer) where {T<:AbstractFloat}
    nu = nunknowns(ch.network)
    return [ChemRowWork(chem_workspace(ch.solver, ch.network, T), zeros(T, nu), zeros(T, nu))
            for _ in 1:nrows]
end

# The photons a band-1 source adds to `N` in one cell over one substep, `phi q dt`.
@inline _photon_source(::Nothing, g, i, j, k, t, phi, dt, ::Type{T}) where {T} = zero(T)
@inline function _photon_source(f, g, i, j, k, t, phi, dt, ::Type{T}) where {T}
    x, y, z = cellcenter(g, i, j, k)
    return phi * T(f(x, y, z, t)) * dt
end

# Every cell of the row `(j, k)`. Returns the steps summed over the row, the most any one
# cell took, and the number of cells that did not converge.
function _chem_row!(U::Array{T,4}, R::Array{T,4}, chatarr::Array{T,4}, ch::Photochemistry,
                    row::ChemRowWork{W,T}, phys, g::CartesianGrid{T}, j::Int, k::Int,
                    dt::T, t::T) where {W,T}
    net = ch.network
    nsp = nspecies(net)
    nb = nbands(net)
    ne = ienergy(net)
    u = row.u
    u0 = row.u0
    c = T(phys.c)
    z = zero(T)
    steps = 0
    worst = 0
    fails = 0
    @inbounds for i in g.lo[1]:g.hi[1]
        rho = U[IRHO, i, j, k]
        mx, my, mz = U[IMX, i, j, k], U[IMY, i, j, k], U[IMZ, i, j, k]
        e = U[IEGAS, i, j, k] - (mx * mx + my * my + mz * mz) / (2 * rho)
        for s in 1:nsp
            u[s] = U[scalar_index(s), i, j, k] / species_mass(net, s)
        end
        u[ne] = e
        # The reduced speed of light of the first band's group is the one the cell's
        # rates see; without bands nothing reads it.
        chat = nb > 0 ? chatarr[band_group(net, 1), i, j, k] : T(phys.chat)
        for b in 1:nb
            gr = band_group(net, b)
            N = R[gbase(gr)+IERAD, i, j, k] / band_photon_energy(net, b)
            # Emission goes into the first band only, scaled by that band's `phi`.
            b == 1 && (N += _photon_source(ch.emission, g, i, j, k, t, chatarr[gr, i, j, k] / c, dt, T))
            u[iband(net, b)] = N
        end
        copyto!(u0, u)

        st = solve_chemistry!(row.w, ch.solver, net, ChemCell(rho, c, chat, dt), u)
        steps += st.steps
        worst = max(worst, st.steps)
        st.converged || (fails += 1)

        for s in 1:nsp
            U[scalar_index(s), i, j, k] = max(u[s], z) * species_mass(net, s)
        end
        U[IEGAS, i, j, k] += chem_internal_energy(net, u) - e
        for b in 1:nb
            base = gbase(band_group(net, b))
            Nstart = u0[iband(net, b)]
            Nend = max(u[iband(net, b)], z)
            # Absorption acts on the flux with the same linear rate as on `N`, so the
            # flux follows the photon density (Quokka's `SKIP_PHOTOCHEMFLUX`).
            ratio = Nstart > z ? Nend / Nstart : z
            R[base+IERAD, i, j, k] = Nend * band_photon_energy(net, b)
            R[base+IFRX, i, j, k] *= ratio
            R[base+IFRY, i, j, k] *= ratio
            R[base+IFRZ, i, j, k] *= ratio
        end
    end
    return steps, worst, fails
end

function _chemistry_step!(U::Array{T,4}, R::Array{T,4}, chatarr::Array{T,4}, ch::Photochemistry,
                          cw::Vector, phys, g::CartesianGrid{T}, dt::T, t::T) where {T}
    nsteps = Threads.Atomic{Int}(0)
    nmax = Threads.Atomic{Int}(0)
    nfail = Threads.Atomic{Int}(0)
    jlo, klo, ny = g.lo[2], g.lo[3], g.n[2]
    threaded_jk(g.lo[2]:g.hi[2], g.lo[3]:g.hi[3]) do j, k
        row = cw[(j-jlo)+(k-klo)*ny+1]
        rsteps, rworst, rfails = _chem_row!(U, R, chatarr, ch, row, phys, g, j, k, dt, t)
        Threads.atomic_add!(nsteps, rsteps)
        Threads.atomic_max!(nmax, rworst)
        rfails > 0 && Threads.atomic_add!(nfail, rfails)
    end
    return ChemStepStats(prod(g.n), nsteps[], nmax[], nfail[])
end

"""
    chemistry_step!(sim, dt, t) -> ChemStepStats

Integrate the chemistry of every interior cell of `sim` over `dt`, from time `t`: the
second half of each radiation substep, which `advance_radiation!` calls right after the IMEX
step. It adds the emission to the photon density, runs the solver of the simulation's
[`Photochemistry`](@ref) in each cell, and writes the species, the gas internal energy and
the band's energy and flux back; see [`Photochemistry`](@ref) for what each does. It uses the
workspaces the simulation built, one per row of cells, so it is safe to call from one place
at a time and allocates nothing per cell with [`BackwardEulerChem`](@ref).

The cells whose solve did not converge are counted in the returned [`ChemStepStats`](@ref)
and keep the solver's last state. It throws an `ArgumentError` for a simulation built
without `chemistry`.
"""
function chemistry_step!(sim, dt::T, t::T) where {T<:AbstractFloat}
    ch = sim.chemistry
    ch === nothing &&
        throw(ArgumentError("chemistry_step! needs a Simulation built with `chemistry`"))
    return _chemistry_step!(sim.U, sim.R, sim.rw.chat, ch, sim.cw, sim.radiation, sim.grid, dt, t)
end
