# The simulation object and the top-level time-stepping driver.
#
# One step is operator-split into
#
#   1. an explicit SSP-RK update of the Euler equations (fluxes from the
#      chosen AbstractRiemannSolver), then
#   2. a sequence of IMEX PD-ARS radiation substeps, each of which transports
#      the radiation moments explicitly (HLL + M1) and couples them to the
#      matter implicitly.
#
# The split is the one used by Quokka: radiation subcycles inside the hydro step
# because its signal speed is `chat`, which is normally much larger than
# `|v| + c_s`.

"""
    Diagnostics

Running counters describing how hard the solver had to work, accumulated from
construction and reported by [`run!`](@ref). Read them off `sim.diag`, or reset
them with [`reset_diagnostics!`](@ref).

| field | meaning |
|:---|:---|
| `steps` | steps taken |
| `rad_substeps` | IMEX radiation substeps taken |
| `floors_hydro` | cell updates that hit a gas density or pressure floor |
| `floors_rad` | cell updates projected back onto the M1 admissible set |
| `newton_cells` | cells passed through the implicit coupling solve |
| `newton_iterations` | Newton iterations summed over those cells |
| `newton_max` | worst-case Newton iterations in any one cell |
| `newton_failures` | cells that hit `max_iterations` without converging |

A handful of floor hits is normal wherever the solution genuinely reaches
vacuum. `newton_failures > 0` is not: it means the implicit solve gave up, and
the run should not be trusted.
"""
mutable struct Diagnostics
    steps::Int
    rad_substeps::Int
    floors_hydro::Int
    floors_rad::Int
    newton_cells::Int
    newton_iterations::Int
    newton_max::Int
    newton_failures::Int
end

Diagnostics() = Diagnostics(0, 0, 0, 0, 0, 0, 0, 0)

"""
    reset_diagnostics!(sim) -> Diagnostics

Zero the counters in `sim.diag`. Useful when timing or scoring a phase of a run
separately from the transient that preceded it.
"""
function reset_diagnostics!(sim)
    d = sim.diag
    d.steps = d.rad_substeps = d.floors_hydro = d.floors_rad = 0
    d.newton_cells = d.newton_iterations = d.newton_max = d.newton_failures = 0
    return d
end

@inline function _record!(d::Diagnostics, s::SourceStats)
    d.newton_cells += s.cells
    d.newton_iterations += s.newton_iterations
    d.newton_max = max(d.newton_max, s.max_newton)
    d.newton_failures += s.failures
    return d
end

function Base.show(io::IO, d::Diagnostics)
    print(io, "Diagnostics(steps = ", d.steps)
    d.rad_substeps > 0 && print(io, ", rad_substeps = ", d.rad_substeps)
    print(io, ", floors = ", d.floors_hydro, " gas / ", d.floors_rad, " rad")
    if d.newton_cells > 0
        @printf(io, ", newton = %.2f mean / %d max", d.newton_iterations / d.newton_cells,
                d.newton_max)
        d.newton_failures > 0 && print(io, ", ", d.newton_failures, " FAILURES")
    end
    print(io, ")")
end

"""
    SourceOrder

Which half of the Strang-split source step is running: `SourceForward` before
the advance, `SourceReverse` after it.

The order matters only when several source terms are applied in sequence and do
not commute -- reversing the sequence on the second half-step is what keeps the
pair second order. A single source term can ignore it.

Quokka's `enum class SourceOrder { forward, reverse }`.
"""
@enum SourceOrder SourceForward SourceReverse

"""
    HydroWorkspace{T}

Scratch arrays for the hydrodynamic sweeps. Allocated once per simulation.
"""
struct HydroWorkspace{T<:AbstractFloat}
    prim::Array{T,4}
    qL::Array{T,4}
    qR::Array{T,4}
    flux::Array{T,4}
    dU::Array{T,4}
    U0::Array{T,4}
end

function HydroWorkspace(g::CartesianGrid{T}) where {T}
    HydroWorkspace(allocate(g, NHYDRO), allocate(g, NHYDRO), allocate(g, NHYDRO),
                   allocate(g, NHYDRO), allocate(g, NHYDRO), allocate(g, NHYDRO))
end

"""
    RadiationWorkspace{T}

Scratch arrays for the radiation sweeps and the IMEX stages.

`chat` is the odd one out: it is `(ngroups, i, j, k)` rather than
`(nradvars, i, j, k)`, and it holds the local reduced speed of light rather than
a state (see [`update_chat!`](@ref)). It is filled with the constant
`phys.chat` when the simulation is built, so a run with the default
[`ConstantChat`](@ref) limiter reads exactly the number it always did and never
writes to it again.
"""
struct RadiationWorkspace{T<:AbstractFloat}
    prim::Array{T,4}
    qL::Array{T,4}
    qR::Array{T,4}
    flux::Array{T,4}
    dR::Array{T,4}
    Rtmp::Array{T,4}
    Utmp::Array{T,4}
    chat::Array{T,4}
end

function RadiationWorkspace(g::CartesianGrid{T}, nrad::Integer=NRAD,
                            ng::Integer=nrad ÷ NRAD) where {T}
    RadiationWorkspace(allocate(g, nrad), allocate(g, nrad), allocate(g, nrad),
                       allocate(g, nrad), allocate(g, nrad), allocate(g, nrad),
                       allocate(g, NHYDRO), allocate(g, ng))
end

"""
    Simulation(; n, xlo = (0,0,0), xhi = (1,1,1), kwargs...)

A three-dimensional (radiation-)hydrodynamics simulation.

# Grid
  - `n`   : interior cell counts, `(nx, ny, nz)` or a single integer. A
    direction with `n[d] == 1` is degenerate and is skipped entirely, so a 1D
    problem is `n = (256, 1, 1)`.
  - `xlo`, `xhi` : domain corners.

# Physics
  - `eos = IdealGas()`  : equation of state.
  - `radiation = nothing` : a [`RadiationPhysics`](@ref) to enable the
    two-moment radiation module.
  - `hydro = true` : evolve the Euler equations. Set to `false` for static
    radiation tests; this also freezes the gas momentum in the coupling.
  - `source = nothing` : callable `source(x, y, z, t)` returning a *radiation*
    energy source density (energy per volume per time), applied inside the
    implicit coupling. For anything else -- gravity, driving, cooling -- use
    `strang_sources`.
  - `strang_sources = nothing` : callable `f(sim, t, dt, order)` applied at
    `dt/2` before the advance and `dt/2` after it, mutating `sim.U` and `sim.R`
    in place. See [`SourceOrder`](@ref).
  - `floors = Floors()` : positivity floors.

# Numerics
  - `reconstruction = PPM()` : [`PCM`](@ref), [`PLM`](@ref) or [`PPM`](@ref).
  - `radiation_reconstruction = reconstruction` : reconstruction for the
    radiation moments. "PPM hydro with PLM radiation" is a standard pairing:
    the M1 system is stiffer and the extra order buys less there. The
    ghost-cell width is taken from whichever needs more.
  - `riemann = HLLC()` : interface flux solver, [`HLLC`](@ref) or [`LLF`](@ref).
  - `integrator = SSPRK3()`  : explicit integrator for the hydro subsystem.
  - `boundaries = Boundaries(Outflow())`, `radiation_boundaries = boundaries`.
  - `cfl = 0.4`, `cfl_radiation = 0.3`.
  - `max_radiation_substeps = 10` : if the hydro step would need more radiation
    substeps than this, the hydro step is shortened instead.

# Example
```julia
sim = Simulation(n = (128, 128, 1), xhi = (1.0, 1.0, 1.0),
                 eos = IdealGas(gamma = 1.4),
                 boundaries = Boundaries(Periodic()))
set_initial_conditions!(sim) do x, y, z
    (rho = 1.0, vx = 0.0, p = 1.0)
end
run!(sim, tmax = 0.5)
```
"""
mutable struct Simulation{T,E<:AbstractEOS,RP,RC<:AbstractReconstruction,
                          RR<:AbstractReconstruction,RS<:AbstractRiemannSolver,
                          IN<:AbstractIntegrator,BH,BR,SRC,SS,RW}
    const grid::CartesianGrid{T}
    const eos::E
    const radiation::RP
    const reconstruction::RC
    const radiation_reconstruction::RR
    const riemann::RS
    const integrator::IN
    const boundaries::BH
    const radiation_boundaries::BR
    const source::SRC
    const strang_sources::SS
    const floors::Floors{T}
    const cfl::T
    const cfl_radiation::T
    const max_radiation_substeps::Int
    const hydro_enabled::Bool
    const U::Array{T,4}
    const R::Array{T,4}
    const hw::HydroWorkspace{T}
    const rw::RW
    const diag::Diagnostics
    t::T
    dt::T
    step::Int
end

function Simulation(; n, xlo=(0.0, 0.0, 0.0), xhi=(1.0, 1.0, 1.0),
                    eos::AbstractEOS=IdealGas(),
                    radiation=nothing,
                    hydro::Bool=true,
                    source=nothing,
                    strang_sources=nothing,
                    floors::Floors=Floors(),
                    reconstruction::AbstractReconstruction=PPM(),
                    radiation_reconstruction::AbstractReconstruction=reconstruction,
                    riemann::AbstractRiemannSolver=HLLC(),
                    integrator::AbstractIntegrator=SSPRK3(),
                    boundaries::Boundaries=Boundaries(Outflow()),
                    radiation_boundaries::Boundaries=boundaries,
                    cfl=0.4, cfl_radiation=0.3,
                    max_radiation_substeps::Integer=10,
                    T::Type{<:AbstractFloat}=Float64)
    hydro || radiation !== nothing ||
        throw(ArgumentError("nothing to evolve: enable hydro, radiation, or both"))
    ng = max(required_ghosts(reconstruction),
             radiation === nothing ? 0 : required_ghosts(radiation_reconstruction))
    grid = CartesianGrid(; n, xlo, xhi, nghost=ng, T)
    U = allocate(grid, NHYDRO)
    nrad = radiation === nothing ? 0 : nradvars(radiation)
    Rarr = radiation === nothing ? Array{T,4}(undef, 0, 0, 0, 0) : allocate(grid, nrad)
    rw = radiation === nothing ? nothing :
         RadiationWorkspace(grid, nrad, ngroups(radiation))
    # The limiter only ever raises `chat` above this floor, and `ConstantChat`
    # leaves it here for the whole run.
    rw === nothing || fill!(rw.chat, T(radiation.chat))
    fl = Floors(T(floors.rho), T(floors.pressure), T(floors.erad))
    Simulation(grid, eos, radiation, reconstruction, radiation_reconstruction,
               riemann, integrator, boundaries,
               radiation_boundaries, source, strang_sources, fl, T(cfl), T(cfl_radiation),
               Int(max_radiation_substeps), hydro, U, Rarr,
               HydroWorkspace(grid), rw, Diagnostics(), zero(T), zero(T), 0)
end

"""
    has_radiation(sim)

Whether the radiation module is enabled (a compile-time property of the type).
"""
@inline has_radiation(::Simulation{T,E,RP}) where {T,E,RP} = RP !== Nothing

Base.eltype(::Simulation{T}) where {T} = T

function Base.show(io::IO, sim::Simulation{T}) where {T}
    g = sim.grid
    println(io, "Simulation{", T, "}")
    println(io, "  grid           : ", g.n, "  (", ndim(g), "D, ", g.ng, " ghost cells)")
    println(io, "  domain         : ", ntuple(d -> (g.xlo[d], g.xhi[d]), 3))
    println(io, "  eos            : ", sim.eos)
    println(io, "  reconstruction : ", sim.reconstruction,
            has_radiation(sim) && sim.radiation_reconstruction !== sim.reconstruction ?
            " (gas), $(sim.radiation_reconstruction) (radiation)" : "")
    println(io, "  riemann        : ", nameof(typeof(sim.riemann)))
    println(io, "  integrator     : ", sim.integrator)
    println(io, "  hydro          : ", sim.hydro_enabled)
    if has_radiation(sim)
        r = sim.radiation
        println(io, "  radiation      : ", nameof(typeof(r.closure)), ", c = ", r.c,
                ", chat/c = ", r.chat / r.c, ", beta_order = ", r.beta_order)
        println(io, "  opacity        : ", r.opacity)
    else
        println(io, "  radiation      : disabled")
    end
    print(io, "  t = ", sim.t, ", step = ", sim.step)
end

# --- initial conditions ------------------------------------------------------

@inline _field(nt::NamedTuple, ::Val{K}, default) where {K} =
    haskey(nt, K) ? getfield(nt, K) : default

"""
    set_initial_conditions!(f, sim)

Fill the interior state from a callable `f(x, y, z)` that returns a `NamedTuple`
of primitive variables at the cell centre.

Recognised keys:

| key                | meaning                                | default |
|:-------------------|:---------------------------------------|:--------|
| `rho`              | mass density                           | required |
| `p` or `T`         | pressure, or gas temperature           | required (one of) |
| `vx`, `vy`, `vz`   | velocity                               | `0` |
| `Erad` or `Trad`   | radiation energy density, or radiation temperature | gas temperature |
| `Fx`, `Fy`, `Fz`   | radiation flux                         | `0` |

Ghost cells are filled from the boundary conditions afterwards, so `f` is only
called on interior cells.

The radiation keys are *totals*, whatever the group structure: `Erad` and the
fluxes are split over the frequency groups by the Planck spectrum at the
radiation temperature `Erad` implies, so a multigroup run starts in equilibrium
rather than with everything piled into group 1. A multigroup problem therefore
needs no group-aware initial condition of its own; a [`Fixed`](@ref) radiation
boundary still does, since nothing splits a boundary state.

```julia
set_initial_conditions!(sim) do x, y, z
    x < 0.5 ? (rho = 1.0, p = 1.0) : (rho = 0.125, p = 0.1)
end
```
"""
function set_initial_conditions!(f::F, sim::Simulation{T}) where {F,T}
    g = sim.grid
    eos = sim.eos
    U = sim.U
    z0 = zero(T)
    for k in g.lo[3]:g.hi[3], j in g.lo[2]:g.hi[2], i in g.lo[1]:g.hi[1]
        x, y, z = cellcenter(g, i, j, k)
        s = f(x, y, z)
        rho = T(_field(s, Val(:rho), NaN))
        isfinite(rho) || throw(ArgumentError("initial condition must provide `rho`"))
        p = T(_field(s, Val(:p), NaN))
        if !isfinite(p)
            tg = T(_field(s, Val(:T), NaN))
            isfinite(tg) || throw(ArgumentError("initial condition must provide `p` or `T`"))
            p = pressure(eos, rho, eint_from_temperature(eos, rho, tg))
        end
        w = SVec(rho, T(_field(s, Val(:vx), z0)), T(_field(s, Val(:vy), z0)),
                 T(_field(s, Val(:vz), z0)), p)
        setstate!(U, prim2cons(eos, w), i, j, k)

        if has_radiation(sim)
            phys = sim.radiation
            er = T(_field(s, Val(:Erad), NaN))
            if !isfinite(er)
                tr = T(_field(s, Val(:Trad), NaN))
                if !isfinite(tr)
                    tr = temperature(eos, rho, internal_energy(eos, rho, p))
                end
                er = phys.a_rad * tr^4
            end
            er = max(er, sim.floors.erad)
            fx = T(_field(s, Val(:Fx), z0))
            fy = T(_field(s, Val(:Fy), z0))
            fz = T(_field(s, Val(:Fz), z0))
            # A scalar `Erad` in an initial condition is a *total*. Split it over
            # the groups by the Planck spectrum at the radiation temperature it
            # implies -- the only spectrum a single number carries -- so that a
            # multigroup run starts in equilibrium rather than with everything
            # piled into group 1 and the rest empty. Exact at one group.
            frac = planck_fractions(phys.groups, sqrt(sqrt(er / phys.a_rad)), phys.k_B)
            for gr in 1:ngroups(phys)
                b = gbase(gr)
                sim.R[b+IERAD, i, j, k] = er * frac[gr]
                sim.R[b+IFRX, i, j, k] = fx * frac[gr]
                sim.R[b+IFRY, i, j, k] = fy * frac[gr]
                sim.R[b+IFRZ, i, j, k] = fz * frac[gr]
            end
        end
    end
    # Ghosts first: `enforce_state!` sweeps the whole array, and until the
    # boundary conditions have run the ghost cells hold uninitialised memory.
    fill_ghosts!(U, sim.boundaries, g, sim.t)
    has_radiation(sim) && fill_ghosts!(sim.R, sim.radiation_boundaries, g, sim.t)
    enforce_state!(sim)
    return sim
end

# --- right-hand sides --------------------------------------------------------

function _hydro_sweep!(dU, sim::Simulation, ::Val{D}) where {D}
    hw = sim.hw
    g = sim.grid
    reconstruct!(hw.qL, hw.qR, hw.prim, sim.reconstruction, Val(D), g, NHYDRO)
    hydro_fluxes!(hw.flux, hw.qL, hw.qR, hw.prim, sim.eos, sim.riemann, Val(D), g)
    flux_divergence!(dU, hw.flux, Val(D), g, NHYDRO)
    return nothing
end

"""
    hydro_rhs!(dU, U, sim)

Method-of-lines right-hand side of the Euler equations: fill ghost cells,
convert to primitives (applying floors), then accumulate `-div F` from a
directional sweep per active dimension.
"""
function hydro_rhs!(dU::Array{T,4}, U::Array{T,4}, sim::Simulation{T}) where {T}
    g = sim.grid
    fill_ghosts!(U, sim.boundaries, g, sim.t)
    sim.diag.floors_hydro += enforce_floors!(U, sim.eos, g, sim.floors)
    cons2prim!(sim.hw.prim, U, sim.eos, g)
    fill!(dU, zero(T))
    g.active[1] && _hydro_sweep!(dU, sim, Val(1))
    g.active[2] && _hydro_sweep!(dU, sim, Val(2))
    g.active[3] && _hydro_sweep!(dU, sim, Val(3))
    return dU
end

function _rad_sweep!(dR, R, sim::Simulation, ::Val{D}) where {D}
    rw = sim.rw
    g = sim.grid
    nrad = nradvars(sim.radiation)
    reconstruct!(rw.qL, rw.qR, rw.prim, sim.radiation_reconstruction, Val(D), g, nrad)
    rad_fluxes!(rw.flux, rw.qL, rw.qR, R, rw.chat, sim.radiation, Val(D), g)
    flux_divergence!(dR, rw.flux, Val(D), g, nrad)
    return nothing
end

"""
    rad_rhs!(dR, R, sim, t)

Method-of-lines right-hand side of the M1 transport operator (the explicit part
of the IMEX scheme). Repairs `R` in place first, so the state handed to the HLL
solver always satisfies `E > 0` and `|F| < cE`.
"""
function rad_rhs!(dR::Array{T,4}, R::Array{T,4}, sim::Simulation{T}, t::T) where {T}
    g = sim.grid
    fill_ghosts!(R, sim.radiation_boundaries, g, t)
    sim.diag.floors_rad += enforce_rad_floors!(R, sim.radiation, g, sim.floors)
    rad_cons2prim!(sim.rw.prim, R, sim.radiation, g)
    fill!(dR, zero(T))
    g.active[1] && _rad_sweep!(dR, R, sim, Val(1))
    g.active[2] && _rad_sweep!(dR, R, sim, Val(2))
    g.active[3] && _rad_sweep!(dR, R, sim, Val(3))
    return dR
end

# --- stage drivers -----------------------------------------------------------

"""
    advance_hydro!(sim, dt)

One explicit SSP-RK step of the Euler equations.
"""
function advance_hydro!(sim::Simulation{T}, dt::T) where {T}
    g = sim.grid
    hw = sim.hw
    copyto!(hw.U0, sim.U)
    for (a, b, c) in tableau(sim.integrator)
        hydro_rhs!(hw.dU, sim.U, sim)
        combine!(sim.U, T(a), hw.U0, T(b), sim.U, T(c) * dt, hw.dU, g, NHYDRO)
    end
    return sim
end

# Recompute the local reduced speed of light for one IMEX stage from the state
# that stage is about to transport. A no-op under `ConstantChat`, down to the
# ghost fills -- which is what makes the default path bit-for-bit unchanged.
@inline _stage_chat!(sim::Simulation, U, R, dt, t) =
    _stage_chat!(sim, sim.radiation.chat_limiter, U, R, dt, t)

@inline _stage_chat!(::Simulation{T}, ::ConstantChat, U::Array{T,4}, R::Array{T,4},
                     dt::T, t::T) where {T} = nothing

function _stage_chat!(sim::Simulation{T}, ::AbstractChatLimiter, U::Array{T,4},
                      R::Array{T,4}, dt::T, t::T) where {T}
    g = sim.grid
    # The limiter is evaluated in ghost cells as well, so that the transport
    # step has a value on both sides of every interface bounding the domain.
    # `rad_rhs!` fills `R`'s ghosts again a moment later, which costs one
    # boundary sweep and keeps the ordering here independent of what it does.
    fill_ghosts!(U, sim.boundaries, g, t)
    fill_ghosts!(R, sim.radiation_boundaries, g, t)
    update_chat!(sim.rw.chat, U, R, sim.eos, sim.radiation, g, sim.floors, dt)
    return nothing
end

"""
    imex_substep!(sim, dt, t)

One IMEX PD-ARS substep of the radiation subsystem.

Stage 1 is the identity and is skipped. Stage 2 takes a forward-Euler transport
step into scratch storage and applies the implicit coupling there; stage 3
forms the Shu-Osher combination of the two stages, adds the transport of the
stage-2 state, and applies the implicit coupling again.

Each stage begins by recomputing the local reduced speed of light from the state
that stage transports, so the transport and the coupling that follows it see the
same `chat` -- the order the improved RSLA is derived for, in which every
explicit stage is immediately followed by the implicit solve that damps what it
amplified. With the default [`ConstantChat`](@ref) limiter the two
[`update_chat!`](@ref) calls are no-ops.
"""
function imex_substep!(sim::Simulation{T}, dt::T, t::T) where {T}
    g = sim.grid
    rw = sim.rw
    phys = sim.radiation
    fl = sim.floors
    upd = sim.hydro_enabled

    # --- stage 2: U(2) = U^n + dt Aex21 s(U^n), then implicit coupling -------
    _stage_chat!(sim, sim.U, sim.R, dt, t)
    rad_rhs!(rw.dR, sim.R, sim, t)
    combine!(rw.Rtmp, one(T), sim.R, zero(T), sim.R, T(IMEX_AEX_21) * dt, rw.dR, g,
             nradvars(sim.radiation))
    copyto!(rw.Utmp, sim.U)
    amend_rad_state!(rw.Rtmp, phys, g, fl)
    s1 = radiation_source_step!(rw.Utmp, rw.Rtmp, rw.chat, sim.eos, phys, g,
                                T(IMEX_AIM_22) * dt,
                                sim.source, t + dt, fl; update_momentum=upd)

    # --- stage 3: Shu-Osher combination + implicit coupling ------------------
    _stage_chat!(sim, rw.Utmp, rw.Rtmp, dt, t + dt)
    rad_rhs!(rw.dR, rw.Rtmp, sim, t + dt)
    combine!(sim.R, T(1 - IMEX_ALPHA), sim.R, T(IMEX_ALPHA), rw.Rtmp,
             T(IMEX_AEX_32) * dt, rw.dR, g, nradvars(sim.radiation))
    combine_gas!(sim.U, T(IMEX_ALPHA), rw.Utmp, g)
    amend_rad_state!(sim.R, phys, g, fl)
    s2 = radiation_source_step!(sim.U, sim.R, rw.chat, sim.eos, phys, g,
                                T(IMEX_AIM_33) * dt,
                                sim.source, t + dt, fl; update_momentum=upd)
    return s1 + s2
end

"""
    advance_radiation!(sim, dt)

Subcycle the radiation subsystem over one hydro step, using as many IMEX
substeps as the radiation CFL condition requires.
"""
function advance_radiation!(sim::Simulation{T}, dt::T) where {T}
    dtr = _rad_dt(sim)
    nsub = max(1, ceil(Int, dt / dtr))
    dts = dt / nsub
    stats = SourceStats()
    tt = sim.t
    for _ in 1:nsub
        stats += imex_substep!(sim, dts, tt)
        tt += dts
    end
    sim.diag.rad_substeps += nsub
    _record!(sim.diag, stats)
    return stats
end

# --- driver ------------------------------------------------------------------

# The radiation transport step: the RSLA CFL condition, plus -- only when a
# `chat` limiter is active -- the radiation-force feedback bound. That bound
# constrains the standard RSLA too, but adding a step restriction to the default
# path would silently change every existing result, so it rides with the
# limiter that also implements the other half of the same constraint.
@inline _rad_dt(sim::Simulation) = _rad_dt(sim, sim.radiation.chat_limiter)

@inline _rad_dt(sim::Simulation, ::ConstantChat) =
    radiation_timestep(sim.radiation, sim.grid, sim.cfl_radiation)

_rad_dt(sim::Simulation, ::AbstractChatLimiter) =
    min(radiation_timestep(sim.radiation, sim.grid, sim.cfl_radiation),
        radiation_pressure_dt(sim.U, sim.R, sim.eos, sim.radiation, sim.grid,
                              sim.cfl_radiation, sim.floors))

"""
    timestep(sim)

The step size allowed by the CFL conditions of the enabled physics.

With radiation enabled the hydro step is additionally capped so that it needs no
more than `sim.max_radiation_substeps` radiation substeps.
"""
function timestep(sim::Simulation{T}) where {T}
    dt = T(Inf)
    if sim.hydro_enabled
        # `sim.U` is admissible here: `step!` and `set_initial_conditions!` both
        # leave it that way, so this conversion needs no floors of its own.
        cons2prim!(sim.hw.prim, sim.U, sim.eos, sim.grid)
        dt = hydro_timestep(sim.hw.prim, sim.eos, sim.grid, sim.cfl)
    end
    if has_radiation(sim)
        dtr = _rad_dt(sim)
        if !sim.hydro_enabled
            dt = dtr
        elseif dt > sim.max_radiation_substeps * dtr
            dt = sim.max_radiation_substeps * dtr
        end
    end
    isfinite(dt) || throw(ErrorException("timestep is not finite; the state may be corrupt"))
    return dt
end

"""
    step!(sim, dt = timestep(sim))

Advance the simulation by one step. Returns the step size taken.

The hydrodynamic and radiation subsystems are advanced in sequence, which is
Lie (first-order) operator splitting -- see the note on splitting error in the
README. The step ends by projecting the state back onto the admissible set, so
that between steps `sim.U` and `sim.R` are always physical and every diagnostic
can read them without repairing anything.
"""
function step!(sim::Simulation{T}, dt::Real=timestep(sim)) where {T}
    dtT = T(dt)
    half = dtT / 2
    apply_strang_sources!(sim, sim.t, half, SourceForward)
    sim.hydro_enabled && advance_hydro!(sim, dtT)
    has_radiation(sim) && advance_radiation!(sim, dtT)
    apply_strang_sources!(sim, sim.t + dtT, half, SourceReverse)
    enforce_state!(sim)
    sim.t += dtT
    sim.dt = dtT
    sim.step += 1
    sim.diag.steps += 1
    return dtT
end

"""
    apply_strang_sources!(sim, t, dt, order) -> sim

Apply the user's operator-split source terms over `dt`, which is *half* the
step: [`step!`](@ref) calls this once before the advance with `SourceForward`
and once after with `SourceReverse`, which is what makes the pair second order.

The Julia counterpart of Quokka's `addStrangSplitSources`, called from the same
two places in the driver and with the same halved step.

The hook mutates `sim.U` and `sim.R` directly, so it has the grid, the equation
of state and the radiation parameters to hand. It is called once per half-step,
not once per cell, so it need not be allocation-free -- but a loop *inside* it
is in the hot path, and the boxing trap described in CLAUDE.md applies there.

```julia
# constant gravity along -y, as a momentum and energy kick
function gravity!(sim, t, dt, order)
    g = sim.grid
    for k in g.lo[3]:g.hi[3], j in g.lo[2]:g.hi[2], i in g.lo[1]:g.hi[1]
        rho = sim.U[Hydro3D.IRHO, i, j, k]
        dp = -0.1 * rho * dt
        sim.U[Hydro3D.IEGAS, i, j, k] +=
            dp * (sim.U[Hydro3D.IMY, i, j, k] + dp / 2) / rho
        sim.U[Hydro3D.IMY, i, j, k] += dp
    end
end
Simulation(n = (64, 64, 1), strang_sources = gravity!)
```

Nothing is implemented on top of this hook: no gravity, no cooling, no driving.
It exists so that adding one does not mean editing `step!`.
"""
@inline function apply_strang_sources!(sim::Simulation{T}, t::T, dt::T,
                                       order::SourceOrder) where {T}
    f = sim.strang_sources
    if f !== nothing
        f(sim, t, dt, order)
        # A user source can push a cell out of the admissible set, and the
        # radiation coupling reads `U` before any right-hand side would floor it.
        enforce_state!(sim)
    end
    return sim
end

"""
    nrepaired(sim) -> Int

Total cell updates that hit a positivity floor, gas and radiation together.
Zero is the invariant worth asserting in a test: a scheme that needs no help
from the floors is one whose positivity you have actually verified.
"""
nrepaired(sim::Simulation) = sim.diag.floors_hydro + sim.diag.floors_rad

"""
    enforce_state!(sim) -> Int

Project both subsystems onto their admissible sets and accumulate the repair
count. Called at the end of every step and after the initial conditions, which
is what lets the conversion kernels and every diagnostic path stay read-only.

The analogue of Quokka's `EnforceLimits`, called from the same place in the
driver.
"""
function enforce_state!(sim::Simulation)
    n = 0
    if sim.hydro_enabled
        m = enforce_floors!(sim.U, sim.eos, sim.grid, sim.floors)
        sim.diag.floors_hydro += m
        n += m
    end
    if has_radiation(sim)
        m = amend_rad_state!(sim.R, sim.radiation, sim.grid, sim.floors)
        sim.diag.floors_rad += m
        n += m
    end
    return n
end

"""
    run!(sim; tmax, dtout = nothing, maxsteps = typemax(Int),
         outdir = nothing, basename = "data", verbose = true, callback = nothing)

Integrate to `tmax`.

  - `dtout`    : if given (together with `outdir`), write a snapshot every
    `dtout` of simulation time; the step size is trimmed to land exactly on the
    output times.
  - `outdir`   : directory for VTK output; created if needed.
  - `callback` : called as `callback(sim)` after each step. Return `false` to
    stop early.

Returns `sim`.
"""
function run!(sim::Simulation{T}; tmax, dtout=nothing, maxsteps::Integer=typemax(Int),
              outdir=nothing, basename::AbstractString="data",
              verbose::Bool=true, callback=nothing) where {T}
    tend = T(tmax)
    sim.t < tend || throw(ArgumentError("tmax = $tmax is not after the current time $(sim.t)"))
    writing = outdir !== nothing && dtout !== nothing
    nout = 0
    if outdir !== nothing
        mkpath(outdir)
        write_vtk(sim, joinpath(outdir, @sprintf("%s_%05d", basename, nout)))
        nout += 1
    end
    tout = writing ? sim.t + T(dtout) : tend

    verbose && _print_header(sim, tend)
    t0 = time()
    while sim.t < tend && sim.step < maxsteps
        dt = timestep(sim)
        dt = min(dt, tend - sim.t)
        writing && (dt = min(dt, tout - sim.t))
        step!(sim, dt)

        if verbose && (sim.step % 10 == 0 || sim.t >= tend)
            @printf("  step %6d   t = %-12.6g dt = %-12.6g\n", sim.step, sim.t, sim.dt)
        end
        if writing && sim.t >= tout - 8 * eps(tout)
            write_vtk(sim, joinpath(outdir, @sprintf("%s_%05d", basename, nout)))
            nout += 1
            tout += T(dtout)
        end
        if callback !== nothing && callback(sim) === false
            verbose && println("  stopped by callback")
            break
        end
    end
    if outdir !== nothing && !writing
        write_vtk(sim, joinpath(outdir, @sprintf("%s_%05d", basename, nout)))
    end
    verbose && _print_summary(sim, time() - t0)
    return sim
end

function _print_summary(sim::Simulation, elapsed)
    d = sim.diag
    ncell = prod(sim.grid.n)
    @printf("done: %d steps, t = %.6g, %.2f s (%.3g cell-updates/s)\n",
            sim.step, sim.t, elapsed, elapsed > 0 ? ncell * sim.step / elapsed : 0.0)
    # A handful of floor hits is normal wherever the solution genuinely reaches
    # vacuum (deep shadows, the interior of a blast). A count comparable to the
    # number of cell updates means the run is unphysical.
    nfl = d.floors_hydro + d.floors_rad
    nfl > 0 && @printf("note: %d cell updates hit a positivity floor (%d gas, %d radiation)\n",
                       nfl, d.floors_hydro, d.floors_rad)
    if d.newton_cells > 0
        @printf("      implicit coupling: %.2f Newton iterations per cell, %d worst case, %d substeps\n",
                d.newton_iterations / d.newton_cells, d.newton_max, d.rad_substeps)
        # Not a note. A cell that exhausted `max_iterations` did not solve the
        # coupling, so the energy exchange in it is whatever the last iterate
        # happened to be.
        d.newton_failures > 0 &&
            @printf("WARNING: the implicit solve failed to converge in %d cell updates\n",
                    d.newton_failures)
    end
    return nothing
end

function _print_header(sim::Simulation, tend)
    g = sim.grid
    phys = has_radiation(sim) ? "radiation-hydrodynamics" :
           sim.hydro_enabled ? "hydrodynamics" : "radiation transport"
    println("Hydro3D: 3D ", phys)
    @printf("  grid %d x %d x %d (%dD), %s + %s, %d thread(s)\n",
            g.n[1], g.n[2], g.n[3], ndim(g),
            nameof(typeof(sim.reconstruction)), nameof(typeof(sim.integrator)),
            Threads.nthreads())
    @printf("  integrating from t = %g to t = %g\n", sim.t, tend)
end

# --- diagnostics -------------------------------------------------------------

"""
    conserved_totals(sim)

Volume integrals of the conserved quantities over the interior, returned as a
`NamedTuple` `(mass, momentum, energy_gas, energy_rad, energy_total)`. Useful as
a conservation check; note that the total energy is only conserved exactly when
every boundary is periodic or reflecting and there are no sources.
"""
function conserved_totals(sim::Simulation{T}) where {T}
    g = sim.grid
    dV = cellvolume(g)
    mass = zero(T)
    px = zero(T)
    py = zero(T)
    pz = zero(T)
    egas = zero(T)
    erad = zero(T)
    @inbounds for k in g.lo[3]:g.hi[3], j in g.lo[2]:g.hi[2], i in g.lo[1]:g.hi[1]
        mass += sim.U[IRHO, i, j, k]
        px += sim.U[IMX, i, j, k]
        py += sim.U[IMY, i, j, k]
        pz += sim.U[IMZ, i, j, k]
        egas += sim.U[IEGAS, i, j, k]
        if has_radiation(sim)
            for gr in 1:ngroups(sim.radiation)
                erad += sim.R[gbase(gr)+IERAD, i, j, k]
            end
        end
    end
    (mass=mass * dV, momentum=(px * dV, py * dV, pz * dV),
     energy_gas=egas * dV, energy_rad=erad * dV,
     energy_total=(egas + erad) * dV)
end

"""
    primitives(sim)

A fresh `(NHYDRO, nx, ny, nz)` array of interior primitive variables
`(rho, vx, vy, vz, p)`. Convenience for analysis and plotting.

Does not modify `sim`. This is load-bearing, not incidental: it used to apply
the positivity floors to `sim.U` as a side effect, so a run that was plotted
every ten steps and one that was not could quietly disagree.
"""
function primitives(sim::Simulation{T}) where {T}
    g = sim.grid
    cons2prim!(sim.hw.prim, sim.U, sim.eos, g)
    sim.hw.prim[:, g.lo[1]:g.hi[1], g.lo[2]:g.hi[2], g.lo[3]:g.hi[3]]
end

"""
    gas_temperature(sim)

A fresh `(nx, ny, nz)` array of gas temperatures over the interior.
"""
function gas_temperature(sim::Simulation{T}) where {T}
    g = sim.grid
    out = Array{T,3}(undef, g.n...)
    @inbounds for k in 1:g.n[3], j in 1:g.n[2], i in 1:g.n[1]
        u = getstate(sim.U, Val(NHYDRO), i + g.ng[1], j + g.ng[2], k + g.ng[3])
        out[i, j, k] = temperature(sim.eos, u[IRHO], internal_energy(u))
    end
    out
end

"""
    radiation_temperature(sim)

A fresh `(nx, ny, nz)` array of radiation temperatures `(E_rad / a_rad)^(1/4)`.
"""
function radiation_temperature(sim::Simulation{T}) where {T}
    has_radiation(sim) || throw(ArgumentError("radiation is not enabled"))
    g = sim.grid
    arad = sim.radiation.a_rad
    out = Array{T,3}(undef, g.n...)
    @inbounds for k in 1:g.n[3], j in 1:g.n[2], i in 1:g.n[1]
        # The radiation temperature is set by the *total* energy density: the
        # Planck fractions sum to one, so a_rad T^4 is the sum over groups.
        # Reading group 1 alone would report the temperature of one bin.
        er = zero(T)
        for gr in 1:ngroups(sim.radiation)
            er += sim.R[gbase(gr)+IERAD, i+g.ng[1], j+g.ng[2], k+g.ng[3]]
        end
        out[i, j, k] = sqrt(sqrt(max(er, zero(T)) / arad))
    end
    out
end
