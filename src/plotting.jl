# Field extraction, and the plotting entry points.
#
# `field` is useful on its own and needs nothing external. The three plotting
# functions are declared here but left without methods: those live in
# `ext/Hydro3DPlotsExt.jl` and appear as soon as the user loads Plots. Keeping
# them out of the package proper means solving does not drag in a plotting
# stack.

"""
    coordinates(grid, d) -> Vector

Cell-centre coordinates of the interior cells along direction `d`.
"""
function coordinates(g::CartesianGrid{T}, d::Integer) where {T}
    [g.xlo[d] + (i - T(0.5)) * g.dx[d] for i in 1:g.n[d]]
end

"""
    available_fields(sim) -> Tuple{Vararg{Symbol}}

The field names [`field`](@ref) accepts for this simulation. Radiation fields
are only present when the radiation module is enabled.
"""
function available_fields(sim::Simulation)
    hydro = (:density, :vx, :vy, :vz, :speed, :pressure, :temperature,
             :internal_energy, :mach)
    has_radiation(sim) ? (hydro..., :Erad, :Fx, :Fy, :Fz, :Trad, :reduced_flux) : hydro
end

"""
    field(sim, name) -> (data, label)

Extract one named field over the interior as a fresh `(nx, ny, nz)` array,
together with a label suitable for an axis or colour bar.

See [`available_fields`](@ref) for the accepted names.

```julia
rho, lbl = field(sim, :density)
```
"""
function field(sim::Simulation{T}, name::Symbol) where {T}
    g = sim.grid
    ir, jr, kr = g.lo[1]:g.hi[1], g.lo[2]:g.hi[2], g.lo[3]:g.hi[3]
    if name === :temperature
        return gas_temperature(sim), "gas temperature"
    elseif name === :Trad
        _require_radiation(sim, name)
        return radiation_temperature(sim), "radiation temperature"
    end

    prim = primitives(sim)
    if name === :density
        return prim[IRHO, :, :, :], "density"
    elseif name === :vx
        return prim[IVX, :, :, :], "x velocity"
    elseif name === :vy
        return prim[IVY, :, :, :], "y velocity"
    elseif name === :vz
        return prim[IVZ, :, :, :], "z velocity"
    elseif name === :pressure
        return prim[IPRE, :, :, :], "pressure"
    elseif name === :speed
        return sqrt.(prim[IVX, :, :, :] .^ 2 .+ prim[IVY, :, :, :] .^ 2 .+
                     prim[IVZ, :, :, :] .^ 2), "speed"
    elseif name === :internal_energy
        return internal_energy.(Ref(sim.eos), prim[IRHO, :, :, :], prim[IPRE, :, :, :]),
               "internal energy"
    elseif name === :mach
        v = sqrt.(prim[IVX, :, :, :] .^ 2 .+ prim[IVY, :, :, :] .^ 2 .+ prim[IVZ, :, :, :] .^ 2)
        return v ./ soundspeed.(Ref(sim.eos), prim[IRHO, :, :, :], prim[IPRE, :, :, :]),
               "Mach number"
    elseif name === :Erad
        _require_radiation(sim, name)
        return sim.R[IERAD, ir, jr, kr], "radiation energy density"
    elseif name === :Fx
        _require_radiation(sim, name)
        return sim.R[IFRX, ir, jr, kr], "x radiation flux"
    elseif name === :Fy
        _require_radiation(sim, name)
        return sim.R[IFRY, ir, jr, kr], "y radiation flux"
    elseif name === :Fz
        _require_radiation(sim, name)
        return sim.R[IFRZ, ir, jr, kr], "z radiation flux"
    elseif name === :reduced_flux
        _require_radiation(sim, name)
        c = sim.radiation.c
        E = sim.R[IERAD, ir, jr, kr]
        F = sqrt.(sim.R[IFRX, ir, jr, kr] .^ 2 .+ sim.R[IFRY, ir, jr, kr] .^ 2 .+
                  sim.R[IFRZ, ir, jr, kr] .^ 2)
        return F ./ (c .* E), "reduced flux |F|/(cE)"
    end
    throw(ArgumentError("unknown field $name; available: $(available_fields(sim))"))
end

function _require_radiation(sim::Simulation, name::Symbol)
    has_radiation(sim) ||
        throw(ArgumentError("field $name needs the radiation module, which is disabled"))
end

const _PLOTS_HINT = "run `using Plots` first; Hydro3D's plotting methods live in a package extension"

"""
    plotstate(sim; var = :density, slice = nothing, kwargs...)

Plot one field of the simulation, choosing the representation from the
dimensionality: a line plot in 1D, a heat map in 2D, and a heat map of the
mid-plane slice in 3D.

`var` is any name from [`available_fields`](@ref). `slice` overrides the slice
index in 3D. Remaining keyword arguments pass through to Plots.

**Requires `using Plots`.**

```julia
using Plots, Hydro3D
sim, tmax = Hydro3D.Problems.kelvin_helmholtz(n = 256)
run!(sim; tmax)
plotstate(sim, var = :density)
```
"""
function plotstate end

"""
    plotpanels(sim; vars = nothing, kwargs...)

A multi-panel overview of the state: density, velocity, pressure and gas
temperature, plus radiation energy and temperature when radiation is enabled.
Pass `vars` to choose the panels yourself.

**Requires `using Plots`.**
"""
function plotpanels end

"""
    animate_run(sim; tmax, nframes = 100, var = :density, fps = 20,
                filename = "hydro3d.gif", kwargs...)

Integrate `sim` to `tmax` while capturing `nframes` evenly spaced frames of
`var`, and write an animated GIF. Returns the file name.

Named `animate_run` rather than `animate` so that it does not collide with
`Plots.animate` when both packages are in scope.

**Requires `using Plots`.**

```julia
using Plots, Hydro3D
sim, tmax = Hydro3D.Problems.kelvin_helmholtz(n = 256)
animate_run(sim; tmax, var = :density, filename = "kh.gif")
```
"""
function animate_run end

"""
    plotcompare(table_or_path...; metric = nothing, kwargs...)

Plot one or more [`ComparisonTable`](@ref)s: each metric against resolution, one
line per variant, on log-log axes with the measured convergence rate in the
legend.

Accepts tables or the paths to tables written by [`compare`](@ref), so a figure
can be redrawn from stored numbers without re-running any simulation:

```julia
using Plots, Hydro3D
plotcompare("radhydro_shock_compare.txt")
```

Give it several tables to put the same schemes on different problems side by
side, each panel labelled with its problem. That comparison is the point --
one problem gives a ranking, two tell you whether the ranking means anything:

```julia
plotcompare("sod_shock_tube_riemann_compare.txt",
            "smooth_advection_riemann_compare.txt")
```

Pass `metric` to draw one panel per table instead of all of them.

**Requires `using Plots`.**
"""
function plotcompare end

# Informative fallbacks. The extension's methods are more specific, so they win
# whenever Plots is loaded.
plotstate(::Any...; kwargs...) = error("plotstate: ", _PLOTS_HINT)
plotpanels(::Any...; kwargs...) = error("plotpanels: ", _PLOTS_HINT)
animate_run(::Any...; kwargs...) = error("animate_run: ", _PLOTS_HINT)
plotcompare(::Any...; kwargs...) = error("plotcompare: ", _PLOTS_HINT)
