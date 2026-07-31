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

# The radiation state variables, one set per frequency group: name, offset
# within a group, label. Both the group sums and the `Erad_2`-style per-group
# names are built from this list.
const _RAD_VARS = ((:Erad, IERAD, "radiation energy density"),
                   (:Fx, IFRX, "x radiation flux"),
                   (:Fy, IFRY, "y radiation flux"),
                   (:Fz, IFRZ, "z radiation flux"))

"""
    available_fields(sim) -> Tuple{Vararg{Symbol}}

The field names [`field`](@ref) accepts for this simulation. Radiation fields
are only present when the radiation module is enabled.

A multigroup run also lists `:Erad_1`, `:Fx_1`, ... one set per group. A grey
run does not: with one group those names are accepted but they name the same
array as `:Erad`, `:Fx`, ..., and advertising both would suggest a distinction
that is not there.
"""
function available_fields(sim::Simulation)
    hydro = (:density, :vx, :vy, :vz, :speed, :pressure, :temperature,
             :internal_energy, :mach)
    has_radiation(sim) || return hydro
    rad = (:Erad, :Fx, :Fy, :Fz, :Trad, :reduced_flux)
    ng = ngroups(sim.radiation)
    ng == 1 && return (hydro..., rad...)
    per = Tuple(Symbol(v, :_, gr) for gr in 1:ng for (v, _, _) in _RAD_VARS)
    return (hydro..., rad..., per...)
end

# One radiation component summed over the groups, as an (nx, ny, nz) array.
_rad_total(sim::Simulation, comp::Int) = dropdims(_rad_group_sum(sim, comp:comp); dims=1)

# `:Erad_2` -> (:Erad, 2); `nothing` for a name with no numeric group suffix, so
# `:internal_energy` and `:reduced_flux` pass straight through.
function _group_suffix(name::Symbol)
    s = String(name)
    i = findlast('_', s)
    i === nothing && return nothing
    gr = tryparse(Int, s[(i+1):end])
    gr === nothing && return nothing
    return (Symbol(s[1:(i-1)]), gr)
end

"""
    field(sim, name) -> (data, label)

Extract one named field over the interior as a fresh `(nx, ny, nz)` array,
together with a label suitable for an axis or colour bar.

See [`available_fields`](@ref) for the accepted names. `:Erad`, `:Fx`, `:Fy`,
`:Fz` and `:reduced_flux` are summed over the frequency groups, so they mean the
same thing in a grey run and a multigroup one; a trailing group number picks one
group out, as in `field(sim, :Erad_2)`.

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

    # Per-group access: `Erad_2` is group 2's energy density. Handled here, ahead
    # of the primitives, because it needs none of them. No plain field name ends
    # in an underscore and a number, so nothing already named can be shadowed.
    sg = _group_suffix(name)
    if sg !== nothing
        base, gr = sg
        i = findfirst(v -> first(v) === base, _RAD_VARS)
        if i !== nothing
            _require_radiation(sim, name)
            _, comp, label = _RAD_VARS[i]
            ng = ngroups(sim.radiation)
            1 <= gr <= ng ||
                throw(ArgumentError("field $name asks for group $gr of $ng"))
            return sim.R[gbase(gr)+comp, ir, jr, kr], "group $gr $label"
        end
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
        return _rad_total(sim, IERAD), "radiation energy density"
    elseif name === :Fx
        _require_radiation(sim, name)
        return _rad_total(sim, IFRX), "x radiation flux"
    elseif name === :Fy
        _require_radiation(sim, name)
        return _rad_total(sim, IFRY), "y radiation flux"
    elseif name === :Fz
        _require_radiation(sim, name)
        return _rad_total(sim, IFRZ), "z radiation flux"
    elseif name === :reduced_flux
        _require_radiation(sim, name)
        c = sim.radiation.c
        # Group sums, and still causal: each group has |F_g| <= c E_g, so the
        # summed flux obeys |sum F_g| <= sum |F_g| <= c sum E_g.
        E = _rad_total(sim, IERAD)
        F = sqrt.(_rad_total(sim, IFRX) .^ 2 .+ _rad_total(sim, IFRY) .^ 2 .+
                  _rad_total(sim, IFRZ) .^ 2)
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

A table with only one resolution -- one built by comparing prebuilt
[`Simulation`](@ref)s -- is drawn as a bar chart instead. There is no trend to
put on a log-log axis, and an axis that implies one would be a claim the run
never measured.

**Requires `using Plots`.**
"""
function plotcompare end

# Informative fallbacks. The extension's methods are more specific, so they win
# whenever Plots is loaded.
plotstate(::Any...; kwargs...) = error("plotstate: ", _PLOTS_HINT)
plotpanels(::Any...; kwargs...) = error("plotpanels: ", _PLOTS_HINT)
animate_run(::Any...; kwargs...) = error("animate_run: ", _PLOTS_HINT)
plotcompare(::Any...; kwargs...) = error("plotcompare: ", _PLOTS_HINT)
