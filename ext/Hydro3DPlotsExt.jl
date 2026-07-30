"""
    Hydro3DPlotsExt

Plotting methods for Hydro3D, loaded automatically when Plots is available.

Provides methods for [`Hydro3D.plotstate`](@ref), [`Hydro3D.plotpanels`](@ref)
[`Hydro3D.animate_run`](@ref) and [`Hydro3D.plotcompare`](@ref), plus a Plots
recipe so that `plot(sim)` works directly.
"""
module Hydro3DPlotsExt

using Hydro3D
using Hydro3D: Simulation, CartesianGrid, ndim, field, coordinates, has_radiation,
               available_fields, ComparisonTable, read_comparison, convergence_rate,
               metric_columns, is_error_metric
using Plots
using Printf

# Pick the two in-plane directions of a 2D or 3D field, and the slice index of
# the third. For 3D the default slice is the middle of the domain.
function _plane(sim::Simulation, slice)
    g = sim.grid
    act = findall(collect(g.active))
    if length(act) <= 1
        return (isempty(act) ? 1 : act[1]), 0, 0
    elseif length(act) == 2
        return act[1], act[2], 1
    else
        return 1, 2, (slice === nothing ? cld(g.n[3], 2) : slice)
    end
end

_axislabel(d) = ("x", "y", "z")[d]

function _title(sim::Simulation, label)
    @sprintf("%s   (t = %.4g, step %d)", label, sim.t, sim.step)
end

function Hydro3D.plotstate(sim::Simulation; var::Symbol=:density, slice=nothing,
                           logscale::Bool=false, kwargs...)
    g = sim.grid
    data, label = field(sim, var)
    d1, d2, k = _plane(sim, slice)

    # Defaults come first so that anything the caller passes overrides them.
    if d2 == 0                                  # one active direction: line plot
        x = coordinates(g, d1)
        kw = (; xlabel=_axislabel(d1), ylabel=label, legend=false,
              title=_title(sim, label), lw=2, kwargs...)
        return plot(x, vec(data); kw...)
    end

    # two or three active directions: heat map of the (d1, d2) plane
    x = coordinates(g, d1)
    y = coordinates(g, d2)
    z = ndim(g) == 2 ? data[:, :, 1] : selectdim(data, 3, k)
    z = logscale ? log10.(max.(z, eps(eltype(z)))) : z
    cb = logscale ? "log10 " * label : label
    ttl = _title(sim, label)
    ndim(g) == 3 && (ttl *= @sprintf("  [%s slice %d]", _axislabel(3), k))
    # the right margin keeps a long colour-bar title off the tick labels
    kw = (; xlabel=_axislabel(d1), ylabel=_axislabel(d2), title=ttl,
          colorbar_title=cb, aspect_ratio=:equal, xlims=extrema(x), ylims=extrema(y),
          right_margin=4Plots.mm, kwargs...)
    # transpose: Plots wants (y, x) for a heat map of z(x, y)
    heatmap(x, y, permutedims(z); kw...)
end

function Hydro3D.plotpanels(sim::Simulation; vars=nothing, slice=nothing,
                            size=(950, 620), kwargs...)
    picked = vars === nothing ?
             (has_radiation(sim) ?
              (:density, :speed, :pressure, :temperature, :Erad, :Trad) :
              (:density, :speed, :pressure, :temperature)) : Tuple(vars)
    panels = [Hydro3D.plotstate(sim; var=v, slice=slice, kwargs...) for v in picked]
    ncol = length(panels) <= 4 ? 2 : 3
    plot(panels...; layout=(cld(length(panels), ncol), ncol), size=size)
end

function Hydro3D.animate_run(sim::Simulation; tmax, nframes::Integer=100,
                             var::Symbol=:density, fps::Integer=20,
                             filename::AbstractString="hydro3d.gif",
                             clims=nothing, verbose::Bool=true, kwargs...)
    t0 = sim.t
    dtf = (tmax - t0) / nframes
    dtf > 0 || throw(ArgumentError("tmax = $tmax is not after the current time $t0"))

    # Pin the colour range so the animation does not flicker as the extrema
    # drift. Taken from the initial state unless the caller supplies it, which
    # they must when plotting on a log scale.
    if clims === nothing
        lo, hi = extrema(first(field(sim, var)))
        clims = lo == hi ? nothing : (lo, hi)
    end
    plotkw = clims === nothing ? kwargs : (; clims=clims, kwargs...)

    anim = Animation()
    frame(anim, Hydro3D.plotstate(sim; var=var, plotkw...))
    for m in 1:nframes
        Hydro3D.run!(sim; tmax=min(t0 + m * dtf, tmax), verbose=false)
        frame(anim, Hydro3D.plotstate(sim; var=var, plotkw...))
        verbose && m % 10 == 0 && @printf("  frame %d/%d  t = %.4g\n", m, nframes, sim.t)
    end
    gif(anim, filename; fps=fps, show_msg=false)
    return filename
end

# `plot(sim)` and `plot!(sim)` go through plotstate's default field.
@recipe function f(sim::Simulation; var=:density)
    g = sim.grid
    data, label = field(sim, var)
    d1, d2, k = _plane(sim, nothing)
    if d2 == 0
        seriestype --> :path
        xguide --> _axislabel(d1)
        yguide --> label
        legend --> false
        linewidth --> 2
        return coordinates(g, d1), vec(data)
    else
        z = ndim(g) == 2 ? data[:, :, 1] : selectdim(data, 3, k)
        seriestype --> :heatmap
        xguide --> _axislabel(d1)
        yguide --> _axislabel(d2)
        aspect_ratio --> :equal
        return coordinates(g, d1), coordinates(g, d2), permutedims(z)
    end
end

# --- comparison tables -------------------------------------------------------

Hydro3D.plotcompare(path::AbstractString, more::AbstractString...; kwargs...) =
    Hydro3D.plotcompare(read_comparison(path), map(read_comparison, more)...; kwargs...)

function Hydro3D.plotcompare(t::ComparisonTable, more::ComparisonTable...;
                             metric=nothing, kwargs...)
    ts = (t, more...)
    any(isempty, ts) && throw(ArgumentError("plotcompare: the table is empty"))
    # One panel per (table, metric). Several tables in one figure is how you
    # answer "do the schemes rank the same way on a different problem", which
    # is the question a single sweep cannot settle.
    pairs = [(u, m) for u in ts
             for m in (metric === nothing ? metric_columns(u) : [String(metric)])]
    isempty(pairs) && throw(ArgumentError("plotcompare: the table has no metric columns"))
    named = length(ts) > 1

    panels = [(_sweeps(u) ? _panel : _barpanel)(u, m; title=(named ? u.problem : ""))
              for (u, m) in pairs]

    # Stamp the code version onto the figure. A convergence plot without one is
    # a number you cannot go back to.
    head = join(filter(!isempty, (named ? "" : t.problem, _stamp(t))), "   ")
    n = length(panels)
    defaults = (; layout=(1, n), size=(460n + 80, 440),
                plot_title=head, plot_titlefontsize=11,
                plot_titlevspan=0.09, bottom_margin=6Plots.mm, left_margin=8Plots.mm,
                right_margin=4Plots.mm)
    plot(panels...; defaults..., kwargs...)
end

function _stamp(t::ComparisonTable)
    i = findfirst(p -> first(p) == "code", t.meta)
    return i === nothing ? "" : "[" * last(t.meta[i]) * "]"
end

_sweeps(t::ComparisonTable) = length(unique(t[:size])) > 1

# A table from prebuilt simulations has one resolution, so there is nothing to
# put on a log-log axis -- every series would be a lone marker and the axis
# would imply a trend that was never measured. A grouped bar is the honest
# picture of a head-to-head, on a linear axis because a bar chart is read from
# zero.
function _barpanel(t::ComparisonTable, m::AbstractString; title="")
    bv = Hydro3D._by_variant(t)
    labels = [v for (v, _) in bv]
    y = [t[m][k][1] for (_, k) in bv]
    bar(labels, y; legend=false, ylabel=m, title=title, titlefontsize=10,
        lw=0, bar_width=0.6)
end

function _panel(t::ComparisonTable, m::AbstractString; title="")
    sizes = t[:size]
    y = t[m]
    # tick at the resolutions actually run: log decades are useless when the
    # sweep spans less than one
    ticks = sort!(unique(sizes))
    # `spike` converges *to* 4.26 rather than to zero, so it gets a linear axis
    # and no slope. Quote a rate only for the variants where the error falls
    # monotonically -- M1 scored against a P1 reference converges to the wrong
    # answer, and its non-monotone error should not be labelled with a rate.
    errorlike = is_error_metric(t, m)
    pl = plot(; xscale=:log10, yscale=(errorlike ? :log10 : :identity),
              xlabel="cells per side", ylabel=m, title=title, titlefontsize=10,
              xticks=(ticks, string.(Int.(ticks))), legend=:best)
    for (i, (v, k)) in enumerate(Hydro3D._by_variant(t))
        lbl = errorlike && Hydro3D._decreasing(y[k]) ?
              @sprintf("%s  (rate %.2f)", v, convergence_rate(t, m; variant=v)) : v
        plot!(pl, sizes[k], y[k]; marker=:circle, ms=4, lw=2, c=i, label=lbl)
    end
    return pl
end

end # module Hydro3DPlotsExt
