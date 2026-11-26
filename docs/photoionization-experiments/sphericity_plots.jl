#!/usr/bin/env julia
#
# Figures from the output of sphericity.jl.
#
#   julia --project=. docs/photoionization-experiments/sphericity_plots.jl [kernel ...] [--n=64] [--solver=be|ros] [--outdir=DIR]
#
# For each kernel, sphericity_<kernel>_<solver>.png: the neutral fraction (linear, 0 to 1) in the mid-plane
# z = -dx/2 (top) and the front radius over all directions, R(phi, cos theta) / R_mean - 1
# (bottom), at 100 and 500 Myr. With `--fig=TAG`, the two figures of the set are named
# sphericity_comparison_TAG_<solver>.png and sphericity_summary_TAG_<solver>.png. And sphericity_summary_<solver>.png: the anisotropy
# (R_body - R_axis) / R_mean and the spread std(R) / R_mean against time, one line per kernel.

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
include(joinpath(@__DIR__, "sphericity.jl"))
using Plots
default(fontfamily="sans-serif", framestyle=:box, dpi=140, titlefontsize=9, guidefontsize=8,
        legendfontsize=7)

const PLOT_TIMES = (100, 500)
# `--fig=TAG` names a set: sphericity_comparison_TAG_<solver>.png and sphericity_summary_TAG_<solver>.png.
const FIG = (t = opt("fig", ""); isempty(t) ? "" : "_" * t)

readx(name, tm) = (a = Array{Float32}(undef, N, N, N);
                   read!(joinpath(OUTDIR, "x_$(name)_$(SOLVER)_$(tm)Myr.f32"), a); Float64.(a))

function rmap(x, xlo, dx; nphi=96, nmu=48)
    phis = range(-pi, pi, length=nphi)
    mus = range(-1, 1, length=nmu)
    R = [front_radius(x, (sqrt(1 - m^2) * cos(p), sqrt(1 - m^2) * sin(p), m), xlo, dx)
         for m in mus, p in phis]
    return phis, mus, R ./ mean(filter(!isnan, R)) .- 1
end

function kernel_figure(name)
    L = 6.6 * KPC
    xlo, dx = -L, 2L / N
    c = [xlo + (i - 0.5) * dx for i in 1:N] ./ KPC
    panels = []
    for tm in PLOT_TIMES
        x = readx(name, tm)
        push!(panels, heatmap(c, c, permutedims(1 .- x[:, :, N ÷ 2]);
                              c=:viridis, clims=(0, 1),
                              xlims=extrema(c), ylims=extrema(c), xlabel="x (kpc)", ylabel="y (kpc)",
                              colorbar_title="x_H0", title="$name, $tm Myr: mid-plane"))
    end
    for tm in PLOT_TIMES
        phis, mus, dR = rmap(readx(name, tm), xlo, dx)
        a = maximum(abs, filter(!isnan, dR))
        push!(panels, heatmap(phis, mus, dR; c=:balance, clims=(-a, a), xlabel="phi",
                              ylabel="cos theta", colorbar_title="R / R_mean - 1",
                              title="$name, $tm Myr: front radius"))
    end
    p = plot(panels...; layout=(2, 2), size=(1100, 900), left_margin=22Plots.mm, right_margin=4Plots.mm, bottom_margin=8Plots.mm, top_margin=3Plots.mm)
    savefig(p, joinpath(OUTDIR, "sphericity_$(name)_$(SOLVER).png"))
end

function readtable(name)
    rows = [parse.(Float64, split(l)) for l in eachline(joinpath(OUTDIR, "sphericity_$(name)_$(SOLVER).txt"))
            if !startswith(l, "#")]
    return reduce(hcat, rows)'
end

function summary_figure(names)
    pa = plot(xlabel="t (Myr)", ylabel="(R_body - R_axis) / R_mean (%)", title="anisotropy",
              legend=:bottomright)
    ps = plot(xlabel="t (Myr)", ylabel="std(R) / R_mean (%)", title="spread", legend=false)
    for name in names
        t = readtable(name)
        plot!(pa, t[:, 1], 100t[:, 10]; m=:circle, label=name)
        plot!(ps, t[:, 1], 100t[:, 4]; m=:circle, label=name)
    end
    hline!(pa, [0]; c=:gray, ls=:dot, label="")
    savefig(plot(pa, ps; layout=(1, 2), size=(1000, 400), xscale=:log10,
                 left_margin=22Plots.mm, right_margin=4Plots.mm, bottom_margin=8Plots.mm, top_margin=3Plots.mm),
            joinpath(OUTDIR, "sphericity_summary$(FIG)_$(SOLVER).png"))
end

# All kernels in one figure, one row each: the mid-plane at 100 and 500 Myr, and the front
# radius at 500 Myr on one colour scale, so that the rows can be compared.
function comparison_figure(names; tm_map=500, lim=0.15)
    L = 6.6 * KPC
    xlo, dx = -L, 2L / N
    c = [xlo + (i - 0.5) * dx for i in 1:N] ./ KPC
    panels = []
    for name in names
        for tm in PLOT_TIMES
            x = readx(name, tm)
            push!(panels, heatmap(c, c, permutedims(1 .- x[:, :, N ÷ 2]);
                                  c=:viridis, clims=(0, 1),
                                  xlims=extrema(c), ylims=extrema(c), colorbar=false,
                                  xlabel="x (kpc)", ylabel="y (kpc)",
                                  title="$name, $tm Myr: x_H0, 0 (dark) to 1"))
        end
        phis, mus, dR = rmap(readx(name, tm_map), xlo, dx)
        push!(panels, heatmap(phis, mus, dR; c=:balance, clims=(-lim, lim), xlabel="phi",
                              ylabel="cos theta", colorbar_title="R / R_mean - 1",
                              title="$name, $tm_map Myr: front radius"))
    end
    p = plot(panels...; layout=grid(length(names), 3, widths=[0.29, 0.29, 0.42]),
             size=(1650, 500 * length(names)),
             left_margin=22Plots.mm, right_margin=4Plots.mm, bottom_margin=8Plots.mm, top_margin=3Plots.mm)
    savefig(p, joinpath(OUTDIR, "sphericity_comparison$(FIG)_$(SOLVER).png"))
end

names = filter(a -> !startswith(a, "--"), ARGS)
names = isempty(names) ? [k for k in ORDER if isfile(joinpath(OUTDIR, "sphericity_$(k)_$(SOLVER).txt"))] : names
foreach(kernel_figure, names)
summary_figure(names)
comparison_figure(names)
println("  wrote figures for ", join(names, ", "), " to $OUTDIR")
