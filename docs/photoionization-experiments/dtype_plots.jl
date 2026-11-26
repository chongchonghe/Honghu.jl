#!/usr/bin/env julia
#
# Figures from the output of dtype.jl (Iliev et al. 2009 Test 5, centred box).
#
#   julia --project=. docs/photoionization-experiments/dtype_plots.jl [kernel ...] [--n=64] [--solver=be|ros] [--outdir=DIR]
#
#   dtype_comparison_<solver>.png       one row per kernel at 500 Myr: mid-plane slices of the
#                                        neutral fraction, n / n0 and T, and the front radius
#                                        over all directions
#   dtype_evolution_<kernel>_<solver>.png  the same panels for one kernel at 100, 300, 500 Myr
#   dtype_radius_<solver>.png           r_I, the mean front and the mean shock radius against
#                                        time, with two approximate D-type references, and
#                                        the spread of the front and of the shock
#   dtype_profiles_<kernel>_<solver>.png   shell-averaged profiles of x_H+, n / n0, T and v_r

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
include(joinpath(@__DIR__, "dtype.jl"))
using Plots
default(fontfamily="sans-serif", framestyle=:box, dpi=140, titlefontsize=9, guidefontsize=8,
        legendfontsize=7)

const L5 = 15.0 * KPC
const XLO5, DX5 = -L5, 2L5 / N
const CC = [XLO5 + (i - 0.5) * DX5 for i in 1:N] ./ KPC

readf(tag, name, tm) = (a = Array{Float32}(undef, N, N, N);
                        read!(joinpath(DT_OUT, "$(tag)_$(name)_$(SOLVER)_$(tm)Myr.f32"), a); Float64.(a))

slice(f) = permutedims(f[:, :, N ÷ 2])
sliceplot(z, title; kw...) = heatmap(CC, CC, z; colorbar=false, aspect_ratio=1, xlims=extrema(CC), ylims=extrema(CC),
                                     xlabel="x (kpc)", ylabel="y (kpc)", title, kw...)

function frontmap(x)
    phis = range(-pi, pi, length=96)
    mus = range(-1, 1, length=48)
    R = [front_radius(x, (sqrt(1 - m^2) * cos(p), sqrt(1 - m^2) * sin(p), m), XLO5, DX5)
         for m in mus, p in phis]
    return phis, mus, R ./ mean(filter(!isnan, R)) .- 1
end

function panel_row(name, tm; lim=0.15)
    x, rho, T = readf("x", name, tm), readf("rho", name, tm), readf("T", name, tm)
    phis, mus, dR = frontmap(x)
    return [sliceplot(slice(1 .- x), "$name, $tm Myr: x_H0, 0 (dark) to 1"; c=:viridis, clims=(0, 1)),
            sliceplot(slice(rho), "n / n0, 0 (dark) to 2.5"; c=:magma, clims=(0, 2.5)),
            sliceplot(slice(log10.(T)), "log10 T (K), 2 (dark) to 4.4"; c=:inferno, clims=(2, 4.4)),
            heatmap(phis, mus, dR; c=:balance, clims=(-lim, lim), xlabel="phi", ylabel="cos theta",
                    title="front radius, R / R_mean - 1")]
end

function grid_figure(rows, file)
    panels = reduce(vcat, rows)
    p = plot(panels...; layout=grid(length(rows), 4, widths=[0.24, 0.24, 0.24, 0.28]),
             size=(2100, 470 * length(rows)), left_margin=22Plots.mm, right_margin=4Plots.mm, bottom_margin=8Plots.mm, top_margin=3Plots.mm)
    savefig(p, joinpath(DT_OUT, file))
end

readtable5(name) = (rows = [parse.(Float64, split(l)) for l in eachline(joinpath(DT_OUT, "dtype_$(name)_$(SOLVER).txt"))
                            if !startswith(l, "#")]; reduce(hcat, rows)')

# Approximate references for the D-type phase, both starting at the Stromgren radius at t = 0:
# Spitzer (1978), R = R_S (1 + 7 c t / 4 R_S)^(4/7), and Hosokawa & Inutsuka (2006), with
# c -> sqrt(4/3) c. R_S = 5.39 kpc (alpha_B at 1e4 K) and c = 12.8 km/s, the isothermal sound
# speed of ionized hydrogen at 1e4 K. Neither has the R-type phase, so both are early at the
# start; they bracket what a D-type front should do, not where it should be.
function references(t_myr)
    RS = 5.39
    c = 12.85e5 * MYR / KPC                         # kpc per Myr
    sp = RS .* (1 .+ 7 .* c .* t_myr ./ (4RS)) .^ (4 / 7)
    hi = RS .* (1 .+ 7 .* sqrt(4 / 3) .* c .* t_myr ./ (4RS)) .^ (4 / 7)
    return sp, hi
end

function radius_figure(names)
    pr = plot(xlabel="t (Myr)", ylabel="radius (kpc)", title="Test 5: radii", legend=:topleft)
    t = range(0, 500, length=200)
    sp, hi = references(t)
    plot!(pr, t, sp; c=:gray, ls=:dash, label="Spitzer (approx.)")
    plot!(pr, t, hi; c=:gray, ls=:dot, label="Hosokawa & Inutsuka (approx.)")
    ps = plot(xlabel="t (Myr)", ylabel="std(R) / R_mean (%)", title="spread", legend=:topright)
    for (i, name) in enumerate(names)
        d = readtable5(name)
        # The densest point is a shock only once the gas is compressed; before that it is noise.
        s = d[:, 11] .> 1.1
        plot!(pr, d[:, 1], d[:, 3]; c=i, m=:circle, label="$name: front")
        plot!(pr, d[s, 1], d[s, 7]; c=i, m=:utriangle, ls=:dash, label="$name: shock")
        plot!(ps, d[:, 1], 100d[:, 4]; c=i, m=:circle, label="$name: front")
        plot!(ps, d[s, 1], 100d[s, 8]; c=i, m=:utriangle, ls=:dash, label="$name: shock")
    end
    savefig(plot(pr, ps; layout=(1, 2), size=(1200, 450), left_margin=22Plots.mm, right_margin=4Plots.mm, bottom_margin=8Plots.mm, top_margin=3Plots.mm),
            joinpath(DT_OUT, "dtype_radius_$(SOLVER).png"))
end

function profile_figure(name)
    rows = [parse.(Float64, split(l)) for l in eachline(joinpath(DT_OUT, "dtype_$(name)_$(SOLVER)_profiles.txt"))
            if !startswith(l, "#")]
    d = reduce(hcat, rows)'
    ps = [plot(xlabel="r (kpc)", ylabel=lab, legend=(k == 1 ? :bottomleft : false), title=lab)
          for (k, lab) in enumerate(("x_H+", "n / n0", "T (K)", "v_r (km/s)"))]
    for (i, tm) in enumerate((100, 200, 300, 400, 500))
        m = d[:, 1] .== tm
        for k in 1:4
            plot!(ps[k], d[m, 2], d[m, k+2]; c=i, label="$tm Myr", yscale=(k == 3 ? :log10 : :identity))
        end
    end
    savefig(plot(ps...; layout=(2, 2), size=(1000, 750), left_margin=22Plots.mm, right_margin=4Plots.mm, bottom_margin=8Plots.mm, top_margin=3Plots.mm),
            joinpath(DT_OUT, "dtype_profiles_$(name)_$(SOLVER).png"))
end

names = filter(a -> !startswith(a, "--"), ARGS)
names = isempty(names) ? [k for k in ("gauss", "wendland2", "gauss3")
                          if isfile(joinpath(DT_OUT, "dtype_$(k)_$(SOLVER).txt"))] : names
grid_figure([panel_row(nm, 500) for nm in names], "dtype_comparison_$(SOLVER).png")
for nm in names
    grid_figure([panel_row(nm, tm) for tm in (100, 300, 500)], "dtype_evolution_$(nm)_$(SOLVER).png")
    profile_figure(nm)
end
radius_figure(names)
println("  wrote figures for ", join(names, ", "), " to $DT_OUT")
