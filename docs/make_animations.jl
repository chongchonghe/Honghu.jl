#!/usr/bin/env julia
#
# Regenerate the animations embedded in README.md.
#
#   julia --project=. -t auto docs/make_animations.jl [which]
#
# `which` selects one of kh / shadow / sedov, or all of them if omitted.
# Takes several minutes; the GIFs are a few MB each.

ENV["GKSwstype"] = "100"          # headless GR

using Honghu
using Plots
using Printf
const P = Honghu.Problems

const OUT = joinpath(@__DIR__, "figures")
mkpath(OUT)
default(fontfamily="sans-serif", framestyle=:box, dpi=100, titlefontsize=10,
        guidefontsize=9)

# --- Kelvin-Helmholtz: the classic roll-up ----------------------------------
function anim_kh(n=256, nframes=90)
    println("Kelvin-Helmholtz ($(n)^2, $nframes frames)")
    sim, tmax = P.kelvin_helmholtz(; n)
    animate_run(sim; tmax, nframes, var=:density, fps=20,
                filename=joinpath(OUT, "kh.gif"),
                c=:magma, size=(470, 430), colorbar=false)
end

# --- Radiation shadow: the beam sweeping past the clump ---------------------
function anim_shadow(nframes=90)
    println("radiation shadow ($nframes frames)")
    sim, tmax = P.radiation_shadow(n=(200, 80))
    # log scale spans 10 decades, so the colour range has to be given explicitly
    animate_run(sim; tmax, nframes, var=:Erad, fps=18,
                filename=joinpath(OUT, "shadow.gif"),
                logscale=true, clims=(-10, 0), c=:inferno,
                size=(700, 320), colorbar=false)
end

# --- Sedov blast: a spherical shell on a Cartesian grid ---------------------
function anim_sedov(n=64, nframes=80)
    println("Sedov blast ($(n)^3, $nframes frames)")
    sim, tmax = P.sedov_blast(; n)
    animate_run(sim; tmax, nframes, var=:density, fps=18,
                filename=joinpath(OUT, "sedov.gif"),
                c=:viridis, size=(520, 460), colorbar=false, clims=(0.0, 4.0))
end

const WHICH = isempty(ARGS) ? ("kh", "shadow", "sedov") : Tuple(ARGS)
"kh" in WHICH && anim_kh()
"shadow" in WHICH && anim_shadow()
"sedov" in WHICH && anim_sedov()
for f in readdir(OUT)
    endswith(f, ".gif") && @printf("  %-12s %6.2f MB\n", f,
                                   filesize(joinpath(OUT, f)) / 2^20)
end
