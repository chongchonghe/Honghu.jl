#!/usr/bin/env julia
#
# Regenerate the animations embedded in README.md.
#
#   julia --project=. -t auto docs/make_animations.jl [which]
#
# `which` selects one of kh / shadow / sedov / orszag_tang, or all of them if omitted.
# Takes several minutes; the GIFs are a few MB each.

ENV["GKSwstype"] = "100"          # headless GR

using Honghu
using Plots
using Printf
const P = Honghu.Problems

const OUT = joinpath(@__DIR__, "figures")
mkpath(OUT)
default(fontfamily="sans-serif", framestyle=:box, dpi=200, titlefontsize=10,
        guidefontsize=9)

# --- Kelvin-Helmholtz: the classic roll-up ----------------------------------
function anim_kh(n=512, nframes=90)
    println("Kelvin-Helmholtz ($(n)^2, $nframes frames)")
    sim, tmax = P.kelvin_helmholtz(; n)
    animate_run(sim; tmax, nframes, var=:density, fps=20,
                filename=joinpath(OUT, "kh.gif"),
                c=:magma, size=(470, 430), colorbar=false)
end

# --- Radiation shadow: the beam sweeping past the clump ---------------------
function anim_shadow(nframes=90)
    println("radiation shadow ($nframes frames)")
    sim, tmax = P.radiation_shadow(n=(500, 200))
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

# --- Orszag-Tang: MHD shocks interacting into turbulence --------------------
function anim_orszag_tang(n=256, nframes=90)
    println("Orszag-Tang ($(n)^2, $nframes frames)")
    sim, tmax = P.orszag_tang(; n, integrator=SSPRK2())
    # the initial density is uniform, so the colour range has to be given;
    # (0.05, 0.6) spans the extrema reached up to t = 0.5
    animate_run(sim; tmax, nframes, var=:density, fps=18,
                filename=joinpath(OUT, "orszag_tang.gif"),
                c=:magma, size=(470, 430), colorbar=false, clims=(0.05, 0.6))
end

const WHICH = isempty(ARGS) ? ("kh", "shadow", "sedov", "orszag_tang") : Tuple(ARGS)
"kh" in WHICH && anim_kh()
"shadow" in WHICH && anim_shadow()
"sedov" in WHICH && anim_sedov()
"orszag_tang" in WHICH && anim_orszag_tang()
for f in readdir(OUT)
    endswith(f, ".gif") && @printf("  %-12s %6.2f MB\n", f,
                                   filesize(joinpath(OUT, f)) / 2^20)
end
