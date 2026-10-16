#!/usr/bin/env julia
#
# The shadow test: a collimated beam strikes an opaque clump.
#
#   julia --project=. -t auto examples/radiation_shadow.jl [--no-plot] [--outdir=DIR]
#
# This is what M1 buys over flux-limited diffusion. FLD makes the radiation flux
# a function of the local energy gradient, so it flows around the obstacle and
# erases the shadow. M1 carries the flux as an independent variable and keeps
# the beam pointing the way it was sent.
#
# Writes shadow.png (a log map plus a cut along the beam), a VTK series under
# shadow/, and an ASCII map for terminals without a plotting stack.

using Honghu
using Printf
include(joinpath(@__DIR__, "common.jl"))
const P = Honghu.Problems

sim, tmax = P.radiation_shadow()
println(sim)
vtkdir = outpath("shadow")
run!(sim; tmax, outdir=vtkdir, basename="shadow", dtout=tmax / 30)

g = sim.grid
er = sim.R[1, g.lo[1]:g.hi[1], g.lo[2]:g.hi[2], g.lo[3]]
mid = g.n[2] ÷ 2
far = g.n[1] - 2
@printf("beam at the far edge   : E = %.4f\n", er[far, 3])
@printf("shadow at the far edge : E = %.4g\n", er[far, mid])
@printf("contrast               : %.4g\n", er[far, 3] / max(er[far, mid], 1e-30))
println("  wrote VTK snapshots to $vtkdir/")

if PLOTTING
    map = plotstate(sim; var=:Erad, logscale=true, c=:inferno,
                    title="M1 shadow behind an opaque clump")
    th = range(0, 2pi; length=200)
    plot!(map, 0.25 .+ 0.10 .* cos.(th), 0.06 .* sin.(th); c=:cyan, lw=1.5, label="")
    cut = plot(coordinates(g, 1), er[:, mid]; lw=2, c=1, yscale=:log10,
               label="behind clump (y = 0)", xlabel="x", ylabel="radiation energy",
               legend=:bottomleft)
    plot!(cut, coordinates(g, 1), er[:, 3]; lw=2, c=2, label="unobstructed beam")
    saveplot(plot(map, cut; layout=@layout([a{0.55h}; b]), size=(760, 700)), "shadow.png")
end

# ASCII map: rows are y, columns are x, shading is log10(E)
levels = " .:-=+*#%@"
println("\nradiation energy density (log scale, @ = 1, blank < 1e-9)")
for j in round.(Int, range(1, g.n[2], length=25))
    print("  ")
    for i in round.(Int, range(1, g.n[1], length=100))
        e = er[i, j]
        k = e <= 1e-9 ? 1 : clamp(round(Int, 10 + log10(e) * 9 / 9), 1, 10)
        print(levels[k])
    end
    println()
end
plotting_note()
