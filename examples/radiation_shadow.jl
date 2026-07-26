#!/usr/bin/env julia
#
# The shadow test: a collimated beam strikes an opaque clump.
#
#   julia --project=. -t auto examples/radiation_shadow.jl
#
# This is what M1 buys over flux-limited diffusion. FLD makes the radiation flux
# a function of the local energy gradient, so it flows around the obstacle and
# erases the shadow. M1 carries the flux as an independent variable and keeps
# the beam pointing the way it was sent.
#
# Writes shadow/shadow_*.vti and prints an ASCII map of the radiation energy.

using Hydro3D
using Printf
const P = Hydro3D.Problems

sim, tmax = P.radiation_shadow()
println(sim)
run!(sim; tmax, outdir="shadow", basename="shadow", dtout=tmax / 30)

g = sim.grid
er = sim.R[1, g.lo[1]:g.hi[1], g.lo[2]:g.hi[2], g.lo[3]]
mid = g.n[2] ÷ 2
far = g.n[1] - 2
@printf("beam at the far edge   : E = %.4f\n", er[far, 3])
@printf("shadow at the far edge : E = %.4g\n", er[far, mid])
@printf("contrast               : %.4g\n", er[far, 3] / max(er[far, mid], 1e-30))

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
