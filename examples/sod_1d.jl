#!/usr/bin/env julia
#
# Sod shock tube, compared against the exact Riemann solution.
#
#   julia --project=. examples/sod_1d.jl [nx]
#
# Writes sod.txt (numerical) and sod_exact.txt (analytic) as plain-text tables
# with a header line, ready for gnuplot, numpy.loadtxt or Plots.jl.

using Hydro3D
using Printf
const P = Hydro3D.Problems

nx = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 512

sim, tmax = P.sod_shock_tube(; nx)
run!(sim; tmax)

prim = primitives(sim)
g = sim.grid

# L1 error against the exact solution
err = 0.0
open("sod_exact.txt", "w") do f
    println(f, "# x density velocity pressure")
    for i in 1:nx
        x = g.xlo[1] + (i - 0.5) * g.dx[1]
        rho, u, p = P.sod_exact(x, sim.t)
        @printf(f, "%.10g %.10g %.10g %.10g\n", x, rho, u, p)
        global err += abs(prim[1, i, 1, 1] - rho)
    end
end
write_line(sim, "sod.txt")

@printf("nx = %d, t = %.4f, L1 density error = %.4e\n", nx, sim.t, err / nx)
println("wrote sod.txt and sod_exact.txt")
