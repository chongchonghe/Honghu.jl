#!/usr/bin/env julia
#
# Sod shock tube, compared against the exact Riemann solution.
#
#   julia --project=. examples/sod_1d.jl [nx] [--no-plot] [--outdir=DIR]
#
# Writes sod.png (four panels against the exact solution) plus sod.txt and
# sod_exact.txt as plain-text tables for gnuplot, numpy.loadtxt or Plots.jl.

using Honghu
using Printf
include(joinpath(@__DIR__, "common.jl"))
const P = Honghu.Problems

nx = isempty(OPTS) ? 512 : parse(Int, OPTS[1])

sim, tmax = P.sod_shock_tube(; nx)
run!(sim; tmax)

prim = primitives(sim)
g = sim.grid

# L1 error against the exact solution, and the exact profile as a data file
err = 0.0
open(outpath("sod_exact.txt"), "w") do f
    println(f, "# x density velocity pressure")
    for i in 1:nx
        x = g.xlo[1] + (i - 0.5) * g.dx[1]
        rho, u, p = P.sod_exact(x, sim.t)
        @printf(f, "%.10g %.10g %.10g %.10g\n", x, rho, u, p)
        global err += abs(prim[1, i, 1, 1] - rho)
    end
end
write_line(sim, outpath("sod.txt"))

@printf("nx = %d, t = %.4f, L1 density error = %.4e\n", nx, sim.t, err / nx)
println("  wrote ", outpath("sod.txt"), " and ", outpath("sod_exact.txt"))

if PLOTTING
    x = coordinates(g, 1)
    xe = range(0, 1; length=2000)
    ex = [P.sod_exact(xi, sim.t) for xi in xe]
    eint(rho, p) = p / (1.4 - 1) / rho

    panels = []
    for (i, (num, exact, lbl)) in enumerate((
        (prim[1, :, 1, 1], [e[1] for e in ex], "density"),
        (prim[2, :, 1, 1], [e[2] for e in ex], "velocity"),
        (prim[5, :, 1, 1], [e[3] for e in ex], "pressure"),
        (eint.(prim[1, :, 1, 1], prim[5, :, 1, 1]), [eint(e[1], e[3]) for e in ex],
         "specific internal energy")))
        pl = plot(xe, exact; lw=2, c=:black, label=i == 1 ? "exact" : "",
                  xlabel="x", ylabel=lbl)
        scatter!(pl, x, num; ms=2, msw=0, c=1, label=i == 1 ? "Honghu ($nx cells)" : "")
        push!(panels, pl)
    end
    saveplot(plot(panels...; layout=(2, 2), size=(900, 620),
                  plot_title=@sprintf("Sod shock tube at t = %.2f", sim.t),
                  plot_titlevspan=0.08), "sod.png")
end
plotting_note()
