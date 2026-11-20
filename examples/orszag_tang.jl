#!/usr/bin/env julia
#
# The Orszag-Tang vortex, a 2D MHD problem whose shocks interact into a turbulent
# pattern, with constrained transport keeping div b at round-off.
#
#   julia --project=. -t auto examples/orszag_tang.jl [n] [--no-plot] [--outdir=DIR]
#
# Runs n x n cells (default 256) to t = 0.5 with HLLD, PPM and SSPRK2, prints the
# largest |div b| dx / |b| and the conservation drifts, and writes orszag_tang.png
# (density, pressure and magnetic pressure).

using Honghu
using Printf
include(joinpath(@__DIR__, "common.jl"))
const P = Honghu.Problems

n = isempty(OPTS) ? 256 : parse(Int, OPTS[1])

sim, tmax = P.orszag_tang(; n, integrator=SSPRK2())
println(sim)
t0 = conserved_totals(sim)
run!(sim; tmax)
t1 = conserved_totals(sim)

bmax = maximum(maximum(abs, sim.B[d]) for d in 1:3)
@printf("max |div b| dx / max |b| = %.2e\n",
        maximum(abs, Honghu.divergence_b(sim)) * sim.grid.dx[1] / bmax)
@printf("mass drift %.2e, energy drift %.2e\n", abs(t1.mass - t0.mass) / t0.mass,
        abs(t1.energy_total - t0.energy_total) / t0.energy_total)

if PLOTTING
    panels = [plotstate(sim; var=v, c=:viridis, title=t)
              for (v, t) in ((:density, "density"), (:pressure, "pressure"),
                             (:magnetic_pressure, "magnetic pressure"))]
    saveplot(plot(panels...; layout=(1, 3), size=(1300, 400),
                  plot_title=@sprintf("Orszag-Tang, %d^2, t = %.2f", n, sim.t),
                  plot_titlevspan=0.1), "orszag_tang.png")
end
plotting_note()
