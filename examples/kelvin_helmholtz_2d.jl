#!/usr/bin/env julia
#
# Kelvin-Helmholtz instability in 2D, written out as a VTK time series.
#
#   julia --project=. -t auto examples/kelvin_helmholtz_2d.jl [n]
#
# Open kh/kh_00000.vti in ParaView; the numbered files are read as one time
# series. Colour by "density".

using Hydro3D
using Printf
const P = Hydro3D.Problems

n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 256

sim, tmax = P.kelvin_helmholtz(; n)
println(sim)

t0 = conserved_totals(sim)
run!(sim; tmax, outdir="kh", basename="kh", dtout=tmax / 100)
t1 = conserved_totals(sim)

@printf("mass   drift: %.3e\n", abs(t1.mass - t0.mass) / t0.mass)
@printf("energy drift: %.3e\n", abs(t1.energy_total - t0.energy_total) / t0.energy_total)
println("wrote 101 snapshots to kh/")
