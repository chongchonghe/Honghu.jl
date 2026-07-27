#!/usr/bin/env julia
#
# Kelvin-Helmholtz instability in 2D, written out as a VTK time series.
#
#   julia --project=. -t auto examples/kelvin_helmholtz_2d.jl [n] [--no-plot] [--outdir=DIR]
#
# Writes kh.png (density at three times) and a VTK series under kh/. Open
# kh/kh_00000.vti in ParaView; the numbered files are read as one time series.

using Hydro3D
using Printf
include(joinpath(@__DIR__, "common.jl"))
const P = Hydro3D.Problems

n = isempty(OPTS) ? 256 : parse(Int, OPTS[1])

sim, tmax = P.kelvin_helmholtz(; n)
println(sim)

t0 = conserved_totals(sim)
vtkdir = outpath("kh")
# stop at a few intermediate times so the figure can show the roll-up
snaps = []
for t in (0.4tmax, 0.7tmax, tmax)
    run!(sim; tmax=t, outdir=vtkdir, basename="kh", dtout=tmax / 100)
    PLOTTING && push!(snaps, plotstate(sim; var=:density, c=:magma, colorbar=false,
                                       title=@sprintf("t = %.2f", sim.t)))
end
t1 = conserved_totals(sim)

@printf("mass   drift: %.3e\n", abs(t1.mass - t0.mass) / t0.mass)
@printf("energy drift: %.3e\n", abs(t1.energy_total - t0.energy_total) / t0.energy_total)
println("  wrote VTK snapshots to $vtkdir/")

PLOTTING && saveplot(plot(snaps...; layout=(1, 3), size=(1050, 380),
                          plot_title="Kelvin-Helmholtz, $(n)^2, PPM + SSPRK3",
                          plot_titlevspan=0.14, bottom_margin=5Plots.mm), "kh.png")
plotting_note()
