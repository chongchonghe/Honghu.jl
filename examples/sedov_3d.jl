#!/usr/bin/env julia
#
# Sedov-Taylor point explosion in 3D, compared against the similarity solution
# R(t) = xi (E t^2 / rho)^(1/5).
#
#   julia --project=. -t auto examples/sedov_3d.jl [n] [--no-plot] [--outdir=DIR]
#
# Writes sedov.png (a mid-plane slice, and every cell in the box collapsed onto
# one radial profile) plus a VTK series under sedov/. In ParaView, a Contour
# filter on "density" shows the spherical shell.

using Honghu
using Printf
include(joinpath(@__DIR__, "common.jl"))
const P = Honghu.Problems

n = isempty(OPTS) ? 96 : parse(Int, OPTS[1])

sim, tmax = P.sedov_blast(; n)
println(sim)

t0 = conserved_totals(sim)
vtkdir = outpath("sedov")
run!(sim; tmax, outdir=vtkdir, basename="sedov", dtout=tmax / 20)
t1 = conserved_totals(sim)

prim = primitives(sim)
g = sim.grid
ic = n ÷ 2

# shock radius along the three axes: the density peak of the swept-up shell
radius(line) = (argmax(line) - 0.5) * g.dx[1]
rx = radius(prim[1, ic:end, ic, ic])
ry = radius(prim[1, ic, ic:end, ic])
rz = radius(prim[1, ic, ic, ic:end])
R_sedov = 1.15 * (1.0 * tmax^2 / 1.0)^(1 / 5)     # xi ~ 1.15 for gamma = 5/3

@printf("shock radius: x = %.4f, y = %.4f, z = %.4f  (similarity solution %.4f)\n",
        rx, ry, rz, R_sedov)
@printf("mass   drift: %.3e\n", abs(t1.mass - t0.mass) / t0.mass)
@printf("energy drift: %.3e\n", abs(t1.energy_total - t0.energy_total) / t0.energy_total)
println("  wrote VTK snapshots to $vtkdir/")

if PLOTTING
    sl = plotstate(sim; var=:density, c=:viridis, title="z = 0 slice",
                   colorbar_title="", left_margin=5Plots.mm)
    # a spherical blast on a Cartesian grid should collapse onto one curve
    rho, _ = field(sim, :density)
    xs, ys, zs = coordinates(g, 1), coordinates(g, 2), coordinates(g, 3)
    r = [sqrt(xs[i]^2 + ys[j]^2 + zs[k]^2) for k in 1:n for j in 1:n for i in 1:n]
    d = [rho[i, j, k] for k in 1:n for j in 1:n for i in 1:n]
    rp = scatter(r, d; ms=0.6, msw=0, alpha=0.15, c=1, label="all cells",
                 xlabel="radius", ylabel="density", xlims=(0, 0.5), legend=:topleft)
    vline!(rp, [R_sedov]; c=:black, lw=2, ls=:dash, label="Sedov-Taylor radius")
    saveplot(plot(sl, rp; layout=(1, 2), size=(1000, 430),
                  plot_title=@sprintf("Sedov blast, %d^3, t = %.3f", n, tmax),
                  plot_titlevspan=0.12, bottom_margin=6Plots.mm), "sedov.png")
end
plotting_note()
