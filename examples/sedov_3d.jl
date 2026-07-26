#!/usr/bin/env julia
#
# Sedov-Taylor point explosion in 3D, compared against the similarity solution
# R(t) = xi (E t^2 / rho)^(1/5).
#
#   julia --project=. -t auto examples/sedov_3d.jl [n]
#
# Writes a VTK series to sedov/. In ParaView, a Contour filter on "density" or a
# slice through the centre shows the spherical shell.

using Hydro3D
using Printf
const P = Hydro3D.Problems

n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 96

sim, tmax = P.sedov_blast(; n)
println(sim)

t0 = conserved_totals(sim)
run!(sim; tmax, outdir="sedov", basename="sedov", dtout=tmax / 20)
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
