#!/usr/bin/env julia
#
# Non-equilibrium Marshak wave: a cold optically thick slab heated from x = 0 by
# a wall in radiative equilibrium at T = 1.
#
#   julia --project=. examples/marshak_wave.jl [nx]
#
# Writes marshak.txt with columns including gas_temperature and
# radiation_temperature. The two separate at the front and merge behind it,
# which is the whole point of solving the two-moment system rather than
# assuming local thermal equilibrium.

using Hydro3D
using Printf
const P = Hydro3D.Problems

nx = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 512

sim, tmax = P.marshak_wave(; nx)
println(sim)
run!(sim; tmax)

Tg = gas_temperature(sim)[:, 1, 1]
Tr = radiation_temperature(sim)[:, 1, 1]
g = sim.grid
x = [(i - 0.5) * g.dx[1] for i in 1:nx]

front = findlast(>(0.5), Tr)
@printf("radiation front (T_rad > 0.5) at x = %.4f\n", x[front])
@printf("largest gas/radiation temperature separation: %.4f at x = %.4f\n",
        maximum(Tr .- Tg), x[argmax(Tr .- Tg)])

write_line(sim, "marshak.txt")
println("wrote marshak.txt")

# a compact ASCII profile, so the example says something without a plotting stack
println("\n  x      T_rad   T_gas")
for i in 1:cld(nx, 24):nx
    bar = repeat("#", round(Int, 20 * Tr[i]))
    @printf("  %.3f  %.4f  %.4f  %s\n", x[i], Tr[i], Tg[i], bar)
end
