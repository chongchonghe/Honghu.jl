#!/usr/bin/env julia
#
# Non-equilibrium Marshak wave: a cold optically thick slab heated from x = 0 by
# a wall in radiative equilibrium at T = 1.
#
#   julia --project=. examples/marshak_wave.jl [nx] [--no-plot] [--outdir=DIR]
#
# Writes marshak.png (the two temperatures at three times) and marshak.txt.
# T_rad and T_gas separate at the front and merge behind it, which is the whole
# point of solving the two-moment system rather than assuming local thermal
# equilibrium.

using Honghu
using Printf
include(joinpath(@__DIR__, "common.jl"))
const P = Honghu.Problems

nx = isempty(OPTS) ? 512 : parse(Int, OPTS[1])

sim, tmax = P.marshak_wave(; nx)
println(sim)

g = sim.grid
x = coordinates(g, 1)
pl = PLOTTING ? plot(; xlabel="x", ylabel="temperature", xlims=(0, 0.12),
                     title="Marshak wave: radiation runs ahead of the matter",
                     legend=:topright) : nothing

for (i, t) in enumerate((0.1tmax, 0.3tmax, tmax))
    run!(sim; tmax=t, verbose=(i == 3))
    if PLOTTING
        Tr, _ = field(sim, :Trad)
        Tg, _ = field(sim, :temperature)
        plot!(pl, x, vec(Tr); lw=2, c=i, label=@sprintf("T_rad, t = %.2f", sim.t))
        plot!(pl, x, vec(Tg); lw=2, c=i, ls=:dash, label=@sprintf("T_gas, t = %.2f", sim.t))
    end
end

Tg = gas_temperature(sim)[:, 1, 1]
Tr = radiation_temperature(sim)[:, 1, 1]
front = findlast(>(0.5), Tr)
@printf("radiation front (T_rad > 0.5) at x = %.4f\n", x[front])
@printf("largest gas/radiation temperature separation: %.4f at x = %.4f\n",
        maximum(Tr .- Tg), x[argmax(Tr .- Tg)])

write_line(sim, outpath("marshak.txt"))
println("  wrote ", outpath("marshak.txt"))
PLOTTING && saveplot(plot(pl; size=(700, 500)), "marshak.png")

# a compact ASCII profile, so the example still says something with no Plots
println("\n  x      T_rad   T_gas")
for i in 1:cld(nx, 24):nx
    bar = repeat("#", round(Int, 20 * Tr[i]))
    @printf("  %.3f  %.4f  %.4f  %s\n", x[i], Tr[i], Tg[i], bar)
end
plotting_note()
