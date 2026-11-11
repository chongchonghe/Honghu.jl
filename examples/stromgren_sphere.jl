#!/usr/bin/env julia
#
# The Stromgren sphere of Iliev et al. (2006): an ionizing point source switched on in a
# uniform, static, neutral hydrogen gas, with the hydrogen network and the photon band of
# `Photochemistry`.
#
#   julia --project=. -t 8 examples/stromgren_sphere.jl [test] [n] [octant] [--no-plot] [--outdir=DIR]
#
# `test` is 1 (isothermal, 1e4 K) or 2 (heating and cooling from 100 K); the default is 1.
# `n` is the number of cells along each side; the default is 64. The source is at the centre
# of a 13.2 kpc box by default. The word `octant` uses the paper's geometry instead: the source
# in the corner of a 6.6 kpc box with reflecting walls on the low faces. `n` counts the cells
# along the side of whichever box it is, so the octant at `n` has the cell size of the centred
# box at `2n`, with one eighth of its cells. The reduced speed of light is 3e-4 c.
#
# Runs to 500 Myr and prints the ionized radius at the output times of the paper. Writes
# stromgren_test<k>_radius.txt (time, ionized radius, the finite-chat front equation, the
# chat = c curve), stromgren_test<k>.png (a slice of the neutral fraction, and of the
# temperature in Test 2) and stromgren_test<k>_radius.png (the radius against both references).
#
# Test 1 has an analytic reference; Test 2 has none, and the reference columns of its file are
# the Test 1 ones, for scale. See docs/photoionization-findings.md for what the numbers mean
# and what a run of this size costs.

using Honghu
using Printf
include(joinpath(@__DIR__, "common.jl"))
const P = Honghu.Problems

octant = "octant" in OPTS
nums = filter(a -> a != "octant", OPTS)
test = isempty(nums) ? 1 : parse(Int, nums[1])
n = length(nums) < 2 ? 64 : parse(Int, nums[2])

const CHAT_FACTOR = 3e-4
const MYR = 1e6 * Honghu.CGS.year
const KPC = 1e3 * Honghu.CGS.parsec
const TIMES_MYR = (10, 30, 100, 200, 500)

sim, tmax = P.stromgren_iliev(; test, n, centered=!octant, chat_factor=CHAT_FACTOR)
println(sim)
x0 = test == 1 ? 1.2e-3 : 0.0          # the background ionization that `ionized_radius` leaves out
chat = CHAT_FACTOR * Honghu.CGS.c_light
r_fin(t) = P.stromgren_radius_finite_c(t; chat) / KPC
r_ana(t) = P.stromgren_radius_analytic(t) / KPC

rows = NTuple{4,Float64}[]
println("\n  t (Myr)   r_I (kpc)   finite-chat (kpc)   chat = c (kpc)")
for tmyr in TIMES_MYR
    run!(sim; tmax=tmyr * MYR, verbose=false)
    r = P.ionized_radius(sim; centered=!octant, x0) / KPC
    push!(rows, (tmyr, r, r_fin(tmyr * MYR), r_ana(tmyr * MYR)))
    @printf("  %7d   %9.4f   %17.4f   %14.4f\n", rows[end]...)
end
d = sim.diag
@printf("\n%d chemistry solves, %.2f steps per cell on average, %d failed\n",
        d.chem_cells, d.chem_steps / d.chem_cells, d.chem_failures)
d.chem_failures > 0 && println("WARNING: some chemistry solves did not converge; do not trust this run")

file = outpath("stromgren_test$(test)_radius.txt")
open(file, "w") do io
    println(io, "# Iliev et al. (2006) Test $test, n = $n, ", octant ? "octant" : "centred",
            ", chat = $(CHAT_FACTOR) c")
    test == 2 && println(io, "# Test 2 has no analytic solution; the last two columns are Test 1's.")
    println(io, "# t (Myr)  r_I (kpc)  finite-chat front equation (kpc)  chat = c curve (kpc)")
    for r in rows
        @printf(io, "%g %.6g %.6g %.6g\n", r...)
    end
end
println("  wrote $file")

if PLOTTING
    # The octant's slice 1 is the plane through the source; the centred box's middle slice is
    # the plane through it as well, to within half a cell. Test 1 is isothermal, so a
    # temperature panel would show round-off: its figure is the neutral fraction alone, on a
    # log scale, where the interior runs from 1e-3 to 1e-1 and the front is a step.
    saveplot(plotpanels(sim; vars=test == 1 ? [:X_1] : [:X_1, :temperature],
                        logscale=test == 1, slice=octant ? 1 : nothing,
                        size=test == 1 ? (620, 520) : (1100, 480), xticks=3, yticks=4),
             "stromgren_test$(test).png")
    t = range(0.5, 500, length=200)
    line = plot(t, r_ana.(t .* MYR); lw=2, c=:gray, ls=:dash, label="chat = c, analytic",
                xlabel="t (Myr)", ylabel="ionized radius (kpc)", legend=:bottomright,
                title="Stromgren Test $test: ionized radius" * (test == 2 ? " (curves are Test 1's)" : ""),
                size=(700, 450))
    plot!(line, t, r_fin.(t .* MYR); lw=2, c=:black, label="finite chat, Shapiro et al. (2006)")
    plot!(line, first.(rows), getindex.(rows, 2); lw=2, c=1, m=:circle, label="Honghu, n = $n")
    saveplot(line, "stromgren_test$(test)_radius.png")
end
plotting_note()
