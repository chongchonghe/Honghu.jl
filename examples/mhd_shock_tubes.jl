#!/usr/bin/env julia
#
# The Brio-Wu and Ryu-Jones 2a MHD shock tubes, against Quokka.
#
#   julia --project=. examples/mhd_shock_tubes.jl [--no-plot] [--outdir=DIR]
#
# Runs both tubes at 256 cells with Quokka's scheme (HLLD, the Q26 EMF with LD04
# averaging, RK2, PLM with Quokka's default Sweby 1.5 limiter) and the constant step
# of the Quokka runs in test/data/quokka, prints the L1 difference of each variable,
# and writes mhd_shock_tubes.png. The two codes differ only by Quokka's shock
# flattening; with PCM they agree to round-off (test/test_mhd.jl).

using Honghu
using Printf
include(joinpath(@__DIR__, "common.jl"))
const P = Honghu.Problems

readline_quokka(name) =
    permutedims(reduce(hcat, [parse.(Float64, split(l))
                              for l in eachline(joinpath(@__DIR__, "..", "test", "data", "quokka", name))
                              if !startswith(l, "#")]))

names = ("density", "x velocity", "y velocity", "z velocity", "pressure", "b_y", "b_z")
panels = []
for (title, problem, file, cols) in (("Brio-Wu", P.brio_wu, "bw_plm.txt", (1, 2, 3, 6)),
                                     ("Ryu-Jones 2a", P.ryu_jones_2a, "rj_plm.txt", (1, 3, 5, 6)))
    ref = readline_quokka(file)
    sim, tmax = problem(n=size(ref, 1), reconstruction=PLM(Sweby(1.5)), integrator=SSPRK2())
    for _ in 1:round(Int, tmax / 2.5e-4)
        step!(sim, 2.5e-4)
    end
    pr = primitives(sim)
    by = first(field(sim, :by))[:, 1, 1]
    bz = first(field(sim, :bz))[:, 1, 1]
    mine = hcat(pr[1, :, 1, 1], pr[2, :, 1, 1], pr[3, :, 1, 1], pr[4, :, 1, 1], pr[5, :, 1, 1], by, bz)
    @printf("%s at t = %.2f, L1 difference from Quokka:\n", title, sim.t)
    for c in 1:7
        q = ref[:, c+1]
        any(!=(0), q) && @printf("  %-11s %.2e\n", names[c], sum(abs, mine[:, c] .- q) / sum(abs, q))
    end
    if PLOTTING
        x = ref[:, 1]
        for (m, c) in enumerate(cols)
            pl = plot(x, ref[:, c+1]; lw=1.5, c=:black, label=m == 1 ? "Quokka" : "",
                      xlabel="x", ylabel=names[c], title=m == 1 ? title : "")
            scatter!(pl, x, mine[:, c]; ms=1.5, msw=0, c=1, label=m == 1 ? "Honghu" : "")
            push!(panels, pl)
        end
    end
end
PLOTTING && saveplot(plot(panels...; layout=(2, 4), size=(1300, 560), left_margin=4Plots.mm,
                          bottom_margin=4Plots.mm), "mhd_shock_tubes.png")
plotting_note()
