#!/usr/bin/env julia
#
# Thread scaling of the 3D Sedov blast. Not part of the test suite: run directly,
# once per thread count.
#
#   for t in 1 2 4 8 16 32 64; do
#       julia --project=. -t $t test/benchmarks_parallel.jl [n] [--passes] [--tile=a,b,c|none]
#   done
#
# Prints one line, `threads n ms/step Mcell/s KiB/step`: the minimum over
# `NSAMPLES` steps of a fixed `dt` after warm-up, which is the number
# `docs/parallel-findings.md` quotes. `--passes` adds the time of each whole-grid
# pass of the hydro right-hand side, the breakdown that says where a step goes.
# `--tile` overrides the `tile` keyword of the simulation.

using Honghu
using Printf

const P = Honghu.Problems
const NSAMPLES = 7

args = filter(a -> !startswith(a, "--"), ARGS)
n = isempty(args) ? 128 : parse(Int, args[1])
passes = "--passes" in ARGS
tilearg = let a = findfirst(s -> startswith(s, "--tile="), ARGS)
    a === nothing ? nothing : split(ARGS[a], "=")[2]
end
kw = tilearg === nothing ? (;) :
     tilearg == "none" ? (; tile=nothing) :
     (; tile=Tuple(parse.(Int, split(tilearg, ","))))

sim, _ = P.sedov_blast(; n, kw...)
for _ in 1:3
    step!(sim)                       # compile, and let the blast leave its first cells
end
dt = Honghu.timestep(sim)

best(f) = (f(); minimum(@elapsed(f()) for _ in 1:NSAMPLES))

GC.gc()
tstep = best(() -> step!(sim, dt))
bytes = @allocated step!(sim, dt)
@printf("threads %3d  n %4d  %9.2f ms/step  %8.2f Mcell/s  %10.1f KiB/step\n",
        Threads.nthreads(), n, 1e3 * tstep, n^3 / tstep / 1e6, bytes / 1024)

if passes
    g, hw, U = sim.grid, sim.hw, sim.U
    nv = Honghu.nhydrovars(sim)
    list = [
        "fill_ghosts" => () -> Honghu.fill_ghosts!(U, sim.boundaries, g, sim.t; momentum_blocks=1),
        "enforce_floors" => () -> Honghu.enforce_floors!(U, sim.eos, g, sim.floors, sim.nscalars),
        "cons2prim" => () -> Honghu.cons2prim!(hw.prim, U, sim.eos, g, sim.nscalars),
        "copy U0" => () -> copyto!(hw.U0, U),
        "sweep x" => () -> Honghu._hydro_sweep!(hw.dU, sim, Val(1)),
        "sweep y" => () -> Honghu._hydro_sweep!(hw.dU, sim, Val(2)),
        "sweep z" => () -> Honghu._hydro_sweep!(hw.dU, sim, Val(3)),
        "combine" => () -> Honghu.combine!(U, 0.5, hw.U0, 0.5, U, 0.0, hw.dU, g, nv),
        "hydro_rhs!" => () -> Honghu.hydro_rhs!(hw.dU, U, sim),
        "timestep" => () -> Honghu.timestep(sim),
        "enforce_state!" => () -> Honghu.enforce_state!(sim),
    ]
    for (name, f) in list
        @printf("    %-15s %9.2f ms\n", name, 1e3 * best(f))
    end
end
