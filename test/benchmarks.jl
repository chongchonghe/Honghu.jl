#!/usr/bin/env julia
#
# Timing benchmarks. Not part of the test suite: run directly.
#
#   julia --project=. -t 1 test/benchmarks.jl
#   julia --project=. -t 8 test/benchmarks.jl
#
# Prints the table that goes in README.md. Uses BenchmarkTools rather than a
# hand-rolled loop so the numbers come from warmed-up, minimum-over-samples
# statistics instead of whatever the first call happened to cost.

using Honghu
using BenchmarkTools
using Printf

"Build a small periodic simulation, with radiation if asked for."
function setup(n; radiation::Bool)
    rad = radiation ?
          RadiationPhysics(c=10.0, chat=1.0, a_rad=1.0, opacity=ConstantOpacity(1.0)) :
          nothing
    sim = Simulation(n=(n, n, n), radiation=rad, boundaries=Boundaries(Periodic()))
    set_initial_conditions!(sim) do x, y, z
        (rho=1 + 0.1 * sinpi(2x) * sinpi(2y) * sinpi(2z),
         vx=0.3, vy=0.1, vz=-0.2, p=1.0, Erad=1.0)
    end
    sim
end

function bench_steps(ns=(32, 64))
    @printf("\n%-8s %-6s %12s %14s %12s\n", "physics", "n^3", "ms/step",
            "Mcell-upd/s", "KiB/step")
    println("-"^58)
    for radiation in (false, true), n in ns
        sim = setup(n; radiation)
        # a fixed dt keeps every sample doing identical work
        dt = Honghu.timestep(sim)
        b = @benchmark step!($sim, $dt) samples = 12 evals = 1 seconds = 20
        ms = minimum(b).time / 1e6
        @printf("%-8s %-6s %12.2f %14.2f %12.1f\n",
                radiation ? "RHD" : "hydro", "$n", ms, n^3 / (ms * 1e3),
                minimum(b).memory / 1024)
    end
end

function bench_kernels(n=64)
    sim = setup(n; radiation=true)
    hw, rw, g = sim.hw, sim.rw, sim.grid
    Honghu.cons2prim!(hw.prim, sim.U, sim.eos, g)
    Honghu.rad_cons2prim!(rw.prim, sim.R, sim.radiation, g)
    cells = prod(g.n)

    kernels = (
        ("cons2prim", () -> Honghu.cons2prim!(hw.prim, sim.U, sim.eos, g)),
        ("reconstruct PLM", () -> Honghu.reconstruct!(hw.qL, hw.qR, hw.prim, PLM(), Val(1), g, 5)),
        ("reconstruct PPM", () -> Honghu.reconstruct!(hw.qL, hw.qR, hw.prim, PPM(), Val(1), g, 5)),
        ("HLLC sweep", () -> Honghu.hydro_fluxes!(hw.flux, hw.qL, hw.qR, hw.prim, sim.eos, sim.riemann, Val(1), g)),
        ("HLL rad sweep", () -> Honghu.rad_fluxes!(rw.flux, rw.qL, rw.qR, sim.R, rw.chat, sim.radiation, Val(1), g)),
        ("implicit source", () -> Honghu.radiation_source_step!(sim.U, sim.R, rw.chat,
                                                                sim.eos, sim.radiation,
                                                                g, 1e-3, nothing, 0.0,
                                                                sim.floors)),
    )
    @printf("\n%-18s %12s %14s %12s\n", "kernel at $(n)^3", "ms", "ns/cell", "KiB")
    println("-"^60)
    for (name, f) in kernels
        b = @benchmark $f() samples = 20 evals = 1 seconds = 15
        t = minimum(b)
        @printf("%-18s %12.3f %14.2f %12.1f\n", name, t.time / 1e6,
                t.time / cells, t.memory / 1024)
    end
end

@printf("Honghu benchmarks -- Julia %s, %d thread(s)\n", VERSION, Threads.nthreads())
bench_steps()
bench_kernels()
println()
