#!/usr/bin/env julia
#
# The hydro sweep on an NVIDIA GPU, against the CPU. Not part of the test suite:
#
#   julia --project=test/gpu -e 'using Pkg; Pkg.instantiate()'    # once
#   julia --project=test/gpu -t 8 test/gpu/gpu_sweep.jl [n]
#
# (On a machine whose home depot is short of space, put the CUDA toolkit
# elsewhere first: `JULIA_DEPOT_PATH=/somewhere/with/space: julia ...`, the
# trailing colon keeping the default depot after it.)
#
# A Sedov blast is run three steps on the CPU, and its state converted to
# primitives. Then each sweep, along x, y and z, in each of its three modes
# (overwrite `dU`, subtract from it, form the Runge-Kutta stage) runs once on the
# CPU with `CPUThreads` and once on the GPU with `KABackend(CUDABackend())`, on
# `CuArray` copies of the same arrays, and the interiors are compared. The cell
# kernels are the same functions in both, so the only room for a difference is
# how the device compiles the arithmetic (fused multiply-adds, its own math
# library), not what is computed. That is round-off, so it is measured against each
# variable's scale: the largest difference of variable `n` over the largest
# `|value|` of variable `n`. (Cell by cell it would be meaningless where two
# nearly equal fluxes cancel: a `dU` of 1e-12 off by 1e-19 is 1e-7 "relative".)
# The script exits non-zero if that exceeds 1e-12; it is about 1e-16. It also prints one sweep's time on each,
# for the record; on a card with a low Float64 rate that number says little.

using Honghu, CUDA, KernelAbstractions, Printf

CUDA.functional() || error("CUDA is not functional on this machine")
n = isempty(ARGS) ? 64 : parse(Int, ARGS[1])

sim, _ = Honghu.Problems.sedov_blast(; n, tile=nothing)
for _ in 1:3
    step!(sim)
end
g, hw = sim.grid, sim.hw
nv = Honghu.nhydrovars(sim)
U = copy(sim.U)
Honghu.fill_ghosts!(U, sim.boundaries, g, sim.t; momentum_blocks=1)
Honghu.cons2prim!(hw.prim, U, sim.eos, g, sim.nscalars)
prim = hw.prim
prim_d = CuArray(prim)

cpu = Honghu.CPUThreads()
gpu = Honghu.KABackend(CUDABackend())
inner(A) = Array(A)[:, g.lo[1]:g.hi[1], g.lo[2]:g.hi[2], g.lo[3]:g.hi[3]]
relerr(a, b) = maximum(maximum(abs, a[v, :, :, :] .- b[v, :, :, :]) / maximum(abs, b[v, :, :, :])
                       for v in axes(b, 1))

println("device: ", CUDA.name(CUDA.device()), "   grid: ", n, "^3   threads: ",
        Threads.nthreads())
worst = 0.0
for D in 1:3, mode in (:overwrite, :subtract, :update)
    dU = fill(0.25, size(U))
    dU_d = CuArray(dU)
    sc = (similar(prim), similar(prim), similar(prim))
    sc_d = (CUDA.zeros(Float64, size(prim)), CUDA.zeros(Float64, size(prim)),
            CUDA.zeros(Float64, size(prim)))
    Uc, U0c = copy(U), 0.9 .* U
    Ud, U0d = CuArray(Uc), CuArray(U0c)
    upd = mode === :update ? (0.5, U0c, 0.5, Uc, 1e-3) : nothing
    upd_d = mode === :update ? (0.5, U0d, 0.5, Ud, 1e-3) : nothing
    ow = mode === :overwrite
    run(dst, p, s, u, backend) = Honghu.hydro_sweep!(dst, p, s, sim.reconstruction, sim.eos,
                                                     sim.riemann, Val(D), g, nv,
                                                     sim.nscalars, ow, u, backend, nothing)
    run(dU, prim, sc, upd, cpu)
    run(dU_d, prim_d, sc_d, upd_d, gpu)
    a, b = mode === :update ? (inner(Ud), inner(Uc)) : (inner(dU_d), inner(dU))
    e = relerr(a, b)
    global worst = max(worst, e)
    tc = minimum(@elapsed(run(dU, prim, sc, upd, cpu)) for _ in 1:3)
    tg = minimum(CUDA.@elapsed(run(dU_d, prim_d, sc_d, upd_d, gpu)) for _ in 1:3)
    @printf("D = %d  %-9s  identical %-5s  max diff / scale %.2e   cpu %7.2f ms  gpu %7.2f ms\n",
            D, mode, a == b, e, 1e3tc, 1e3tg)
end
println(worst <= 1e-12 ? "GPU AGREES WITH CPU" : "GPU DIFFERS FROM CPU")
exit(worst <= 1e-12 ? 0 : 1)
