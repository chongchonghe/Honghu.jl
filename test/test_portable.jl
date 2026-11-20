# The device path of the hydro sweep, checked without a device: the same
# simulations, once with the default `CPUThreads` backend and once with their
# sweeps launched as KernelAbstractions kernels on KA's own CPU backend. The cell
# kernels and their arithmetic are the same, so the answer must be the same to
# the bit. `test/gpu/gpu_sweep.jl` runs the same kernels on a CUDA GPU.

# KernelAbstractions comes from `test/ka`, which `runtests.jl` puts on the load
# path; instantiate it once with `julia --project=test/ka -e 'using Pkg;
# Pkg.instantiate()'`.
using KernelAbstractions: KernelAbstractions

function _portable_blast(; kw...)
    sim = Simulation(; n=(14, 11, 9), nscalars=2, kw...)
    set_initial_conditions!(sim) do x, y, z
        r2 = (x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2
        (rho=1.0, p=r2 < 0.04 ? 10.0 : 0.1, X=(r2 < 0.04 ? 0.9 : 0.2, r2 < 0.04 ? 0.1 : 0.8))
    end
    return sim
end

@testset "KernelAbstractions backend equals CPUThreads" begin
    P = Honghu.Problems
    ka = Honghu.KABackend(KernelAbstractions.CPU())
    makers = [
        "sedov 3D" => () -> P.sedov_blast(; n=20)[1],
        "sedov 3D tiled" => () -> P.sedov_blast(; n=20, tile=(4, 5, 3))[1],
        "kh 2D" => () -> P.kelvin_helmholtz(; n=32)[1],
        "sod 1D" => () -> P.sod_shock_tube(; n=64)[1],
        "scalars" => () -> _portable_blast(),
        "plm" => () -> P.sedov_blast(; n=16, reconstruction=PLM())[1],
    ]
    for (name, mk) in makers
        a, b = mk(), mk()
        b.hw.backend = ka
        # a tile asked for is not used off the CPU backend: the GPU's box is the grid
        @test !Honghu._uses_tiles(b.hw, b.reconstruction)
        for _ in 1:3
            dt = Honghu.timestep(a)
            step!(a, dt)
            step!(b, dt)
        end
        ok = a.U == b.U
        ok || @info "KernelAbstractions and CPUThreads differ" name maximum(abs, a.U .- b.U)
        @test ok
    end
end
