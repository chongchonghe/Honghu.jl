# The parallel drivers: each must do exactly what its serial counterpart does,
# whatever the number of threads. Bit-for-bit (`==`) is the requirement, because
# the drivers only change who computes what, never the arithmetic.
#
# Run these with several threads (the suite's `-t 4`). With one, the tiled sweep
# has one task and one scratch, and a bug that lets two tasks share a scratch --
# it happened once, through a boxed variable in the spawn loop -- cannot show.

@testset "threaded fill and copy" begin
    A = rand(5, 10, 11, 13)
    B = similar(A)
    Honghu.threaded_copyto!(B, A)
    @test B == A
    Honghu.threaded_fill!(A, 0.0)
    @test all(iszero, A)
    # a single slab along the last index, which leaves nothing to split
    C = rand(3, 4, 5, 1)
    D = similar(C)
    Honghu.threaded_copyto!(D, C)
    @test D == C
end

# `dU` from scratch: zero it, then subtract each active direction's divergence,
# exactly as `hydro_rhs!` did before the zeroing was folded into the first sweep.
function reference_rhs(sim)
    U = copy(sim.U)
    g, hw, nv = sim.grid, sim.hw, Honghu.nhydrovars(sim)
    Honghu.fill_ghosts!(U, sim.boundaries, g, sim.t; momentum_blocks=1)
    Honghu.enforce_floors!(U, sim.eos, g, sim.floors, sim.nscalars)
    Honghu.enforce_scalar_floors!(U, g, sim.nscalars, false)
    prim = similar(U)
    Honghu.cons2prim!(prim, U, sim.eos, g, sim.nscalars)
    qL, qR, flux = similar(U), similar(U), similar(U)
    dU = zeros(size(U))
    for D in 1:3
        g.active[D] || continue
        Honghu.reconstruct!(qL, qR, prim, sim.reconstruction, Val(D), g, nv)
        Honghu.hydro_fluxes!(flux, qL, qR, prim, sim.eos, sim.riemann, Val(D), g, sim.nscalars)
        Honghu.flux_divergence!(dU, flux, Val(D), g, nv)
    end
    return dU
end
interior(A, g) = A[:, g.lo[1]:g.hi[1], g.lo[2]:g.hi[2], g.lo[3]:g.hi[3]]

@testset "flux_divergence! overwrite" begin
    sim, _ = Honghu.Problems.sedov_blast(; n=12)
    g, hw = sim.grid, sim.hw
    Honghu.cons2prim!(hw.prim, sim.U, sim.eos, g)
    Honghu.reconstruct!(hw.qL, hw.qR, hw.prim, PPM(), Val(2), g, 5)
    Honghu.hydro_fluxes!(hw.flux, hw.qL, hw.qR, hw.prim, sim.eos, sim.riemann, Val(2), g)
    sub = zeros(size(sim.U))
    Honghu.flux_divergence!(sub, hw.flux, Val(2), g, 5)
    over = fill(NaN, size(sim.U))                 # whatever was there is replaced
    Honghu.flux_divergence!(over, hw.flux, Val(2), g, 5, true)
    @test interior(over, g) == interior(sub, g)
end

@testset "hydro_rhs! matches zero-then-subtract" begin
    for sim in (Honghu.Problems.sedov_blast(; n=13)[1],
                Simulation(n=(1, 16, 16)),               # x inactive: y overwrites
                Simulation(n=(10, 1, 9)))                # y inactive
        if sim.grid.n[1] == 1
            set_initial_conditions!((x, y, z) -> (rho=1 + 0.3sinpi(2y), vz=0.2, p=1.0), sim)
        elseif sim.grid.n[2] == 1
            set_initial_conditions!((x, y, z) -> (rho=1 + 0.3sinpi(2z), vx=0.2, p=1.0), sim)
        end
        ref = reference_rhs(sim)
        fill!(sim.hw.dU, NaN)
        Honghu.hydro_rhs!(sim.hw.dU, sim.U, sim)
        @test interior(sim.hw.dU, sim.grid) == interior(ref, sim.grid)
    end
end

# A reconstruction with no tiled method: the tiled driver must leave it alone.
struct UntiledPCM <: Honghu.AbstractReconstruction end
Honghu.required_ghosts(::UntiledPCM) = 2
Honghu.reconstruct!(qL, qR, q, ::UntiledPCM, ::Val{D}, g, nvar) where {D} =
    Honghu.reconstruct!(qL, qR, q, PCM(), Val(D), g, nvar)

function scalar_blast(; kw...)
    sim = Simulation(; n=(14, 11, 9), nscalars=2, kw...)
    set_initial_conditions!(sim) do x, y, z
        r2 = (x - 0.5)^2 + (y - 0.5)^2 + (z - 0.5)^2
        (rho=1.0, p=r2 < 0.04 ? 10.0 : 0.1, X=(r2 < 0.04 ? 0.9 : 0.2, r2 < 0.04 ? 0.1 : 0.8))
    end
    return sim
end

@testset "tiled sweep equals untiled" begin
    P = Honghu.Problems
    makers = [
        "sedov 37" => kw -> P.sedov_blast(; n=37, kw...)[1],
        "sedov 48" => kw -> P.sedov_blast(; n=48, kw...)[1],
        "kh 2D" => kw -> P.kelvin_helmholtz(; n=40, kw...)[1],
        "sod 1D" => kw -> P.sod_shock_tube(; n=64, kw...)[1],
        "scalars" => kw -> scalar_blast(; kw...),
    ]
    for (name, mk) in makers, recon in (PCM(), PLM(), PPM()), tile in ((4, 5, 3), Honghu.DEFAULT_TILE)
        a = mk((; reconstruction=recon, tile))
        b = mk((; reconstruction=recon, tile=nothing))
        @test a.hw.tiling !== nothing && b.hw.tiling === nothing
        for _ in 1:5
            dt = Honghu.timestep(b)
            step!(a, dt)
            step!(b, dt)
        end
        ok = a.U == b.U
        ok || @info "tiled and untiled differ" name recon tile maximum(abs, a.U .- b.U)
        @test ok
    end
    # `:auto` tiles from `TILE_MIN_THREADS` threads, and is untiled below
    auto = P.sedov_blast(; n=12)[1]
    @test (auto.hw.tiling !== nothing) == (Threads.nthreads() >= Honghu.TILE_MIN_THREADS)
    @test_throws ArgumentError P.sedov_blast(; n=12, tile=:fast)
    @test !Honghu.has_tiled_method(UntiledPCM())
    @test all(Honghu.has_tiled_method, (PCM(), PLM(), PPM()))
    a = P.sedov_blast(; n=12, reconstruction=UntiledPCM())[1]
    b = P.sedov_blast(; n=12, reconstruction=PCM(), tile=nothing)[1]
    for _ in 1:3
        dt = Honghu.timestep(b)
        step!(a, dt); step!(b, dt)
    end
    @test a.U == b.U
end

@testset "tile boxes partition the interior" begin
    for n in ((37, 20, 9), (16, 1, 1), (1, 13, 7)), tile in ((4, 5, 3), (8, 8, 8)), D in 1:3
        g = CartesianGrid(n=n)
        g.active[D] || continue
        seen = zeros(Int, g.dims)
        for (ir, jr, kr) in Honghu.tile_boxes(g, D, tile)
            @test length((ir, jr, kr)[D]) == g.n[D]       # whole extent along the sweep
            seen[ir, jr, kr] .+= 1
        end
        @test all(==(1), seen[g.lo[1]:g.hi[1], g.lo[2]:g.hi[2], g.lo[3]:g.hi[3]])
        @test sum(seen) == prod(g.n)
    end
end
