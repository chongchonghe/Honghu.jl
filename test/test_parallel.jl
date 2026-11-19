# The parallel drivers: each must do exactly what its serial counterpart does,
# whatever the number of threads. Bit-for-bit (`==`) is the requirement, because
# the drivers only change who computes what, never the arithmetic.

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
