@testset "mass scalars: storage and boundaries" begin
    mk(; kw...) = Simulation(; n=(8, 1, 1), nscalars=2, kw...)
    sim = mk()
    @test size(sim.U, 1) == Honghu.NHYDRO + 2
    @test Honghu.nhydrovars(sim) == 7
    @test Honghu.scalar_index(1) == 6
    @test size(Simulation(n=(8, 1, 1)).U, 1) == Honghu.NHYDRO
    @test_throws ArgumentError Simulation(n=(8, 1, 1), nscalars=-1)

    set_initial_conditions!((x, y, z) -> (rho=2.0, p=1.0, X=(0.25, 0.75)), sim)
    i = sim.grid.lo[1]
    @test sim.U[6, i, 1, 1] == 0.5
    @test sim.U[7, i, 1, 1] == 1.5
    @test conserved_totals(sim).scalars ≈ [0.5, 1.5]
    @test_throws ArgumentError set_initial_conditions!((x, y, z) -> (rho=1.0, p=1.0, X=(1.0,)), mk())

    # Reflecting flips the normal momentum only, never a scalar row. Eight rows
    # on purpose: 8 ÷ NRAD == 2 is what the old stride rule would have flipped.
    g = sim.grid
    A = Honghu.allocate(g, 8)
    for ii in g.lo[1]:g.hi[1], n in 1:8
        A[n, ii, 1, 1] = 10n + ii
    end
    Honghu.fill_ghosts!(A, Boundaries(Reflecting()), g; momentum_blocks=1)
    @test A[2, g.lo[1]-1, 1, 1] == -A[2, g.lo[1], 1, 1]
    @test A[6, g.lo[1]-1, 1, 1] == A[6, g.lo[1], 1, 1]

    # a Fixed hydro boundary has to cover the scalars too
    bad = Simulation(n=(8, 1, 1), nscalars=1,
                     boundaries=Boundaries(Fixed((1.0, 0.0, 0.0, 0.0, 1.0))))
    @test_throws DimensionMismatch set_initial_conditions!((x, y, z) -> (rho=1.0, p=1.0, X=(1.0,)), bad)

    # the IMEX stages never touch the scalar rows
    rad = RadiationPhysics(c=1.0, a_rad=1.0, opacity=ConstantOpacity(1.0))
    rs = Simulation(n=(8, 1, 1), nscalars=1, hydro=false, radiation=rad)
    set_initial_conditions!((x, y, z) -> (rho=1.0, T=1.0, Trad=0.5, X=(0.3,)), rs)
    S0 = rs.U[6, :, :, :]
    step!(rs)
    @test rs.U[6, :, :, :] == S0
end
