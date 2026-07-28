@testset "grid geometry" begin
    g = CartesianGrid(n=(8, 4, 1), xlo=(0.0, -1.0, 0.0), xhi=(2.0, 1.0, 1.0), nghost=3)
    @test g.n == (8, 4, 1)
    @test g.active == (true, true, false)
    @test g.ng == (3, 3, 0)                 # degenerate directions carry no ghosts
    @test g.dims == (14, 10, 1)
    @test g.lo == (4, 4, 1)
    @test g.hi == (11, 7, 1)
    @test Hydro3D.ndim(g) == 2
    @test all(g.dx .≈ (0.25, 0.5, 1.0))

    # cell centres line up with the domain
    @test all(cellcenter(g, g.lo...) .≈ (0.125, -0.75, 0.5))
    @test all(cellcenter(g, g.hi...) .≈ (1.875, 0.75, 0.5))
    @test Hydro3D.cellvolume(g) ≈ 0.25 * 0.5 * 1.0
    @test length(Hydro3D.interior(g)) == 8 * 4 * 1

    @test_throws ArgumentError CartesianGrid(n=(0, 1, 1))
    @test_throws ArgumentError CartesianGrid(n=(4, 1, 1), xlo=(1.0, 0.0, 0.0), xhi=(0.0, 1.0, 1.0))
end

@testset "sweep ranges" begin
    g = CartesianGrid(n=(8, 4, 1), nghost=3)
    # the reconstruction sweep covers the interior extended by one cell in D
    @test cell_ranges(g, Val(1)) == (3:12, 4:7, 1:1)
    @test cell_ranges(g, Val(2)) == (4:11, 3:8, 1:1)
    # faces bound the interior cells
    @test face_ranges(g, Val(1)) == (4:12, 4:7, 1:1)
    @test face_ranges(g, Val(2)) == (4:11, 4:8, 1:1)
    # writing qL at cell hi+1 must stay inside the array
    @test last(cell_ranges(g, Val(1))[1]) + 1 <= g.dims[1]
end

@testset "boundary conditions" begin
    nv = NHYDRO
    for (recon, ng) in ((PCM(), 2), (PLM(), 2), (PPM(), 3))
        @test required_ghosts(recon) == ng
    end

    g = CartesianGrid(n=(4, 1, 1), nghost=2)
    A = allocate(g, nv)
    for i in g.lo[1]:g.hi[1], n in 1:nv
        A[n, i, 1, 1] = 10n + (i - g.lo[1])
    end

    B = copy(A)
    fill_ghosts!(B, Boundaries(Periodic()), g)
    @test B[1, g.lo[1]-1, 1, 1] == A[1, g.hi[1], 1, 1]
    @test B[1, g.lo[1]-2, 1, 1] == A[1, g.hi[1]-1, 1, 1]
    @test B[1, g.hi[1]+1, 1, 1] == A[1, g.lo[1], 1, 1]

    B = copy(A)
    fill_ghosts!(B, Boundaries(Outflow()), g)
    @test B[1, g.lo[1]-1, 1, 1] == B[1, g.lo[1]-2, 1, 1] == A[1, g.lo[1], 1, 1]
    @test B[1, g.hi[1]+2, 1, 1] == A[1, g.hi[1], 1, 1]

    B = copy(A)
    fill_ghosts!(B, Boundaries(Reflecting()), g)
    @test B[1, g.lo[1]-1, 1, 1] == A[1, g.lo[1], 1, 1]        # scalar mirrored
    @test B[2, g.lo[1]-1, 1, 1] == -A[2, g.lo[1], 1, 1]       # normal momentum flipped
    @test B[3, g.lo[1]-1, 1, 1] == A[3, g.lo[1], 1, 1]        # transverse untouched
    @test B[2, g.lo[1]-2, 1, 1] == -A[2, g.lo[1]+1, 1, 1]

    B = copy(A)
    fill_ghosts!(B, Boundaries((Fixed((1.0, 2.0, 3.0, 4.0, 5.0)), Outflow()),
                               (Outflow(), Outflow()), (Outflow(), Outflow())), g)
    @test B[:, g.lo[1]-1, 1, 1] == [1.0, 2.0, 3.0, 4.0, 5.0]

    # a reflecting wall must leave a uniform state untouched apart from the sign
    @test_throws DimensionMismatch fill_ghosts!(allocate(g, NRAD),
                                                Boundaries(Fixed((1.0, 2.0, 3.0, 4.0, 5.0))), g)
end

@testset "FunctionBC" begin
    # the time-dependent counterpart of Fixed, which drops the `t` it is handed
    ramp = FunctionBC((x, y, z, t) -> (1.0, min(t, 1.0), 0.0, 0.0, 2.5 + 0.5 * min(t, 1.0)^2))
    mk() = begin
        s = Simulation(n=(32, 1, 1), eos=IdealGas(gamma=1.4),
                       boundaries=Boundaries((ramp, Outflow()), (Outflow(), Outflow()),
                                             (Outflow(), Outflow())))
        set_initial_conditions!(s) do x, y, z
            (rho=1.0, p=1.0)
        end
        s
    end

    # at t = 0 the ghost momentum is zero; later it is not, which `Fixed` cannot do
    sim = mk()
    g = sim.grid
    @test sim.U[2, 1, 1, 1] == 0.0
    fill_ghosts!(sim.U, sim.boundaries, g, 0.5)
    @test sim.U[2, 1, 1, 1] ≈ 0.5
    fill_ghosts!(sim.U, sim.boundaries, g, 2.0)
    @test sim.U[2, 1, 1, 1] ≈ 1.0            # clamped by the ramp

    # and it drives the interior
    sim = mk()
    run!(sim; tmax=0.3, verbose=false)
    @test conserved_totals(sim).momentum[1] > 0
    @test nrepaired(sim) == 0

    # a constant FunctionBC must agree with the equivalent Fixed
    state = (1.0, 0.4, 0.0, 0.0, 2.6)
    a = Simulation(n=(16, 1, 1), boundaries=Boundaries((Fixed(state), Outflow()),
                                                       (Outflow(), Outflow()), (Outflow(), Outflow())))
    b = Simulation(n=(16, 1, 1), boundaries=Boundaries((FunctionBC((x, y, z, t) -> state), Outflow()),
                                                       (Outflow(), Outflow()), (Outflow(), Outflow())))
    for s in (a, b)
        set_initial_conditions!(s) do x, y, z
            (rho=1.0, p=1.0)
        end
        run!(s; tmax=0.2, verbose=false)
    end
    @test a.U == b.U

    @test_throws DimensionMismatch fill_ghosts!(mk().U,
        Boundaries((FunctionBC((x, y, z, t) -> (1.0, 2.0)), Outflow()),
                   (Outflow(), Outflow()), (Outflow(), Outflow())),
        Simulation(n=(32, 1, 1)).grid, 0.0)
end

@testset "per-subsystem reconstruction" begin
    rad() = RadiationPhysics(c=10.0, chat=10.0, a_rad=1.0, opacity=ConstantOpacity(1.0),
                             beta_order=0)
    # ghost width is the max over both subsystems, so the higher-order one is safe
    s = Simulation(n=(16, 1, 1), radiation=rad(), reconstruction=PPM(),
                   radiation_reconstruction=PLM())
    @test s.grid.ng[1] == 3
    s2 = Simulation(n=(16, 1, 1), radiation=rad(), reconstruction=PLM(),
                    radiation_reconstruction=PPM())
    @test s2.grid.ng[1] == 3
    # defaults to the hydro choice
    @test Simulation(n=(16, 1, 1), radiation=rad(),
                     reconstruction=PLM()).radiation_reconstruction isa PLM

    # and the radiation sweep really uses its own choice. Compare the error
    # against the analytic Gaussian rather than the raw arrays: the state decays
    # to ~1e-21 in the tails, where a relative comparison means nothing.
    err(rr) = begin
        sim, tmax = P.radiation_diffusion(nx=64, radiation_reconstruction=rr)
        run!(sim; tmax, verbose=false)
        default_metrics(P.radiation_diffusion).l1_Erad(sim)
    end
    e = map(err, (PCM(), PLM(), PPM()))
    @test e[1] > e[2] > e[3]                        # higher order, lower error
    @test e[3] < 0.01                               # and PPM actually solves it

    # hydro reconstruction must not be what is moving: hold it fixed while the
    # radiation choice varies
    a = Simulation(n=(16, 1, 1), radiation=rad(), reconstruction=PPM(),
                   radiation_reconstruction=PPM())
    b = Simulation(n=(16, 1, 1), radiation=rad(), reconstruction=PPM(),
                   radiation_reconstruction=PLM())
    @test a.reconstruction === b.reconstruction
    @test typeof(a.radiation_reconstruction) !== typeof(b.radiation_reconstruction)
end
