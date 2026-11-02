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

@testset "mass scalars: advection" begin
    for riemann in (HLLC(), LLF()), recon in (PLM(), PPM())
        sim = Simulation(n=(64, 1, 1), nscalars=2, riemann=riemann, reconstruction=recon,
                         boundaries=Boundaries(Periodic()))
        set_initial_conditions!((x, y, z) -> (rho=1 + 0.5sinpi(2x), vx=1.0, p=1.0,
                                              X=(0.25, 0.75)), sim)
        m0 = conserved_totals(sim).scalars
        run!(sim, tmax=0.5, verbose=false)
        ir = sim.grid.lo[1]:sim.grid.hi[1]
        X1 = sim.U[6, ir, 1, 1] ./ sim.U[1, ir, 1, 1]
        X2 = sim.U[7, ir, 1, 1] ./ sim.U[1, ir, 1, 1]
        @test maximum(abs.(X1 .- 0.25)) < 1e-12          # uniform stays uniform
        @test maximum(abs.(X1 .+ X2 .- 1)) < 1e-12
        @test conserved_totals(sim).scalars ≈ m0 rtol = 1e-13

        sim = Simulation(n=(64, 1, 1), nscalars=1, riemann=riemann, reconstruction=recon,
                         boundaries=Boundaries(Periodic()))
        set_initial_conditions!((x, y, z) -> (rho=1.0, vx=1.0, p=1.0,
                                              X=(0.25 < x < 0.5 ? 1.0 : 0.0,)), sim)
        run!(sim, tmax=0.3, verbose=false)
        X = sim.U[6, ir, 1, 1] ./ sim.U[1, ir, 1, 1]
        @test minimum(X) >= -1e-14                         # monotone contact
        @test maximum(X) <= 1 + 1e-14
        @test sim.diag.floors_scalar == 0
    end

    # scalars are passive: the hydro rows are identical with and without them,
    # with reflecting walls and three scalars (eight rows) on purpose
    a = Simulation(n=(32, 32, 1), boundaries=Boundaries(Reflecting()))
    b = Simulation(n=(32, 32, 1), nscalars=3, boundaries=Boundaries(Reflecting()))
    w(x, y) = (rho=1 + 0.2sinpi(2x) * cospi(2y), vx=0.3, vy=-0.2, p=1.0)
    set_initial_conditions!((x, y, z) -> w(x, y), a)
    set_initial_conditions!((x, y, z) -> (; w(x, y)..., X=(0.2, 0.3, 0.5)), b)
    run!(a, tmax=1.0, maxsteps=10, verbose=false)
    run!(b, tmax=1.0, maxsteps=10, verbose=false)
    @test a.U == b.U[1:Honghu.NHYDRO, :, :, :]

    # the density floor keeps the mass fraction; a negative scalar is clipped and counted
    s = Simulation(n=(4, 1, 1), nscalars=1)
    set_initial_conditions!((x, y, z) -> (rho=1.0, p=1.0, X=(0.4,)), s)
    i = s.grid.lo[1]
    s.U[1, i, 1, 1] = 1e-6
    s.U[6, i, 1, 1] = 0.4e-6
    Honghu.enforce_floors!(s.U, s.eos, s.grid, Floors(rho=1e-3), 1)
    @test s.U[1, i, 1, 1] == 1e-3
    @test s.U[6, i, 1, 1] / s.U[1, i, 1, 1] ≈ 0.4
    s.U[6, i+1, 1, 1] = -1e-3
    @test Honghu.enforce_scalar_floors!(s.U, s.grid, 1) == 1
    @test s.U[6, i+1, 1, 1] == 0.0
end

@testset "mass scalars: floors without hydro" begin
    # with the Euler update off, the scalars are still projected after the initial
    # conditions, and the clip is counted apart from the gas floors
    rad = RadiationPhysics(c=1.0, a_rad=1.0, opacity=ConstantOpacity(1.0))
    sim = Simulation(n=(8, 1, 1), nscalars=2, hydro=false, radiation=rad)
    set_initial_conditions!((x, y, z) -> (rho=1.0, T=1.0, Trad=0.5,
                                          X=(x < 0.5 ? -0.1 : 0.2, 0.3)), sim)
    ir = sim.grid.lo[1]:sim.grid.hi[1]
    @test minimum(sim.U[6, ir, 1, 1]) == 0.0
    @test sim.U[7, ir, 1, 1] == fill(0.3, 8)
    # the sweep covers the ghost cells, which copy the negative edge
    @test sim.diag.floors_scalar == 4 + sim.grid.ng[1]
    @test sim.diag.floors_hydro == 0
    # a scalar clip is a positivity repair like any other, so `nrepaired` sees it
    @test nrepaired(sim) == sim.diag.floors_scalar
end

@testset "mass scalars: output, fields, checkpoints" begin
    sim = Simulation(n=(8, 4, 1), nscalars=2)
    set_initial_conditions!((x, y, z) -> (rho=1 + x, p=1.0, X=(x, 1 - x)), sim)
    U0 = copy(sim.U)
    @test (:X_1, :X_2) ⊆ available_fields(sim)
    X1, lbl = field(sim, :X_1)
    @test lbl == "mass fraction 1"
    @test X1[3, 2, 1] ≈ coordinates(sim.grid, 1)[3]
    @test_throws ArgumentError field(sim, :X_3)
    @test_throws ArgumentError field(sim, :X_0)
    @test_throws ArgumentError field(Simulation(n=(8, 4, 1)), :X_1)
    @test size(primitives(sim), 1) == Honghu.NHYDRO + 2
    gas_temperature(sim); conserved_totals(sim); field(sim, :density)
    @test sim.U == U0                                   # reading does not perturb

    # the file holds the mass fraction, not rho X_s
    mf = only(f for f in Honghu.output_fields(sim) if f[1] == "mass_fraction_1")
    @test dropdims(mf[3]; dims=1) ≈ X1

    mktempdir() do dir
        fn = write_vtk(sim, joinpath(dir, "s"))
        @test occursin("Name=\"mass_fraction_2\"", read(fn, String))

        cp = save_checkpoint(sim, joinpath(dir, "c"))
        sim.U .= 0
        load_checkpoint!(sim, cp)
        @test sim.U == U0
        other = Simulation(n=(8, 4, 1), nscalars=1)
        @test_throws ErrorException load_checkpoint!(other, cp)

        # a version-1 file (no variable counts) still loads into a scalar-free simulation
        plain = Simulation(n=(8, 4, 1))
        set_initial_conditions!((x, y, z) -> (rho=1.0, p=1.0), plain)
        v1 = joinpath(dir, "old.honghu")
        open(v1, "w") do f
            write(f, b"HONGHU\x00\x01", Int64(8))
            foreach(d -> write(f, Int64(plain.grid.n[d])), 1:3)
            foreach(d -> write(f, Int64(plain.grid.ng[d])), 1:3)
            write(f, 0.0, 0.0, Int64(0), Int64(0), plain.U)
        end
        Uplain = copy(plain.U); plain.U .= 0
        load_checkpoint!(plain, v1)
        @test plain.U == Uplain
        @test_throws ErrorException load_checkpoint!(sim, v1)
    end
end

@testset "mass scalars: renormalisation is for chemistry only" begin
    s = Simulation(n=(4, 1, 1), nscalars=2)
    set_initial_conditions!((x, y, z) -> (rho=2.0, p=1.0, X=(0.3, 0.5)), s)     # sum X = 0.8
    i = s.grid.lo[1]
    U = s.U
    # the default leaves passive scalars alone
    @test Honghu.enforce_scalar_floors!(U, s.grid, 2) == 0
    @test U[6, i, 1, 1] + U[7, i, 1, 1] ≈ 1.6
    # renormalising makes them add up to rho, keeps their ratio, and is not a repair
    @test Honghu.enforce_scalar_floors!(U, s.grid, 2, true) == 0
    @test U[6, i, 1, 1] + U[7, i, 1, 1] ≈ 2.0 rtol = 1e-15
    @test U[6, i, 1, 1] / U[7, i, 1, 1] ≈ 0.3 / 0.5 rtol = 1e-15
    # a negative one is clipped and counted first, then the rest add up to rho
    U[6, i+1, 1, 1] = -0.1
    @test Honghu.enforce_scalar_floors!(U, s.grid, 2, true) == 1
    @test U[6, i+1, 1, 1] == 0
    @test U[7, i+1, 1, 1] ≈ 2.0 rtol = 1e-15
    # a cell with nothing to scale, or no mass, is left as it is
    U[6, i+2, 1, 1] = U[7, i+2, 1, 1] = 0
    @test Honghu.enforce_scalar_floors!(U, s.grid, 2, true) == 0
    @test U[6, i+2, 1, 1] == 0 && U[7, i+2, 1, 1] == 0
end
