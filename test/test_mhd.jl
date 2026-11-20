# Magnetohydrodynamics: the face-centred field, its boundary conditions, and the
# magnetic energy that every internal-energy site must subtract and add back.

# The largest |div b| dx / max|b| over the interior: the constrained-transport invariant.
function divb_norm(sim)
    d = Honghu.divergence_b(sim)
    bmax = maximum(maximum(abs, sim.B[e]) for e in 1:3)
    return maximum(abs, d) * minimum(sim.grid.dx[e] for e in 1:3 if sim.grid.active[e]) / bmax
end

# The internal energy of every interior cell, `E - Ekin - Emag`.
function eint_field(sim)
    g = sim.grid
    [begin
         u = Honghu.getstate(sim.U, Val(5), i, j, k)
         Honghu.internal_energy(u) - Honghu.magnetic_energy(sim.B, i, j, k)
     end for i in g.lo[1]:g.hi[1], j in g.lo[2]:g.hi[2], k in g.lo[3]:g.hi[3]]
end

# The interior of a state array: an MHD grid has more ghost cells than a plain one.
interior(sim, A) = A[:, sim.grid.lo[1]:sim.grid.hi[1], sim.grid.lo[2]:sim.grid.hi[2],
                     sim.grid.lo[3]:sim.grid.hi[3]]

# A smooth periodic vector potential with every component varying in every direction.
mhd_potential(x, y, z) = (0.1 * sinpi(2y) * cospi(2z), 0.1 * sinpi(2z) * cospi(2x),
                          0.1 * sinpi(2x) * cospi(2y))

@testset "MHD face field layout" begin
    for n in ((16, 1, 1), (16, 8, 1), (8, 6, 4))
        sim = Simulation(n=n, mhd=true, boundaries=Boundaries(Periodic()))
        g = sim.grid
        for d in 1:3
            # one more face than cells along its own direction, if that direction is active
            @test size(sim.B[d]) == (1, ntuple(e -> g.dims[e] + (e == d && g.active[d]), 3)...)
        end
    end
    @test Simulation(n=(8, 1, 1)).B === nothing
    # only periodic and outflow faces are defined for the field
    for bc in (Reflecting(), Fixed(ntuple(_ -> 1.0, 5)))
        @test_throws ArgumentError Simulation(n=(8, 8, 1), mhd=true, boundaries=Boundaries(bc))
    end
end

@testset "MHD initial conditions" begin
    gam = 5 / 3
    # a uniform oblique field from the keys, in every dimensionality
    for n in ((16, 1, 1), (16, 8, 1), (8, 6, 4))
        sim = Simulation(n=n, mhd=true, eos=IdealGas(gamma=gam), boundaries=Boundaries(Periodic()))
        set_initial_conditions!(sim) do x, y, z
            (rho=1.3, vx=0.2, vy=-0.1, vz=0.3, p=0.7, bx=0.3, by=-0.4, bz=0.5)
        end
        @test all(==(0.3), sim.B[1]) && all(==(-0.4), sim.B[2]) && all(==(0.5), sim.B[3])
        g = sim.grid
        i, j, k = g.lo
        emag = (0.3^2 + 0.4^2 + 0.5^2) / 2
        @test Honghu.magnetic_energy(sim.B, i, j, k) ≈ emag
        @test sim.U[5, i, j, k] ≈ 0.7 / (gam - 1) + 1.3 * (0.04 + 0.01 + 0.09) / 2 + emag
        # cons2prim takes the magnetic energy out again
        @test all(≈(0.7), primitives(sim)[IPRE, :, :, :])
        @test divb_norm(sim) == 0
    end

    # a field from a vector potential is divergence-free to round-off, in 2D and 3D
    for n in ((32, 32, 1), (16, 12, 10))
        sim = Simulation(n=n, mhd=true, boundaries=Boundaries(Periodic()))
        set_initial_conditions!(sim; vector_potential=mhd_potential) do x, y, z
            (rho=1.0, p=1.0, bx=0.1)
        end
        @test divb_norm(sim) < 1e-13
        @test maximum(abs, sim.B[2]) > 0.1           # the potential did set a field
    end

    # ... and `divergence_b` sees a field that is not
    sim = Simulation(n=(16, 1, 1), mhd=true)
    set_initial_conditions!((x, y, z) -> (rho=1.0, p=1.0, bx=x), sim)
    @test all(≈(1.0), Honghu.divergence_b(sim))

    # a field asked of a simulation without MHD is an error, not a field dropped
    nomhd = Simulation(n=(8, 1, 1))
    @test_throws ArgumentError set_initial_conditions!((x, y, z) -> (rho=1.0, p=1.0, by=1.0), nomhd)
    @test_throws ArgumentError set_initial_conditions!((x, y, z) -> (rho=1.0, p=1.0), nomhd;
                                                       vector_potential=mhd_potential)
end

@testset "MHD face boundary conditions" begin
    # periodic: the ghost faces are images of interior faces, along the field's own
    # direction (face lo - m is face hi + 1 - m) and across it (as cells)
    sim = Simulation(n=(8, 6, 4), mhd=true, boundaries=Boundaries(Periodic()))
    set_initial_conditions!((x, y, z) -> (rho=1.0, p=1.0), sim; vector_potential=mhd_potential)
    g = sim.grid
    B1 = sim.B[1]
    lo, hi, ng = g.lo, g.hi, g.ng
    for m in 1:ng[1]
        @test B1[:, lo[1]-m, :, :] == B1[:, hi[1]+1-m, :, :]
        @test B1[:, hi[1]+1+m, :, :] == B1[:, lo[1]+m, :, :]
    end
    for m in 1:ng[2]
        @test B1[:, :, lo[2]-m, :] == B1[:, :, hi[2]+1-m, :]
        @test B1[:, :, hi[2]+m, :] == B1[:, :, lo[2]-1+m, :]
    end

    # outflow: a ghost face copies the boundary face, the face itself along the
    # field's direction and the boundary row across it
    sim = Simulation(n=(8, 6, 1), mhd=true, boundaries=Boundaries(Outflow()))
    set_initial_conditions!((x, y, z) -> (rho=1.0, p=1.0), sim; vector_potential=mhd_potential)
    g = sim.grid
    B1, B2 = sim.B[1], sim.B[2]
    for m in 1:g.ng[1]
        @test B1[:, g.lo[1]-m, :, :] == B1[:, g.lo[1], :, :]
        @test B1[:, g.hi[1]+1+m, :, :] == B1[:, g.hi[1]+1, :, :]
        @test B2[:, g.lo[1]-m, :, :] == B2[:, g.lo[1], :, :]
        @test B2[:, g.hi[1]+m, :, :] == B2[:, g.hi[1], :, :]
    end
end

@testset "MHD floors and time step see the magnetic energy" begin
    gam = 5 / 3
    sim = Simulation(n=(4, 1, 1), mhd=true, eos=IdealGas(gamma=gam), boundaries=Boundaries(Periodic()),
                     floors=Floors(rho=1e-3, pressure=1e-3))
    set_initial_conditions!((x, y, z) -> (rho=1.0, vx=0.5, p=1.0, bx=1.0, by=2.0), sim)
    g = sim.grid
    i, j, k = g.lo
    emag = Honghu.magnetic_energy(sim.B, i, j, k)
    @test emag ≈ 2.5
    ekin = 0.5 * 0.5^2
    emin = Honghu.internal_energy(sim.eos, 1.0, 1e-3)

    # a cell with no thermal energy left keeps its magnetic energy when floored
    sim.U[5, i, j, k] = ekin + emag
    n = Honghu.enforce_floors!(sim.U, sim.eos, g, sim.floors; bfield=sim.B)
    @test n == 1
    @test sim.U[5, i, j, k] ≈ ekin + emin + emag
    # a vacuum cell is reset to the floor state with the field's energy on top
    sim.U[1, i+1, j, k] = 0.0
    Honghu.enforce_floors!(sim.U, sim.eos, g, sim.floors; bfield=sim.B)
    @test sim.U[5, i+1, j, k] ≈ emin + emag

    # the time step uses the fast magnetosonic speed along each direction
    set_initial_conditions!((x, y, z) -> (rho=1.0, vx=0.5, p=1.0, bx=1.0, by=2.0), sim)
    gp = gam * 1.0
    b2 = 1.0 + 4.0
    cfast(bt2) = sqrt((b2 + gp + sqrt((b2 - gp)^2 + 4gp * bt2)) / 2)
    @test timestep(sim) ≈ sim.cfl * g.dx[1] / (0.5 + cfast(4.0))
end

# Two copies of one radiating problem, with and without a magnetic field, step for
# step: radiation leaves the field alone, so the gas internal energy and the
# radiation must come out the same, and the total gas energy higher by exactly the
# field's energy. A site that read the field's energy as heat would make the
# magnetised gas hotter and the two runs part.
@testset "radiation coupling in a magnetised cell" begin
    function rad_box(; mhd, groups=Honghu.GREY, limiter=ConstantChat(), beta=0)
        rad = RadiationPhysics(c=10.0, chat=2.0, a_rad=1.0, opacity=ConstantOpacity(5.0),
                               groups=groups, chat_limiter=limiter, beta_order=beta)
        sim = Simulation(n=(6, 4, 1), mhd=mhd, hydro=false, radiation=rad,
                         eos=IdealGas(gamma=5 / 3, cv=1.0), boundaries=Boundaries(Periodic()))
        field = mhd ? (bx=1.2, by=-0.7, bz=0.5) : NamedTuple()
        set_initial_conditions!(sim) do x, y, z
            merge((rho=1.0 + 0.3 * sinpi(2x), vx=0.4, vy=-0.2, T=1.0 + 0.5 * cospi(2y),
                   Erad=0.2 * (1 + 0.5 * sinpi(2x))), field)
        end
        sim
    end
    for (groups, limiter, beta) in ((Honghu.GREY, ConstantChat(), 0),
                                    (Honghu.GREY, StabilityBound(), 1),
                                    (RadiationGroups(0.1, 20.0, 2), ConstantChat(), 0),
                                    (RadiationGroups(0.1, 20.0, 2), StabilityBound(), 0))
        a = rad_box(mhd=false; groups, limiter, beta)
        b = rad_box(mhd=true; groups, limiter, beta)
        for _ in 1:4
            dt = timestep(a)
            step!(a, dt)
            step!(b, dt)
        end
        @test eint_field(b) ≈ eint_field(a) rtol = 1e-12
        @test interior(b, b.R) ≈ interior(a, a.R) rtol = 1e-12
        @test interior(b, b.U)[2:4, :, :, :] ≈ interior(a, a.U)[2:4, :, :, :] rtol = 1e-12
        @test gas_temperature(b) ≈ gas_temperature(a) rtol = 1e-12
    end
end

@testset "photochemistry in a magnetised cell" begin
    net = net2()
    # A field whose energy is about the gas's thermal energy. BackwardEulerChem, because
    # the adaptive RosenbrockChem turns the round-off of adding and removing the field's
    # energy into 3e-10 in T after two steps: so does adding and removing the same number
    # with no field at all.
    box(mhd) = chem_box(net=net, n=4, mhd=mhd, solver=BackwardEulerChem(),
                        emission=(x, y, z, t) -> 1e-17,
                        ic=(x, y, z) -> merge((rho=hrho(net, 1e-3), vx=2e5, T=100.0,
                                               X=hX(net, 0.01), Erad=0.0),
                                              mhd ? (bx=5e-9, by=-4e-9) : NamedTuple()))
    a, b = box(false), box(true)
    for _ in 1:3
        dt = timestep(a)
        step!(a, dt)
        step!(b, dt)
    end
    @test a.diag.chem_cells > 0
    @test eint_field(b) ≈ eint_field(a) rtol = 1e-12
    @test interior(b, b.U)[6:end, :, :, :] ≈ interior(a, a.U)[6:end, :, :, :] rtol = 1e-12
    @test gas_temperature(b) ≈ gas_temperature(a) rtol = 1e-12
end

# --- the MHD Riemann solvers ---------------------------------------------------

# One side of an interface in canonical form, with `E` the total energy including
# the field's.
function mhd_state(rho, u, v, w, p, by, bz, bx; gam=5 / 3)
    E = p / (gam - 1) + rho * (u^2 + v^2 + w^2) / 2 + (bx^2 + by^2 + bz^2) / 2
    HydroState(rho, u, v, w, p, sqrt(gam * p / rho), E, by, bz)
end

# The ideal-MHD flux along x of one state, written independently of the solvers.
function mhd_physical_flux(s, bx)
    pt = s.P + (bx^2 + s.by^2 + s.bz^2) / 2
    (s.rho * s.u, s.rho * s.u^2 + pt - bx^2, s.rho * s.u * s.v - bx * s.by,
     s.rho * s.u * s.w - bx * s.bz, s.u * (s.E + pt) - bx * (s.u * bx + s.v * s.by + s.w * s.bz))
end

@testset "HydroState carries the transverse field" begin
    s = HydroState(1.0, 0.1, 0.2, 0.3, 1.0, 1.2, 2.5)            # the hydro constructor
    @test s.by == 0 && s.bz == 0
    s = HydroState(1.0, 0.1, 0.2, 0.3, 1.0, 1.2, 2.5, -0.4, 0.6)
    @test (s.by, s.bz) == (-0.4, 0.6)
    @test LLFMHD() isa Honghu.AbstractMHDRiemannSolver
end

@testset "LLFMHD" begin
    gam = 5 / 3
    cf(s, bx) = Honghu.fast_magnetosonic_speed(gam, s.rho, s.P, bx, s.by, s.bz)
    # consistency: equal states give the physical flux, and bounding speeds that
    # enclose the fast waves
    for (st, bx) in ((mhd_state(1.0, 0.3, -0.2, 0.1, 0.8, 0.5, -0.7, 0.9), 0.9),
                     (mhd_state(0.4, -1.1, 0.6, 0.0, 2.0, 0.0, 1.3, -0.5), -0.5))
        F, am, ap = Honghu.mhd_flux(LLFMHD(), st, st, gam, bx, 0.0)
        @test all(Tuple(F) .≈ mhd_physical_flux(st, bx))
        @test am ≈ max(0.0, cf(st, bx) - st.u)
        @test ap ≈ max(0.0, st.u + cf(st, bx))
    end
    # mirror symmetry: reflecting x maps the flux of (L, R) to minus that of
    # (mirror R, mirror L) in mass, energy and transverse momentum, and swaps the
    # two bounding speeds
    sL = mhd_state(1.0, 0.4, 0.3, -0.2, 1.0, 1.0, 0.2, 0.75)
    sR = mhd_state(0.125, -0.1, 0.0, 0.5, 0.1, -1.0, 0.4, 0.75)
    mir(s) = HydroState(s.rho, -s.u, s.v, s.w, s.P, s.cs, s.E, s.by, s.bz)
    F, am, ap = Honghu.mhd_flux(LLFMHD(), sL, sR, gam, 0.75, 0.0)
    G, bm, bp = Honghu.mhd_flux(LLFMHD(), mir(sR), mir(sL), gam, -0.75, 0.0)
    @test F[1] ≈ -G[1] && F[2] ≈ G[2] && F[3] ≈ -G[3] && F[4] ≈ -G[4] && F[5] ≈ -G[5]
    @test (am, ap) == (bp, bm)
    # without a field it is the hydrodynamic LLF
    hL = mhd_state(1.0, 0.4, 0.3, -0.2, 1.0, 0.0, 0.0, 0.0)
    hR = mhd_state(0.125, -0.1, 0.0, 0.5, 0.1, 0.0, 0.0, 0.0)
    F, _, _ = Honghu.mhd_flux(LLFMHD(), hL, hR, gam, 0.0, 0.0)
    @test all(Tuple(F) .≈ Tuple(riemann_flux(LLF(), hL, hR, gam, 0.0, 0.0)))
end

@testset "MHD takes an MHD Riemann solver" begin
    @test Simulation(n=(8, 1, 1), mhd=true).riemann isa Honghu.AbstractMHDRiemannSolver
    @test_throws ArgumentError Simulation(n=(8, 1, 1), mhd=true, riemann=HLLC())
    @test_throws ArgumentError Simulation(n=(8, 1, 1), riemann=LLFMHD())
end

# --- constrained transport ---------------------------------------------------------

# Orszag-Tang, in a box with `n` cells along x and y and `nz` along z (a thin
# periodic 3D box when nz > 1). Each field component varies only across its own
# direction, so the keys set it divergence-free.
function orszag_tang(; n=32, nz=1, riemann=LLFMHD(), reconstruction=PPM(), integrator=SSPRK2())
    sim = Simulation(n=(n, n, nz), mhd=true, riemann=riemann, reconstruction=reconstruction,
                     integrator=integrator, eos=IdealGas(gamma=5 / 3),
                     boundaries=Boundaries(Periodic()))
    b0 = 1 / sqrt(4pi)
    set_initial_conditions!(sim) do x, y, z
        (rho=25 / (36pi), p=5 / (12pi), vx=-sinpi(2y), vy=sinpi(2x),
         bx=-b0 * sinpi(2y), by=b0 * sinpi(4x))
    end
    sim
end

# Brio & Wu (1988) along direction D, with the transverse field in the first
# transverse direction of the canonical order (y for x, z for y, x for z). `nt`
# cells across, periodic there.
function brio_wu(D; n=64, nt=1, riemann=LLFMHD(), reconstruction=PLM(), integrator=SSPRK2())
    nn = ntuple(d -> d == D ? n : nt, 3)
    t1 = mod1(D + 1, 3)
    sim = Simulation(n=nn, mhd=true, riemann=riemann, reconstruction=reconstruction,
                     integrator=integrator, eos=IdealGas(gamma=2.0),
                     boundaries=Boundaries(ntuple(d -> d == D ? (Outflow(), Outflow()) :
                                                       (Periodic(), Periodic()), 3)...))
    key = (:bx, :by, :bz)
    set_initial_conditions!(sim) do x, y, z
        s = (x, y, z)[D]
        left = s < 0.5
        NamedTuple{(:rho, :p, key[D], key[t1])}((left ? 1.0 : 0.125, left ? 1.0 : 0.1, 0.75,
                                                  left ? 1.0 : -1.0))
    end
    sim
end

@testset "CT keeps div b at round-off" begin
    for riemann in (LLFMHD(), HLLD()), reconstruction in (PCM(), PLM(), PPM()),
        integrator in (ForwardEuler(), SSPRK2(), SSPRK3())
        sim = orszag_tang(n=32; riemann, reconstruction, integrator)
        for _ in 1:30
            step!(sim)
        end
        @test divb_norm(sim) < 1e-13
        @test nrepaired(sim) == 0
    end
    # 3D, every component varying in every direction
    sim = Simulation(n=(12, 10, 8), mhd=true, boundaries=Boundaries(Periodic()), integrator=SSPRK2())
    set_initial_conditions!(sim; vector_potential=mhd_potential) do x, y, z
        (rho=1.0 + 0.2sinpi(2x) * cospi(2z), vx=0.3sinpi(2y), vy=-0.2, vz=0.1cospi(2x), p=1.0, bx=0.2)
    end
    for _ in 1:20
        step!(sim)
    end
    @test divb_norm(sim) < 1e-13
end

@testset "MHD conservation on a periodic box" begin
    sim = orszag_tang(n=32)
    t0 = conserved_totals(sim)
    for _ in 1:20
        step!(sim)
    end
    t1 = conserved_totals(sim)
    @test t1.mass ≈ t0.mass rtol = 1e-13
    @test all(abs.(t1.momentum .- t0.momentum) .< 1e-13 * t0.mass)
    @test t1.energy_gas ≈ t0.energy_gas rtol = 1e-13
end

@testset "MHD preserves a uniform oblique state" begin
    for n in ((8, 1, 1), (8, 6, 1), (6, 6, 6))
        sim = Simulation(n=n, mhd=true, boundaries=Boundaries(Periodic()))
        set_initial_conditions!(sim) do x, y, z
            (rho=1.3, vx=0.4, vy=-0.3, vz=0.2, p=0.9, bx=0.5, by=-0.6, bz=0.7)
        end
        U0 = copy(sim.U)
        B0 = deepcopy(sim.B.b)
        for _ in 1:10
            step!(sim)
        end
        @test interior(sim, sim.U) ≈ interior(sim, U0) rtol = 1e-14
        @test all(maximum(abs, sim.B[d] .- B0[d]) < 1e-15 for d in 1:3)
    end
end

@testset "periodic face images stay bitwise equal" begin
    sim = orszag_tang(n=24)
    for _ in 1:25
        step!(sim)
    end
    g = sim.grid
    @test sim.B[1][:, g.hi[1]+1, :, :] == sim.B[1][:, g.lo[1], :, :]
    @test sim.B[2][:, :, g.hi[2]+1, :] == sim.B[2][:, :, g.lo[2], :]
end

@testset "MHD shock tube along x, y and z" begin
    # density, pressure, normal velocity and transverse field along the tube
    function line(sim, D)
        g = sim.grid
        t1 = mod1(D + 1, 3)
        idx(m) = ntuple(d -> d == D ? g.lo[d] + m - 1 : g.lo[d], 3)
        [(sim.U[1, idx(m)...], primitives(sim)[IPRE, ntuple(d -> d == D ? m : 1, 3)...],
          sim.U[D+1, idx(m)...], Honghu.cell_centred_field(sim.B, t1, idx(m)...)) for m in 1:g.n[D]]
    end
    lines = map(1:3) do D
        sim = brio_wu(D; n=64)
        run!(sim; tmax=0.08, verbose=false)
        @test nrepaired(sim) == 0
        line(sim, D)
    end
    for D in 2:3, q in 1:4
        @test getindex.(lines[D], q) ≈ getindex.(lines[1], q) rtol = 1e-13
    end
end

@testset "inactive directions equal a thin periodic 3D box" begin
    # 1D: Brio-Wu, against the same tube 4 cells wide, periodic across
    for recon in (PLM(), PPM())
        a = brio_wu(1; n=64, reconstruction=recon)
        b = brio_wu(1; n=64, nt=4, reconstruction=recon)
        for _ in 1:20
            dt = timestep(a)
            step!(a, dt)
            step!(b, dt)
        end
        pa, pb = primitives(a), primitives(b)
        for j in 1:4, k in 1:4
            @test pb[:, :, j, k] ≈ pa[:, :, 1, 1] rtol = 1e-12
        end
        by_a = [Honghu.cell_centred_field(a.B, 2, i, 1, 1) for i in a.grid.lo[1]:a.grid.hi[1]]
        by_b = [Honghu.cell_centred_field(b.B, 2, i, b.grid.lo[2], b.grid.lo[3]) for i in b.grid.lo[1]:b.grid.hi[1]]
        @test by_b ≈ by_a rtol = 1e-12
    end
    # 2D: Orszag-Tang, against the same problem 4 cells deep
    a = orszag_tang(n=24)
    b = orszag_tang(n=24, nz=4)
    for _ in 1:15
        dt = timestep(a)
        step!(a, dt)
        step!(b, dt)
    end
    pa, pb = primitives(a), primitives(b)
    for k in 1:4
        @test pb[:, :, :, k] ≈ pa[:, :, :, 1] rtol = 1e-12
    end
end

# A circularly polarised Alfven wave along x (Toth 2000), an exact nonlinear solution
# with b_par = 1 and rho = 1, so v_A = 1 and one period is t = 1. Returns the L1
# error of b_y after one period, relative to its L1 norm.
function cpaw_1d(n; reconstruction=PLM(), riemann=LLFMHD())
    sim = Simulation(n=(n, 1, 1), mhd=true, reconstruction=reconstruction, riemann=riemann,
                     integrator=SSPRK2(), boundaries=Boundaries(Periodic()))
    A = 0.1
    set_initial_conditions!(sim) do x, y, z
        (rho=1.0, p=0.1, vy=-A * sinpi(2x), vz=-A * cospi(2x), bx=1.0, by=A * sinpi(2x),
         bz=A * cospi(2x))
    end
    by(s) = [Honghu.cell_centred_field(s.B, 2, i, 1, 1) for i in s.grid.lo[1]:s.grid.hi[1]]
    b0 = by(sim)
    run!(sim, tmax=1.0, verbose=false)
    return sum(abs, by(sim) .- b0) / sum(abs, b0)
end

@testset "a 1D Alfven wave comes back after one period" begin
    e32, e64 = cpaw_1d(32), cpaw_1d(64)
    @test e64 < 0.015
    @test log2(e32 / e64) > 1.5
end

# --- HLLD ----------------------------------------------------------------------------

@testset "HLLD" begin
    gam = 5 / 3
    cf(s, bx) = Honghu.fast_magnetosonic_speed(gam, s.rho, s.P, bx, s.by, s.bz)
    # consistency, also with no normal field, where the Alfven waves merge with the contact
    for (st, bx) in ((mhd_state(1.0, 0.3, -0.2, 0.1, 0.8, 0.5, -0.7, 0.9), 0.9),
                     (mhd_state(0.4, -1.1, 0.6, 0.0, 2.0, 0.0, 1.3, -0.5), -0.5),
                     (mhd_state(0.7, 0.2, 0.3, -0.4, 1.0, 0.8, 0.6, 0.0), 0.0))
        F, am, ap = Honghu.mhd_flux(HLLD(), st, st, gam, bx, 0.0)
        @test all(Tuple(F) .≈ mhd_physical_flux(st, bx))
        @test am ≈ max(0.0, cf(st, bx) - st.u) && ap ≈ max(0.0, st.u + cf(st, bx))
    end
    # every wave moving right: the flux is the left state's
    sL = mhd_state(1.0, 5.0, 0.1, -0.2, 1.0, 0.3, 0.2, 0.5)
    sR = mhd_state(0.5, 5.0, 0.0, 0.1, 0.4, -0.3, 0.1, 0.5)
    F, am, ap = Honghu.mhd_flux(HLLD(), sL, sR, gam, 0.5, 0.0)
    @test all(Tuple(F) .≈ mhd_physical_flux(sL, 0.5)) && am == 0
    # mirror symmetry, as for LLFMHD
    sL = mhd_state(1.0, 0.4, 0.3, -0.2, 1.0, 1.0, 0.2, 0.75)
    sR = mhd_state(0.125, -0.1, 0.0, 0.5, 0.1, -1.0, 0.4, 0.75)
    mir(s) = HydroState(s.rho, -s.u, s.v, s.w, s.P, s.cs, s.E, s.by, s.bz)
    F, am, ap = Honghu.mhd_flux(HLLD(), sL, sR, gam, 0.75, 0.0)
    G, bm, bp = Honghu.mhd_flux(HLLD(), mir(sR), mir(sL), gam, -0.75, 0.0)
    @test F[1] ≈ -G[1] && F[2] ≈ G[2] && F[3] ≈ -G[3] && F[4] ≈ -G[4] && F[5] ≈ -G[5]
    @test (am, ap) == (bp, bm)
    # a stationary contact: no mass or energy crosses it, and the momentum flux is the
    # total pressure, for HLLD and not for LLFMHD
    cL = mhd_state(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 1.0)
    cR = mhd_state(0.25, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 1.0)
    F, _, _ = Honghu.mhd_flux(HLLD(), cL, cR, gam, 1.0, 0.0)
    @test abs(F[1]) < 1e-15 && abs(F[5]) < 1e-15
    F, _, _ = Honghu.mhd_flux(LLFMHD(), cL, cR, gam, 1.0, 0.0)
    @test abs(F[1]) > 0.1
    @test Simulation(n=(8, 1, 1), mhd=true).riemann isa HLLD

    # a stationary rotational discontinuity (rho = p = 1, u = c_A = 1, b_t turned by 90
    # degrees, v_t jumping by sign(b_n) delta(b_t) / sqrt(rho)): HLLD's flux is the
    # upwind physical flux, LLFMHD's is not. In a simulation the RD still smears, with
    # either solver: constrained transport advances the field with the LD04 EMF, whose
    # dissipation term alpha+ alpha- / (alpha+ + alpha-) (b_y^L - b_y^R) is not zero there.
    rL = mhd_state(1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0)
    rR = mhd_state(1.0, 1.0, -1.0, 1.0, 1.0, 0.0, 1.0, 1.0)
    F, _, _ = Honghu.mhd_flux(HLLD(), rL, rR, gam, 1.0, 0.0)
    @test all(isapprox.(Tuple(F), mhd_physical_flux(rL, 1.0); atol=1e-14))
    @test all(isapprox.(Tuple(F), mhd_physical_flux(rR, 1.0); atol=1e-14))
    F, _, _ = Honghu.mhd_flux(LLFMHD(), rL, rR, gam, 1.0, 0.0)
    @test !isapprox(F[3], mhd_physical_flux(rL, 1.0)[3]; atol=1e-3)
end

# Isolated discontinuities that HLLD resolves and LLFMHD smears: the cells of the
# profile `q` that are neither the left nor the right value after `nsteps` with PCM.
function smeared_cells(ic, q; riemann, n=32, nsteps=40)
    sim = Simulation(n=(n, 1, 1), mhd=true, riemann=riemann, reconstruction=PCM(),
                     integrator=SSPRK2(), boundaries=Boundaries(Outflow()))
    set_initial_conditions!(ic, sim)
    for _ in 1:nsteps
        step!(sim)
    end
    v = q(sim)
    count(x -> !(isapprox(x, v[1]; rtol=1e-10) || isapprox(x, v[end]; rtol=1e-10)), v)
end

@testset "HLLD keeps an isolated contact sharp" begin
    contact(x, y, z) = (rho=x < 0.5 ? 1.0 : 0.25, p=1.0, bx=1.0, by=0.5)
    rho(sim) = interior(sim, sim.U)[1, :, 1, 1]
    @test smeared_cells(contact, rho; riemann=HLLD()) == 0
    @test smeared_cells(contact, rho; riemann=LLFMHD()) > 4
end

# Along one direction LD04 is the HLL flux of the transverse field: the reason an
# isolated rotational discontinuity smears whatever the Riemann solver. In 1D the
# edge EMF must be, to round-off, the HLL flux of b_y (for eps_z) and minus that of
# b_z (for eps_y), (a+ F_L + a- F_R - a+ a- (b_R - b_L)) / (a+ + a-) with
# F = u* b_t - b_x u_t, from the same reconstructed states and bounding speeds.
@testset "LD04 is the HLL flux of b_t in 1D" begin
    for recon in (PCM(), PLM(), PPM()), riemann in (HLLD(), LLFMHD())
        sim = Simulation(n=(32, 1, 1), mhd=true, reconstruction=recon, riemann=riemann,
                         boundaries=Boundaries(Periodic()))
        set_initial_conditions!(sim) do x, y, z
            (rho=1.0 + 0.3sinpi(2x), p=0.5, vx=0.2, vy=0.4cospi(2x), vz=-0.1sinpi(4x),
             bx=0.8, by=0.6sinpi(2x), bz=0.3cospi(2x))
        end
        for _ in 1:5
            step!(sim)
        end
        Honghu.hydro_rhs!(sim.hw.dU, sim.U, sim)      # the face arrays and EMFs of this state
        mw, B = sim.mw, sim.B
        o = (1, 0, 0)
        worst = 0.0
        for i in sim.grid.lo[1]:(sim.grid.hi[1]+1)
            am, ap = mw.fspd[1][1, i, 1, 1], mw.fspd[1][2, i, 1, 1]
            us, bx = mw.ustar[1][1, i, 1, 1], B[1][1, i, 1, 1]
            side(A) = (last(Honghu._recon_pair(recon, A, i - 1, 1, 1, o)),
                       first(Honghu._recon_pair(recon, A, i, 1, 1, o)))
            hll(bt, ut) = begin
                FL, FR = us * bt[1] - bx * ut[1], us * bt[2] - bx * ut[2]
                (ap * FL + am * FR - ap * am * (bt[2] - bt[1])) / (ap + am)
            end
            worst = max(worst, abs(mw.emf[3][1, i, 1, 1] - hll(side(B[2]), side(mw.ustar[2]))),
                        abs(mw.emf[2][1, i, 1, 1] + hll(side(B[3]), side(mw.ustar[3]))))
        end
        @test worst < 1e-14
    end
end

@testset "MHD fields and output" begin
    sim = Simulation(n=(8, 6, 1), mhd=true, boundaries=Boundaries(Periodic()))
    set_initial_conditions!((x, y, z) -> (rho=1.0, p=1.0, bx=0.3, by=-0.4, bz=0.5), sim)
    @test all(f -> f in available_fields(sim), (:bx, :by, :bz, :magnetic_pressure))
    @test :bx ∉ available_fields(Simulation(n=(8, 1, 1)))
    @test all(==(-0.4), first(field(sim, :by)))
    @test all(≈(0.25), first(field(sim, :magnetic_pressure)))
    f = Dict(name => data for (name, _, data) in Honghu.output_fields(sim))
    @test size(f["magnetic_field"]) == (3, 8, 6, 1)
    @test all(==(0.5), f["magnetic_field"][3, :, :, :])
    @test_throws ArgumentError field(Simulation(n=(8, 1, 1)), :bx)
end

@testset "oblique Alfven wave converges at second order" begin
    # one period of Toth's circularly polarised wave at 63 degrees to the grid, PLM +
    # SSPRK2 with Quokka's default limiter; L1 error of b_z over the amplitude
    function err(n)
        sim, tmax = P.alfven_wave(n=n, reconstruction=PLM(Sweby(1.5)), integrator=SSPRK2())
        g = sim.grid
        bz(s) = [Honghu.cell_centred_field(s.B, 3, i, j, 1) for i in g.lo[1]:g.hi[1], j in g.lo[2]:g.hi[2]]
        b0 = bz(sim)
        run!(sim; tmax, verbose=false)
        @test divb_norm(sim) < 1e-13
        @test nrepaired(sim) == 0
        return sum(abs, bz(sim) .- b0) / length(b0) / 0.1
    end
    e32, e64 = err(32), err(64)
    @test e64 < 0.01
    @test log2(e32 / e64) > 1.8          # measured 1.87; 2.00 from 64 to 128
end

# The x line of a Quokka MHD shock tube, as test/data/quokka holds it: columns
# x rho vx vy vz p by bz.
read_quokka_line(name) =
    permutedims(reduce(hcat, [parse.(Float64, split(l))
                              for l in eachline(joinpath(@__DIR__, "data", "quokka", name * ".txt"))
                              if !startswith(l, "#")]))

@testset "shock tubes against Quokka" begin
    # Quokka's MHDBrioWuShockTube and MHDRyuJones2aShockTube at 256 cells (in 256x16x16
    # boxes) with Q26 + LD04, RK2 and a constant step of 2.5e-4, against the same in 1D.
    # With PCM the two codes are the same scheme and agree to round-off. With PLM (same
    # limiter, Sweby 1.5) and PPM they differ by Quokka's shock flattening, which is
    # always on there and absent here: the bound is the measured L1 difference with a
    # margin. Quokka holds the initial states at the boundaries and this run has outflow,
    # which differs once Ryu-Jones's fast shock reaches x = 1, so that tube is compared
    # on x < 0.9.
    cases = (("bw", P.brio_wu, 1.0), ("rj", P.ryu_jones_2a, 0.9))
    recons = (("pcm", PCM(), nothing), ("plm", PLM(Sweby(1.5)), (bw=0.025, rj=0.003)),
              ("ppm", PPM(), (bw=0.04, rj=0.005)))
    for (tag, problem, xmax) in cases, (rtag, recon, bound) in recons
        ref = read_quokka_line(tag * "_" * rtag)
        sim, tmax = problem(n=size(ref, 1), reconstruction=recon, integrator=SSPRK2())
        for _ in 1:round(Int, tmax / 2.5e-4)
            step!(sim, 2.5e-4)
        end
        @test nrepaired(sim) == 0
        g = sim.grid
        pr = primitives(sim)
        bcc(d) = [Honghu.cell_centred_field(sim.B, d, i, 1, 1) for i in g.lo[1]:g.hi[1]]
        mine = hcat(pr[1, :, 1, 1], pr[2, :, 1, 1], pr[3, :, 1, 1], pr[4, :, 1, 1],
                    pr[5, :, 1, 1], bcc(2), bcc(3))
        w = ref[:, 1] .< xmax
        for c in 1:7
            q, m = ref[w, c+1], mine[w, c]
            if bound === nothing
                @test maximum(abs, m .- q) < 1e-9
            elseif any(!=(0), q)
                @test sum(abs, m .- q) / sum(abs, q) < bound[Symbol(tag)]
            end
        end
    end
end

@testset "MHD checkpoints are refused, not written without the field" begin
    sim = Simulation(n=(8, 1, 1), mhd=true)
    set_initial_conditions!((x, y, z) -> (rho=1.0, p=1.0, by=1.0), sim)
    path = joinpath(mktempdir(), "mhd")
    @test_throws ArgumentError save_checkpoint(sim, path)
    plain = Simulation(n=(8, 1, 1))
    set_initial_conditions!((x, y, z) -> (rho=1.0, p=1.0), plain)
    file = save_checkpoint(plain, path)
    @test_throws ArgumentError load_checkpoint!(sim, file)
end
