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
        @test b.R ≈ a.R rtol = 1e-12
        @test b.U[2:4, :, :, :] ≈ a.U[2:4, :, :, :] rtol = 1e-12
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
    @test b.U[6:end, :, :, :] ≈ a.U[6:end, :, :, :] rtol = 1e-12
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
