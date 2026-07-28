"Build a Sod shock tube running along direction `D`."
function sod_along(D::Int; nx=128, kwargs...)
    n = ntuple(d -> d == D ? nx : 1, 3)
    sim = Simulation(; n=n, xhi=(1.0, 1.0, 1.0), eos=IdealGas(gamma=1.4),
                     boundaries=Boundaries(Outflow()), kwargs...)
    set_initial_conditions!(sim) do x, y, z
        s = (x, y, z)[D]
        s < 0.5 ? (rho=1.0, p=1.0) : (rho=0.125, p=0.1)
    end
    sim
end

"Interior density profile along the single active direction."
function profile(sim, D::Int, var=IRHO)
    pr = primitives(sim)
    g = sim.grid
    [pr[var, ntuple(d -> d == D ? i : 1, 3)...] for i in 1:g.n[D]]
end

@testset "Sod shock tube vs exact solution" begin
    for nx in (128, 256)
        sim, tmax = P.sod_shock_tube(nx=nx)
        run!(sim; tmax, verbose=false)
        @test sim.t ≈ tmax
        pr = primitives(sim)
        g = sim.grid
        e = 0.0
        for i in 1:g.n[1]
            xc = g.xlo[1] + (i - 0.5) * g.dx[1]
            rho, _, _ = P.sod_exact(xc, sim.t)
            e += abs(pr[IRHO, i, 1, 1] - rho)
        end
        # the error is dominated by the smeared contact, so it falls like 1/nx
        @test e / g.n[1] < 4.0 / nx
        @test all(pr[IRHO, :, 1, 1] .> 0)
        @test all(pr[IPRE, :, 1, 1] .> 0)
        # no new extrema
        @test maximum(pr[IRHO, :, 1, 1]) <= 1.0 + 1e-9
        @test minimum(pr[IRHO, :, 1, 1]) >= 0.125 - 1e-9
    end
end

@testset "Toro Riemann problem battery" begin
    # The five shock tubes of Toro (2009) table 4.1: strong rarefactions and a
    # near vacuum (2), very strong shocks (3, 4), and two colliding shocks (5).
    # The scheme must stay positive without help from the floors.
    cases = (("Sod", (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.2, 0.01),
             ("123", (1.0, -2.0, 0.4), (1.0, 2.0, 0.4), 0.15, 0.01),
             ("blast L", (1.0, 0.0, 1000.0), (1.0, 0.0, 0.01), 0.012, 0.10),
             ("blast R", (1.0, 0.0, 0.01), (1.0, 0.0, 100.0), 0.035, 0.10),
             ("collision", (5.99924, 19.5975, 460.894), (5.99242, -6.19633, 46.095), 0.035, 0.30))
    for (name, L, R, tmax, tol) in cases
        sim = Simulation(n=(200, 1, 1), eos=IdealGas(gamma=1.4),
                         boundaries=Boundaries(Outflow()))
        set_initial_conditions!(sim) do x, y, z
            x < 0.5 ? (rho=L[1], vx=L[2], p=L[3]) : (rho=R[1], vx=R[2], p=R[3])
        end
        run!(sim; tmax, verbose=false)
        pr = primitives(sim)
        g = sim.grid
        @test all(pr[IRHO, :, 1, 1] .> 0)
        @test all(pr[IPRE, :, 1, 1] .> 0)
        @test sim.nrepaired == 0            # no help from the positivity floors
        e = 0.0
        for i in 1:g.n[1]
            xc = (i - 0.5) * g.dx[1]
            r, _, _ = P.sod_exact(xc, sim.t; rhoL=L[1], uL=L[2], pL=L[3],
                                  rhoR=R[1], uR=R[2], pR=R[3], gamma=1.4)
            e += abs(pr[IRHO, i, 1, 1] - r)
        end
        @test e / g.n[1] < tol
    end
end

@testset "directional isotropy" begin
    # the same shock tube along x, y and z must give identical answers
    profiles = map(1:3) do D
        sim = sod_along(D; nx=128)
        run!(sim; tmax=0.2, verbose=false)
        profile(sim, D)
    end
    @test profiles[1] ≈ profiles[2]
    @test profiles[1] ≈ profiles[3]
end

@testset "uniform flow is preserved" begin
    # a uniform state advected through a periodic 3D box must stay exactly uniform
    sim = Simulation(n=(8, 8, 8), boundaries=Boundaries(Periodic()))
    set_initial_conditions!(sim) do x, y, z
        (rho=1.5, vx=0.4, vy=-0.2, vz=0.3, p=2.0)
    end
    run!(sim; tmax=0.2, verbose=false)
    pr = primitives(sim)
    @test all(isapprox.(pr[IRHO, :, :, :], 1.5; rtol=1e-12))
    @test all(isapprox.(pr[IVX, :, :, :], 0.4; rtol=1e-12))
    @test all(isapprox.(pr[IPRE, :, :, :], 2.0; rtol=1e-12))
end

@testset "conservation" begin
    # periodic boundaries and no sources: mass, momentum and energy are exact
    sim, _ = P.kelvin_helmholtz(n=64)
    t0 = conserved_totals(sim)
    run!(sim; tmax=0.3, verbose=false)
    t1 = conserved_totals(sim)
    @test isapprox(t1.mass, t0.mass; rtol=1e-12)
    @test isapprox(t1.energy_total, t0.energy_total; rtol=1e-12)
    @test isapprox(t1.momentum[1], t0.momentum[1]; atol=1e-12 * t0.mass)
    @test sim.nrepaired == 0
end

@testset "diagnostics do not perturb the state" begin
    # `cons2prim!` used to enforce the floors on `U` in place, so *reading* a
    # field repaired the state being read -- silently, and without counting the
    # repairs. Two runs that differed only in how often they were plotted could
    # then disagree. Cells below the floors are what makes this observable, so
    # the state deliberately contains some.
    mk() = begin
        s = Simulation(n=(8, 1, 1), floors=Floors(rho=1e-3, pressure=1e-3))
        set_initial_conditions!(s) do x, y, z
            x > 0.75 ? (rho=1e-8, p=1e-8) : (rho=1.0, p=1.0)
        end
        s
    end

    sim = mk()
    step!(sim)                       # a state the floors have already touched
    U0, R0, n0 = copy(sim.U), copy(sim.R), sim.nrepaired

    for f in (s -> primitives(s), s -> field(s, :density), s -> field(s, :pressure),
              s -> field(s, :temperature), s -> field(s, :mach), s -> gas_temperature(s),
              s -> timestep(s), s -> conserved_totals(s))
        f(sim)
    end
    mktempdir() do dir
        write_vtk(sim, joinpath(dir, "snap"))
        write_line(sim, joinpath(dir, "line.txt"); dir=1)
    end

    @test sim.U == U0                # bit-identical, not merely close
    @test sim.R == R0
    @test sim.nrepaired == n0

    # ... and the trajectory itself is independent of how often you looked
    a, b = mk(), mk()
    for _ in 1:20
        step!(a)
        step!(b)
        field(b, :density)           # b is "plotted" every step
    end
    @test a.U == b.U
end

@testset "reflecting wall" begin
    # a symmetric problem in a reflecting box must stay symmetric
    sim = Simulation(n=(64, 1, 1), boundaries=Boundaries(Reflecting()))
    set_initial_conditions!(sim) do x, y, z
        (rho=1.0, vx=0.5 * sinpi(2x), p=1.0)
    end
    t0 = conserved_totals(sim)
    run!(sim; tmax=0.3, verbose=false)
    t1 = conserved_totals(sim)
    @test isapprox(t1.mass, t0.mass; rtol=1e-12)
    rho = profile(sim, 1)
    @test rho ≈ reverse(rho) rtol = 1e-10
end

@testset "3D Sedov blast" begin
    sim, tmax = P.sedov_blast(n=32)
    t0 = conserved_totals(sim)
    run!(sim; tmax, verbose=false)
    t1 = conserved_totals(sim)
    @test isapprox(t1.mass, t0.mass; rtol=1e-12)
    @test isapprox(t1.energy_total, t0.energy_total; rtol=1e-10)
    pr = primitives(sim)
    @test all(pr[IRHO, :, :, :] .> 0)
    @test all(pr[IPRE, :, :, :] .> 0)
    # the blast must be spherical: the shock radius along x, y and z agrees
    g = sim.grid
    ic = g.n[1] ÷ 2
    rx = argmax(pr[IRHO, ic:end, ic, ic])
    ry = argmax(pr[IRHO, ic, ic:end, ic])
    rz = argmax(pr[IRHO, ic, ic, ic:end])
    @test rx == ry == rz
    # and close to the Sedov-Taylor similarity radius, xi ~ 1.15 for gamma = 5/3
    R = (rx - 0.5) * g.dx[1]
    @test 0.9 < R / (1.15 * (tmax^2)^(1 / 5)) < 1.3
end

@testset "integrators" begin
    # all three integrators are stable and agree to their formal accuracy
    ref = nothing
    for integ in (ForwardEuler(), SSPRK2(), SSPRK3())
        sim, tmax = P.smooth_advection(nx=64, reconstruction=PLM(), integrator=integ)
        run!(sim; tmax, verbose=false)
        pr = primitives(sim)
        @test all(pr[IRHO, :, 1, 1] .> 0)
        @test Hydro3D.nstages(integ) == length(Hydro3D.tableau(integ))
        e = maximum(abs.(pr[IRHO, :, 1, 1] .-
                         [1 + 0.5 * sinpi(2 * (i - 0.5) / 64) for i in 1:64]))
        @test e < 0.2
    end
end
