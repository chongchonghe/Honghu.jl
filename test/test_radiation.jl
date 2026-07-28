@testset "free streaming" begin
    # the front must advance at exactly chat, and the state stay causal
    for chat_over_c in (1.0, 0.5)
        sim, tmax = P.radiation_streaming(nx=256, chat_over_c=chat_over_c)
        run!(sim; tmax, verbose=false)
        g = sim.grid
        er = [sim.R[IERAD, i+g.ng[1], 1, 1] for i in 1:g.n[1]]
        xs = [(i - 0.5) * g.dx[1] for i in 1:g.n[1]]
        front = xs[findlast(>(0.5), er)]
        @test front ≈ chat_over_c * tmax rtol = 0.02
        @test er[4] ≈ 1.0 rtol = 1e-6            # plateau behind the front
        @test er[end] < 1e-6                     # nothing ahead of it
        # causality: |F| <= c E everywhere
        @test all(1:g.n[1]) do i
            abs(sim.R[IFRX, i+g.ng[1], 1, 1]) <= sim.R[IERAD, i+g.ng[1], 1, 1] * (1 + 1e-10)
        end
    end
end

@testset "static diffusion limit" begin
    # the M1 system must reduce to D = c/(3 rho kappa) independently of how many
    # mean free paths a cell spans: this is the asymptotic-preserving property
    D = 1 / (3 * 300.0)
    errs = Float64[]
    for nx in (64, 128, 256)
        sim, tmax = P.radiation_diffusion(nx=nx)
        run!(sim; tmax, verbose=false)
        g = sim.grid
        num = 0.0
        den = 0.0
        for i in 1:g.n[1]
            xc = (i - 0.5) * g.dx[1]
            ex = P.diffusion_exact(xc, sim.t; D=D)
            num += abs(sim.R[IERAD, i+g.ng[1], 1, 1] - ex)
            den += ex
        end
        push!(errs, num / den)
    end
    @test errs[1] < 0.02
    @test errs[3] < errs[2] < errs[1]
    @test errs[3] < 2e-3

    # energy is conserved exactly: no absorption, periodic boundaries
    sim, tmax = P.radiation_diffusion(nx=128)
    e0 = conserved_totals(sim).energy_rad
    run!(sim; tmax, verbose=false)
    @test isapprox(conserved_totals(sim).energy_rad, e0; rtol=1e-12)
end

@testset "stiff matter-radiation coupling" begin
    # equilibrium of cv*T + a*T^4 = cv*T_gas0 + a*T_rad0^4 with cv = a = 1
    Teq = let lo = 0.0, hi = 1.0
        for _ in 1:200
            mid = (lo + hi) / 2
            (mid + mid^4 - (1.0 + 1e-4)) < 0 ? (lo = mid) : (hi = mid)
        end
        (lo + hi) / 2
    end
    for kappa in (1e3, 1e6, 1e9)
        sim, tmax = P.matter_coupling(kappa=kappa)
        t0 = conserved_totals(sim)
        run!(sim; tmax, verbose=false)
        t1 = conserved_totals(sim)
        # the implicit solve conserves E_int + (c/chat) E_rad to round-off
        @test isapprox(t1.energy_total, t0.energy_total; rtol=1e-10)
        Tg = gas_temperature(sim)[1, 1, 1]
        Tr = radiation_temperature(sim)[1, 1, 1]
        # stiffer coupling -> closer to equilibrium, and to the analytic value
        @test isapprox(Tg, Tr; rtol=10 / sqrt(kappa))
        @test isapprox(Tg, Teq; rtol=10 / sqrt(kappa))
    end
    # a single enormous step must still land on equilibrium (L-stability)
    sim, _ = P.matter_coupling(kappa=1e9)
    step!(sim, 1.0)
    @test isapprox(gas_temperature(sim)[1, 1, 1], Teq; rtol=1e-4)
end

@testset "radiation force" begin
    # an optically thick slab pushed by a radiation flux. Integrating
    #   dp/dt = rho kappa F/c,  dF/dt = -chat rho kappa F
    # gives  dp = (rho kappa/c) F0 (1 - exp(-chat rho kappa t))/(chat rho kappa)
    c, chat, kappa, rho = 100.0, 10.0, 1.0, 1.0
    F0 = 0.9 * c * 1.0
    tmax = 0.05
    dp_exact = (rho * kappa / c) * F0 * (1 - exp(-chat * rho * kappa * tmax)) /
               (chat * rho * kappa)

    function drive(nx)
        rad = RadiationPhysics(; c, chat, a_rad=1e-6, opacity=ConstantOpacity(kappa),
                               beta_order=1)
        sim = Simulation(n=(nx, 1, 1), radiation=rad, eos=IdealGas(gamma=5 / 3, cv=1.0),
                         boundaries=Boundaries(Periodic()))
        set_initial_conditions!(sim) do x, y, z
            (rho=rho, p=1.0, Erad=1.0, Fx=F0)
        end
        run!(sim; tmax, verbose=false)
        sim
    end

    s1 = drive(32)
    s2 = drive(64)
    t1 = conserved_totals(s1)
    t2 = conserved_totals(s2)
    e1 = abs(t1.momentum[1] / t1.mass - dp_exact)
    e2 = abs(t2.momentum[1] / t2.mass - dp_exact)
    @test e1 / dp_exact < 1e-2
    # halving the step halves the error at least: the IMEX coupling converges
    @test e2 < 0.6 * e1
    # E_gas + (c/chat) E_rad is conserved up to the O(beta^2) truncation of the
    # velocity-dependent terms
    @test isapprox(t1.energy_gas + (c / chat) * t1.energy_rad, 1.5 + (c / chat) * 1.0;
                   rtol=1e-5)
    # radiation with no flux and no velocity must not push anything
    rad0 = RadiationPhysics(; c, chat, a_rad=1e-6, opacity=ConstantOpacity(kappa),
                            beta_order=1)
    still = Simulation(n=(16, 1, 1), radiation=rad0, eos=IdealGas(gamma=5 / 3, cv=1.0),
                       boundaries=Boundaries(Periodic()))
    set_initial_conditions!(still) do x, y, z
        (rho=1.0, p=1.0, Erad=1.0, Fx=0.0)
    end
    run!(still; tmax=0.05, verbose=false)
    @test abs(conserved_totals(still).momentum[1]) < 1e-14
end

@testset "problem factories expose their physics" begin
    # A factory builds its own RadiationPhysics from constants that exist only
    # in its body, so overriding the whole object means re-deriving them -- and
    # getting one wrong gives a *running* simulation with different physics
    # rather than an error. The knobs worth varying are named keywords instead.
    @test P.radhydro_shock(nx=16)[1].radiation.closure isa EddingtonClosure
    @test P.radhydro_shock(nx=16, closure=LevermoreClosure())[1].radiation.closure isa
          LevermoreClosure
    @test P.radhydro_shock(nx=16, beta_order=0)[1].radiation.beta_order == 0

    # chat_factor scales the reduced speed of light, so it scales the substep count
    chat(f) = P.radhydro_shock(nx=16, chat_factor=f)[1].radiation.chat
    @test chat(20.0) ≈ 2 * chat(10.0)

    for f in (P.radiation_streaming, P.radiation_diffusion, P.marshak_wave,
              P.radiation_shadow)
        @test f(closure=EddingtonClosure())[1].radiation.closure isa EddingtonClosure
        @test f(beta_order=1)[1].radiation.beta_order == 1
        @test f()[1].radiation.closure isa LevermoreClosure   # default unchanged
    end

    # and the override reaches the solver, not just the struct
    spike(cl) = begin
        sim, tmax = P.radhydro_shock(nx=128, closure=cl)
        run!(sim; tmax, verbose=false)
        maximum(gas_temperature(sim)[:, 1, 1])
    end
    # optically thick and nearly isotropic here, so M1 and P1 agree closely --
    # but the run must actually differ, or the keyword is doing nothing
    sM1, sP1 = spike(LevermoreClosure()), spike(EddingtonClosure())
    @test sM1 != sP1
    @test isapprox(sM1, sP1; rtol=1e-3)
end

@testset "diagnostics" begin
    # The implicit solve's convergence behaviour used to be computed and thrown
    # away: `radiation_source_step!` built a SourceStats, `advance_radiation!`
    # summed it, and `step!` dropped the value. Nothing in the package read it.
    sim, tmax = P.radhydro_shock(nx=64)
    run!(sim; tmax, verbose=false)
    d = sim.diag

    @test d.steps == sim.step
    @test d.rad_substeps >= d.steps          # radiation subcycles inside the hydro step
    @test d.newton_cells == d.rad_substeps * 2 * prod(sim.grid.n)   # two IMEX stages
    @test d.newton_failures == 0
    @test d.newton_max >= 1
    # the reported mean can never exceed the reported worst case -- these used
    # to be counted per outer iteration and per cell respectively, so on this
    # problem the mean (4.95) came out above the max (3)
    @test d.newton_iterations / d.newton_cells <= d.newton_max

    @test nrepaired(sim) == d.floors_hydro + d.floors_rad
    reset_diagnostics!(sim)
    @test d.steps == 0 && d.newton_cells == 0 && nrepaired(sim) == 0
    # ... and the counters resume from zero rather than being frozen
    step!(sim)
    @test d.steps == 1 && d.newton_cells > 0
end

@testset "Marshak wave" begin
    sim, tmax = P.marshak_wave(nx=128)
    run!(sim; tmax, verbose=false)
    Tg = gas_temperature(sim)[:, 1, 1]
    Tr = radiation_temperature(sim)[:, 1, 1]
    g = sim.grid
    # the wall heats the near side towards T_wall = 1 and the far side stays cold
    @test 0.9 < Tr[1] <= 1.0
    @test Tr[end] < 1e-2
    # radiation runs ahead of the matter: T_rad > T_gas through the front
    front = findlast(>(0.5), Tr)
    @test front !== nothing && 1 < front < g.n[1]
    @test maximum(Tr .- Tg) > 0.05
    @test all(Tr .>= -1e-30)
    # monotone decrease away from the wall
    @test issorted(Tr; rev=true)
end

@testset "closures" begin
    # The Eddington closure is isotropic at every reduced flux; Levermore is
    # not, and that difference is the whole point of using M1.
    for f in (0.0, 0.3, 0.7, 0.999)
        @test Hydro3D.eddington_factor(EddingtonClosure(), f) ≈ 1 / 3
    end
    @test Hydro3D.eddington_factor(LevermoreClosure(), 0.0) ≈ 1 / 3
    @test Hydro3D.eddington_factor(LevermoreClosure(), 1.0) ≈ 1.0
    # the one-argument form must keep meaning Levermore
    @test Hydro3D.eddington_factor(0.6) == Hydro3D.eddington_factor(LevermoreClosure(), 0.6)

    T = Hydro3D.eddington_tensor(EddingtonClosure(), 1.0, 0.0, 0.0)
    @test T[1, 1] ≈ 1 / 3
    @test T[2, 2] ≈ 1 / 3
    @test T[1, 2] ≈ 0.0 atol = 1e-14

    # P1 cannot free-stream: a beam under the Eddington closure spreads, while
    # under M1 it stays collimated with |F| = cE.
    function beam_front(cl)
        c = 1.0
        rad = RadiationPhysics(c=c, chat=c, a_rad=1.0, beta_order=0, closure=cl,
                               opacity=ConstantOpacity(0.0))
        b = Fixed((1.0, 0.999 * c, 0.0, 0.0))
        sim = Simulation(n=(256, 1, 1), radiation=rad, hydro=false,
                         boundaries=Boundaries(Outflow()),
                         radiation_boundaries=Boundaries((b, Outflow()),
                                                         (Outflow(), Outflow()),
                                                         (Outflow(), Outflow())))
        set_initial_conditions!(sim) do x, y, z
            (rho=1.0, p=1.0, Erad=1e-10)
        end
        run!(sim; tmax=0.5, verbose=false)
        g = sim.grid
        er = [sim.R[IERAD, i+g.ng[1], 1, 1] for i in 1:g.n[1]]
        xs = [(i - 0.5) * g.dx[1] for i in 1:g.n[1]]
        xs[findlast(>(0.5), er)]
    end
    @test beam_front(LevermoreClosure()) ≈ 0.5 rtol = 0.02   # front travels at chat
    @test beam_front(EddingtonClosure()) < 0.45              # P1 runs slow, c/sqrt(3)
end

@testset "radiative shock (Lowrie & Edwards Mach 3)" begin
    # Quokka's RadhydroShock: the most demanding 1D radiation-hydrodynamics
    # problem here. A Mach 3 shock in an optically thick, strongly coupled
    # medium, with a radiative precursor ahead of the front and a Zel'dovich
    # spike behind it where the matter temperature overshoots its downstream
    # value. The spike only appears if the velocity-dependent coupling, the
    # implicit solve and the shock capturing are all correct together.
    xe, rhoe, ve, Tme, Tre = P.radhydro_shock_exact()
    @test issorted(xe)
    @test maximum(Tme) ≈ 4.2618 rtol = 1e-4      # the spike, in the reference
    @test Tre[end] ≈ 3.6617 rtol = 1e-3          # downstream radiation temperature

    errs = Float64[]
    spikes = Float64[]
    for nx in (128, 256)
        sim, tmax = P.radhydro_shock(; nx)
        run!(sim; tmax, verbose=false)
        @test sim.t ≈ tmax
        g = sim.grid
        x = coordinates(g, 1)
        Tg = gas_temperature(sim)[:, 1, 1]
        Tr = radiation_temperature(sim)[:, 1, 1]
        prim = primitives(sim)

        # relative L1 error in the radiation temperature, Quokka's metric
        push!(errs, sum(abs, P.interpolate_to(xe, x, Tr) .- Tre) / sum(abs, Tre))
        push!(spikes, maximum(Tg))

        # the scheme must hold this together without help from the floors
        @test nrepaired(sim) == 0
        @test all(prim[IRHO, :, 1, 1] .> 0)
        @test all(prim[IPRE, :, 1, 1] .> 0)
        @test all(Tr .> 0)

        # upstream and downstream Rankine-Hugoniot states are held by the
        # boundaries and must survive the interior solve
        @test prim[IRHO, 2, 1, 1] ≈ 1.0 rtol = 1e-3
        @test prim[IVX, 2, 1, 1] ≈ 3.0 rtol = 1e-3
        @test Tg[2] ≈ 1.0 rtol = 1e-2
        @test prim[IRHO, end-1, 1, 1] ≈ 3.00217 rtol = 5e-3
        @test Tg[end-1] ≈ 3.6619 rtol = 1e-2
        # mass flux rho*v is conserved across the shock
        @test prim[IRHO, 2, 1, 1] * prim[IVX, 2, 1, 1] ≈
              prim[IRHO, end-1, 1, 1] * prim[IVX, end-1, 1, 1] rtol = 5e-3

        # radiative precursor: the gas ahead of the shock is preheated well
        # above its upstream temperature
        ishock = argmax(Tg)
        @test Tg[max(ishock - 12, 1)] > 1.5
        # Zel'dovich spike: matter hotter than the downstream state, and hotter
        # than the radiation at the same place
        @test Tg[ishock] > 1.03 * Tg[end-1]
        @test Tg[ishock] > Tr[ishock]
        # ... and it is a local feature, not a monotone ramp
        @test Tg[ishock] > Tg[min(ishock + 8, length(Tg))]
    end

    # Quokka asserts a relative L1 error below 1% at 256 cells
    @test errs[2] < 0.01
    @test errs[2] < errs[1]                       # and it converges
    @test spikes[1] < spikes[2] < 4.2618          # spike sharpens towards the exact value
    @test spikes[2] > 4.15
end

@testset "radiation directional isotropy" begin
    # a streaming beam along x, y and z must give identical profiles
    function beam_along(D)
        c = 1.0
        rad = RadiationPhysics(c=c, chat=c, a_rad=1.0,
                               opacity=ConstantOpacity(0.0), beta_order=0)
        b = Fixed((1.0, ntuple(d -> d == D ? 0.999 * c : 0.0, 3)...))
        bcs = ntuple(3) do d
            d == D ? (b, Outflow()) : (Outflow(), Outflow())
        end
        sim = Simulation(n=ntuple(d -> d == D ? 96 : 1, 3), radiation=rad, hydro=false,
                         boundaries=Boundaries(Outflow()),
                         radiation_boundaries=Boundaries(bcs...))
        set_initial_conditions!(sim) do x, y, z
            (rho=1.0, p=1.0, Erad=1e-10)
        end
        run!(sim; tmax=0.4, verbose=false)
        g = sim.grid
        [sim.R[IERAD, ntuple(d -> d == D ? i + g.ng[d] : 1, 3)...] for i in 1:g.n[D]]
    end
    px, py, pz = beam_along(1), beam_along(2), beam_along(3)
    @test px ≈ py
    @test px ≈ pz
end

@testset "shadow" begin
    sim, tmax = P.radiation_shadow(n=(140, 56))
    run!(sim; tmax, verbose=false)
    g = sim.grid
    er = sim.R[IERAD, g.lo[1]:g.hi[1], g.lo[2]:g.hi[2], g.lo[3]]
    mid = g.n[2] ÷ 2
    edge = 3
    ahead = g.n[1] ÷ 8                  # upstream of the clump
    behind = round(Int, 0.5 * g.n[1])   # just downstream of it
    far = g.n[1] - 2

    # the beam crosses the transparent gas essentially unattenuated
    @test er[ahead, mid] > 0.95
    @test er[far, edge] > 0.95
    # and is almost entirely removed directly behind the clump
    @test er[behind, mid] < 1e-4 * er[behind, edge]
    # the shadow survives all the way to the far boundary
    @test er[far, mid] < 0.1 * er[far, edge]
    # no radiation appears where none was sent, and the state stays causal
    @test all(er .>= 0)
    @test all(Iterators.product(1:g.n[1], 1:g.n[2])) do (i, j)
        E = sim.R[IERAD, i+g.ng[1], j+g.ng[2], 1]
        F = sqrt(sim.R[IFRX, i+g.ng[1], j+g.ng[2], 1]^2 +
                 sim.R[IFRY, i+g.ng[1], j+g.ng[2], 1]^2)
        F <= E * (1 + 1e-9)
    end
end
