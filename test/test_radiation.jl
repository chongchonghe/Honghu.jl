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
