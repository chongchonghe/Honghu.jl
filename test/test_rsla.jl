# The variable reduced speed of light.
#
# Most of these are invariants rather than error thresholds, for the reason
# CLAUDE.md gives: a limiter is a numerical knob, and a test that pins a number
# it happens to produce has to be retuned every time the knob improves. The three
# that are not invariants are the three that measure the point of the thing --
# that a local `chat` beats a global one on a shock and on a Marshak wave, and
# that a per-group `chat` does not conserve energy.

using Hydro3D: chat_factor, per_group, update_chat!, radiation_pressure_dt,
               _signal_speed, ConstantChat, StabilityBound, OpticalDepthFit,
               CouplingState, solve_coupling, gbase, IERAD, IEGAS, gas_scale

const ISO = 1 / sqrt(3)          # the signal speed of an isotropic radiation field

@testset "the limiter rule" begin
    phi0, nu0 = 0.03, 0.8

    @testset "ConstantChat is the standard RSLA" begin
        for tau in (0.0, 1.0, 1e6), prat in (0.0, 1.0)
            @test chat_factor(ConstantChat(), phi0, nu0, tau, prat, ISO) == phi0
        end
    end

    @testset "StabilityBound stays inside [phi0, 1]" begin
        lim = StabilityBound()
        for tau in (0.0, 0.1, 1.0, 1.15, 2.0, 1e8), prat in (0.0, 1e-6, 1.0, 1e6),
            sig in (ISO, 0.5, 1.0)
            phi = chat_factor(lim, phi0, nu0, tau, prat, sig)
            @test phi0 <= phi <= 1
        end
    end

    @testset "... rises with optical depth and reaches one" begin
        lim = StabilityBound()
        taus = (0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 2.0)
        phis = [chat_factor(lim, phi0, nu0, t, 0.0, ISO) for t in taus]
        @test issorted(phis)
        # Unbounded for tau >= 2 sig: beyond that the source damping wins at any
        # step, which is the whole idea of the improved RSLA.
        @test chat_factor(lim, phi0, nu0, 2ISO + 1e-9, 0.0, ISO) == 1
        @test chat_factor(lim, phi0, nu0, 1e8, 0.0, ISO) == 1
    end

    @testset "... reduces to the CFL bound as tau -> 0" begin
        lim = StabilityBound(beta=1.0)
        # nu_phi = (phi/phi0) nu0 must equal the largest CFL number the closure's
        # signal speed allows, 1/sig.
        for sig in (ISO, 0.5, 1.0), nu in (0.1, 0.4, 0.8)
            phi = chat_factor(lim, phi0, nu, 0.0, 0.0, sig)
            phi < 1 || continue                       # clipped, nothing to check
            @test (phi / phi0) * nu ≈ 1 / sig rtol = 1e-12
        end
        # ... and with the isotropic signal speed that is the sqrt(3) of the
        # published bound, so the generalisation contains the original.
        @test chat_factor(lim, phi0, 0.8, 0.0, 0.0, ISO) ≈ phi0 * sqrt(3) / 0.8
    end

    @testset "... a beamed cell is bounded sqrt(3) tighter than an isotropic one" begin
        lim = StabilityBound(beta=1.0)
        iso = chat_factor(lim, phi0, 0.8, 0.0, 0.0, ISO)
        beam = chat_factor(lim, phi0, 0.8, 0.0, 0.0, 1.0)
        @test iso ≈ sqrt(3) * beam rtol = 1e-12
    end

    @testset "... the radiation-force channel caps it" begin
        lim = StabilityBound(beta=1.0)
        # Deep in the optically thick limit the tau channel is unbounded, so
        # whatever survives is phi_P.
        phi = chat_factor(lim, phi0, nu0, 1e8, 0.1, ISO)
        @test phi ≈ phi0 / (2 * nu0 * 0.1) rtol = 1e-12
        # Below phi0 the floor takes over: the limiter never asks for less than
        # the reduction the user requested.
        @test chat_factor(lim, phi0, nu0, 1e8, 100.0, ISO) == phi0
        # ... and a cell with no radiation in it must not divide by zero.
        @test isfinite(chat_factor(lim, phi0, nu0, 1e8, 0.0, ISO))
    end

    @testset "OpticalDepthFit interpolates between phi0 and one" begin
        lim = OpticalDepthFit(s=1.0, p=2.0)
        @test chat_factor(lim, phi0, nu0, 0.0, 0.0, ISO) ≈ phi0
        @test chat_factor(lim, phi0, nu0, 1e8, 0.0, ISO) ≈ 1
        @test chat_factor(lim, phi0, nu0, 1.0, 0.0, ISO) ≈ (phi0 + 1) / 2
        # A tau large enough to overflow `(s tau)^p` must still give the limit
        # rather than a NaN out of Inf/Inf.
        @test chat_factor(lim, phi0, nu0, 1e200, 0.0, ISO) == 1
    end

    @testset "per_group defaults to false" begin
        @test !per_group(ConstantChat())
        @test !per_group(StabilityBound())
        @test !per_group(OpticalDepthFit(s=1.0, p=2.0))
        @test per_group(StabilityBound(per_group=true))
        @test per_group(OpticalDepthFit(s=1.0, p=2.0, per_group=true))
    end
end

@testset "the signal speed comes from the closure" begin
    c = 2.0
    R = zeros(4, 3, 1, 1)
    R[IERAD, 1, 1, 1] = 1.0                       # isotropic: f = 0
    R[IERAD, 2, 1, 1] = 1.0
    R[2, 2, 1, 1] = 0.9999 * c                    # nearly free-streaming
    R[IERAD, 3, 1, 1] = 1.0
    R[2, 3, 1, 1] = 0.5 * c
    @test _signal_speed(LevermoreClosure(), R, 1, 1, 1, 1, c) ≈ ISO rtol = 1e-9
    @test _signal_speed(LevermoreClosure(), R, 2, 1, 1, 1, c) > 0.99
    @test ISO < _signal_speed(LevermoreClosure(), R, 3, 1, 1, 1, c) < 1
    # The Eddington closure cannot free-stream, and the bound must not pretend
    # otherwise: it is 1/sqrt(3) whatever the flux.
    for i in 1:3
        @test _signal_speed(EddingtonClosure(), R, i, 1, 1, 1, c) ≈ ISO
    end
    # An empty or inadmissible cell -- which ghost cells filled by a user
    # boundary condition can be -- must not produce a NaN.
    Rbad = zeros(4, 1, 1, 1)
    @test isfinite(_signal_speed(LevermoreClosure(), Rbad, 1, 1, 1, 1, c))
    Rbad[IERAD, 1, 1, 1] = 1.0
    Rbad[2, 1, 1, 1] = 5.0 * c                    # |f| > 1
    @test _signal_speed(LevermoreClosure(), Rbad, 1, 1, 1, 1, c) == 1
end

@testset "the limiter is a no-op where it should be" begin
    # `beta -> 0` drives the bound below phi0, and the floor takes over, so the
    # limiter must reproduce the standard RSLA it is a generalisation of --
    # exactly, not approximately. This is the whole plumbing under test at once.
    for lim in (StabilityBound(beta=1e-12), StabilityBound(beta=1e-12, per_group=true))
        a, tmax = P.marshak_vaytet(n=24, ngroups=3, chat_over_c=0.05)
        b, _ = P.marshak_vaytet(n=24, ngroups=3, chat_over_c=0.05, chat_limiter=lim)
        run!(a; tmax=tmax / 50, verbose=false)
        run!(b; tmax=tmax / 50, verbose=false)
        @test a.R == b.R
        @test a.diag.steps == b.diag.steps
    end

    # At chat = c there is nothing left to reduce: phi is pinned at one by its
    # own floor, so an IRSLA run is the exact-limit no-op the paper's battery
    # checks for.
    a, tmax = P.marshak_vaytet(n=24, ngroups=3)
    b, _ = P.marshak_vaytet(n=24, ngroups=3, chat_limiter=StabilityBound())
    run!(a; tmax=tmax / 50, verbose=false)
    run!(b; tmax=tmax / 50, verbose=false)
    @test a.R == b.R
end

@testset "chat is filled in ghost cells too" begin
    # The transport step needs a value on both sides of the faces bounding the
    # domain, and the limiter is local enough to supply one without
    # communicating. If the ghosts were left at their initial fill this would
    # still "work" and would quietly use the wrong speed at the boundary.
    sim, _ = P.marshak_vaytet(n=16, ngroups=2, chat_over_c=0.02,
                              chat_limiter=StabilityBound())
    step!(sim)
    ch = sim.rw.chat
    @test all(>(0), ch)
    # The wall ghost sees the 1000 K boundary spectrum and the interior sees a
    # 300 K one, so they cannot agree -- which is what proves the ghosts were
    # actually visited.
    @test ch[:, 1, 1, 1] != ch[:, sim.grid.lo[1], 1, 1]
end

@testset "the interface takes the minimum of its two cells" begin
    # Paper section 5 argues at length for the minimum -- an optically thick cell
    # must not impose a CFL-violating speed on a transparent neighbour -- and
    # then its own algorithm summary says maximum. The minimum is what is
    # implemented, and this is the test that says so.
    c = 3.0
    phys = RadiationPhysics(c=c, chat=c, a_rad=1.0, opacity=ConstantOpacity(0.0))
    g = CartesianGrid(n=(4, 1, 1), xlo=(0.0, 0.0, 0.0), xhi=(1.0, 1.0, 1.0), nghost=2)
    R = allocate(g, 4)
    qL = allocate(g, 4)
    qR = allocate(g, 4)
    flux = allocate(g, 4)
    chat = allocate(g, 1)
    for i in axes(R, 2)
        R[IERAD, i, 1, 1] = i <= 4 ? 1.0 : 0.25     # a jump the flux can see
    end
    # Piecewise constant, in this package's convention: face `f` is the low-side
    # face of cell `f`, so `qL[f]` comes from cell `f-1` and `qR[f]` from cell `f`.
    for i in 2:size(R, 2)
        qL[IERAD, i, 1, 1] = R[IERAD, i-1, 1, 1]
        qR[IERAD, i, 1, 1] = R[IERAD, i, 1, 1]
    end
    slow, fast = 0.4c, c
    fluxes = map(((l, r),) -> begin
                     chat[1, :, 1, 1] .= fast
                     chat[1, 4, 1, 1] = l          # cell 4 -- left of face 5
                     chat[1, 5, 1, 1] = r          # cell 5 -- right of face 5
                     Hydro3D.rad_fluxes!(flux, qL, qR, R, chat, phys, Val(1), g)
                     flux[IERAD, 5, 1, 1]
                 end, ((slow, fast), (fast, slow), (slow, slow), (fast, fast)))
    lo_hi, hi_lo, lo_lo, hi_hi = fluxes
    @test lo_hi == hi_lo == lo_lo      # min picked the slow side in all three
    @test hi_hi != lo_lo               # ... and the speed genuinely matters
end

@testset "a per-group chat breaks the RSLA's energy identity" begin
    # No transport, one cell, two groups, relaxing from Tgas = 1.3 against a
    # Trad = 0.5 spectrum. With a *uniform* phi the coupling obeys
    #
    #     Delta [e + sum_g E_g] = (1 - phi) Delta e
    #
    # exactly: the RSLA's non-conservation is a known function of the gas energy
    # change and of nothing else. With a per-group phi that identity fails, and
    # what replaces it depends on the path energy took through the spectrum. See
    # `per_group` for why this is the reason the default is one phi per cell.
    c, arad, kB, rho, cv, NG = 1.0, 1.0, 1.0, 1.0, 1.0, 2
    grp = RadiationGroups(0.1, 20.0, NG)
    op = ConstantOpacity(1e3)
    z = zero(SVec{NG,Float64})

    "Returns (physical drift, gas energy change, weighted drift), all over E0."
    function relax(chats)
        Egas = cv * 1.3
        Erad = (arad * 0.5^4) * planck_fractions(grp, 0.5, kB)
        state(eg, er) = CouplingState{NG,NG + 1,Float64,typeof(op)}(eg, er, z, z, rho, cv,
                                                                    op, 1e-3, c, chats,
                                                                    arad, kB, grp, 1e-10,
                                                                    1e-30)
        E0 = Egas + sum(Tuple(Erad))
        g0 = Egas
        w0 = Egas + dot(gas_scale(state(Egas, Erad)), Erad)
        s = state(Egas, Erad)
        for _ in 1:400
            s = state(Egas, Erad)
            r = solve_coupling(QuokkaCoupling(), s)
            Egas, Erad = r.Egas, r.Erad
        end
        w1 = Egas + dot(gas_scale(s), Erad)
        ((Egas + sum(Tuple(Erad)) - E0) / E0, (Egas - g0) / E0, w1 / w0 - 1)
    end

    # Whatever phi is, the quantity the coupling actually conserves must be
    # conserved -- that is the solver working, and it has to hold in both cases
    # for the comparison below to mean anything.
    for chats in (SVec(1.0, 1.0), SVec(0.1, 0.1), SVec(1.0, 0.1), SVec(1.0, 0.01))
        @test abs(relax(chats)[3]) < 1e-6
    end

    for phi in (1.0, 0.1, 0.01)
        dphys, dgas, _ = relax(SVec(phi, phi))
        # `atol` because at phi = 1 the identity predicts exactly zero, and the
        # solver's own convergence tolerance is what is left.
        @test isapprox(dphys, (1 - phi) * dgas; rtol=1e-6, atol=1e-8)
    end

    # phi_1 = 1 makes the identity predict *no* drift; the measured drift is
    # around half the cell's energy, and grows as the two groups' phi separate.
    for (chats, atleast) in ((SVec(1.0, 0.1), 0.4), (SVec(1.0, 0.01), 0.5))
        dphys, dgas, _ = relax(chats)
        @test abs(dphys - (1 - chats[1]) * dgas) > atleast
    end
end

@testset "a local chat beats a global one" begin
    # The measurements the limiter exists for. Both are scored against a
    # chat = c reference at the same resolution, and both run the same number of
    # steps as the RSLA they are compared with, so the comparison is at equal
    # cost.
    l1(T, ref) = sum(abs, T .- ref) / sum(abs, ref)

    @testset "Lowrie-Edwards radiative shock" begin
        n = 128
        r, tmax = P.radhydro_shock(; n, chat_factor=433.0)      # chat = c
        run!(r; tmax, verbose=false)
        ref = vec(gas_temperature(r))
        a, _ = P.radhydro_shock(; n, chat_factor=3.0)
        b, _ = P.radhydro_shock(; n, chat_factor=3.0, chat_limiter=StabilityBound())
        run!(a; tmax, verbose=false)
        run!(b; tmax, verbose=false)
        ea = l1(vec(gas_temperature(a)), ref)
        eb = l1(vec(gas_temperature(b)), ref)
        @test eb < ea / 2                                   # measured ~4.6x
        @test b.diag.steps <= a.diag.steps + 2              # ... at the same cost
        @test b.diag.newton_failures == 0
    end

    @testset "Vaytet multigroup Marshak wave" begin
        n, ng = 32, 4
        r, tmax = P.marshak_vaytet(; n, ngroups=ng)
        run!(r; tmax, verbose=false)
        ref = vec(gas_temperature(r))
        a, _ = P.marshak_vaytet(; n, ngroups=ng, chat_over_c=0.01)
        b, _ = P.marshak_vaytet(; n, ngroups=ng, chat_over_c=0.01,
                                chat_limiter=StabilityBound())
        run!(a; tmax, verbose=false)
        run!(b; tmax, verbose=false)
        Tb = vec(gas_temperature(b))
        @test l1(Tb, ref) < l1(vec(gas_temperature(a)), ref)
        @test b.diag.steps == a.diag.steps
        # The slab is heated by a 1000 K wall and cannot exceed it. Before the
        # bound carried the closure's signal speed this reached 2158 K.
        @test maximum(Tb) < 1000
    end
end

@testset "the radiation-pressure step constraint" begin
    # Only bites where radiation pressure is dynamically significant, and is not
    # evaluated at all where the gas cannot move.
    sim, _ = P.radhydro_shock(n=32, chat_limiter=StabilityBound())
    phys = sim.radiation
    dt_rsla = Hydro3D.radiation_timestep(phys, sim.grid, sim.cfl_radiation)
    dt_P = radiation_pressure_dt(sim.U, sim.R, sim.eos, phys, sim.grid,
                                 sim.cfl_radiation, sim.floors)
    @test dt_P <= dt_rsla * (1 + 1e-12)   # never longer than the RSLA step
    @test dt_P > 0

    # A static problem must not pay for it: `advance_radiation!` subcycles on
    # `_rad_dt`, and with hydro disabled that has to be the plain RSLA step.
    stat, _ = P.marshak_vaytet(n=16, ngroups=2, chat_over_c=0.02,
                               chat_limiter=StabilityBound())
    @test Hydro3D._rad_dt(stat) ==
          Hydro3D.radiation_timestep(stat.radiation, stat.grid, stat.cfl_radiation)
end
