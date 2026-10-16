# The variable reduced speed of light.
#
# Most of these are invariants rather than error thresholds, for the reason
# CLAUDE.md gives: a limiter is a numerical knob, and a test that pins a number
# it happens to produce has to be retuned every time the knob improves. The three
# that are not invariants are the three that measure the point of the thing --
# that a local `chat` beats a global one on a shock and on a Marshak wave, and
# that a per-group `chat` does not conserve energy.

using Hydro3D: chat_factor, per_group, flux_form, update_chat!, energy_exchange_dt, LocalCell,
               stability_denominator, _signal_sum, ConstantChat, StabilityBound,
               OpticalDepthFit, CouplingState, solve_coupling, gbase, IERAD, IEGAS,
               gas_scale

const ISO = 1 / sqrt(3)          # the signal speed of an isotropic radiation field

# A 1D cell of unit width: `sdx = sig`, `chiE = chiF = tau`.
cell1d(cdt, tau; sig=ISO, hrat=0.0) = LocalCell(cdt, sig, tau, tau, hrat, 1.0)

@testset "the limiter rule" begin
    phi0 = 0.03
    cdt = 0.8 / phi0                                # a step at CFL number 0.8

    @testset "ConstantChat is the standard RSLA" begin
        for tau in (0.0, 1.0, 1e6), hrat in (0.0, 1.0)
            @test chat_factor(ConstantChat(), phi0, cell1d(cdt, tau; hrat)) == phi0
        end
    end

    @testset "StabilityBound stays inside [phi0, 1]" begin
        lim = StabilityBound()
        for tau in (0.0, 0.1, 1.0, 1.15, 2.0, 1e8), hrat in (0.0, 1e-6, 1.0, 1e6),
            sig in (ISO, 0.5, 1.0)
            phi = chat_factor(lim, phi0, cell1d(cdt, tau; sig, hrat))
            @test phi0 <= phi <= 1
        end
    end

    @testset "... is the 1D closed form nu (2 sig - tau) <= 2 when K -> 0" begin
        lim = StabilityBound(beta=1.0)
        for tau in (0.0, 0.2, 0.5, 0.8, 1.0), sig in (ISO, 1.0)
            2sig > tau || continue
            phi = chat_factor(lim, phi0, cell1d(cdt, tau; sig))
            phi < 1 || continue
            @test phi * cdt * (2sig - tau) ≈ 2 rtol = 1e-12
        end
        # No bound at all beyond tau = 2 sig: the source damping wins at any step.
        @test chat_factor(lim, phi0, cell1d(cdt, 2ISO + 1e-9)) == 1
        @test chat_factor(lim, phi0, cell1d(cdt, 1e8)) == 1
        # ... and there `beta` has nothing to multiply: phi is exactly one, not
        # beta, so thick interiors run the unmodified scheme.
        @test chat_factor(StabilityBound(beta=0.5), phi0, cell1d(cdt, 1e8)) == 1
    end

    @testset "... reduces to the CFL bound of the closure's signal speed" begin
        lim = StabilityBound(beta=1.0)
        for sig in (ISO, 0.5, 1.0)
            phi = chat_factor(lim, phi0, cell1d(cdt, 0.0; sig))
            @test phi * cdt * sig ≈ 1 rtol = 1e-12
        end
        iso = chat_factor(lim, phi0, cell1d(cdt, 0.0))
        beam = chat_factor(lim, phi0, cell1d(cdt, 0.0; sig=1.0))
        @test iso ≈ sqrt(3) * beam rtol = 1e-12
    end

    @testset "... the heat-capacity channel caps it in radiation-rich cells" begin
        lim = StabilityBound(beta=1.0)
        # Thick limit: the damping chiE/(2 + K) saturates at 1/(hrat cdt), so the
        # bound is phi cdt (sig - 1/(hrat cdt)) <= 1.
        hrat = 10 / cdt                               # sig hrat cdt = 5.8 > 1
        phi = chat_factor(lim, phi0, cell1d(cdt, 1e12; hrat))
        @test phi * cdt * (ISO - 1 / (hrat * cdt)) ≈ 1 rtol = 1e-9
        # Where sig hrat cdt < 1 it does not: a thick enough cell reaches one.
        @test chat_factor(lim, phi0, cell1d(cdt, 1e12; hrat=0.5 / cdt)) == 1
        # The standard RSLA is never excluded: at phi = phi0 the bound is at
        # most the CFL number times sig, whatever hrat is.
        for hrat in (0.0, 1.0, 1e6), tau in (0.0, 1.0, 1e6)
            w = cell1d(cdt, tau; hrat)
            @test phi0 * cdt * stability_denominator(w) <= 0.8 * ISO + 1e-12
        end
    end

    @testset "OpticalDepthFit interpolates between phi0 and one" begin
        lim = OpticalDepthFit(s=1.0, p=2.0)
        @test chat_factor(lim, phi0, cell1d(cdt, 0.0)) ≈ phi0
        @test chat_factor(lim, phi0, cell1d(cdt, 1e8)) ≈ 1
        @test chat_factor(lim, phi0, cell1d(cdt, 1.0)) ≈ (phi0 + 1) / 2
        # A tau large enough to overflow `(s tau)^p` must still give the limit
        # rather than a NaN out of Inf/Inf.
        @test chat_factor(lim, phi0, cell1d(cdt, 1e200)) == 1
    end

    @testset "per_group: always for the stability bound, off for the fitting form" begin
        @test !per_group(ConstantChat())
        @test per_group(StabilityBound())
        @test !per_group(OpticalDepthFit(s=1.0, p=2.0))
        @test per_group(OpticalDepthFit(s=1.0, p=2.0, per_group=true))
    end
end

@testset "the signal speed comes from the closure, per direction" begin
    c = 2.0
    inv3 = (1.0, 1.0, 1.0)                        # three active unit-width directions
    R = zeros(4, 3, 1, 1)
    R[IERAD, 1, 1, 1] = 1.0                       # isotropic: f = 0
    R[IERAD, 2, 1, 1] = 1.0
    R[2, 2, 1, 1] = 0.9999 * c                    # nearly free-streaming along x
    R[IERAD, 3, 1, 1] = 1.0
    R[2, 3, 1, 1] = 0.5 * c
    @test _signal_sum(LevermoreClosure(), R, 1, 1, 1, 1, c, inv3) ≈ 3ISO rtol = 1e-9
    # A beam along x: speed one along it, the 0.1 floor across it -- not one in
    # every direction, which is what bounding by the largest speed would say.
    beam = _signal_sum(LevermoreClosure(), R, 2, 1, 1, 1, c, inv3)
    @test 0.99 + 2 * 0.1 - 0.02 < beam < 1 + 2 * 0.1 + 0.02
    @test _signal_sum(LevermoreClosure(), R, 2, 1, 1, 1, c, (1.0, 0.0, 0.0)) > 0.99
    # The Eddington closure cannot free-stream, and the bound must not pretend
    # otherwise: it is 1/sqrt(3) in every direction whatever the flux.
    for i in 1:3
        @test _signal_sum(EddingtonClosure(), R, i, 1, 1, 1, c, inv3) ≈ 3ISO
    end
    # An empty or inadmissible cell -- which ghost cells filled by a user
    # boundary condition can be -- must not produce a NaN, and an inadmissible
    # one gets the fastest speed the closure has.
    Rbad = zeros(4, 1, 1, 1)
    @test isfinite(_signal_sum(LevermoreClosure(), Rbad, 1, 1, 1, 1, c, inv3))
    Rbad[IERAD, 1, 1, 1] = 1.0
    Rbad[2, 1, 1, 1] = 5.0 * c                    # |f| > 1
    @test _signal_sum(LevermoreClosure(), Rbad, 1, 1, 1, 1, c, inv3) == 3
end

# A uniform static medium carrying a grid-scale checkerboard in E_rad, at a
# fixed local phi (or under a real limiter). Returns the growth factor per
# substep of the checkerboard, which the von Neumann analysis predicts exactly.
struct FixedPhi <: Hydro3D.AbstractChatLimiter
    phi::Float64
end
Hydro3D.chat_factor(l::FixedPhi, phi0, cell::LocalCell) = l.phi
Hydro3D._convert_limiter(l::FixedPhi, ::Type{T}) where {T} = l

function checkerboard_growth(lim; D, n=8, phi0=1e-3, tau, Pcal, nst=24, nmeas=12)
    cv, gamma = 1.0, 5 / 3
    a_rad = Pcal * (gamma - 1) * cv               # P = a T^4 / (rho a_iso^2), T = rho = 1
    dx = 1 / n
    rad = RadiationPhysics(; c=1.0, chat=phi0, a_rad, beta_order=0,
                           opacity=ConstantOpacity(tau / dx), chat_limiter=lim)
    sim = Simulation(; n=ntuple(d -> d <= D ? n : 1, 3), eos=IdealGas(; gamma, cv),
                     radiation=rad, hydro=false, boundaries=Boundaries(Periodic()),
                     cfl_radiation=0.4, radiation_reconstruction=PCM())
    cell(x) = floor(Int, x / dx)
    set_initial_conditions!(sim) do x, y, z
        s = iseven(cell(x) + (D > 1) * cell(y) + (D > 2) * cell(z)) ? 1.0 : -1.0
        (rho=1.0, T=1.0, Erad=a_rad * (1 + 1e-3 * s))
    end
    g = sim.grid
    I = (g.lo[1]:g.hi[1], g.lo[2]:g.hi[2], g.lo[3]:g.hi[3])
    sgn = [(-1.0)^((i - g.lo[1]) + (D > 1) * (j - g.lo[2]) + (D > 2) * (k - g.lo[3]))
           for i in I[1], j in I[2], k in I[3]]
    proj() = abs(sum(sim.R[IERAD, I...] .* sgn))
    dt = Hydro3D.radiation_timestep(sim.radiation, g, sim.cfl_radiation)
    a1 = 0.0
    for s in 1:nst
        step!(sim, dt)
        s == nst - nmeas && (a1 = proj())
    end
    return (proj() / a1)^(1 / nmeas)
end

@testset "the bound is exact: a checkerboard in 1, 2 and 3 dimensions" begin
    # The derivation says the checkerboard's growth factor crosses one exactly at
    # the bound. So a fixed phi a tenth above it must grow, a tenth below it must
    # decay, and the limiter at beta = 1 must sit on the boundary itself -- in
    # every dimension, which is what the old `rho kappa dx_min` got wrong by a
    # factor D.
    for D in (1, 2, 3), (tau, Pcal) in ((0.3, 1e-6), (1.0, 1e-1))
        n = 8
        dt = 0.4 / (1e-3 * D * n)                 # the RSLA step at phi0 = 1e-3
        K = 4 * Pcal * (2 / 3) * dt * tau * n
        den = dt * (D * ISO * n - tau * n / (2 + K))
        phib = 1 / den
        @test checkerboard_growth(FixedPhi(0.9phib); D, tau, Pcal) < 1
        @test checkerboard_growth(FixedPhi(1.1phib); D, tau, Pcal) > 1.05
        @test checkerboard_growth(StabilityBound(beta=1.0); D, tau, Pcal) ≈ 1 atol = 1e-3
        @test checkerboard_growth(StabilityBound(); D, tau, Pcal) < 0.98
    end
    # ... and the standard RSLA is stable at its own step however radiation-rich
    # the gas: no step restriction is needed for stability.
    for D in (1, 2), Pcal in (1.0, 100.0)
        @test checkerboard_growth(ConstantChat(); D, tau=10.0, Pcal) < 1
    end
end

@testset "the limiter is a no-op where it should be" begin
    # A fitting form with `s = 0` is `phi = phi0` in every cell, so it must
    # reproduce the standard RSLA -- exactly, not approximately -- through the
    # whole variable-chat path. This is the plumbing under test at once. (A
    # vanishing `beta` no longer does this: thick cells run at `phi = 1` whatever
    # `beta` is, because there is no bound there for it to multiply.)
    for lim in (OpticalDepthFit(s=0.0, p=1.0, flux_form=true),
                OpticalDepthFit(s=0.0, p=1.0, per_group=true, flux_form=true))
        a, tmax = P.marshak_vaytet(n=24, ngroups=3, chat_over_c=0.05)
        b, _ = P.marshak_vaytet(n=24, ngroups=3, chat_over_c=0.05, chat_limiter=lim)
        run!(a; tmax=tmax / 50, verbose=false)
        run!(b; tmax=tmax / 50, verbose=false)
        @test a.R == b.R
        @test a.diag.steps == b.diag.steps
    end
    # The default time-derivative form is the same scheme when phi is uniform --
    # F_HLL(chat) = (chat/c) F_HLL(c) term by term -- but reaches it through a
    # different product, so it agrees to round-off rather than bit for bit.
    a, tmax = P.marshak_vaytet(n=24, ngroups=3, chat_over_c=0.05)
    b, _ = P.marshak_vaytet(n=24, ngroups=3, chat_over_c=0.05,
                            chat_limiter=OpticalDepthFit(s=0.0, p=1.0))
    run!(a; tmax=tmax / 50, verbose=false)
    run!(b; tmax=tmax / 50, verbose=false)
    @test isapprox(a.R, b.R; rtol=1e-10)
    @test a.diag.steps == b.diag.steps

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
    # `chat` is computed from the state at the start of a substep, which in the
    # first step is the uniform initial state; the second step starts from one in
    # which the wall's radiation has entered the domain.
    step!(sim)
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
    # implemented wherever `rad_fluxes!` is handed a `chat` array -- the standard
    # RSLA and `flux_form = true` -- and this is the test that says so. The
    # default, time-derivative form needs no interface value at all.
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
    # `per_group` for why this identity is lost by any local phi anyway.
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

@testset "the energy-exchange step is opt-in and only shortens" begin
    sim, _ = P.radhydro_shock(n=32, chat_limiter=StabilityBound())
    dt_rsla = Hydro3D.radiation_timestep(sim.radiation, sim.grid, sim.cfl_radiation)
    # Off by default: the limiter runs at the plain RSLA step.
    @test Hydro3D._rad_dt(sim) == dt_rsla
    on, _ = P.radhydro_shock(n=32, chat_limiter=StabilityBound(energy_step=true))
    dt_E = energy_exchange_dt(on.U, on.R, on.eos, on.radiation, on.grid, 0.5, on.floors)
    @test dt_E > 0
    dt_c = dt_rsla * on.radiation.chat / on.radiation.c   # the unreduced CFL step
    @test Hydro3D._rad_dt(on) == min(dt_rsla, max(dt_E, dt_c))
    # ... and in a radiation-dominated gas it stops at the unreduced step rather
    # than asking for more than running at the true speed of light would cost.
    rich, _ = P.radhydro_pulse_dyn(n=32, chat_over_c=0.01,
                                   chat_limiter=StabilityBound(energy_step=true))
    dtr = Hydro3D.radiation_timestep(rich.radiation, rich.grid, rich.cfl_radiation)
    @test Hydro3D._rad_dt(rich) ≈ dtr * 0.01
end

# A limiter that reads the gas temperature through `hrat` (which rises through a
# shock) and ripples an otherwise uniform `phi` by `amp`.
struct Ripple <: Hydro3D.AbstractChatLimiter
    phi::Float64
    amp::Float64
    flux::Bool
end
Hydro3D._convert_limiter(l::Ripple, ::Type{T}) where {T} = l
Hydro3D.chat_factor(l::Ripple, phi0, w::LocalCell) = l.phi * (1 - l.amp * min(1.0, w.hrat / 0.08))
Hydro3D.flux_form(l::Ripple) = l.flux

@testset "phi belongs on the time derivative, not inside the divergence" begin
    # The RSLA changes how fast radiation relaxes, not where it relaxes to. With
    # phi on the time derivative that holds cell by cell for any phi(x); with phi
    # inside the divergence the steady state picks up a spurious F . grad(ln phi),
    # and a few per cent of ripple in phi costs more than the RSLA itself.
    l1(T, ref) = sum(abs, T .- ref) / sum(abs, ref)
    n = 64
    r, tmax = P.radhydro_shock_super(; n, chat_over_c=1.0)
    run!(r; tmax, verbose=false)
    ref = vec(gas_temperature(r))
    err(lim) = (s = first(P.radhydro_shock_super(; n, chat_limiter=lim));
                run!(s; tmax, verbose=false); l1(vec(gas_temperature(s)), ref))
    flat = err(Ripple(0.12, 0.0, false))
    @test err(Ripple(0.12, 0.0, true)) ≈ flat rtol = 1e-6   # uniform phi: the same scheme
    @test err(Ripple(0.12, 0.05, false)) < 1.5 * flat
    @test err(Ripple(0.12, 0.05, true)) > 5 * flat
    @test !flux_form(StabilityBound()) && flux_form(StabilityBound(flux_form=true))
end

# phi = 0.1 where the absorption coefficient is below 1.5e-12 and 0.02 above it;
# with a negligible opacity the density profile is a position switch.
struct PhiStep <: Hydro3D.AbstractChatLimiter
    flux::Bool
end
Hydro3D._convert_limiter(l::PhiStep, ::Type{T}) where {T} = l
Hydro3D.chat_factor(l::PhiStep, phi0, w::LocalCell) = w.chiE < 1.5e-12 ? 0.1 : 0.02
Hydro3D.flux_form(l::PhiStep) = l.flux

@testset "a steady beam crosses a jump in phi unchanged" begin
    # Free streaming through vacuum, phi dropping five-fold halfway across. The
    # steady state of the unmodified equations is a uniform beam, and the RSLA
    # must not move it. On the time derivative it does not; inside the divergence
    # div(phi F) = 0 forces F ~ 1/phi, and the beam comes out five times brighter.
    function beam(flux)
        rad = RadiationPhysics(; c=1.0, chat=0.1, a_rad=1.0, beta_order=0,
                               opacity=ConstantOpacity(1e-12), chat_limiter=PhiStep(flux))
        sim = Simulation(; n=(32, 1, 1), radiation=rad, hydro=false,
                         boundaries=Boundaries(Outflow()),
                         radiation_boundaries=Boundaries((Fixed((1.0, 0.999, 0.0, 0.0)),
                                                          Outflow()),
                                                         (Outflow(), Outflow()),
                                                         (Outflow(), Outflow())))
        set_initial_conditions!(sim) do x, y, z
            (rho=x < 0.5 ? 1.0 : 2.0, p=1.0, Erad=1e-10, Fx=0.0)
        end
        run!(sim; tmax=200.0, verbose=false)       # many crossing times at 0.02 c
        E = sim.R[IERAD, sim.grid.lo[1]:sim.grid.hi[1], 1, 1]
        return E[8], E[24]
    end
    lo, hi = beam(false)
    @test lo ≈ 1 rtol = 1e-3
    @test hi ≈ 1 rtol = 1e-3
    lo, hi = beam(true)
    @test lo ≈ 1 rtol = 1e-3
    @test hi ≈ 5 rtol = 1e-2
end

@testset "the wave-speed correction is refused with a local chat" begin
    # Its stability bound does not hold (paper Sec. 3.2): a face's correction is
    # set by both cells beside it, so a thick cell beside a thin one ran at
    # phi = 1 behind an almost undamped face and grew by ~1e43 in 200 steps; and
    # in a uniform medium the lowered dissipation lets modes near a quarter of the
    # grid frequency grow wherever the radiation's inertia matters.
    op = ConstantOpacity(1.0)
    for lim in (StabilityBound(), OpticalDepthFit(s=1.0, p=2.0))
        @test_throws ArgumentError RadiationPhysics(; c=1.0, chat=1e-3, a_rad=1.0,
                                                    opacity=op, chat_limiter=lim,
                                                    wavespeed_correction=true)
    end
    @test RadiationPhysics(; c=1.0, chat=1e-3, a_rad=1.0, opacity=op,
                           wavespeed_correction=true).wavespeed_correction
end

@testset "a negative energy after transport is left for the coupling solve" begin
    # Ensman's supercritical shock (Melon Fuksman et al. 2021) on 64 cells: the
    # dense cell at the wall (tau_cell ~ 5) runs at phi ~ 0.8 beside thin cells at
    # phi ~ 0.02, and at a local CFL number of ~70 the transport stage drives E
    # negative there. Flooring E before the solve added energy every substep and
    # made the IRSLA ~17 times worse than the RSLA against chat = c; the solve
    # conserves e + E/phi and absorbs the negative E.
    CGS = Hydro3D.CGS
    function ensman(; phi0, lim=ConstantChat(), cfl=0.9)
        rho0, T0, gamma = 7.78e-10, 10.0, 1.4
        rad = RadiationPhysics(; c=CGS.c_light, chat=phi0 * CGS.c_light, a_rad=CGS.a_rad,
                               k_B=CGS.k_B, opacity=ConstantOpacity(3.1e-10 / rho0),
                               chat_limiter=lim)
        sim = Simulation(; n=(64, 1, 1), xhi=(7e10, 1.0, 1.0),
                         eos=IdealGas(; gamma, cv=CGS.k_B / (CGS.m_u * (gamma - 1))),
                         radiation=rad, cfl=0.4, cfl_radiation=cfl,
                         boundaries=Boundaries((Reflecting(), Outflow()), (Periodic(), Periodic()),
                                               (Periodic(), Periodic())),
                         floors=Floors(erad=CGS.a_rad * 1e-4))
        set_initial_conditions!(sim) do x, y, z
            (rho=rho0, vx=-20e5, T=T0, Erad=CGS.a_rad * T0^4, Fx=0.0)
        end
        run!(sim; tmax=7.5e3, verbose=false)
        return vec(gas_temperature(sim)), sim
    end
    ref, _ = ensman(phi0=1.0)
    err(T) = sum(abs, T .- ref) / sum(abs, ref)
    rsla, _ = ensman(phi0=0.01)
    irsla, sim = ensman(phi0=0.01, lim=StabilityBound())
    @test sim.diag.floors_presolve == 0
    @test err(irsla) < err(rsla) / 2
end

@testset "the flux is not clipped before the coupling solve" begin
    # A thick (tau_cell = 20), gas-dominated hot spot at phi0 = 0.01: the limiter
    # sets phi = 1, so the radiation runs at a local CFL number of ~90, and the
    # transport stage leaves |F| >> cE. The implicit solve turns that flux into the
    # diffusion flux; clipping it to cE first slowed the diffusion and made the
    # IRSLA twice as wrong as the RSLA (4.8% against 2.2%) instead of 1000 times
    # better.
    function hotspot(; phi0, lim=ConstantChat(), cfl=0.9, tmax=300.0)
        n = 128
        rad = RadiationPhysics(; c=1.0, chat=phi0, a_rad=1.0, beta_order=0,
                               opacity=ConstantOpacity(20.0 * n), chat_limiter=lim)
        sim = Simulation(; n=(n, 1, 1), eos=IdealGas(; gamma=5 / 3, cv=4000.0),
                         radiation=rad, hydro=false, boundaries=Boundaries(Periodic()),
                         cfl_radiation=cfl)
        set_initial_conditions!(sim) do x, y, z
            T = 1 + exp(-(x - 0.5)^2 / (2 * 0.08^2))
            (rho=1.0, T=T, Erad=T^4)
        end
        run!(sim; tmax, verbose=false)
        return vec(gas_temperature(sim)) .- 1, sim
    end
    ref, _ = hotspot(phi0=1.0, cfl=0.4)
    err(d) = sum(abs, d .- ref) / sum(abs, ref)
    rsla, _ = hotspot(phi0=0.01)
    irsla, sim = hotspot(phi0=0.01, lim=StabilityBound())
    @test all(==(1.0), sim.rw.chat[1, sim.grid.lo[1]:sim.grid.hi[1], 1, 1])
    @test err(irsla) < err(rsla) / 100
end
