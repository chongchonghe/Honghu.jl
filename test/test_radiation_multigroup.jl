# Multigroup radiation: the source-term driver, the group-aware reflecting
# boundary, and the Vaytet Marshak wave.
#
# The tests are mostly of the invariant kind CLAUDE.md asks for. Three are worth
# pointing at:
#
#   * at one group the multigroup driver must reproduce the grey driver to
#     round-off. They solve the same equations by different routes, so a
#     disagreement above round-off means one of them is wrong.
#   * `QuokkaCoupling` and `DirectCoupling` are two formulations of one backward
#     Euler step, so they must agree to their tolerances. That they do is the
#     reason the `AbstractCoupling` interface exists.
#   * a cell left alone in an optically thick medium must relax onto a Planck
#     spectrum at its own temperature, group by group. That is the one thing a
#     multigroup solver can get wrong while every total still looks right.

using Hydro3D: gbase, multigroup_source_step!, group_opacities, group_centers,
               enforce_state!, radiation_source_step!, IMX, IEGAS, IFRZ

"Group energies of a Planck spectrum at `Trad`, written into every interior cell."
function set_planck!(sim, Trad)
    phys = sim.radiation
    g = sim.grid
    NG = ngroups(phys)
    frac = planck_fractions(phys.groups, Trad, phys.k_B)
    E = phys.a_rad * Trad^4
    for k in g.lo[3]:g.hi[3], j in g.lo[2]:g.hi[2], i in g.lo[1]:g.hi[1], gr in 1:NG
        b = gbase(gr)
        sim.R[b+IERAD, i, j, k] = E * frac[gr]
        sim.R[b+IFRX, i, j, k] = 0.0
        sim.R[b+IFRY, i, j, k] = 0.0
        sim.R[b+IFRZ, i, j, k] = 0.0
    end
    fill_ghosts!(sim.R, sim.radiation_boundaries, g, sim.t)
    enforce_state!(sim)
    reset_diagnostics!(sim)
    return sim
end

"A uniform multigroup box in code units (`c = a_rad = k_B = 1`), out of equilibrium."
function mg_box(; NG=4, n=8, kappa=1.0, chat=1.0, T_gas=1.0, T_rad=0.1,
                coupling=QuokkaCoupling(), beta_order=0, hydro=false)
    groups = RadiationGroups(0.1, 20.0, NG)      # edges in units of k_B T, since energy_unit = 1
    rad = RadiationPhysics(; c=1.0, chat=chat, a_rad=1.0, k_B=1.0, groups=groups,
                           opacity=ConstantOpacity(kappa), beta_order=beta_order,
                           coupling=coupling)
    sim = Simulation(; n=(n, 1, 1), radiation=rad, hydro=hydro,
                     eos=IdealGas(gamma=5 / 3, cv=1.0),
                     boundaries=Boundaries(Periodic()))
    set_initial_conditions!(sim) do x, y, z
        (rho=1.0, T=T_gas, Erad=T_rad^4)
    end
    set_planck!(sim, T_rad)
end

"`E_gas + (c/chat) * sum_g E_rad,g` over the interior: what the coupling conserves."
function mg_energy(sim)
    g = sim.grid
    NG = ngroups(sim.radiation)
    eg = 0.0
    er = 0.0
    for k in g.lo[3]:g.hi[3], j in g.lo[2]:g.hi[2], i in g.lo[1]:g.hi[1]
        eg += sim.U[IEGAS, i, j, k]
        for gr in 1:NG
            er += sim.R[gbase(gr)+IERAD, i, j, k]
        end
    end
    eg + (sim.radiation.c / sim.radiation.chat) * er
end

@testset "one group reproduces the grey driver" begin
    # The two drivers solve the same equations by different routes: the grey one
    # with the 2x2 Newton written out inline, the multigroup one through
    # `solve_coupling`. They are not bit-for-bit identical, and cannot be: the
    # grey iteration opens with the linearised solution as its first guess while
    # `QuokkaCoupling` opens at `R = 0`, so the two walk to the same root along
    # different paths and stop on different sides of the last ulp. They must
    # agree to round-off, which is far tighter than the 1e-9 they are asked to
    # converge to, and that is the assertion worth making.
    sim, _ = P.marshak_wave(nx=32)
    for _ in 1:5
        step!(sim)                      # a state with real structure in it
    end
    U0, R0 = copy(sim.U), copy(sim.R)
    for kappa in (0.0, 1e-3, 1.0, 3e2, 1e6, 1e9)
        rad = RadiationPhysics(c=1.0, chat=1.0, a_rad=1.0, beta_order=0,
                               opacity=ConstantOpacity(kappa))
        U, R = copy(U0), copy(R0)
        radiation_source_step!(U, R, sim.eos, rad, sim.grid, 1e-3, nothing, 0.0,
                               sim.floors; update_momentum=false)
        Um, Rm = copy(U0), copy(R0)
        multigroup_source_step!(Um, Rm, sim.eos, rad, sim.grid, 1e-3, nothing, 0.0,
                                sim.floors; update_momentum=false)
        @test isapprox(Um, U; rtol=1e-14)
        @test isapprox(Rm, R; rtol=1e-14)
        @test Rm != R0 || kappa == 0.0    # something actually happened
    end

    # With the velocity-dependent terms on there is a lagged outer iteration on
    # the `v.F` work term as well, and the momentum kick and the flux damping to
    # get right; the two drivers must still track each other. The velocity
    # changes sign across the box so both signs of the work term are exercised.
    #
    # `failures == 0` is asserted rather than assumed: `QuokkaCoupling` opens its
    # iteration at `R = 0`, so a work term more negative than the cell's thermal
    # emission makes its first implied group energy negative and the iteration
    # never recovers (see the note at the end of this file). Every regime this
    # comparison covers is one where it does converge.
    c, chat = 100.0, 10.0
    for kappa in (1e-2, 1.0, 1e2, 1e4)
        rad = RadiationPhysics(; c, chat, a_rad=0.2, opacity=ConstantOpacity(kappa),
                               beta_order=1)
        sim2 = Simulation(n=(16, 1, 1), radiation=rad, eos=IdealGas(gamma=5 / 3, cv=1.0),
                          boundaries=Boundaries(Periodic()))
        set_initial_conditions!(sim2) do x, y, z
            (rho=1.0, p=1.0, vx=0.2 * sinpi(2x), Erad=1.0, Fx=0.5 * c)
        end
        U, R = copy(sim2.U), copy(sim2.R)
        sg = radiation_source_step!(sim2.U, sim2.R, sim2.eos, rad, sim2.grid, 1e-3, nothing,
                                    0.0, sim2.floors)
        Um, Rm = copy(U), copy(R)
        sm = multigroup_source_step!(Um, Rm, sim2.eos, rad, sim2.grid, 1e-3, nothing, 0.0,
                                     sim2.floors)
        @test sg.failures == 0 && sm.failures == 0
        @test isapprox(Um, sim2.U; rtol=1e-12)
        @test isapprox(Rm, sim2.R; rtol=1e-12)
        @test Um[IMX, 8, 1, 1] != U[IMX, 8, 1, 1]  # the momentum kick really happened
    end

    # ... and the group machinery itself is inert at one group: any single group,
    # whatever its edges, must give the run it had before multigroup existed.
    function grey_run(groups)
        rad = RadiationPhysics(; c=1.0, chat=1.0, a_rad=1.0, beta_order=0, groups=groups,
                               opacity=ConstantOpacity(3e2))
        s = Simulation(n=(32, 1, 1), radiation=rad, hydro=false,
                       eos=IdealGas(gamma=5 / 3, cv=1.0),
                       boundaries=Boundaries(Periodic()))
        set_initial_conditions!(s) do x, y, z
            (rho=1.0, T=0.5 + sinpi(2x)^2, Erad=0.1 * exp(-(x - 0.5)^2 / 0.02))
        end
        run!(s; tmax=0.05, verbose=false)
        (copy(s.U), copy(s.R))
    end
    default = grey_run(Hydro3D.GREY)
    @test grey_run(RadiationGroups([0.0, Inf])) == default
    @test grey_run(RadiationGroups([1e-4, 1e4])) == default
end

@testset "the two couplings agree" begin
    # Same backward Euler step in two sets of base variables. `DirectCoupling`'s
    # residual is `tau (B - E)`, whose cancellation error grows with `tau`, so its
    # achievable tolerance degrades where the medium is thick -- which is exactly
    # the conditioning that Quokka's replaced base variables avoid, and why this
    # comparison is worth making rather than assuming.
    for kappa in (1e-2, 1.0, 1e2, 1e4)
        a = mg_box(; kappa, coupling=QuokkaCoupling())
        b = mg_box(; kappa, coupling=DirectCoupling())
        run!(a; tmax=1e-2, verbose=false)
        run!(b; tmax=1e-2, verbose=false)
        @test a.diag.newton_failures == 0
        @test b.diag.newton_failures == 0
        @test isapprox(a.R, b.R; rtol=1e-6)
        @test isapprox(a.U, b.U; rtol=1e-9)
        # and both land on the same equilibrium temperature
        @test isapprox(gas_temperature(a)[1, 1, 1], gas_temperature(b)[1, 1, 1]; rtol=1e-8)
    end
end

@testset "multigroup total energy is conserved" begin
    # Periodic box, no sources: transport moves `sum_g E_g` around without
    # changing it, and the coupling only trades energy between the gas and the
    # groups. `E_gas + (c/chat) sum_g E_g` must therefore be constant to
    # round-off, whatever the groups do among themselves.
    #
    # The coupling conserves energy only as well as it converges -- the residual
    # it stops on *is* the energy imbalance -- so the tolerance is tightened here
    # to make round-off, rather than 1e-9, the thing being measured. That is a
    # property of the backward Euler step and not of the group structure: the
    # grey driver has the same 7e-10 at the default tolerance.
    for chat in (1.0, 0.25)
        groups = RadiationGroups(0.1, 20.0, 5)
        rad = RadiationPhysics(; c=1.0, chat=chat, a_rad=1.0, k_B=1.0, groups=groups,
                               opacity=ConstantOpacity(30.0), beta_order=0,
                               coupling=QuokkaCoupling(tolerance=1e-13))
        sim = Simulation(n=(48, 1, 1), radiation=rad, hydro=false,
                         eos=IdealGas(gamma=5 / 3, cv=1.0),
                         boundaries=Boundaries(Periodic()))
        set_initial_conditions!(sim) do x, y, z
            (rho=1.0, T=0.4 + 0.3 * sinpi(2x), Erad=1e-3)
        end
        # a spatially varying spectrum, so transport actually has work to do
        g = sim.grid
        for i in g.lo[1]:g.hi[1]
            x = cellcenter(g, i, 1, 1)[1]
            set_local = planck_fractions(groups, 0.2 + 0.4 * sinpi(2x)^2, 1.0)
            E = 1e-2 * (1 + 0.5 * sinpi(2x))
            for gr in 1:5
                sim.R[gbase(gr)+IERAD, i, 1, 1] = E * set_local[gr]
            end
        end
        fill_ghosts!(sim.R, sim.radiation_boundaries, g, sim.t)
        enforce_state!(sim)
        reset_diagnostics!(sim)

        e0 = mg_energy(sim)
        run!(sim; tmax=0.05, verbose=false)
        @test sim.diag.newton_failures == 0
        @test isapprox(mg_energy(sim), e0; rtol=1e-12)
        # the groups did move energy between themselves, so the invariant is not
        # holding because nothing happened
        @test !isapprox(sim.R[gbase(1)+IERAD, g.lo[1], 1, 1],
                        1e-2 * planck_fractions(groups, 0.2, 1.0)[1]; rtol=1e-6)
    end
end

@testset "an optically thick cell relaxes onto a Planck spectrum" begin
    # The group energies of a relaxed cell must be `a T^4 f_g(T)` at the cell's
    # *own* temperature. A solver that got the group weights wrong could still
    # conserve energy and still reach the right total, so this is the assertion
    # that the frequency resolution means anything.
    for NG in (2, 4, 8)
        sim = mg_box(; NG, kappa=1e6, T_gas=1.0, T_rad=0.1)
        run!(sim; tmax=1e-2, verbose=false)
        Tg = gas_temperature(sim)[1, 1, 1]
        f = planck_fractions(sim.radiation.groups, Tg, 1.0)
        Eg = [sim.R[gbase(gr)+IERAD, sim.grid.lo[1], 1, 1] for gr in 1:NG]
        @test sum(Eg) ≈ Tg^4 rtol = 1e-3               # radiative equilibrium
        for gr in 1:NG
            @test Eg[gr] ≈ Tg^4 * f[gr] rtol = 1e-3
        end
        # and the spectrum is not the one it started from
        @test !isapprox(Eg[1], 0.1^4 * planck_fractions(sim.radiation.groups, 0.1, 1.0)[1];
                        rtol=1e-3)
    end
end

@testset "reflecting boundaries flip every group's flux" begin
    # The reflecting condition used to flip index `d + 1`, which is the normal
    # flux of group 1 and an *energy density* of some other group. Radiation
    # would then leak through a mirror in every group but the first.
    NG = 3
    g = CartesianGrid(n=(6, 5, 4), nghost=2)
    R = allocate(g, Hydro3D.NRAD * NG)
    for k in 1:g.dims[3], j in 1:g.dims[2], i in 1:g.dims[1], gr in 1:NG
        b = gbase(gr)
        R[b+IERAD, i, j, k] = 10.0 * gr + i
        R[b+IFRX, i, j, k] = 1.0 + gr
        R[b+IFRY, i, j, k] = 2.0 + gr
        R[b+IFRZ, i, j, k] = 3.0 + gr
    end
    fill_ghosts!(R, Boundaries(Reflecting()), g, 0.0)
    for dir in 1:3, gr in 1:NG
        b = gbase(gr)
        lo, ng = g.lo[dir], g.ng[dir]
        idx(n) = ntuple(d -> d == dir ? n : 1, 3)
        ghost = idx(lo - 1)
        inner = idx(lo)
        # the normal flux of *this* group is mirrored ...
        @test R[b+dir+1, ghost...] == -R[b+dir+1, inner...]
        # ... and nothing else in the group is
        for comp in 1:Hydro3D.NRAD
            comp == dir + 1 && continue
            @test R[b+comp, ghost...] == R[b+comp, inner...]
        end
    end
    # the five-variable hydro vector still gets exactly one component flipped
    U = allocate(g, NHYDRO)
    fill!(U, 1.0)
    for n in 1:NHYDRO, i in 1:g.dims[1]
        U[n, i, 1, 1] = n + 0.1 * i
    end
    fill_ghosts!(U, Boundaries(Reflecting()), g, 0.0)
    @test U[IMX, g.lo[1]-1, 1, 1] == -U[IMX, g.lo[1], 1, 1]
    for n in (IRHO, 3, 4, NHYDRO)
        @test U[n, g.lo[1]-1, 1, 1] == U[n, g.lo[1], 1, 1]
    end
end

@testset "the multigroup source step does not allocate per cell" begin
    # Same bound and same reason as `test_performance.jl`: `Threads.@threads`
    # allocates task objects, which is O(threads), while a boxed closure variable
    # would be O(cells * groups). The multigroup driver builds a `CouplingState`
    # per cell and unrolls several `ntuple`s inside it, so there is plenty to get
    # wrong here.
    LIMIT = 8.0
    for beta_order in (0, 1)
        groups = RadiationGroups(0.1, 20.0, 4)
        rad = RadiationPhysics(; c=10.0, chat=1.0, a_rad=1.0, k_B=1.0, groups=groups,
                               opacity=ConstantOpacity(1.0), beta_order=beta_order)
        sim = Simulation(n=(24, 24, 24), radiation=rad, hydro=beta_order > 0,
                         eos=IdealGas(gamma=5 / 3, cv=1.0),
                         boundaries=Boundaries(Periodic()))
        set_initial_conditions!(sim) do x, y, z
            (rho=1.0, p=1.0, vx=0.3, Erad=1.0, Fx=0.5)
        end
        f() = radiation_source_step!(sim.U, sim.R, sim.eos, rad, sim.grid, 1e-3, nothing,
                                     0.0, sim.floors; update_momentum=beta_order > 0)
        f()
        @test minimum(@allocated(f()) for _ in 1:3) / prod(sim.grid.n) < LIMIT
    end
end

@testset "Marshak wave with a frequency-dependent opacity (Vaytet)" begin
    sim, tmax = P.marshak_vaytet(nx=32)
    phys = sim.radiation
    @test ngroups(phys) == 4
    # Quokka's edges for four groups, and its bin-centre opacities
    @test collect(Tuple(phys.groups.edges)) ≈ [6e10, 6e11, 6e12, 6e13, 6e14] rtol = 1e-12
    nu = group_centers(phys.groups)
    kap = group_opacities(phys.opacity, phys.groups, 1e-3, 300.0)
    for gr in 1:4
        @test nu[gr] ≈ sqrt(phys.groups.edges[gr] * phys.groups.edges[gr+1])
        @test kap[gr] ≈ 2000.0 * (nu[gr] / 4.0e13)^-2
    end
    # four decades of opacity across the spectrum is the whole point of the test
    @test kap[1] / kap[4] ≈ 1e6 rtol = 1e-6

    # the wall is a Planck spectrum at T_L with no net flux
    g = sim.grid
    wall = [sim.R[gbase(gr)+IERAD, 1, 1, 1] for gr in 1:4]
    @test sum(wall) ≈ phys.a_rad * 1000.0^4 rtol = 1e-12
    @test wall ≈ phys.a_rad * 1000.0^4 .* collect(Tuple(planck_fractions(phys.groups, 1000.0, phys.k_B)))
    @test all(sim.R[gbase(gr)+IFRX, 1, 1, 1] == 0.0 for gr in 1:4)
    # ... and the interior starts as a Planck spectrum at 300 K
    interior = [sim.R[gbase(gr)+IERAD, g.lo[1], 1, 1] for gr in 1:4]
    @test sum(interior) ≈ phys.a_rad * 300.0^4 rtol = 1e-12
    @test nrepaired(sim) == 0

    m = default_metrics(P.marshak_vaytet)
    @test m.front(sim) == 0.0                       # nothing has been heated yet

    run!(sim; tmax=tmax / 4, verbose=false)
    front1 = m.front(sim)
    run!(sim; tmax, verbose=false)
    front2 = m.front(sim)

    Tg = gas_temperature(sim)[:, 1, 1]
    Er = P.total_erad(sim)[:, 1, 1]
    Tr = (Er ./ phys.a_rad) .^ 0.25

    @test sim.diag.newton_failures == 0
    @test 0 < front1 < front2                       # the wave front advances
    @test front2 < 0.75 * (g.xhi[1] - g.xlo[1])     # ... and stays inside the domain
    @test issorted(Tg; rev=true)                    # monotone away from the wall
    @test 900 < Tg[1] < 1000                        # the wall drives it to nearly T_L
    @test m.Trad_wall(sim) > 900
    # radiation runs ahead of the matter, as in the grey Marshak wave. Not in the
    # cell against the wall, where the thin groups stream away faster than they
    # are replaced and the gas ends up the hotter of the two.
    @test all(Tr[2:end] .> Tg[2:end])
    @test maximum(Tr .- Tg) > 50

    # The signature of the test, and the thing no grey calculation can produce:
    # with four decades of opacity across the spectrum, three groups sit in LTE
    # with the gas while the fourth does not. Compare each group against the
    # Planck spectrum at the *local* gas temperature.
    lte(i, gr) = phys.a_rad * Tg[i]^4 *
                 planck_fractions(phys.groups, Tg[i], phys.k_B)[gr]
    egroup(i, gr) = sim.R[gbase(gr)+IERAD, i+g.ng[1], 1, 1]
    far = g.n[1] - 2
    for i in (4, g.n[1] ÷ 2, far), gr in 1:3
        @test egroup(i, gr) ≈ lte(i, gr) rtol = 0.02      # optically thick: in LTE
    end
    @test egroup(far, 4) > 10 * lte(far, 4)               # thin: far above it
    @test egroup(1, 4) < lte(1, 4)                        # ... and below it at the wall
end

# --- a known limitation, recorded here rather than in a commit message --------
#
# `QuokkaCoupling` opens its Newton iteration at `R = 0`, which makes the group
# energy it implies on the first pass `kappa_P/kappa_E * (4piB/c + work/tau)`.
# When the lagged `v.F` work term is negative and larger in magnitude than the
# thermal emission -- a cell moving against the radiation flux, deep enough in
# the medium that `tau` is large -- that first energy is negative, the
# equilibrium-temperature guard then snaps the gas to its floor, and the
# iteration is trapped there. It reports `converged = false`, so the failure is
# visible in `sim.diag.newton_failures` and in `run!`'s summary rather than
# silent, but the cell is wrong.
#
# Reproduce with `beta_order = 1`, `c = 100`, `chat = 10`, `a_rad = 1e-6`,
# `kappa = 100`, `rho = 1`, `p = 1`, `Erad = 1`, `Fx = 0.9c` and `vx < 0`: every
# cell with a negative velocity loses all its energy. `DirectCoupling` gets the
# same problem right, because it starts from `Erad0` instead.
#
# The grey driver in `source_terms.jl` does not have the problem: its first
# iterate is the linearised solution `R = (4piB/c - E0 kappa_E/kappa_P) tau +
# work`, with `Erad` held at `Erad0`. That is also what Quokka's
# `AddSourceTermsMultiGroup` does, so the fix is to give `QuokkaCoupling` the
# same opening guess. It lives in `coupling_solvers.jl`, which this change does
# not own.
