# Multigroup radiation: the source-term driver, the group-aware reflecting
# boundary, and the Vaytet Marshak wave.
#
# The tests are mostly of the invariant kind CLAUDE.md asks for. Three are worth
# pointing at:
#
#   * at one group the multigroup driver must reproduce the grey driver to
#     round-off. They solve the same equations by different routes, so a
#     disagreement above round-off means one of them is wrong.
#   * `BracketedCoupling` and `QuokkaCoupling` are two formulations of one
#     backward Euler step -- one unknown against `1 + NG`, sharing no residual,
#     no iteration and no linear solve -- so they must agree to their
#     tolerances. That they do is the reason the interface exists.
#   * a cell left alone in an optically thick medium must relax onto a Planck
#     spectrum at its own temperature, group by group. That is the one thing a
#     multigroup solver can get wrong while every total still looks right.

using Honghu: gbase, multigroup_source_step!, group_opacities, group_centers,
              enforce_state!, radiation_source_step!, IMX, IEGAS, IFRZ,
              CouplingState, coupling_rhs, coupling_jacobian, solve_coupling, sumvec,
              solve_jacobian, maxabs
using LinearAlgebra: cond          # only, so Honghu's own `dot`/`norm` still win
using .BracketingNonlinearSolve: Bisection, ITP, ModAB, Brent   # alternatives to the default ZBrent

"""
A constant `chat` array of the shape the source drivers want.

The reduced speed of light is a per-cell, per-group field now (see
`update_chat!`), and these tests drive the source step directly rather than
through `step!`, so they have to supply one. Filling it with `phys.chat` is
exactly what the default `ConstantChat` limiter does.
"""
const_chat(phys, grid) = fill!(Honghu.allocate(grid, ngroups(phys)), phys.chat)

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
    # `coupling = QuokkaCoupling()` is passed explicitly throughout, and has to
    # be: this testset compares the two *drivers*, so the formulation must be
    # held fixed or it compares formulations instead. The grey driver's inline
    # 2x2 is Quokka's replaced base variables, so that is the one that can be
    # held to round-off. The default `BracketedCoupling` solves the same
    # equations by a different route and agrees only to its own tolerance --
    # measured, 4e-11 over this state -- which is asserted separately below.
    #
    # The two drivers solve the same equations by different routes: the grey one
    # with the 2x2 Newton written out inline, the multigroup one through
    # `solve_coupling`. They are not bit-for-bit identical, and cannot be: they
    # open from the same linearised transfer but form and test their residuals
    # in a different order, so the two walk to the same root along paths that
    # stop on different sides of the last ulp. They must agree to round-off,
    # which is far tighter than the 1e-9 they are asked to converge to, and that
    # is the assertion worth making.
    sim, _ = P.marshak_wave(nx=32)
    for _ in 1:5
        step!(sim)                      # a state with real structure in it
    end
    U0, R0 = copy(sim.U), copy(sim.R)
    for kappa in (0.0, 1e-3, 1.0, 3e2, 1e6, 1e9)
        rad = RadiationPhysics(c=1.0, chat=1.0, a_rad=1.0, beta_order=0,
                               opacity=ConstantOpacity(kappa),
                               coupling=QuokkaCoupling())
        U, R = copy(U0), copy(R0)
        radiation_source_step!(U, R, const_chat(rad, sim.grid), sim.eos, rad, sim.grid,
                               1e-3, nothing, 0.0, sim.floors; update_momentum=false)
        Um, Rm = copy(U0), copy(R0)
        multigroup_source_step!(Um, Rm, const_chat(rad, sim.grid), sim.eos, rad, sim.grid,
                                1e-3, nothing, 0.0, sim.floors; update_momentum=false)
        @test isapprox(Um, U; rtol=1e-14)
        @test isapprox(Rm, R; rtol=1e-14)
        @test Rm != R0 || kappa == 0.0    # something actually happened
    end

    # ... and with an opacity that depends on temperature, which is the case that
    # separates the two drivers. Both re-evaluate `kappa` at the current iterate,
    # so both converge to the root of the same residual. When the multigroup path
    # froze the opacity for the whole solve instead, it converged to a *different*
    # root and the two drivers disagreed by 1e-4 here -- a discrepancy no constant
    # opacity can show, which is why this loop exists alongside the one above.
    #
    # They agree to their convergence tolerance rather than to round-off, because
    # the two Jacobians differ: the grey driver drops `d(kappa_P/kappa_E)/dT` and
    # the multigroup solvers keep it, so they take different paths to the same
    # answer and stop at different points inside the tolerance ball.
    for n in (-3.0, 2.0), kappa0 in (1e-1, 1e3)
        rad = RadiationPhysics(c=1.0, chat=1.0, a_rad=1.0, beta_order=0,
                               opacity=PowerLawOpacity(; kappa0, rho0=1.0, T0=1.0, n),
                               coupling=QuokkaCoupling())
        U, R = copy(U0), copy(R0)
        radiation_source_step!(U, R, const_chat(rad, sim.grid), sim.eos, rad, sim.grid,
                               1e-3, nothing, 0.0, sim.floors; update_momentum=false)
        Um, Rm = copy(U0), copy(R0)
        multigroup_source_step!(Um, Rm, const_chat(rad, sim.grid), sim.eos, rad, sim.grid,
                                1e-3, nothing, 0.0, sim.floors; update_momentum=false)
        @test isapprox(Um, U; rtol=1e-8)
        @test isapprox(Rm, R; rtol=1e-8)
        @test Rm != R0
    end

    # With the velocity-dependent terms on there is a lagged outer iteration on
    # the `v.F` work term as well, and the momentum kick and the flux damping to
    # get right; the two drivers must still track each other. The velocity
    # changes sign across the box so both signs of the work term are exercised.
    #
    # `failures == 0` is asserted rather than assumed: a work term more negative
    # than the cell's thermal emission used to make `QuokkaCoupling`'s first
    # implied group energy negative, and the iteration never recovered (see the
    # note at the end of this file). Every regime this comparison covers is one
    # where it does converge.
    c, chat = 100.0, 10.0
    for kappa in (1e-2, 1.0, 1e2, 1e4)
        rad = RadiationPhysics(; c, chat, a_rad=0.2, opacity=ConstantOpacity(kappa),
                               beta_order=1, coupling=QuokkaCoupling())
        sim2 = Simulation(n=(16, 1, 1), radiation=rad, eos=IdealGas(gamma=5 / 3, cv=1.0),
                          boundaries=Boundaries(Periodic()))
        set_initial_conditions!(sim2) do x, y, z
            (rho=1.0, p=1.0, vx=0.2 * sinpi(2x), Erad=1.0, Fx=0.5 * c)
        end
        U, R = copy(sim2.U), copy(sim2.R)
        sg = radiation_source_step!(sim2.U, sim2.R, const_chat(rad, sim2.grid), sim2.eos, rad,
                                    sim2.grid, 1e-3, nothing, 0.0, sim2.floors)
        Um, Rm = copy(U), copy(R)
        sm = multigroup_source_step!(Um, Rm, const_chat(rad, sim2.grid), sim2.eos, rad,
                                     sim2.grid, 1e-3, nothing, 0.0, sim2.floors)
        @test sg.failures == 0 && sm.failures == 0
        @test isapprox(Um, sim2.U; rtol=1e-12)
        @test isapprox(Rm, sim2.R; rtol=1e-12)
        @test Um[IMX, 8, 1, 1] != U[IMX, 8, 1, 1]  # the momentum kick really happened
    end

    # ... and the default formulation against the same grey driver, to the
    # tolerance it converges to rather than to round-off. This is the weaker
    # assertion of the two and it is the one that covers what actually runs:
    # `BracketedCoupling` reduces the step to one scalar unknown, so it shares
    # no residual, no unknown and no iteration with the grey 2x2 -- only the
    # equations. Agreement here is therefore a check on the equations.
    for kappa in (0.0, 1e-3, 1.0, 3e2, 1e6, 1e9)
        rad = RadiationPhysics(c=1.0, chat=1.0, a_rad=1.0, beta_order=0,
                               opacity=ConstantOpacity(kappa))
        U, R = copy(U0), copy(R0)
        radiation_source_step!(U, R, const_chat(rad, sim.grid), sim.eos, rad, sim.grid,
                               1e-3, nothing, 0.0, sim.floors; update_momentum=false)
        Um, Rm = copy(U0), copy(R0)
        multigroup_source_step!(Um, Rm, const_chat(rad, sim.grid), sim.eos, rad, sim.grid,
                                1e-3, nothing, 0.0, sim.floors; update_momentum=false)
        @test isapprox(Um, U; rtol=1e-8)
        @test isapprox(Rm, R; rtol=1e-8)
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
    default = grey_run(Honghu.GREY)
    @test grey_run(RadiationGroups([0.0, Inf])) == default
    @test grey_run(RadiationGroups([1e-4, 1e4])) == default
end

"One cell's `CouplingState`, out of equilibrium, in code units."
function cell_state(; NG=4, opacity, Tgas=1.3, Trad=0.5, rho=1.0, cv=1.0, dt=1e-3,
                    c=1.0, chat=1.0, arad=1.0, kB=1.0)
    grp = NG == 1 ? Honghu.GREY : RadiationGroups(0.1, 20.0, NG)
    Erad = (arad * Trad^4) * planck_fractions(grp, Trad, kB)
    z = zero(SVec{NG,Float64})
    # `chat` is per group now; a scalar argument means the same speed for all of
    # them, which is what every test here wants unless it says otherwise.
    chatv = chat isa SVec ? chat : SVec{NG,Float64}(ntuple(_ -> Float64(chat), NG)...)
    CouplingState{NG,NG + 1,Float64,typeof(opacity)}(cv * Tgas, Erad, z, z, rho, cv,
                                                     opacity, dt, c, chatv, arad, kB, grp,
                                                     1e-10, 1e-30)
end

# A temperature-dependent opacity of adjustable strength, so that `dkappa/dT`
# fills the first row and column of every Jacobian these tests build. Defined
# once at file scope rather than inside each testset that wants it, and not only
# to avoid three copies: an anonymous function gets a fresh type per syntactic
# location, that type lands in `CouplingState`'s fourth parameter, and every
# distinct `CouplingState` makes OrdinaryDiffEq specialise its entire integrator
# stack again. Three copies of this line cost about six seconds of compilation.
law(k0) = UserOpacity(planck=(rho, T) -> k0 * T^2, dplanck_dT=(rho, T) -> 2k0 * T)

@testset "the coupling Jacobian is the derivative of the coupling right-hand side" begin
    # The test that pins the refactor. `coupling_rhs` and `coupling_jacobian` are
    # the single source of truth for the coupling rate: `AdaptiveCoupling` builds
    # its Newton matrix as `I - dt J`, and the adaptive solver hands `J` to its
    # integrator. If `J` ever stops being the derivative of the right-hand side,
    # both are quietly solving something other than what they report.
    #
    # It is written with a *temperature-dependent* opacity on purpose. The
    # `dkappa/dT` terms are the whole content of "exact" here, and with a constant
    # opacity they are all zero and the test would pass without ever seeing them.
    kap(rho, T) = 1e2 * T^2
    dkap(rho, T) = 2e2 * T
    law = UserOpacity(planck=kap, dplanck_dT=dkap)

    "The (1+NG)x(1+NG) Jacobian as a dense matrix."
    function dense(s, Egas, Erad, NG)
        j = coupling_jacobian(s, Egas, Erad)
        J = zeros(NG + 1, NG + 1)
        J[1, 1] = j.gas_gas
        for g in 1:NG
            J[1, 1+g] = j.gas_rad[g]
            J[1+g, 1] = j.rad_gas[g]
            J[1+g, 1+g] = j.rad_rad[g]
        end
        J
    end

    # Central differences of `coupling_rhs`. The step is scaled by the largest
    # component rather than by each one: the Wien-tail groups hold 1e-30 of the
    # spectrum, and a step proportional to *that* differences a quantity of order
    # one against itself and measures nothing but round-off.
    function fd(s, Egas, Erad, NG; h_rel=1e-6)
        u = [Egas; [Erad[g] for g in 1:NG]]
        rhs(v) = begin
            E = SVec{NG,Float64}(ntuple(g -> v[1+g], NG)...)
            a, b = coupling_rhs(s, v[1], E)
            [a; [b[g] for g in 1:NG]]
        end
        J = zeros(NG + 1, NG + 1)
        step = h_rel * maximum(abs, u)
        for k in 1:(NG+1)
            up = copy(u); up[k] += step
            um = copy(u); um[k] -= step
            J[:, k] = (rhs(up) - rhs(um)) / (2step)
        end
        J
    end

    for NG in (1, 4)
        s = cell_state(; NG, opacity=law)
        # away from the state's own temperature, so a Jacobian that silently
        # froze the opacity at `s.Egas` would be caught
        Egas = 1.4 * s.Egas
        Erad = 1.3 * s.Erad
        A = dense(s, Egas, Erad, NG)
        B = fd(s, Egas, Erad, NG)
        @test maximum(abs, A - B) / maximum(abs, B) < 1e-6

        # ... and the `dkappa/dT` terms are actually present: an opacity with the
        # same law but no declared derivative gives a *different* Jacobian.
        blind = cell_state(; NG, opacity=UserOpacity(kap))
        C = dense(blind, Egas, Erad, NG)
        @test maximum(abs, A - C) / maximum(abs, B) > 1e-3
        @test maximum(abs, C - B) / maximum(abs, B) > 1e-3
    end
end

@testset "the exact Jacobian changes the rate, not the root" begin
    # Newton stops on the residual of the true equations however good the matrix
    # that got it there, so declaring `dkappa/dT` must not move the answer. What
    # it moves is the iteration count, and by enough to be worth the interface.
    #
    # The iteration claim is `QuokkaCoupling`'s alone now. `BracketedCoupling`
    # hands its equation to a derivative-free root-finder, so `dkappa/dT` enters
    # only its tolerance conversion and cannot change how many evaluations the
    # root-finder takes. The root-invariance half applies to both, and is the
    # half that matters.
    kap(rho, T) = 1e2 * T^2
    law = UserOpacity(planck=kap, dplanck_dT=(rho, T) -> 2e2 * T)
    blind = UserOpacity(kap)

    for method in (QuokkaCoupling(), BracketedCoupling()), NG in (1, 4)
        exact = solve_coupling(method, cell_state(; NG, opacity=law))
        zeroed = solve_coupling(method, cell_state(; NG, opacity=blind))
        @test exact.converged && zeroed.converged
        @test isapprox(exact.Egas, zeroed.Egas; rtol=1e-8)
        @test isapprox(sumvec(exact.Erad), sumvec(zeroed.Erad); rtol=1e-8)
        method isa QuokkaCoupling && @test exact.iterations < zeroed.iterations
    end
end

@testset "the two Jacobian solves agree" begin
    # This is the testset that licenses the default. `solve_arrowhead` is an
    # `O(NG)` elimination that is exact only because the Jacobian's group block
    # is diagonal, which is a property of the terms in `coupling_rhs` and not of
    # radiation -- and both solvers use it unless told otherwise. A default may
    # encode an assumption when something checks the assumption; these are the
    # assertions that do, so they are load-bearing rather than decorative.
    #
    # Until `DenseSolve` existed the linear solve was never compared with
    # anything: both in-tree solvers called the elimination, so they could agree
    # with each other and both be wrong. What checked it before was indirect --
    # the grey driver's own inline 2x2 solve, at one group, through a converged
    # Newton answer -- and Newton stops on the residual of the true equations, so
    # a damaged matrix mostly hides in the iteration count. A general LU of the
    # same matrix is the direct check. Detuning the elimination by one part in a
    # thousand fails the two assertions below, which is what "direct" means.
    #
    # The file-scope `law` is used throughout, so that `dkappa/dT` fills the
    # Jacobian's first row and column. With a constant opacity the elimination
    # would be tested on a matrix simpler than the one it meets.

    "The Newton matrix `I - dt J` in blocks, and a right-hand side with something in it."
    function newton_system(s, NG)
        J = coupling_jacobian(s, 1.4 * s.Egas, 1.3 * s.Erad)
        (one(Float64) - s.dt * J.gas_gas, -s.dt * J.gas_rad, -s.dt * J.rad_gas,
         one(Float64) - s.dt * J.rad_rad, 0.37 * s.Egas,
         SVec{NG,Float64}(ntuple(g -> (0.1 + 0.9g / NG) * s.Erad[g], NG)...))
    end

    for NG in (1, 4, 8), k0 in (1e-6, 1.0, 1e3, 1e9)   # thin, marginal, thick, very thick
        s = cell_state(; NG, opacity=law(k0))
        J00, J0g, Jg0, Jgg, F0, Fg = newton_system(s, NG)
        a0, ag = solve_jacobian(ArrowheadSolve(), J00, J0g, Jg0, Jgg, F0, Fg)
        d0, dg = solve_jacobian(DenseSolve(), J00, J0g, Jg0, Jgg, F0, Fg)
        x = [a0; [ag[g] for g in 1:NG]]
        y = [d0; [dg[g] for g in 1:NG]]
        rel = maximum(abs, x - y) / maximum(abs, y)

        A = zeros(NG + 1, NG + 1)
        A[1, 1] = J00
        for g in 1:NG
            A[1, 1+g] = J0g[g]
            A[1+g, 1] = Jg0[g]
            A[1+g, 1+g] = Jgg[g]
        end

        # Two algorithms for one system agree to `eps * cond`, not to `eps`: each
        # is backward stable, so each returns the exact answer to a slightly
        # different matrix, and the conditioning is what turns that into a
        # difference in the answers. The bound is therefore written with `cond`
        # in it. It matters only in the last row of this loop -- at `tau = 1e6`
        # the Newton matrix has a condition number of 2e8 and no fixed 1e-12
        # would be honest -- and everywhere else the two agree to 1e-16.
        @test rel <= 8 * eps() * cond(A)
        k0 <= 1e3 && @test rel < 1e-12
        # ... and both are actually solving the stated system, not agreeing on a
        # shared misreading of it
        @test maximum(abs, A * y - [F0; [Fg[g] for g in 1:NG]]) <=
              1e-12 * cond(A) * maximum(abs, y)
    end

    # End to end, and for *both* solvers, since both hand-written Newton loops
    # default to the elimination: each takes the same Newton path under either
    # linear solve -- same matrix, same update to round-off, so the same iterate
    # and the same iteration count. This is the assertion that would fail if
    # `linsolve` ever stopped being a choice of how to solve and became a choice
    # of what to solve. It covers `QuokkaCoupling` too, whose `tau <= 0` sentinel
    # is a large *finite* number precisely so that a general factorisation of its
    # matrix is possible.
    #
    # `QuokkaCoupling` is the only solver left that forms a Newton system at all:
    # `BracketedCoupling` has one unknown and `AdaptiveCoupling` hands its matrix
    # to OrdinaryDiffEq. So this is now a one-solver loop, and the elimination is
    # checked against a general factorisation of the same matrix rather than
    # against a second caller of itself. (That the *default* linear solve is
    # still the elimination is pinned by the allocation testset below, where only
    # it can be zero bytes.)
    handwritten = (ls -> QuokkaCoupling(linsolve=ls),)
    for NG in (1, 4, 8), k0 in (1e-6, 1.0, 1e9), M in handwritten
        s = cell_state(; NG, opacity=law(k0))
        a = solve_coupling(M(ArrowheadSolve()), s)
        d = solve_coupling(M(DenseSolve()), s)
        @test a.converged && d.converged
        @test a.iterations == d.iterations
        @test isapprox(a.Egas, d.Egas; rtol=1e-12)
        @test maxabs(a.Erad - d.Erad) <= 1e-12 * maxabs(d.Erad)
    end

    # ... including where `QuokkaCoupling` writes its `tau <= 0` sentinel, which
    # is `-1/eps` rather than `-Inf` so that the elimination does not make NaN.
    # A group that is exactly transparent while others are thick is what reaches
    # it with a residual still to solve, and a large finite entry is also what
    # lets `DenseSolve` factorise the same matrix instead of throwing.
    op = FrequencyOpacity((nu, rho, T) -> nu < 2.0 ? 1e3 * T^2 : 0.0)
    for NG in (4, 8), M in handwritten
        s = cell_state(; NG, opacity=op)
        a = solve_coupling(M(ArrowheadSolve()), s)
        d = solve_coupling(M(DenseSolve()), s)
        @test a.converged && d.converged
        @test isapprox(a.Egas, d.Egas; rtol=1e-12)
        @test maxabs(a.Erad - d.Erad) <= 1e-12 * maxabs(d.Erad)
    end
end

@testset "what each coupling path costs in allocation" begin
    # Asserted rather than described, for every combination of solver, stepper
    # and linear solve, because each of the three has a zero-byte path and a
    # path that cannot be zero, and which is which is the whole argument for the
    # defaults. Measured inside a function: at global scope `@allocated` would be
    # measuring dynamic dispatch on a non-const global and not the solver.
    function coupling_bytes(method, s)
        solve_coupling(method, s)                        # compile and warm up
        minimum(@allocated(solve_coupling(method, s)) for _ in 1:3)
    end

    for NG in (1, 4, 8)
        s = cell_state(; NG, opacity=law(1.0))

        # The package default, and the one every production run takes. Zero, and
        # it stays zero: `test_performance.jl` asserts the same thing through the
        # whole source step, and this is the per-cell statement of it.
        # `BracketedCoupling` reduces the step to one scalar unknown, so it forms
        # no Newton matrix at all and there is nothing here that could allocate
        # -- but the claim is in its docstring, so it is asserted rather than
        # argued.
        @test coupling_bytes(BracketedCoupling(), s) == 0
        @test coupling_bytes(BracketedCoupling(root_finder=Bisection()), s) == 0
        @test coupling_bytes(QuokkaCoupling(), s) == 0
        # `DenseSolve` allocates and cannot not: a `Matrix` is heap storage
        # however small `NG` is, and `QuokkaCoupling` pays that bill every Newton
        # iteration if it is asked for.
        qd = coupling_bytes(QuokkaCoupling(linsolve=DenseSolve()), s)
        @test qd > 0

        # The reference solver allocates, and cannot not: `solve` builds an
        # integrator, a nonlinear solver, their caches and a solution object per
        # cell. Bounded on both sides -- below, because the whole point is that
        # it does; above, so a regression that doubles it is caught rather than
        # shrugged at -- and it is far more than the one path here that forms a
        # matrix at all.
        # The bound depends on the Julia version, because OrdinaryDiffEq's
        # allocation does: at NG = 8 it is 13.0 kB on 1.12 and 20.8 kB on 1.10.
        ad = coupling_bytes(AdaptiveCoupling(), s)
        @test 4_000 < ad < (VERSION < v"1.11" ? 32_000 : 20_000)
        @test ad > 4 * qd
    end
end

@testset "BracketedCoupling: the root-finder chooses how, not what" begin
    # `BracketedCoupling` is an equation handed to a root-finder, and the three
    # testsets below each pin one thing the solve adds on top of a bare call.

    # 1. Any bracketing method lands on the same root. `Bisection` is the
    #    reference: slow, and impossible to get wrong on a valid bracket.
    for NG in (2, 4), k0 in (1e-3, 1.0, 1e3, 1e9), (Tg, Tr) in ((1.3, 0.5), (0.5, 1.3))
        s = cell_state(; NG, opacity=law(k0), Tgas=Tg, Trad=Tr)
        Etot = s.Egas + sumvec(s.Erad)
        # `Any[...]`, not `[...]`: an untyped comprehension asks the compiler for
        # the common result type over five root-finder types, and inside the full
        # suite that inference overflows the stack on Julia 1.10.
        results = Any[solve_coupling(BracketedCoupling(root_finder=rf), s)
                      for rf in (Honghu.ZBrent(), Brent(), ModAB(), ITP(), Bisection())]
        @test all(r.converged for r in results)
        for r in results[2:end]
            @test abs(r.Egas - results[1].Egas) <= 2e-9 * Etot
        end
    end

    # 2. Energy is conserved to the tolerance in a radiation-dominated cell,
    #    whichever method is used. Every method stops on bracket width in `E_gas`,
    #    and here `dG/dE_gas` is ~4000, so an unconverted width tolerance leaves a
    #    conservation error thousands of times the promised one -- measured
    #    3e-6 of the cell's energy with `Bisection`. This is the assertion the
    #    tolerance conversion exists for.
    for rf in (Honghu.ZBrent(), Brent(), ModAB(), Bisection()), k0 in (1.0, 1e3, 1e9)
        s = cell_state(; NG=4, opacity=ConstantOpacity(k0), Tgas=10.0, Trad=1.0)
        r = solve_coupling(BracketedCoupling(root_finder=rf), s)
        Etot = s.Egas + sumvec(s.Erad)
        @test r.converged
        @test abs(r.Egas + sumvec(r.Erad) - Etot) <= 1e-9 * Etot
    end

    # 3. Where the step has several roots, the one returned is the one connected
    #    to where the cell started. With `kappa ~ T^2`, gas at 0.1 and radiation
    #    at 3, `G` changes sign at 0.110, 1.17 and 2.52; the error-controlled ODE
    #    ends at 0.109. From the wide bracket `[cv Tfloor, E_tot]`, `ModAB` and
    #    `Bisection` both returned 2.52 -- the gas handed 23 times the energy it
    #    physically reaches. Marching the bracket out from the old gas energy is
    #    what makes the answer independent of the method.
    s = cell_state(; NG=4, opacity=PowerLawOpacity(kappa0=10.0, n=2.0), Tgas=0.1, Trad=3.0)
    ode = solve_coupling(AdaptiveCoupling(reltol=1e-8, abstol=1e-14), s).Egas
    for rf in (Honghu.ZBrent(), Brent(), ModAB(), ITP(), Bisection())
        r = solve_coupling(BracketedCoupling(root_finder=rf), s)
        @test r.converged
        @test r.Egas < 0.2
        # ... and within one backward-Euler step's discretisation error of the
        # trajectory, which at this optical depth is about one per cent
        @test abs(r.Egas - ode) / ode < 0.02
    end
end

@testset "the two couplings agree" begin
    # Same backward Euler step in two sets of base variables. `BracketedCoupling`'s
    # residual is `tau (B - E)`, whose cancellation error grows with `tau`, so its
    # achievable tolerance degrades where the medium is thick -- which is exactly
    # the conditioning that Quokka's replaced base variables avoid, and why this
    # comparison is worth making rather than assuming.
    for kappa in (1e-2, 1.0, 1e2, 1e4)
        a = mg_box(; kappa, coupling=QuokkaCoupling())
        b = mg_box(; kappa, coupling=BracketedCoupling())
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
    R = allocate(g, Honghu.NRAD * NG)
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
        for comp in 1:Honghu.NRAD
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
        cc = const_chat(rad, sim.grid)
        f() = radiation_source_step!(sim.U, sim.R, cc, sim.eos, rad, sim.grid, 1e-3, nothing,
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
@testset "a negative work term does not drain a cell" begin
    # `QuokkaCoupling` used to open its Newton iteration at `R = 0`, which makes
    # the group energy implied on the first pass `kappaP/kappaE (4piB/c +
    # work/tau)`. When the lagged `v.F` work term is negative and larger than the
    # thermal emission -- a cell moving against the radiation flux, deep enough
    # that `tau` is large -- that first energy is negative, the
    # equilibrium-temperature guard snapped the gas to its floor, and the
    # iteration was trapped: every negative-velocity cell lost all its energy.
    #
    # It now opens from the linearised transfer with `Erad` held at `Erad0`,
    # which is what the grey driver and Quokka's `AddSourceTermsMultiGroup` both
    # do. `BracketedCoupling` never had the problem, because it starts from `Erad0`,
    # so the two agreeing here is the assertion that matters.
    groups = RadiationGroups(1.0, 100.0, 3)
    function drained(coupling)
        rad = RadiationPhysics(; c=100.0, chat=10.0, a_rad=1e-6, groups, coupling,
                               beta_order=1, opacity=ConstantOpacity(100.0))
        sim = Simulation(; n=(8, 1, 1), radiation=rad,
                         boundaries=Boundaries(Periodic()),
                         radiation_boundaries=Boundaries(Periodic()))
        set_initial_conditions!(sim) do x, y, z
            (rho=1.0, vx=-0.5, p=1.0, Erad=1.0, Fx=0.9 * 100.0)
        end
        run!(sim; tmax=1e-3, verbose=false)
        return minimum(first(field(sim, :Erad)))   # group sum
    end
    q = drained(QuokkaCoupling())
    d = drained(BracketedCoupling())
    @test q > 1e-6                      # the cell keeps its energy at all
    @test isapprox(q, d; rtol=1e-6)     # ... and the two solvers agree on it
end
