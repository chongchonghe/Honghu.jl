# A -> B -> C, first order: an exact solution, three species, no energy, no bands.
struct ChainNet <: Honghu.AbstractNetwork
    k1::Float64
    k2::Float64
end
Honghu.nspecies(::ChainNet) = 3
Honghu.nbands(::ChainNet) = 0
Honghu.species_mass(::ChainNet, s) = 1.0
function Honghu.chem_rhs!(du, u, n::ChainNet, cell)
    du[1] = -n.k1 * u[1]; du[2] = n.k1 * u[1] - n.k2 * u[2]; du[3] = n.k2 * u[2]; du[4] = 0.0
    return nothing
end
function Honghu.chem_jacobian!(J, u, n::ChainNet, cell)
    fill!(J, 0.0); J[1, 1] = -n.k1; J[2, 1] = n.k1; J[2, 2] = -n.k2; J[3, 2] = n.k2
    return nothing
end
Honghu.chem_scales!(s, u, ::ChainNet, cell) = (fill!(s, 1.0); nothing)
chain_exact(t, k1, k2) = (exp(-k1 * t), k1 / (k2 - k1) * (exp(-k1 * t) - exp(-k2 * t)))

# G(u) = atan(u - a) as a backward-Euler equation: f(u) = (u - u0 - atan(u - a)) / dt, root a.
# Newton on atan diverges from any start more than 1.39 from the root, so only a damped
# iteration finds it.
struct AtanNet <: Honghu.AbstractNetwork
    a::Float64
    u0::Float64
    dt::Float64
end
Honghu.nspecies(::AtanNet) = 1
Honghu.nbands(::AtanNet) = 0
Honghu.species_mass(::AtanNet, s) = 1.0
function Honghu.chem_rhs!(du, u, n::AtanNet, cell)
    du[1] = (u[1] - n.u0 - atan(u[1] - n.a)) / n.dt; du[2] = 0.0
    return nothing
end
function Honghu.chem_jacobian!(J, u, n::AtanNet, cell)
    fill!(J, 0.0); J[1, 1] = (1 - 1 / (1 + (u[1] - n.a)^2)) / n.dt
    return nothing
end
Honghu.chem_scales!(s, u, ::AtanNet, cell) = (fill!(s, 1.0); nothing)

function be_bytes(w, s, net, cell, u)
    Honghu.solve_chemistry!(w, s, net, cell, u)
    @allocated Honghu.solve_chemistry!(w, s, net, cell, u)
end

@testset "chemistry: integrators on an exact network" begin
    net = ChainNet(1.0, 10.0)
    cell(dt) = Honghu.ChemCell(1.0, 1.0, 1.0, dt)
    u0 = [1.0, 0.0, 0.0, 0.0]
    @test Honghu.nunknowns(net) == 4
    @test Honghu.ienergy(net) == 4

    for (rtol, bound) in ((1e-2, 2e-2), (1e-6, 1e-5))
        s = RosenbrockChem(reltol=rtol, abstol=1e-12)
        w = Honghu.chem_workspace(s, net, Float64)
        @test w.integ.opts.abstol === w.abstol       # per-cell abstol is mutated in place
        u = copy(u0)
        st = Honghu.solve_chemistry!(w, s, net, cell(1.0), u)
        @test st.converged && st.steps > 0
        @test abs(u[1] - chain_exact(1.0, 1.0, 10.0)[1]) < bound
        @test abs(u[2] - chain_exact(1.0, 1.0, 10.0)[2]) < bound
        @test sum(u[1:3]) ≈ 1 atol = 1e-12
        v = copy(u0)                                  # a reused workspace gives the same answer
        Honghu.solve_chemistry!(w, s, net, cell(1.0), v)
        @test v == u
    end

    # A step far longer than the placeholder time span the integrator is built on. Its
    # `dtmax` defaulted to that span, one time unit, so a step of 1e6 took a million
    # internal steps and a step of 1e10 s, an ordinary hydrodynamic one, never finished.
    for dt in (1e3, 1e6)
        s = RosenbrockChem()
        w = Honghu.chem_workspace(s, net, Float64)
        u = copy(u0)
        st = Honghu.solve_chemistry!(w, s, net, cell(dt), u)
        @test st.converged && st.steps < 200
        @test u[3] ≈ 1 atol = 1e-6
    end

    s = BackwardEulerChem()
    w = Honghu.chem_workspace(s, net, Float64)
    function be_run(n)                   # the error at t = 0.5, and whether every step converged
        u = copy(u0)
        ok = true
        for _ in 1:n
            ok &= Honghu.solve_chemistry!(w, s, net, cell(0.5 / n), u).converged
        end
        abs(u[2] - chain_exact(0.5, 1.0, 10.0)[2]), ok
    end
    # First order. The error ratio under halving the step tends to 2 from above, and at
    # 50 against 100 steps (k2 dt = 0.1) it is still 2.47, as one backward-Euler step
    # applied by hand gives; it is 2.08 at 400 against 800.
    (e400, ok400), (e800, ok800) = be_run(400), be_run(800)
    @test ok400 && ok800
    @test 1.8 < e400 / e800 < 2.2
    @test be_bytes(w, s, net, cell(0.01), copy(u0)) == 0

    # The residual ends the iteration as well as the update. A linear problem is solved by
    # the first Newton step, after which the residual is round-off while the update that
    # was taken is of order one; stopping on the update alone takes a second iteration.
    st = Honghu.solve_chemistry!(w, s, net, cell(0.5), copy(u0))
    @test st.converged && st.steps == 1
end

@testset "chemistry: backward Euler damps a Newton step that overshoots" begin
    net = AtanNet(100.0, 105.0, 1.0)
    cell = Honghu.ChemCell(1.0, 1.0, 1.0, 1.0)
    s = BackwardEulerChem()
    w = Honghu.chem_workspace(s, net, Float64)
    u = [105.0, 0.0]
    st = Honghu.solve_chemistry!(w, s, net, cell, u)
    @test st.converged
    @test u[1] ≈ 100 atol = 1e-8
    # the start 5 from the root is far outside the 1.39 that undamped Newton can recover from
    @test abs(net.u0 - net.a) > 1.39
    @test be_bytes(w, s, net, cell, [105.0, 0.0]) == 0
end

@testset "chemistry: defaults of the network interface" begin
    net = ChainNet(1.0, 10.0)
    n = [1e3, 2e3, 3e3, 5.0]            # the fourth entry is not a species and must not count
    T = 1e4
    e = Honghu.network_eint(net, n, T)
    @test e ≈ 1.5 * Honghu.CGS.k_B * 6e3 * T
    @test Honghu.network_temperature(net, n, e) ≈ T
    @test Honghu.chem_internal_energy(net, [1.0, 2.0, 3.0, 7.0]) == 7.0
    @test Honghu.iband(net, 1) == 5
    @test_throws ArgumentError Honghu.initial_mass_fractions(net, (rho=1.0,))
end

# du/dt = u^2 from u = 1 blows up at t = 1, so a step of 2 has no answer.
struct BlowNet <: Honghu.AbstractNetwork end
Honghu.nspecies(::BlowNet) = 1
Honghu.nbands(::BlowNet) = 0
Honghu.species_mass(::BlowNet, s) = 1.0
Honghu.chem_rhs!(du, u, ::BlowNet, cell) = (du[1] = u[1]^2; du[2] = 0.0; nothing)
Honghu.chem_jacobian!(J, u, ::BlowNet, cell) = (fill!(J, 0.0); J[1, 1] = 2u[1]; nothing)
Honghu.chem_scales!(s, u, ::BlowNet, cell) = (fill!(s, 1.0); nothing)

@testset "chemistry: a cell that cannot be solved is reported, not hidden" begin
    net = BlowNet()
    cell(dt) = Honghu.ChemCell(1.0, 1.0, 1.0, dt)

    s = RosenbrockChem()
    w = Honghu.chem_workspace(s, net, Float64)
    # The failure is in the result and nowhere else: OrdinaryDiffEq's own warning, one
    # long line per failing cell, is switched off, so a threaded grid run does not
    # print one for every cell the driver is going to count anyway.
    @test_logs min_level = Base.CoreLogging.Warn begin
        @test !Honghu.solve_chemistry!(w, s, net, cell(2.0), [1.0, 0.0]).converged
    end
    # The failure leaves the workspace usable: the next cell gets the answer a fresh
    # workspace gives. The grid driver counts the failure and carries on with it.
    u = [1.0, 0.0]
    @test Honghu.solve_chemistry!(w, s, net, cell(0.5), u).converged
    v = [1.0, 0.0]
    Honghu.solve_chemistry!(Honghu.chem_workspace(s, net, Float64), s, net, cell(0.5), v)
    @test u == v

    b = BackwardEulerChem()
    wb = Honghu.chem_workspace(b, net, Float64)
    @test !Honghu.solve_chemistry!(wb, b, net, cell(2.0), [1.0, 0.0]).converged   # no root
    # Too few iterations is a failure to converge, and says so.
    b1 = BackwardEulerChem(max_iterations=1)
    st = Honghu.solve_chemistry!(Honghu.chem_workspace(b1, net, Float64), b1, net, cell(0.1), [1.0, 0.0])
    @test !st.converged && st.steps == 1
end

@testset "chemistry: the linear solve" begin
    solve(A, b) = (x = copy(b); ok = Honghu._gauss_solve!(copy(A), x); (ok, x))
    # A zero and a tiny leading pivot both need the row swap; `\` is the reference.
    for A in ([0.0 2.0 1.0; 1.0 1.0 1.0; 3.0 0.0 1.0],
              [1e-17 1.0 0.0; 1.0 1.0 2.0; 0.0 3.0 1.0],
              [4.0 -2.0 1.0; -2.0 4.0 -2.0; 1.0 -2.0 4.0])
        b = [1.0, -2.0, 3.0]
        ok, x = solve(A, b)
        @test ok
        @test x ≈ A \ b rtol = 1e-12
    end
    @test !solve([1.0 2.0; 2.0 4.0], [1.0, 2.0])[1]      # singular: reports it
    @test !solve([0.0 1.0; 0.0 1.0], [1.0, 2.0])[1]      # a zero column
    @test !solve([NaN 1.0; 1.0 1.0], [1.0, 2.0])[1]      # a broken matrix
end

# The chain again, but with a scale that comes from the cell, so two cells can ask for
# very different tolerances through one workspace.
struct ScaledChain <: Honghu.AbstractNetwork
    inner::ChainNet
end
Honghu.nspecies(n::ScaledChain) = Honghu.nspecies(n.inner)
Honghu.nbands(n::ScaledChain) = Honghu.nbands(n.inner)
Honghu.species_mass(n::ScaledChain, s) = Honghu.species_mass(n.inner, s)
Honghu.chem_rhs!(du, u, n::ScaledChain, cell) = Honghu.chem_rhs!(du, u, n.inner, cell)
Honghu.chem_jacobian!(J, u, n::ScaledChain, cell) = Honghu.chem_jacobian!(J, u, n.inner, cell)
Honghu.chem_scales!(s, u, ::ScaledChain, cell) = (fill!(s, cell.rho); nothing)

@testset "chemistry: per-cell scales reach the error control" begin
    net = ScaledChain(ChainNet(1.0, 10.0))
    s = RosenbrockChem(reltol=1e-9, abstol=1e-3)         # abstol * scale sets the tolerance
    cell(scale) = Honghu.ChemCell(scale, 1.0, 1.0, 1.0)  # the scale is the cell's density
    function run(w, scale)
        u = [1.0, 0.0, 0.0, 0.0]
        st = Honghu.solve_chemistry!(w, s, net, cell(scale), u)
        (st.steps, u)
    end
    fresh(scale) = run(Honghu.chem_workspace(s, net, Float64), scale)
    loose, tight = fresh(1e3), fresh(1e-6)
    @test tight[1] > 5 * loose[1]                        # the tolerance really differs
    w = Honghu.chem_workspace(s, net, Float64)
    for scale in (1e3, 1e-6, 1e-6, 1e3, 1e-6)            # alternating, through one workspace
        @test run(w, scale) == fresh(scale)
    end
end

const CGS = Honghu.CGS
hstate(nH, x, T, N) = [nH * (1 - x), nH * x, 1.5 * CGS.k_B * nH * (1 + x) * T, N]
# A central difference in 256-bit arithmetic. In Float64 the step of 1e-6 u_j costs the
# difference of rec - ion, whose two terms differ by 1e7 in a weakly ionized cell, about
# three digits of the column of e, and the check would measure round-off, not the Jacobian.
function fd_jacobian(net, u, cell)
    n = length(u); J = zeros(n, n); fp = zeros(BigFloat, n); fm = zeros(BigFloat, n)
    for j in 1:n
        h = 1e-6 * abs(big(u[j])); up = big.(u); um = big.(u); up[j] += h; um[j] -= h
        Honghu.chem_rhs!(fp, up, net, cell); Honghu.chem_rhs!(fm, um, net, cell)
        J[:, j] = Float64.((fp - fm) / 2h)
    end
    J
end
invariants(u, phi, eps) = (u[1] + u[2], u[2] + u[4] / phi,
                           u[3] + Honghu.CHI_H * u[2] + eps * u[4] / phi)

@testset "chemistry: hydrogen network" begin
    c = CGS.c_light
    eps = 29.65 * CGS.eV
    net = HydrogenOTSA(sigma=1.63e-18, photon_energy=eps)
    cell = Honghu.ChemCell(1e-3 * CGS.m_H, c, 1e-3c, 1e12)

    # exact Jacobian, row by row, at three states (150 K, not 100 K: the
    # recombination cooling switches on at 100 K and a difference across it is noise)
    for u in (hstate(1e-3, 0.01, 150.0, 1e-4), hstate(1e-3, 0.5, 8e3, 1e-5),
              hstate(1e-3, 0.999, 2e4, 1e-6))
        J = zeros(4, 4); Honghu.chem_jacobian!(J, u, net, cell)
        Jfd = fd_jacobian(net, u, cell)
        for i in 1:4
            @test isapprox(J[i, :], Jfd[i, :]; rtol=1e-5)
        end
    end

    # photoionization alone conserves three linear invariants exactly
    pure = HydrogenOTSA(sigma=1.63e-18, photon_energy=eps, alpha0=0.0,
                        collisional_ionization=false, cooling=false)
    for s in (RosenbrockChem(), BackwardEulerChem())
        w = Honghu.chem_workspace(s, pure, Float64)
        u = hstate(1e-3, 0.01, 100.0, 1e-4); I0 = invariants(u, 1e-3, eps)
        @test Honghu.solve_chemistry!(w, s, pure, Honghu.ChemCell(1e-3 * CGS.m_H, c, 1e-3c, 1e13), u).converged
        @test all(isapprox.(invariants(u, 1e-3, eps), I0; rtol=1e-10))
        @test u[2] > 1e-5                                  # it did ionize
    end

    # a dark, fully neutral, cold cell does not move
    for s in (RosenbrockChem(), BackwardEulerChem())
        w = Honghu.chem_workspace(s, net, Float64)
        u = hstate(1e-3, 0.0, 100.0, 0.0); u0 = copy(u)
        @test Honghu.solve_chemistry!(w, s, net, cell, u).converged
        @test all(isfinite, u)
        @test u ≈ u0 rtol = 1e-12
    end

    # a photon density at the radiation floor stays non-negative
    for s in (RosenbrockChem(), BackwardEulerChem())
        w = Honghu.chem_workspace(s, net, Float64)
        u = hstate(1e-3, 1e-6, 100.0, 1e-40)
        Honghu.solve_chemistry!(w, s, net, cell, u)
        @test u[2] >= 0 && u[4] >= 0
    end

    # isothermal mode: e is not evolved; the energy written back follows n_tot
    iso = HydrogenOTSA(sigma=6.3e-18, photon_energy=13.6 * CGS.eV, isothermal=1e4)
    w = Honghu.chem_workspace(RosenbrockChem(), iso, Float64)
    u = hstate(1e-3, 0.01, 1e4, 1e-4); e0 = u[3]
    Honghu.solve_chemistry!(w, RosenbrockChem(), iso, cell, u)
    @test u[3] ≈ e0 rtol = 1e-12
    @test Honghu.chem_internal_energy(iso, u) ≈ 1.5 * CGS.k_B * (u[1] + 2u[2]) * 1e4

    # ionization equilibrium x^2/(1-x) = Gamma/(alpha n) = 0.5 gives x = 0.5. A tiny
    # chat keeps N fixed while the gas sees c sigma N.
    eq = HydrogenOTSA(sigma=6.3e-18, photon_energy=13.6 * CGS.eV, isothermal=1e4,
                      alpha0=2.59e-13, alpha_exp=0.0, collisional_ionization=false)
    N = 0.5 * 2.59e-13 * 1.0 / (c * 6.3e-18)
    ceq = Honghu.ChemCell(CGS.m_H, c, 1e-14c, 1e4 / 2.59e-13)
    for s in (RosenbrockChem(reltol=1e-6), BackwardEulerChem())
        w = Honghu.chem_workspace(s, eq, Float64)
        u = hstate(1.0, 0.01, 1e4, N)
        Honghu.solve_chemistry!(w, s, eq, ceq, u)
        @test u[2] / (u[1] + u[2]) ≈ 0.5 rtol = 1e-3
    end

    # the two integrators agree at a small step. Backward Euler is first order, and its
    # energy, which is the integral of a heating rate that falls as the gas ionizes, is
    # low by 2.1e-3 at dt = 1e9, 1.05e-3 at 5e8 and 4.2e-4 here, in step with dt; the
    # adaptive integrator is the reference.
    small = Honghu.ChemCell(1e-3 * CGS.m_H, c, 1e-3c, 2e8)
    ua = hstate(1e-3, 0.01, 100.0, 1e-4); ub = copy(ua)
    Honghu.solve_chemistry!(Honghu.chem_workspace(RosenbrockChem(reltol=1e-8), net, Float64),
                            RosenbrockChem(reltol=1e-8), net, small, ua)
    Honghu.solve_chemistry!(Honghu.chem_workspace(BackwardEulerChem(), net, Float64),
                            BackwardEulerChem(), net, small, ub)
    @test ua[2] ≈ ub[2] rtol = 1e-3
    @test ua[3] ≈ ub[3] rtol = 1e-3
end

# Entry by entry, in units of f per relative change of u_j so that the columns, which differ
# by 1e19 between e and n, are comparable: 1e-7 of the entry, and an entry that is below 1e-10
# of the largest in its row may be anything, because Float64 underflows where the 256-bit
# difference does not (a Boltzmann factor of exp(-1000) is 0 in one and 1e-440 in the other).
function jacobian_matches(J, Jfd, u)
    S, Sfd = J .* abs.(u'), Jfd .* abs.(u')
    all(1:size(J, 1)) do i
        floor = 1e-10 * max(maximum(abs, S[i, :]), maximum(abs, Sfd[i, :]))
        all(abs(S[i, j] - Sfd[i, j]) <= 1e-7 * max(abs(S[i, j]), abs(Sfd[i, j])) + floor
            for j in axes(J, 2))
    end
end

@testset "chemistry: hydrogen Jacobian in every mode" begin
    c = CGS.c_light
    cell = Honghu.ChemCell(1e-3 * CGS.m_H, c, 1e-3c, 1e12)
    kw = (sigma=1.63e-18, photon_energy=29.65 * CGS.eV)
    nets = (HydrogenOTSA(; kw...),
            HydrogenOTSA(; kw..., cooling=false),
            HydrogenOTSA(; kw..., collisional_ionization=false),
            HydrogenOTSA(; kw..., alpha_exp=0.0, heating_energy=1e-11),
            HydrogenOTSA(; kw..., isothermal=1e4),
            HydrogenOTSA(; kw..., Tmin=500.0))
    # 150 K and 2e4 K are on either side of the recombination-cooling switch and of the
    # excitation and ionization onsets; 3e5 K is where free-free and excitation compete
    for net in nets, u in (hstate(1e-3, 0.01, 150.0, 1e-4), hstate(1e-3, 0.5, 8e3, 1e-5),
                           hstate(1e-3, 0.999, 2e4, 1e-6), hstate(1e-3, 0.9, 3e5, 1e-7),
                           hstate(1e-3, 0.3, 300.0, 1e-5))
        J = zeros(4, 4); Honghu.chem_jacobian!(J, u, net, cell)
        Jfd = fd_jacobian(net, u, cell)
        @test jacobian_matches(J, Jfd, u)
    end

    # the energy row of an isothermal network is zero, in the rate and in the Jacobian
    iso = HydrogenOTSA(; kw..., isothermal=1e4)
    u = hstate(1e-3, 0.5, 8e3, 1e-5)
    du = zeros(4); J = zeros(4, 4)
    Honghu.chem_rhs!(du, u, iso, cell); Honghu.chem_jacobian!(J, u, iso, cell)
    @test du[3] == 0 && all(==(0), J[3, :])

    # a negative n_H0, n_H+ or N counts as zero in every rate, and its column vanishes
    net = nets[1]
    for j in (1, 2, 4)
        u = hstate(1e-3, 0.5, 8e3, 1e-5); u[j] = -1e-9
        v = copy(u); v[j] = -3e-9
        du, dv = zeros(4), zeros(4)
        Honghu.chem_rhs!(du, u, net, cell); Honghu.chem_rhs!(dv, v, net, cell)
        @test du == dv
        J = zeros(4, 4); Honghu.chem_jacobian!(J, u, net, cell)
        @test all(==(0), J[:, j])
        @test jacobian_matches(J, fd_jacobian(net, u, cell), u)
    end
end

@testset "chemistry: hydrogen network interface" begin
    net = HydrogenOTSA(sigma=1.63e-18, photon_energy=29.65 * CGS.eV)
    @test Honghu.nspecies(net) == 2 && Honghu.nbands(net) == 1
    @test Honghu.nunknowns(net) == 4 && Honghu.ienergy(net) == 3 && Honghu.iband(net, 1) == 4
    @test Honghu.species_mass(net, 1) == Honghu.species_mass(net, 2) == CGS.m_H
    @test Honghu.band_group(net, 1) == 1
    @test Honghu.band_photon_energy(net, 1) == 29.65 * CGS.eV
    @test net.heating_energy ≈ 29.65 * CGS.eV - 13.6 * CGS.eV
    @test HydrogenOTSA(sigma=1, photon_energy=2, heating_energy=1, group=3).group == 3

    # electrons count: a fully ionized gas has twice the particles of a neutral one
    n = [3e-4, 7e-4]
    e = Honghu.network_eint(net, n, 1e4)
    @test e ≈ 1.5 * CGS.k_B * (3e-4 + 2 * 7e-4) * 1e4
    @test Honghu.network_temperature(net, n, e) ≈ 1e4
    @test Honghu.chem_internal_energy(net, [1.0, 2.0, 3.0, 4.0]) == 3.0

    @test Honghu.initial_mass_fractions(net, (rho=1.0,)) == (1.0, 0.0)
    X = Honghu.initial_mass_fractions(net, (x_HII=0.25,))
    @test X == (0.75, 0.25) && sum(X) == 1

    sc = zeros(4)
    Honghu.chem_scales!(sc, hstate(1e-3, 0.5, 8e3, 1e-5), net, Honghu.ChemCell(1e-3 * CGS.m_H, 1.0, 1e-3, 1.0))
    @test all(>(0), sc) && sc[1] == sc[2] == 1e-3 && sc[4] ≈ 1e-6

    # backward Euler allocates nothing on this network either: no boxed Union field, no
    # tuple of rates that escapes
    b = BackwardEulerChem(); w = Honghu.chem_workspace(b, net, Float64)
    for iso in (nothing, 1e4)
        nt = HydrogenOTSA(sigma=1.63e-18, photon_energy=29.65 * CGS.eV, isothermal=iso)
        wt = Honghu.chem_workspace(b, nt, Float64)
        @test be_bytes(wt, b, nt, Honghu.ChemCell(1e-3 * CGS.m_H, CGS.c_light, 1e-3 * CGS.c_light, 1e9),
                       hstate(1e-3, 0.01, 100.0, 1e-4)) == 0
    end
end

@testset "chemistry: hydrogen network, cells the solvers must handle" begin
    c = CGS.c_light
    eps = 29.65 * CGS.eV
    net = HydrogenOTSA(sigma=1.63e-18, photon_energy=eps)

    # A cell on a front: one photon per hydrogen atom, Gamma dt = 4.9, T = 30 K. Undamped
    # Newton cycled here for the whole 50 iterations; with the line search it takes a handful.
    # The answer is a root of the backward-Euler equation, checked with the rates and not with
    # the solver, and it keeps the hydrogen it started with.
    cell = Honghu.ChemCell(1e-2 * CGS.m_H, c, 1e-3c, 1e13)
    u0 = hstate(1e-2, 0.0, 30.0, 1e-5)
    s = BackwardEulerChem()
    u = copy(u0)
    st = Honghu.solve_chemistry!(Honghu.chem_workspace(s, net, Float64), s, net, cell, u)
    @test st.converged && st.steps < 20
    f = zeros(4); Honghu.chem_rhs!(f, u, net, cell)
    sc = zeros(4); Honghu.chem_scales!(sc, u0, net, cell)
    @test maximum(abs.(u .- u0 .- cell.dt .* f) ./ sc) < 1e-9
    @test u[1] + u[2] ≈ 1e-2 rtol = 1e-10
    @test u[2] > 5e-3                                   # mostly ionized

    # A dark, mostly ionized cell at 1e4 K recombining over 2.6 recombination times, so a long
    # step with cooling. The residual rises along the Newton direction before it falls: a
    # search that takes a tiny step when no trial lowers it stalls here for all 50 iterations,
    # the whole update converges in 10.
    cellr = Honghu.ChemCell(1.0 * CGS.m_H, c, 1e-3c, 1e13)
    u0r = hstate(1.0, 0.999, 1e4, 0.0)
    ur = copy(u0r)
    st = Honghu.solve_chemistry!(Honghu.chem_workspace(s, net, Float64), s, net, cellr, ur)
    @test st.converged && st.steps < 20
    fr = zeros(4); Honghu.chem_rhs!(fr, ur, net, cellr)
    scr = zeros(4); Honghu.chem_scales!(scr, u0r, net, cellr)
    @test maximum(abs.(ur .- u0r .- cellr.dt .* fr) ./ scr) < 1e-9
    @test ur[1] + ur[2] ≈ 1.0 rtol = 1e-10
    @test 0.3 < ur[2] < 0.6                              # recombined, but not to neutral

    # a dark neutral cold cell already satisfies the equation: no Newton iteration at all
    dark = Honghu.ChemCell(1e-3 * CGS.m_H, c, 1e-3c, 1e12)
    st = Honghu.solve_chemistry!(Honghu.chem_workspace(s, net, Float64), s, net, dark,
                                 hstate(1e-3, 0.0, 100.0, 0.0))
    @test st.converged && st.steps == 0

    # a cell with no hydrogen is a no-op, not a NaN: the temperature is Tmin, every rate is
    # zero, and the scales stay positive
    u = [0.0, 0.0, 1e-15, 1e-5]; u0 = copy(u)
    cell = Honghu.ChemCell(1e-3 * CGS.m_H, c, 1e-3c, 1e12)
    du = fill(NaN, 4); J = fill(NaN, 4, 4); sc = zeros(4)
    Honghu.chem_rhs!(du, u, net, cell); Honghu.chem_jacobian!(J, u, net, cell)
    Honghu.chem_scales!(sc, u, net, cell)
    @test all(==(0), du) && all(isfinite, J) && all(>(0), sc)
    for s in (RosenbrockChem(), BackwardEulerChem())
        v = copy(u0)
        @test Honghu.solve_chemistry!(Honghu.chem_workspace(s, net, Float64), s, net, cell, v).converged
        @test v == u0
    end
end

# Absolute values of the rate coefficients, from the formulas of the docstring evaluated
# independently (Python, same k_B), so that a constant or an exponent typed wrongly cannot hide
# behind the finite-difference test, which only sees that rhs and Jacobian agree with each other.
@testset "chemistry: hydrogen rate coefficients" begin
    net = HydrogenOTSA(sigma=1.63e-18, photon_energy=29.65 * CGS.eV)
    val(f, T) = f(net, T)[1]
    #                    T       alpha_B      beta         L_rec        L_ff         L_ce
    for (T, a, b, lr, lf, lc) in ((1e4, 2.630000e-13, 8.110429e-16, 2.319600e-25, 1.855100e-25, 4.129922e-24),
                                  (1e5, 5.247540e-14, 3.807816e-09, 2.988224e-25, 5.866341e-25, 1.148292e-19),
                                  (3e4, 1.218908e-13, 5.238221e-11, 2.617559e-25, 3.213127e-25, 9.377895e-21))
        @test val(Honghu._hydrogen_alpha, T) ≈ a rtol = 1e-5
        @test val(Honghu._hydrogen_beta, T) ≈ b rtol = 1e-5
        @test val(Honghu._hydrogen_cool_recombination, T) ≈ lr rtol = 1e-5
        @test val(Honghu._hydrogen_cool_freefree, T) ≈ lf rtol = 1e-5
        @test val(Honghu._hydrogen_cool_excitation, T) ≈ lc rtol = 1e-5
    end
    # recombination cooling and excitation are zero or negligible well below 100 K and 1e4 K
    @test val(Honghu._hydrogen_cool_recombination, 50.0) == 0
    @test val(Honghu._hydrogen_cool_recombination, 100.0) ≈ 6.1e-10 * CGS.k_B * 100.0^0.11
    @test val(Honghu._hydrogen_alpha, 50.0) ≈ 1.073190e-11 rtol = 1e-5      # 2.63e-13 (T / 1e4)^-0.7
    # the switches
    off = HydrogenOTSA(sigma=1.63e-18, photon_energy=29.65 * CGS.eV, cooling=false,
                       collisional_ionization=false)
    for f in (Honghu._hydrogen_beta, Honghu._hydrogen_cool_recombination,
              Honghu._hydrogen_cool_freefree, Honghu._hydrogen_cool_excitation)
        @test f(off, 1e4) == (0.0, 0.0)
    end
    @test Honghu._hydrogen_alpha(off, 1e4)[1] == 2.63e-13
end
