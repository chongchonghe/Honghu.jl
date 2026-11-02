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

    s = BackwardEulerChem()
    w = Honghu.chem_workspace(s, net, Float64)
    function be_error(n)
        u = copy(u0)
        for _ in 1:n
            @test Honghu.solve_chemistry!(w, s, net, cell(0.5 / n), u).converged
        end
        abs(u[2] - chain_exact(0.5, 1.0, 10.0)[2])
    end
    # First order. The error ratio under halving the step tends to 2 from above, and at
    # 50 against 100 steps (k2 dt = 0.1) it is still 2.47, as one backward-Euler step
    # applied by hand gives; it is 2.08 at 400 against 800.
    @test 1.8 < be_error(400) / be_error(800) < 2.2
    @test be_bytes(w, s, net, cell(0.01), copy(u0)) == 0
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
    @test_logs (:warn,) match_mode = :any begin       # OrdinaryDiffEq says why it gave up
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
