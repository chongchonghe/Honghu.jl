# Dust-gas thermal coupling (`BateKetoDust`), solved by `BracketedCoupling` as
# one equation in the dust temperature. `docs/coupling-new-method-dust.md`
# derives it; these are the properties it has to keep.

using Hydro3D: CouplingState, SVec, solve_coupling, gas_scale, dust_energy_balance

"One cell with a dust model, in code units: `K = coefficient rho^2`."
function dust_cell(; NG=4, opacity=ConstantOpacity(1.0), Tgas=1.0, Trad=0.5, rho=10.0,
                   coefficient=1.0, dt=1.0)
    grp = NG == 1 ? Hydro3D.GREY : RadiationGroups(0.1, 20.0, NG)
    Erad = (Trad^4) * planck_fractions(grp, max(Trad, 1e-10), 1.0) +
           SVec{NG,Float64}(ntuple(_ -> 1e-20, NG)...)
    z = zero(SVec{NG,Float64})
    one_ = SVec{NG,Float64}(ntuple(_ -> 1.0, NG)...)
    dust = BateKetoDust(; coefficient)
    CouplingState{NG,NG + 1,Float64,typeof(opacity),typeof(dust)}(Tgas, Erad, z, z, rho,
                                                                  1.0, opacity, dt, 1.0,
                                                                  one_, 1.0, 1.0, grp,
                                                                  1e-10, 1e-30, dust)
end

# the same cell without dust
nodust(s::CouplingState{NG,M,T,O}) where {NG,M,T,O} =
    CouplingState{NG,M,T,O}(s.Egas, s.Erad, s.src, s.work, s.rho, s.cv, s.opacity, s.dt,
                            s.c, s.chat, s.a_rad, s.k_B, s.groups, s.Tfloor, s.Efloor)

total_energy(s, Egas, Erad) = Egas + Hydro3D.dot(gas_scale(s), Erad)

const DUST_OPACITIES = (ConstantOpacity(1.0),
                        PowerLawOpacity(kappa0=1.0, rho0=1.0, T0=1.0, m=0.0, n=2.0))

@testset "dust-gas coupling" begin
    @testset "conserves energy exactly and converges everywhere" begin
        # Energy is conserved by construction at every iterate, so to round-off;
        # the tolerance is on the *share* of it between gas and radiation, and is
        # checked against a solve a thousand times tighter.
        nfail = 0
        worst_energy = 0.0
        worst_state = 0.0
        for NG in (1, 4), op in DUST_OPACITIES, Tg in (0.01, 1.0, 10.0), Tr in (0.0, 1.0, 5.0),
            rho in (1e-4, 1.0, 1e4, 1e8), K in (0.0, 1.0, 1e8)
            s = dust_cell(; NG, opacity=op, Tgas=Tg, Trad=Tr, rho, coefficient=K / rho^2)
            r = solve_coupling(BracketedCoupling(), s)
            ref = solve_coupling(BracketedCoupling(tolerance=1e-12), s)
            E0 = total_energy(s, s.Egas, s.Erad)
            nfail += !r.converged
            worst_energy = max(worst_energy, abs(total_energy(s, r.Egas, r.Erad) / E0 - 1))
            worst_state = max(worst_state, abs(r.Egas - ref.Egas) / E0)
        end
        @test nfail == 0
        @test worst_energy < 1e-13
        @test worst_state < 1.01e-9
    end

    @testset "strong coupling recovers the dust-free solve" begin
        # K -> infinity locks the dust to the gas, and the radiation then couples
        # at the gas temperature with the same opacity law: the dust-free problem.
        for NG in (1, 4), op in (ConstantOpacity(1.0),
                                 PowerLawOpacity(kappa0=1.0, rho0=1.0, T0=1.0, m=0.0, n=-1.5)),
            (Tg, Tr) in ((1.0, 0.1), (0.3, 2.0))
            s = dust_cell(; NG, opacity=op, Tgas=Tg, Trad=Tr, coefficient=1e12 / 100)
            r = solve_coupling(BracketedCoupling(tolerance=1e-12), s)
            plain = solve_coupling(BracketedCoupling(tolerance=1e-12), nodust(s))
            @test r.Egas ≈ plain.Egas rtol = 1e-11
            @test r.Tdust ≈ r.Egas / s.cv rtol = 1e-11
        end
    end

    @testset "no coupling leaves the gas alone" begin
        # K = 0: the gas is isolated, and the dust comes to radiative equilibrium
        # with the field. There is no regime switch for this; it is the same
        # equation with a zero coefficient.
        for NG in (1, 4), op in DUST_OPACITIES, (Tg, Tr) in ((1.0, 0.1), (0.3, 2.0))
            s = dust_cell(; NG, opacity=op, Tgas=Tg, Trad=Tr, coefficient=0.0)
            r = solve_coupling(BracketedCoupling(tolerance=1e-12), s)
            @test r.converged
            @test abs(r.Egas - s.Egas) <= 1e-12 * total_energy(s, s.Egas, s.Erad)
            # and the dust balance holds: the net exchange with the radiation is zero
            H, = dust_energy_balance(s, r.Tdust)
            @test abs(H) <= 1e-12 * total_energy(s, s.Egas, s.Erad)
        end
    end

    @testset "the march picks the root connected to the old state" begin
        # With kappa ~ T_d^2 the dust balance itself has three roots here, at
        # T_d = 0.0114, 0.134 and 0.738 -- equilibria of the model, not of the
        # discretisation. Starting from the gas temperature, as Quokka does,
        # returns the cold one.
        s = dust_cell(; NG=4, opacity=DUST_OPACITIES[2], Tgas=0.01, Trad=1.0, rho=1.0,
                      coefficient=1.0)
        r = solve_coupling(BracketedCoupling(), s)
        @test r.converged
        @test 0.0113 < r.Tdust < 0.0115
    end

    @testset "only BracketedCoupling solves it" begin
        dust = BateKetoDust(coefficient=1.0)
        for coupling in (QuokkaCoupling(), AdaptiveCoupling())
            @test_throws ArgumentError RadiationPhysics(; c=1.0, a_rad=1.0,
                                                        opacity=ConstantOpacity(1.0),
                                                        coupling, dust)
            @test_throws ArgumentError solve_coupling(coupling, dust_cell())
        end
        @test_throws ArgumentError BateKetoDust(coefficient=-1.0)
        @test_throws ArgumentError BateKetoDust(coefficient=1.0, particle_mass=0.0)
    end

    @testset "allocates nothing" begin
        run(s) = solve_coupling(BracketedCoupling(), s).Egas
        s = dust_cell()
        run(s)
        @test minimum(@allocated(run(s)) for _ in 1:3) == 0
    end

    @testset "RadDust against the error-controlled solve" begin
        # Quokka's RadDust (grey) and RadDustMG (four groups), with B = a T^4;
        # the reference integrates the model's own ODE. Compared over the whole
        # history, as Quokka's test does; Quokka's tolerance is 3e-3, and first
        # order in dt = 1e-8 gives 2e-4 here. With a constant opacity the answer
        # does not depend on the grouping.
        for groups in (Hydro3D.GREY, RadiationGroups([1e-3, 0.1, 1.0, 10.0, 1e3]))
            sim, tmax = P.rad_dust(; groups)
            ts = Float64[]
            Tg = Float64[]
            Er = Float64[]
            while sim.t < tmax * (1 - 1e-12)
                step!(sim, min(timestep(sim), tmax - sim.t))
                push!(ts, sim.t)
                push!(Tg, gas_temperature(sim)[1])
                push!(Er, first(field(sim, :Erad))[1])
            end
            Tref, Eref = P.rad_dust_reference(ts; groups)
            err = (sum(abs.(Tg .- Tref)) + sum(abs.(Er .- Eref))) /
                  (sum(abs.(Tref)) + sum(abs.(Eref)))
            @test err < 1e-3
            @test sim.diag.newton_failures == 0
            @test Tg[end] ≈ Tref[end] rtol = 1e-6
        end
    end
end
