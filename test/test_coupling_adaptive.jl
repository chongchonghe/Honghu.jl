# The third coupling solver, and the three-way agreement that is the reason the
# `AbstractCoupling` interface exists at all.
#
# This file is NOT included by `runtests.jl` directly. `AdaptiveCoupling`'s
# method lives in a package extension, so testing it means `using OrdinaryDiffEq`
# -- which also exports `step!` and would make that name ambiguous for every
# other test file in the session. `runtests.jl` therefore runs this one in a
# subprocess, and only when OrdinaryDiffEq is installed; the same arrangement,
# and the same reason, as the verification figures.
#
# Run it directly with
#
#     julia --project=<env-with-OrdinaryDiffEq> test/test_coupling_adaptive.jl

using Test
using Hydro3D
using Hydro3D: CouplingState, SVec, solve_coupling, sumvec, GREY

using OrdinaryDiffEq

"One cell's `CouplingState`, out of equilibrium, in code units."
function cell_state(; NG=4, opacity, Tgas=1.0, Trad=0.3, rho=1.0, cv=1.0, dt=1e-3,
                    c=1.0, chat=1.0, arad=1.0, kB=1.0)
    grp = NG == 1 ? GREY : RadiationGroups(0.1, 20.0, NG)
    Erad = (arad * Trad^4) * planck_fractions(grp, Trad, kB)
    z = zero(SVec{NG,Float64})
    CouplingState{NG,NG + 1,Float64,typeof(opacity)}(cv * Tgas, Erad, z, z, rho, cv,
                                                     opacity, dt, c, chat, arad, kB, grp,
                                                     1e-10, 1e-30)
end

@testset "the three coupling solvers agree" begin
    ref = AdaptiveCoupling(reltol=1e-10, abstol=1e-14)

    @testset "deep in equilibrium (tau = 1e9)" begin
        # A step a hundred million times longer than the coupling time. All three
        # must land on the radiative equilibrium temperature of the cell, which is
        # also the only answer that conserves `E_gas + (c/chat) sum_g E_g`.
        for NG in (1, 4), law in (ConstantOpacity(1e9),
                                  PowerLawOpacity(kappa0=1e9, n=-3.0))
            s = cell_state(; NG, opacity=law)
            E0 = s.Egas + sumvec(s.Erad)
            rs = [solve_coupling(m, s) for m in (QuokkaCoupling(), DirectCoupling(), ref)]
            @test all(r.converged for r in rs)
            Ts = [r.Egas / s.cv for r in rs]
            @test maximum(Ts) - minimum(Ts) < 1e-6 * maximum(Ts)
            for r in rs
                @test isapprox(r.Egas + sumvec(r.Erad), E0; rtol=1e-9)
                @test isapprox(sumvec(r.Erad), (r.Egas / s.cv)^4; rtol=1e-5)
            end
        end
    end

    @testset "optically thin (tau = 1e-6)" begin
        # Nothing can be exchanged in a step this short, so all three must leave
        # the cell essentially alone -- and agree on the little that does move.
        for NG in (1, 4), law in (ConstantOpacity(1e-3),
                                  PowerLawOpacity(kappa0=1e-3, n=2.0))
            s = cell_state(; NG, opacity=law)
            rs = [solve_coupling(m, s) for m in (QuokkaCoupling(), DirectCoupling(), ref)]
            @test all(r.converged for r in rs)
            Ts = [r.Egas / s.cv for r in rs]
            @test maximum(Ts) - minimum(Ts) < 1e-9 * maximum(Ts)
            @test all(isapprox(T, 1.0; rtol=1e-4) for T in Ts)
        end
    end

    @testset "the one-step solvers converge onto the reference as dt shrinks" begin
        # Away from both limits the two backward-Euler solvers are not expected to
        # *agree* with the reference; they are expected to converge onto it as the
        # step shrinks. That is what makes `AdaptiveCoupling` a reference rather
        # than a fourth opinion.
        #
        # The rate is two, not one. `QuokkaCoupling` and `DirectCoupling` are first
        # order as *integrators*, but here they take a single step across the whole
        # interval, and one backward-Euler step has a local error of `O(dt^2)`.
        # There is no accumulation over many steps to turn that into `O(dt)`.
        for NG in (1, 4), law in (ConstantOpacity(1e2), PowerLawOpacity(kappa0=1e2, n=-3.0))
            for method in (QuokkaCoupling(), DirectCoupling())
                errs = Float64[]
                for dt in (1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5)
                    s = cell_state(; NG, opacity=law, dt)
                    e = solve_coupling(ref, s).Egas
                    push!(errs, abs(solve_coupling(method, s).Egas - e) / abs(e))
                end
                @test issorted(errs; rev=true)
                @test 1.7 < log2(errs[end-1] / errs[end]) < 2.3
            end
        end
    end
end
