# The error-controlled coupling solver, and the three-way agreement that is the
# reason the `AbstractCoupling` interface exists at all.
#
# This ran in a subprocess until OrdinaryDiffEq became a hard dependency, because
# `using OrdinaryDiffEq` would have shadowed `step!` for the whole session. The
# package now imports six names rather than the package, so this is an ordinary
# `include` from `runtests.jl` and `step!` still means what it means everywhere
# else in the suite. `cell_state` comes from `test_radiation_multigroup.jl`,
# which runs before this one.
#
# The two temperatures below are load-bearing, not decoration. `QuokkaCoupling`
# does not converge on the `n = -3` power-law opacity at `tau = 1e9` from
# `Tgas = 1.3, Trad = 0.5` -- it exhausts its hundred iterations and loses two
# thirds of the cell's energy -- while from `1.0, 0.3` it converges in six. That
# is a robustness limit of those base variables and is untouched by anything
# here; it is recorded so the next person to change a number knows one is being
# avoided rather than that none exists.
ref_cell(; kw...) = cell_state(; Tgas=1.0, Trad=0.3, kw...)

@testset "the three coupling solvers agree" begin
    ref = AdaptiveCoupling(reltol=1e-10, abstol=1e-14)

    @testset "deep in equilibrium (tau = 1e9)" begin
        # A step a hundred million times longer than the coupling time. All three
        # must land on the radiative equilibrium temperature of the cell, which is
        # also the only answer that conserves `E_gas + (c/chat) sum_g E_g`.
        for NG in (1, 4), law in (ConstantOpacity(1e9),
                                  PowerLawOpacity(kappa0=1e9, n=-3.0))
            s = ref_cell(; NG, opacity=law)
            E0 = s.Egas + sumvec(s.Erad)
            rs = [solve_coupling(m, s) for m in (QuokkaCoupling(), BracketedCoupling(), ref)]
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
            s = ref_cell(; NG, opacity=law)
            rs = [solve_coupling(m, s) for m in (QuokkaCoupling(), BracketedCoupling(), ref)]
            @test all(r.converged for r in rs)
            Ts = [r.Egas / s.cv for r in rs]
            @test maximum(Ts) - minimum(Ts) < 1e-9 * maximum(Ts)
            @test all(isapprox(T, s.Egas / s.cv; rtol=1e-4) for T in Ts)
        end
    end

    @testset "the one-step solvers converge onto the reference as dt shrinks" begin
        # Away from both limits the two backward-Euler solvers are not expected to
        # *agree* with the reference; they are expected to converge onto it as the
        # step shrinks. That is what makes `AdaptiveCoupling` a reference rather
        # than a fourth opinion.
        #
        # The rate is two, not one. `QuokkaCoupling` and `BracketedCoupling` are first
        # order as *integrators*, but here they take a single step across the whole
        # interval, and one backward-Euler step has a local error of `O(dt^2)`.
        # There is no accumulation over many steps to turn that into `O(dt)`.
        for NG in (1, 4), law in (ConstantOpacity(1e2), PowerLawOpacity(kappa0=1e2, n=-3.0))
            for method in (QuokkaCoupling(), BracketedCoupling())
                errs = Float64[]
                for dt in (1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5)
                    s = ref_cell(; NG, opacity=law, dt)
                    e = solve_coupling(ref, s).Egas
                    push!(errs, abs(solve_coupling(method, s).Egas - e) / abs(e))
                end
                @test issorted(errs; rev=true)
                @test 1.7 < log2(errs[end-1] / errs[end]) < 2.3
            end
        end
    end
end
