# The bracketing root-finder `BracketedCoupling` hands its equation to by default:
# BracketingNonlinearSolve's `Brent`.
#
# Two properties make it the right default, and each has an assertion here that
# fails without it: it evaluates each point once, and it has the minimum step of
# Brent (1973) / Numerical Recipes' `zbrent`. The library's `Brent` had neither
# before 1.12.8, and an in-tree `ZBrent` stood in for it until
# SciML/NonlinearSolve.jl#1340 took the method upstream; `Project.toml` requires
# 1.12.8, and these tests are what would notice if a later release lost either.

using Honghu: IntervalNonlinearProblem, solve
using SciMLBase: ReturnCode
using BracketingNonlinearSolve: Brent

"Solve `f(x) = 0` on `(lo, hi)`, recording every point `f` was evaluated at."
function traced_solve(f, lo, hi, alg; abstol=nothing)
    xs = Float64[]
    g = (x, p) -> (push!(xs, x); f(x))
    sol = solve(IntervalNonlinearProblem{false}(g, (lo, hi), nothing), alg; abstol)
    (sol, xs)
end

@testset "the default root-finder, Brent" begin
    @testset "finds the root, evaluating each point once" begin
        # At a positive tolerance, as `BracketedCoupling` always passes one. These
        # are the counts the in-tree `ZBrent` made, and the library now visits
        # the same points and returns the same root to the last bit.
        for (lo, hi, nmax) in ((1.0, 2.0, 9), (0.0, 2.0, 10), (0.0, 10.0, 14))
            sol, xs = traced_solve(x -> x^2 - 2, lo, hi, Brent(); abstol=1e-14)
            @test sol.retcode == ReturnCode.Success
            @test abs(sol.u - sqrt(2)) <= 1e-14
            @test sol.left <= sqrt(2) <= sol.right       # the returned bracket brackets
            # No point is evaluated twice. Before 1.12.8 the library's `Brent`
            # re-evaluated `f(c)` every iteration: on [1, 2] it made 20
            # evaluations at 9 distinct points.
            @test length(unique(xs)) == length(xs)
            @test length(xs) <= nmax
        end
    end

    @testset "the minimum step" begin
        # When an interpolated step would move the estimate by less than
        # `tol1 = 2 eps |b| + abstol`, the method moves it by `tol1` towards the
        # contrapoint instead. On a convex function the interpolated iterates all
        # land on one side of the root, and without that step the far end of the
        # bracket is only ever replaced by bisection. `x^4 - 2` from [0, 4] is the
        # clean case -- the same `T^4` shape as the coupling's energy balance.
        # Measured: 12 evaluations with the minimum step, 50 without, and 62 for
        # the library's `Brent` before 1.12.8.
        sol, xs = traced_solve(x -> x^4 - 2, 0.0, 4.0, Brent(); abstol=5e-11)
        @test abs(sol.u - 2^0.25) <= 1e-10
        @test length(xs) <= 14
    end

    @testset "allocates nothing" begin
        f(x, p) = x^2 - p
        run(lo, hi, p) = solve(IntervalNonlinearProblem{false}(f, (lo, hi), p), Brent();
                               abstol=1e-12).u
        run(1.0, 2.0, 2.0)
        @test minimum(@allocated(run(1.0, 2.0, 2.0)) for _ in 1:3) == 0
    end
end
