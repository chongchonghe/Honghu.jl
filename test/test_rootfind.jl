# `ZBrent`, the bracketing root-finder `BracketedCoupling` hands its equation to.
#
# Two properties distinguish it from the `Brent` in BracketingNonlinearSolve, and
# each has an assertion here that fails without it: it evaluates each point once,
# and it has the minimum step of Brent (1973) / Numerical Recipes' `zbrent`.

using Honghu: ZBrent, IntervalNonlinearProblem, solve
using .BracketingNonlinearSolve: Brent

"Solve `f(x) = 0` on `(lo, hi)`, recording every point `f` was evaluated at."
function traced_solve(f, lo, hi, alg; abstol=nothing)
    xs = Float64[]
    g = (x, p) -> (push!(xs, x); f(x))
    sol = solve(IntervalNonlinearProblem{false}(g, (lo, hi), nothing), alg; abstol)
    (sol, xs)
end

@testset "ZBrent" begin
    @testset "finds the root, evaluating each point once" begin
        for (lo, hi, nmax) in ((1.0, 2.0, 9), (0.0, 2.0, 10), (0.0, 10.0, 14))
            sol, xs = traced_solve(x -> x^2 - 2, lo, hi, ZBrent())
            @test sol.retcode == Honghu.ReturnCode.Success
            @test abs(sol.u - sqrt(2)) <= 2 * eps(sqrt(2))
            @test sol.left <= sqrt(2) <= sol.right       # the returned bracket brackets
            # No point is evaluated twice. BracketingNonlinearSolve's `Brent`
            # re-evaluated `f(c)` every iteration before version 1.12.8: on
            # [1, 2] it made 20 evaluations at 9 distinct points.
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
        # Measured: 12 evaluations with the minimum step, 50 without.
        sol, xs = traced_solve(x -> x^4 - 2, 0.0, 4.0, ZBrent(); abstol=5e-11)
        @test abs(sol.u - 2^0.25) <= 1e-10
        @test length(xs) <= 14
        # ... and it is the minimum step that the library's `Brent` was missing,
        # not just its repeated evaluations: without it, far more distinct points.
        # BracketingNonlinearSolve 1.12.8 took this method upstream
        # (SciML/NonlinearSolve.jl#1340), and from then on the two must agree;
        # the release a fresh install resolves to decides which is checked.
        _, ys = traced_solve(x -> x^4 - 2, 0.0, 4.0, Brent(); abstol=5e-11)
        if pkgversion(BracketingNonlinearSolve) >= v"1.12.8"
            @test length(unique(ys)) <= 14
        else
            @test length(unique(ys)) > 2 * length(xs)
        end
    end

    @testset "edge cases" begin
        @test solve(IntervalNonlinearProblem{false}((x, p) -> x - 1, (1.0, 2.0), nothing),
                    ZBrent()).retcode == Honghu.ReturnCode.ExactSolutionLeft
        @test solve(IntervalNonlinearProblem{false}((x, p) -> x^2 - 2, (2.0, 3.0), nothing),
                    ZBrent()).retcode == Honghu.ReturnCode.InitialFailure
        # generic in the number type
        s = solve(IntervalNonlinearProblem{false}((x, p) -> x^2 - p, (big"1.0", big"2.0"), big"2.0"),
                  ZBrent(); abstol=big"1e-60")
        @test abs(s.u - sqrt(big"2")) < big"1e-60"
    end

    @testset "allocates nothing" begin
        f(x, p) = x^2 - p
        run(lo, hi, p) = solve(IntervalNonlinearProblem{false}(f, (lo, hi), p), ZBrent();
                               abstol=1e-12).u
        run(1.0, 2.0, 2.0)
        @test minimum(@allocated(run(1.0, 2.0, 2.0)) for _ in 1:3) == 0
    end
end
