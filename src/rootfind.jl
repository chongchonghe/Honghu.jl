# A bracketing root-finder for one scalar equation, and nothing else.
#
# Nothing here knows about radiation. `BracketedCoupling` reduces a coupling step
# to one scalar equation and hands it to whatever bracketing method it is given;
# this file supplies the default. It is written against SciMLBase's interface --
# `solve(IntervalNonlinearProblem(f, (lo, hi), p), alg; abstol, maxiters)` -- so it
# is interchangeable with the methods in BracketingNonlinearSolve (`ModAB`, `ITP`,
# `Bisection`, ...) and the caller cannot tell which it was handed.
#
# It exists because the `Brent` in BracketingNonlinearSolve 1.12 is not Brent's
# method as Brent (1973) wrote it, and the differences cost this code evaluations
# of an equation whose every evaluation is a Planck integral. See `ZBrent`.

"""
    ZBrent()

Brent's method for a root of a scalar function on a bracket, as Brent (1973)
wrote it and as Numerical Recipes transcribes it in `zbrent`: inverse quadratic
interpolation or a secant step when either would shrink the bracket fast enough,
bisection otherwise, and a **minimum step**. The default `root_finder` of
[`BracketedCoupling`](@ref).

Use it through SciMLBase's interface, exactly like a BracketingNonlinearSolve
method:

```julia
sol = solve(IntervalNonlinearProblem{false}(f, (lo, hi), p), ZBrent();
            abstol = 1e-12, maxiters = 100)
sol.u, sol.left, sol.right
```

`f(x, p)` must change sign on `(lo, hi)`. It stops when the half-width of the
bracket is at most `abstol + 2 eps |x|` -- the same width criterion the
BracketingNonlinearSolve methods use, plus the round-off floor below which no
bracket can shrink -- and returns the best estimate as `u` and the final bracket
as `left`, `right`. With `abstol = nothing` it runs to that round-off floor.

**Why not `BracketingNonlinearSolve.Brent`.** That implementation transcribes a
textbook pseudocode rather than Brent's algorithm, and differs in two ways, both
measured on `BracketedCoupling`'s equation over the 240-cell sweep of
`docs/coupling-new-method.md`:

  - **It has no minimum step.** When an interpolated step would move the
    estimate by less than `tol1 = 2 eps |b| + abstol`, Brent's algorithm moves it
    by `tol1` towards the other end of the bracket instead. Without that, once the
    interpolated iterates all land on the same side of the root -- which a smooth
    convex function makes routine -- the far end of the bracket is never replaced
    except by bisection, and the method degrades towards bisection's cost. On the
    sweep, the same algorithm with the minimum step removed took 13.0 evaluations
    on average and 61 in the worst cell; with it, 7.7 and 12.
  - **It evaluates the function twice per iteration.** It recomputes `f(c)` at
    the top of every iteration, where `c` is always an earlier point whose value
    is already known. On `x^2 - 2` from `[1, 2]` it makes 20 evaluations at 9
    distinct points. This method evaluates each point once.

Together: 20.1 evaluations on average and 70 in the worst cell there, against
7.7 and 12 here -- fewer than `ModAB`'s 9.5 and 12.

A third report, that with FMA contraction the missing minimum step makes
`x^2 - 2` take 37 evaluations instead of 8, is about compiled C and CUDA; Julia
does not contract `x^2 - 2` by default, and on that function the minimum step
makes no difference here. On `BracketedCoupling`'s equation it does, as above.

Brent, R. P. (1973), *Algorithms for Minimization without Derivatives*, ch. 4;
Press et al., *Numerical Recipes*, 3rd ed., section 9.3.
"""
struct ZBrent <: SciMLBase.AbstractNonlinearAlgorithm end

function SciMLBase.solve(prob::IntervalNonlinearProblem, alg::ZBrent;
                         abstol=nothing, maxiters::Integer=1000, kwargs...)
    f, p = prob.f, prob.p
    a, b = promote(float.(prob.tspan)...)
    T = typeof(b)
    tol = abstol === nothing ? zero(T) : T(2 * abstol)   # full width; see tol1 below
    fa, fb = f(a, p), f(b, p)

    iszero(fa) && return _zbrent_solution(prob, alg, a, fa, a, a, ReturnCode.ExactSolutionLeft)
    iszero(fb) && return _zbrent_solution(prob, alg, b, fb, b, b, ReturnCode.ExactSolutionRight)
    signbit(fa) == signbit(fb) &&
        return _zbrent_solution(prob, alg, a, fa, a, b, ReturnCode.InitialFailure)

    # `b` is the best estimate, `c` the contrapoint on the other side of the root,
    # `a` the previous `b`. `d` is the step just taken and `e` the one before.
    c, fc = b, fb
    d = e = b - a
    for _ in 1:maxiters
        # keep the root between b and c
        if signbit(fb) == signbit(fc)
            c, fc = a, fa
            d = e = b - a
        end
        # keep b the better of the two
        if abs(fc) < abs(fb)
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        end

        tol1 = 2 * eps(T) * abs(b) + tol / 2
        xm = (c - b) / 2
        (abs(xm) <= tol1 || iszero(fb)) &&
            return _zbrent_solution(prob, alg, b, fb, min(b, c), max(b, c), ReturnCode.Success)

        if abs(e) >= tol1 && abs(fa) > abs(fb)
            # interpolate: secant if only two distinct points, else inverse quadratic
            s = fb / fa
            if a == c
                P = 2 * xm * s
                Q = one(T) - s
            else
                q = fa / fc
                r = fb / fc
                P = s * (2 * xm * q * (q - r) - (b - a) * (r - one(T)))
                Q = (q - one(T)) * (r - one(T)) * (s - one(T))
            end
            P > 0 && (Q = -Q)
            P = abs(P)
            # accept it only if it stays inside the bracket and shrinks faster
            # than the step before last
            if 2 * P < min(3 * xm * Q - abs(tol1 * Q), abs(e * Q))
                e = d
                d = P / Q
            else
                d = xm
                e = d
            end
        else
            d = xm
            e = d
        end

        a, fa = b, fb
        # The minimum step: never move by less than tol1, and when the step would
        # be smaller, move by tol1 towards the contrapoint. This is what forces the
        # far end of the bracket to be replaced when every interpolated iterate
        # lands on the same side of the root.
        b += abs(d) > tol1 ? d : copysign(tol1, xm)
        fb = f(b, p)
    end
    return _zbrent_solution(prob, alg, b, fb, min(b, c), max(b, c), ReturnCode.MaxIters)
end

@inline _zbrent_solution(prob, alg, u, fu, left, right, retcode) =
    SciMLBase.build_solution(prob, alg, u, fu; retcode, left, right)
