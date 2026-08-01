# Newton's method for a bordered ("arrowhead") nonlinear system, and the two
# linear solves it can use.
#
# Nothing here integrates anything or knows what the unknowns mean. The caller
# hands over a residual, its Jacobian, an opening iterate, a linear solve, a
# tolerance and an iteration cap, and gets back a root. That the residual happens
# to be `u - u0 - dt f(u)` -- one backward-Euler step -- is the caller's business
# and is written down in `coupling_solvers.jl`, which is why this stepper can be
# tuned, damped or replaced without touching a line of coupling physics.
#
# The one thing this file does assume is the *shape* of the system: `1 + NG`
# unknowns split as one scalar and one `SVec{NG}`, with a Jacobian that is dense
# in the first row and column and diagonal in the rest. That is a shape, not a
# physics -- it is what `solve_arrowhead` inverts in `O(NG)` -- and
# `AbstractJacobianSolve` below says which caller relies on it and what checks it.

"""
    solve_arrowhead(J00, J0g, Jg0, Jgg, F0, Fg) -> (dx0, dxg)

Solve the bordered ("arrowhead") linear system

    [ J00   J0g^T ] [dx0]   [F0]
    [ Jg0   diag  ] [dxg] = [Fg]

for the Newton update, by eliminating the group block into the gas row. Exact,
`O(NG)`, and forms no matrix -- which matters because this runs per cell, per
Newton step, per timestep.

A group with `Jgg[g] == 0` is decoupled from the gas (zero optical depth), and
contributes nothing to the elimination; its update is zero.
"""
@inline function solve_arrowhead(J00::T, J0g::SVec{NG,T}, Jg0::SVec{NG,T},
                                 Jgg::SVec{NG,T}, F0::T, Fg::SVec{NG,T}) where {NG,T}
    # dxg = (Fg - Jg0 dx0) / Jgg, substituted into J00 dx0 + sum(J0g dxg) = F0
    inv_d = SVec{NG,T}(ntuple(g -> iszero(Jgg[g]) ? zero(T) : one(T) / Jgg[g], Val(NG))...)
    a = J00 - sumvec(mul(J0g, mul(Jg0, inv_d)))
    b = F0 - sumvec(mul(J0g, mul(Fg, inv_d)))
    dx0 = iszero(a) ? zero(T) : b / a
    dxg = mul(Fg - dx0 * Jg0, inv_d)
    return (dx0, dxg)
end

"""
    AbstractJacobianSolve

How [`QuokkaCoupling`](@ref) and [`DirectCoupling`](@ref) solve the linear system
of one Newton step. Two are provided:

  - [`ArrowheadSolve`](@ref) -- eliminate the group block into the gas row.
    `O(NG)`, forms no matrix, allocates nothing. The default for both.
  - [`DenseSolve`](@ref)     -- assemble the Newton matrix and factorise it.
    `O(NG^3)`, allocates, assumes nothing. The reference.

The arrowhead is exact for the coupling as it stands, and it is not a small
saving: at four groups it is the difference between roughly 0.9 kB a call and
none, and between 25 and 70 per cent of the wall time of the whole cell solve
over `NG = 4` to `16`. That is why it is the default.

It is exact because of a *physical assumption*: groups exchange energy with the
gas and never directly with each other, so the group block of
[`CouplingJacobian`](@ref) is diagonal. That is a property of the terms in
[`coupling_rhs`](@ref) today, not a truth about radiation. Compton
redistribution between groups, or overlapping lines, would give that block
off-diagonal entries, and the elimination has nowhere to put them.

What makes it acceptable as a default is that the assumption is now *checked*.
`DenseSolve` solves the same system without it, and the suite asserts that the
two agree -- block by block on a matrix built from the real Jacobian, and end to
end through a converged cell. A default may encode an assumption when a checked
alternative exists; it may not when the assumption is silent, and until
`DenseSolve` existed this one was silent: both in-tree solvers called the
elimination, so they could agree with each other and both be wrong.

The cross-check has to exist because the failure would be quiet. Newton stops on
the residual of the true equations, so a damaged matrix cannot converge to a
wrong answer -- it can only fail to converge, and a cell that exhausts
`max_iterations` returns its last iterate and increments a counter rather than
stopping the run. What a broken elimination looks like from outside is a slower,
slightly wrong simulation with a nonzero `newton_failures`, which is exactly the
kind of symptom that gets blamed on the physics. Nothing about a run reports it;
a test has to.

`DenseSolve` stays the reference implementation, and is what to switch to the day
a term is added that couples groups directly. That day is not free there either
-- `rad_rad` is an `SVec` today, so the dense assembly writes a diagonal where it
would have to write a block -- but that is one line of assembly against an
elimination that has no place for the entries at all.

This choice belongs to [`newton_solve`](@ref), the hand-written stepper. The
other backward-Euler stepper, [`ImplicitEulerStepper`](@ref), brings its own
linear solve and ignores it; see [`AbstractBackwardEuler`](@ref).
"""
abstract type AbstractJacobianSolve end

"""
    DenseSolve()

Assemble the Newton matrix `I - dt J` as an ordinary `(1+NG)x(1+NG)` matrix and
solve it by LU factorisation (LinearAlgebra's `lu!`/`ldiv!`, in place on the
freshly built matrix and right-hand side).

It makes no use of the arrowhead structure, which is what makes it the
reference: it solves the system that [`ArrowheadSolve`](@ref) assumes something
about, so the two agreeing is a direct test of the assumption rather than of two
copies of it. Pass `linsolve = DenseSolve()` to either solver to run that
comparison, and switch to it for good if a term is added that couples groups
directly -- see [`AbstractJacobianSolve`](@ref).

**It allocates, and it must**: `NG` is a compile-time constant but a `Matrix` is
heap storage, so every Newton iteration of every cell buys a matrix, a
right-hand side and the factorisation's pivot vector. Measured: 272 bytes per
linear solve at one group, 464 at four, 976 at eight, 3.0 kB at sixteen, which
makes a converged three-iteration cell cost 0.54, 0.93, 1.9 and 5.9 kB
respectively. Wall time is 1.3 to 1.7 times [`ArrowheadSolve`](@ref) at four to
sixteen groups, and 4.7 times at one, where the linear solve is most of what a
grey cell does.

Nothing is on this path unless it is asked for: [`QuokkaCoupling`](@ref) defaults
to `ArrowheadSolve()`, which is exactly zero bytes per cell.
"""
struct DenseSolve <: AbstractJacobianSolve end

"""
    ArrowheadSolve()

Solve the Newton step with [`solve_arrowhead`](@ref): eliminate the diagonal
group block into the gas row, `O(NG)`, no matrix formed, nothing allocated. The
default for [`QuokkaCoupling`](@ref) and for [`DirectCoupling`](@ref)'s
hand-written stepper.

Exact while the coupling is the one in [`coupling_rhs`](@ref), where groups meet
only through the gas. [`AbstractJacobianSolve`](@ref) says what that assumption
is, what checks that it still holds, and what to do the day it stops.
"""
struct ArrowheadSolve <: AbstractJacobianSolve end

"""
    solve_jacobian(linsolve, J00, J0g, Jg0, Jgg, F0, Fg) -> (dx0, dxg)

Solve one Newton step's system, given the four blocks of the Newton matrix and
the two blocks of the right-hand side, in the layout [`solve_arrowhead`](@ref)
draws. Dispatches on an [`AbstractJacobianSolve`](@ref).

The two paths differ once the matrix is singular: the arrowhead treats a zero
group entry as a decoupled group and returns a zero update for it, while an LU
factorisation of the same matrix throws. Neither solver produces one.
[`DirectCoupling`](@ref)'s group entries are `1 + dt chat rho kappa_E >= 1`, and
[`QuokkaCoupling`](@ref) writes a large finite sentinel rather than `-Inf` for a
group of zero optical depth, so its matrix factorises too.
"""
@inline solve_jacobian(::ArrowheadSolve, J00::T, J0g::SVec{NG,T}, Jg0::SVec{NG,T},
                       Jgg::SVec{NG,T}, F0::T, Fg::SVec{NG,T}) where {NG,T} =
    solve_arrowhead(J00, J0g, Jg0, Jgg, F0, Fg)

function solve_jacobian(::DenseSolve, J00::T, J0g::SVec{NG,T}, Jg0::SVec{NG,T},
                        Jgg::SVec{NG,T}, F0::T, Fg::SVec{NG,T}) where {NG,T}
    A = zeros(T, NG + 1, NG + 1)
    b = Vector{T}(undef, NG + 1)
    @inbounds A[1, 1] = J00
    @inbounds b[1] = F0
    @inbounds for g in 1:NG
        A[1, 1+g] = J0g[g]
        A[1+g, 1] = Jg0[g]
        A[1+g, 1+g] = Jgg[g]      # the block the arrowhead assumes is diagonal
        b[1+g] = Fg[g]
    end
    # In place on both, so no third copy of the matrix is made; `lu!` still buys
    # its pivot vector, and that is counted in the figures quoted for
    # `DenseSolve`. `b` is never rebound, only written through, so the `ntuple`
    # below captures a value rather than a box.
    ldiv!(lu!(A), b)
    return (@inbounds(b[1]), SVec{NG,T}(ntuple(g -> @inbounds(b[1+g]), Val(NG))...))
end

# --- the iteration ------------------------------------------------------------

"""
    newton_solve(residual, jacobian, u0, linsolve, tolerance, max_iterations)
        -> (u, iterations, converged)

Newton's method on a bordered system of `1 + NG` unknowns, carried as a pair
`u = (x0, xg)` of one scalar and one [`SVec`](@ref) of `NG`.

  - `residual(u) -> (F0, Fg)` : the equations to zero, in the same layout.
  - `jacobian(u) -> (J00, J0g, Jg0, Jgg)` : `dF/du` in the four blocks
    [`solve_arrowhead`](@ref) draws -- scalar, row, column, diagonal.
  - `u0` : the opening iterate.
  - `linsolve` : an [`AbstractJacobianSolve`](@ref), how each step's system is
    inverted.
  - `tolerance` : *absolute*, tested against `max(|F0|, maxabs(Fg))`. Absolute
    rather than relative because scaling a residual is a statement about what the
    unknowns mean, and this function does not know; the caller multiplies by
    whatever its own scale is before handing the number over.
  - `max_iterations` : the cap. Reaching it returns the last iterate with
    `converged = false` rather than throwing, so one stubborn cell in a large
    grid shows up in a run's diagnostics instead of killing the run.

The residual is tested *before* the first step, so a state that already solves
its equations costs one evaluation and no linear solve. That is the common case
in an optically thin cell.

Allocation-free with `ArrowheadSolve` for any residual and Jacobian that are
themselves allocation-free: `u` is a tuple of `isbits` values, and both callables
are specialised on rather than boxed.

This is the whole of the hand-written backward-Euler stepper
([`NewtonStepper`](@ref)): what makes it *backward Euler* is that
[`DirectCoupling`](@ref) hands it `u - u0 - dt f(u)` and `I - dt J`. Nothing
here knows that, which is the point -- damping, a line search or a different
convergence test can be tried here without touching the coupling.
"""
function newton_solve(residual::F, jacobian::J, u0::Tuple{T,SVec{NG,T}},
                      linsolve::AbstractJacobianSolve, tolerance::T,
                      max_iterations::Int) where {F,J,NG,T}
    x0, xg = u0
    n = 0
    converged = false
    while n < max_iterations
        n += 1
        F0, Fg = residual((x0, xg))
        if max(abs(F0), maxabs(Fg)) <= tolerance
            converged = true
            break
        end
        J00, J0g, Jg0, Jgg = jacobian((x0, xg))
        dx0, dxg = solve_jacobian(linsolve, J00, J0g, Jg0, Jgg, -F0, -Fg)
        x0 += dx0
        xg += dxg
    end
    return ((x0, xg), n, converged)
end
