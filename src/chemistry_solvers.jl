# The two chemistry integrators, written against `chem_rhs!` and `chem_jacobian!`.
#
#   `RosenbrockChem`     -- an adaptive stiff integrator from OrdinaryDiffEq, one
#                           reused integrator per workspace. The default: it takes
#                           as many internal steps as the cell needs.
#   `BackwardEulerChem`  -- one backward-Euler step, Newton on the same right-hand
#                           side and Jacobian. No adaptivity and no sub-cycling; if
#                           Newton does not converge it says so. The comparison,
#                           and the only chemistry path that allocates nothing.
#
# Neither restates any physics, so they integrate the same problem and can be
# checked against each other. Which of them runs is the user's choice.

# --- the Rosenbrock integrator ------------------------------------------------

# What the ODE functions read through the integrator's parameter slot. Mutable so
# that the one integrator a workspace holds is pointed at a new cell by assigning
# `cell`, not by building a new problem: `ChemCell` is a bit-type, so the
# assignment allocates nothing.
mutable struct ChemParams{N,T<:AbstractFloat}
    net::N
    cell::ChemCell{T}
end

_chem_ode!(du, u, p::ChemParams, _) = (chem_rhs!(du, u, p.net, p.cell); nothing)
_chem_jac!(J, u, p::ChemParams, _) = (chem_jacobian!(J, u, p.net, p.cell); nothing)

# The network is autonomous: its rates depend on the state and the cell's
# constants, never on `t`, so `df/dt` is exactly zero. Saying so keeps the
# integrator from differentiating the right-hand side with respect to `t` by any
# other means, as `_coupling_tgrad!` does for the coupling.
function _chem_tgrad!(dT, u, p::ChemParams, _)
    fill!(dT, zero(eltype(dT)))
    return nothing
end

# One integrator, created once and re-aimed at each cell with `reinit!`. Creating
# one per cell costs about 8 kB and 19 us; reusing it costs about 1 kB and 11 us
# (design spec, section 5, measured on a three-unknown stiff cell), so the
# difference is most of the allocation and a good part of the time.
#
# `abstol` is the integrator's *own* vector, not the one handed to `init`:
# `init` copies it, so a reference kept to the original would be updated per cell
# and never read. The solve overwrites it in place instead of re-initialising,
# because a new integrator per cell is what this struct exists to avoid.
struct RosenbrockWork{I,P,T<:AbstractFloat}
    integ::I
    p::P
    scale::Vector{T}
    abstol::Vector{T}
end

"""
    RosenbrockChem(; alg = Rosenbrock23(linsolve = SimpleLUFactorization()),
                   reltol = 1e-2, abstol = 1e-8)

Integrate the chemistry of each cell over the step with an adaptive, error-controlled
stiff ODE integrator from OrdinaryDiffEq. This is the default chemistry integrator.
It takes as many internal steps as the cell needs, so a cell that changes little
costs a few and a cell on the ionization front costs more, and it needs no Newton
iteration to converge before it can take a step.

  - `alg`    : the algorithm. Any stiff OrdinaryDiffEq algorithm works: `Rodas5P()`
    and `FBDF()` come with the package, and anything from a sub-package you load
    yourself does too. An explicit method will stall. It is handed the network's
    exact Jacobian and an exact `df/dt = 0`, so it never differentiates the
    right-hand side numerically. The default solves its linear systems with
    LinearSolve's `SimpleLUFactorization`, a dense LU written in Julia, and not with
    the integrator's own default, which calls OpenBLAS: OpenBLAS takes a global lock
    to allocate its buffers on every call, so a threaded chemistry step ran slower
    than a single-threaded one (a 24^3 step took 85 ms on one thread and 184 ms on
    four, against 72 ms and 20 ms with this one). An algorithm you pass keeps its own
    default; give it `linsolve = SimpleLUFactorization()` (from LinearSolve, which you
    load yourself) to get the same.
  - `reltol` : the relative tolerance. The default `1e-2` is deliberately loose,
    as for [`AdaptiveCoupling`](@ref).
  - `abstol` : the absolute tolerance, as a **fraction of [`chem_scales!`](@ref)**: the
    integrator is given `abstol * scale[i]` for unknown `i`, so it means the same in
    any unit system and a species with a negligible share of the cell does not
    drive the step size to zero on its own relative error.

OrdinaryDiffEq's own warnings are switched off (`verbose = SciMLLogging.None()`).
A cell that fails is reported through `ChemStats.converged`, which the grid
driver counts and summarises, and a threaded run would otherwise print one long
warning for every failing cell.

It allocates about a kilobyte a call, all inside OrdinaryDiffEq. The workspace
holds one reused integrator, so give each thread its own and never share one.

```julia
solver = RosenbrockChem(alg = Rodas5P(), reltol = 1e-4)
```
"""
struct RosenbrockChem{A,T<:AbstractFloat} <: AbstractChemSolver
    alg::A
    reltol::T
    abstol::T
end

RosenbrockChem(; alg=Rosenbrock23(linsolve=SimpleLUFactorization()), reltol=1e-2, abstol=1e-8) =
    RosenbrockChem(alg, promote(float(reltol), float(abstol))...)

function chem_workspace(s::RosenbrockChem, net::AbstractNetwork, ::Type{T}) where {T<:AbstractFloat}
    n = nunknowns(net)
    p = ChemParams(net, ChemCell(one(T), one(T), one(T), one(T)))
    f = ODEFunction{true}(_chem_ode!; jac=_chem_jac!, tgrad=_chem_tgrad!)
    prob = ODEProblem{true}(f, ones(T, n), (zero(T), one(T)), p)
    # The initial values are placeholders. Every solve re-aims the integrator with
    # `reinit!`, which also chooses the first step afresh for the new cell, so a
    # cell's answer does not depend on which cell came before it; the test of a
    # reused workspace pins that.
    # `dtmax` defaults to the length of the problem's `tspan`, which is the placeholder
    # (0, 1) here, and `reinit!` does not raise it: left alone, no step could exceed one
    # second whatever the cell's `dt`, and a hydrodynamic step of 1e10 s would not finish.
    integ = init(prob, s.alg; reltol=T(s.reltol), abstol=fill(T(s.abstol), n),
                 dtmax=floatmax(T), save_everystep=false, dense=false,
                 verbose=SciMLLogging.None())
    return RosenbrockWork(integ, p, ones(T, n), integ.opts.abstol)
end

function solve_chemistry!(w::RosenbrockWork{I,P,T}, s::RosenbrockChem, net::AbstractNetwork,
                          cell::ChemCell{T}, u::Vector{T}) where {I,P,T}
    integ = w.integ
    w.p.cell = cell
    chem_scales!(w.scale, u, net, cell)
    atol = T(s.abstol)
    @inbounds for i in eachindex(w.abstol)
        w.abstol[i] = atol * w.scale[i]
    end
    # The integrator's statistics accumulate across `reinit!`, so what this cell
    # cost is the difference.
    n0 = integ.stats.naccept + integ.stats.nreject
    reinit!(integ, u; t0=zero(T), tf=cell.dt)
    solve!(integ)
    copyto!(u, integ.u)
    return ChemStats(integ.stats.naccept + integ.stats.nreject - n0,
                     successful_retcode(integ.sol.retcode))
end

# --- the backward-Euler integrator --------------------------------------------

struct BackwardEulerWork{T<:AbstractFloat}
    u0::Vector{T}       # the state at the start of the step
    f::Vector{T}        # the right-hand side at the current iterate
    du::Vector{T}       # the right-hand side of the linear system, then the Newton update
    scale::Vector{T}
    J::Matrix{T}        # the Jacobian, then the matrix I - dt J
    ut::Vector{T}       # the trial iterate of the line search
    ft::Vector{T}       # the right-hand side at the trial iterate
end

"""
    BackwardEulerChem(; tol = 1e-10, max_iterations = 50)

Integrate the chemistry of each cell with one backward-Euler step over the whole
`dt`: Newton's method on `G(u) = u - u0 - dt f(u)` with the matrix `I - dt J`,
using the network's right-hand side and exact Jacobian. There is no adaptivity
and no sub-cycling, so the step is first order in `dt` and is as accurate as one
step of that length can be. It is the comparison for [`RosenbrockChem`](@ref),
and it allocates nothing.

Species and band unknowns are clamped at zero after each Newton update, so an
iterate never carries a negative abundance into a rate. The Newton update is
damped by a backtracking line search on the scaled residual, `max_i |G_i| / scale_i`:
it tries the whole update, then a half, a quarter and so on up to 20 halvings, and
keeps the first trial, clamped as above, whose residual is smaller than the
current one, or the whole update if none is. It is still one backward-Euler step:
the line search changes how the nonlinear equation of that step is solved, never
the equation.

Newton can still fail, and it is the one-step equation and not the damping that is
at fault. Three cases are known. (a) A stiff growth term, such as collisional
ionization with `beta n_H0 dt > 1`: the backward-Euler root is positive, but the
Newton direction from the old state points into the zero clamp, which a residual
line search cannot cross. (b) Photoionization-front cells at `Gamma dt` of order
`1e2` to `1e3`, where the temperature and the clamps interact inside the step.
(c) A discontinuous rate, such as the 100 K switch of the recombination cooling,
can make the iteration cycle across it. Every such cell is reported through
`ChemStats.converged`, which the grid driver counts, and is never hidden; if Newton
does not converge in `max_iterations`, or meets a singular matrix or a non-finite
update, the result says so and `u` holds the last iterate.

  - `tol`            : stop when the largest Newton update, or the largest residual
    `G`, in units of [`chem_scales!`](@ref), is at most this. The scales are taken
    once, at the state the step starts from. The residual test is what ends a
    solve whose update is stuck a rounding error above `tol`.
  - `max_iterations` : the most Newton iterations to take.

`ChemStats.steps` is the number of Newton iterations taken, 0 if the start state
already satisfies the equation to `tol`.
"""
struct BackwardEulerChem{T<:AbstractFloat} <: AbstractChemSolver
    tol::T
    max_iterations::Int
end

BackwardEulerChem(; tol=1e-10, max_iterations=50) =
    BackwardEulerChem(float(tol), Int(max_iterations))

function chem_workspace(::BackwardEulerChem, net::AbstractNetwork, ::Type{T}) where {T<:AbstractFloat}
    n = nunknowns(net)
    return BackwardEulerWork{T}(zeros(T, n), zeros(T, n), zeros(T, n), ones(T, n), zeros(T, n, n),
                                zeros(T, n), zeros(T, n))
end

# Solve `A x = b` in place, `x` overwriting `b` and `A` destroyed, by Gaussian
# elimination with partial pivoting. Returns `false` if a pivot is zero or not
# finite, which is a singular or broken matrix and not a state to carry on from.
# `lu!` is not used because it allocates its pivot vector, and this path is
# meant to allocate nothing; `n` is the number of unknowns, a handful, so there is
# nothing for a blocked factorisation to win.
function _gauss_solve!(A::Matrix{T}, b::Vector{T}) where {T}
    n = length(b)
    @inbounds for k in 1:n
        p = k
        pmax = abs(A[k, k])
        for i in k+1:n
            a = abs(A[i, k])
            if a > pmax
                pmax = a
                p = i
            end
        end
        (pmax > zero(T) && isfinite(pmax)) || return false
        if p != k
            for j in k:n
                A[k, j], A[p, j] = A[p, j], A[k, j]
            end
            b[k], b[p] = b[p], b[k]
        end
        piv = A[k, k]
        for i in k+1:n
            m = A[i, k] / piv
            m == zero(T) && continue
            for j in k+1:n
                A[i, j] -= m * A[k, j]
            end
            b[i] -= m * b[k]
        end
    end
    @inbounds for i in n:-1:1
        s = b[i]
        for j in i+1:n
            s -= A[i, j] * b[j]
        end
        b[i] = s / A[i, i]
    end
    return true
end

# The largest `|G_i| / scale_i` of `G(u) = u - u0 - dt f(u)`, given `f = f(u)`. NaN-propagating
# on purpose: `max` returns NaN if either argument is, so a broken iterate is seen as one.
function _be_residual(u::Vector{T}, u0::Vector{T}, f::Vector{T}, scale::Vector{T}, dt::T) where {T}
    r = zero(T)
    @inbounds for i in eachindex(u)
        r = max(r, abs(u[i] - u0[i] - dt * f[i]) / scale[i])
    end
    return r
end

# Twenty halvings reach a factor of 1e-6 of the Newton update: past that the trial is the
# current iterate to within round-off in anything a rate can see.
const _BE_MAX_HALVINGS = 20

# Put the trial `u + lambda du` in `w.ut`, clamped exactly as an iterate is (the energy may go
# negative in a bad trial and recover; a number density or a photon count may not, because the
# rates read it), its right-hand side in `w.ft`, and return its scaled residual.
function _be_trial!(w::BackwardEulerWork{T}, u::Vector{T}, net::AbstractNetwork, cell::ChemCell{T},
                    lambda::T) where {T}
    ne = ienergy(net)
    @inbounds for i in eachindex(u)
        v = u[i] + lambda * w.du[i]
        w.ut[i] = i == ne ? v : max(v, zero(T))
    end
    chem_rhs!(w.ft, w.ut, net, cell)
    return _be_residual(w.ut, w.u0, w.ft, w.scale, cell.dt)
end

function solve_chemistry!(w::BackwardEulerWork{T}, s::BackwardEulerChem, net::AbstractNetwork,
                          cell::ChemCell{T}, u::Vector{T}) where {T}
    n = length(u)
    dt = cell.dt
    copyto!(w.u0, u)
    chem_scales!(w.scale, u, net, cell)
    chem_rhs!(w.f, u, net, cell)
    r = _be_residual(u, w.u0, w.f, w.scale, dt)
    r <= s.tol && return ChemStats(0, true)
    for iter in 1:s.max_iterations
        chem_jacobian!(w.J, u, net, cell)
        # `du` is -G(u) on the way in and the Newton update on the way out; the
        # matrix is I - dt J.
        @inbounds for j in 1:n
            for i in 1:n
                w.J[i, j] = -dt * w.J[i, j]
            end
            w.J[j, j] += one(T)
            w.du[j] = w.u0[j] + dt * w.f[j] - u[j]
        end
        _gauss_solve!(w.J, w.du) || return ChemStats(iter, false)
        worst = zero(T)
        @inbounds for i in 1:n
            worst = max(worst, abs(w.du[i]) / w.scale[i])
        end
        isfinite(worst) || return ChemStats(iter, false)
        # Backtrack on the residual. If no trial lowers it, take the whole Newton update, as an
        # undamped iteration would: it is often the right move through a stretch where the
        # residual rises before it falls, and stalling on a tiny step is worse. Either way the
        # iteration moves, and `max_iterations` ends it.
        lambda = one(T)
        rt = r
        found = false
        for _ in 0:_BE_MAX_HALVINGS
            rt = _be_trial!(w, u, net, cell, lambda)
            if rt < r
                found = true
                break
            end
            lambda /= 2
        end
        found || (rt = _be_trial!(w, u, net, cell, one(T)))
        copyto!(u, w.ut)
        copyto!(w.f, w.ft)
        r = rt
        isfinite(r) || return ChemStats(iter, false)
        (worst <= s.tol || r <= s.tol) && return ChemStats(iter, true)
    end
    return ChemStats(s.max_iterations, false)
end
