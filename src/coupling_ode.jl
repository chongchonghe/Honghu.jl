# The two coupling solvers that are backed by OrdinaryDiffEq, and the ODE
# description they share.
#
#   `ImplicitEulerStepper` -- one non-adaptive step of `ImplicitEuler`, which is
#                             backward Euler and nothing else. `DirectCoupling`'s
#                             default stepper.
#   `AdaptiveCoupling`     -- an error-controlled stiff integrator over the same
#                             step. A reference solver: it says what the coupling
#                             ODE's answer *is*, so that the one-step solvers can
#                             be checked against a controlled error rather than an
#                             assumed one.
#
# Both are written against the same `_coupling_ode!` / `_coupling_jac!` /
# `_coupling_tgrad!` triple below, and that triple is written against
# `coupling_rhs` and `coupling_jacobian`, so nothing here restates any physics.
#
# This used to be a package extension, loaded only if the user happened to have
# an ODE stack installed. OrdinaryDiffEq is now a hard dependency, so the split
# is gone: there is no fallback to write, no `MethodError` to dress up, and
# nothing to skip in the test suite -- which is what the arrangement was actually
# costing, since the extension's tests skipped themselves on any checkout that
# did not have the package, and went unexercised without complaining.

# --- the ODE, once, for both solvers ------------------------------------------

# The coupling ODE, written in place so that the right-hand side allocates
# nothing however many times an integrator calls it.
#
# The state is a plain `Vector{T}` of length `1 + NG`, not an `SVector`: a
# static vector would mean a StaticArrays dependency, which CLAUDE.md's
# dependency policy already declined once for `SVec`, and it would buy nothing
# here because `solve` allocates its own caches regardless. That allocation is
# inherent to going through a general integrator, and is the price both solvers
# in this file charge.
#
# `src` and `work` are increments over the *whole* step, held constant, which is
# exactly what the hand-written backward-Euler stepper adds to its residual.
# Dividing them by `dt` here and integrating over `[0, dt]` returns them
# unchanged, so every solver demonstrably integrates the same problem.
function _coupling_ode!(du, u, s::CouplingState{NG,M,T}, _) where {NG,M,T}
    Erad = SVec{NG,T}(ntuple(g -> @inbounds(u[1+g]), Val(NG))...)
    dEgas, dErad = coupling_rhs(s, @inbounds(u[1]), Erad)
    invdt = one(T) / s.dt
    @inbounds du[1] = dEgas - dot(gas_scale(s), s.work) * invdt
    @inbounds for g in 1:NG
        du[1+g] = dErad[g] + (s.src[g] + s.work[g]) * invdt
    end
    return nothing
end

# The exact Jacobian, the same one `DirectCoupling` builds its Newton matrix
# from. Dense `(1+NG)^2`, because that is what an integrator factorises and NG
# is small; the arrowhead structure is still there (every off-diagonal entry of
# the group block is zero) but exploiting it would mean a sparse prototype and a
# specialised factorisation, which for `1 + NG <= 10` costs more than it saves.
#
# `src` and `work` are constants of the step, so they do not appear: they
# differentiate away.
function _coupling_jac!(J, u, s::CouplingState{NG,M,T}, _) where {NG,M,T}
    Erad = SVec{NG,T}(ntuple(g -> @inbounds(u[1+g]), Val(NG))...)
    j = coupling_jacobian(s, @inbounds(u[1]), Erad)
    fill!(J, zero(T))
    @inbounds J[1, 1] = j.gas_gas
    @inbounds for g in 1:NG
        J[1, 1+g] = j.gas_rad[g]
        J[1+g, 1] = j.rad_gas[g]
        J[1+g, 1+g] = j.rad_rad[g]
    end
    return nothing
end

# A Rosenbrock method also needs `df/dt` at fixed `u`. The coupling ODE is
# autonomous -- `src` and `work` are constant rates over the step and the
# opacities are functions of the state alone -- so it is exactly zero, and saying
# so keeps the integrator from trying to differentiate the right-hand side with
# respect to `t` by any other means.
function _coupling_tgrad!(dT, u, s::CouplingState{NG,M,T}, _) where {NG,M,T}
    fill!(dT, zero(T))
    return nothing
end

# Supplying both derivatives analytically is also what makes any of this work at
# all: `planck_fractions` returns an `SVec{NG,T}` whose element type is fixed by
# the group structure, so a dual number cannot flow through the right-hand side
# and no `autodiff` specifier would help.
_ode_function() = ODEFunction{true}(_coupling_ode!; jac=_coupling_jac!,
                                    tgrad=_coupling_tgrad!)

function _ode_initial(s::CouplingState{NG,M,T}) where {NG,M,T}
    u0 = Vector{T}(undef, NG + 1)
    @inbounds u0[1] = s.Egas
    @inbounds for g in 1:NG
        u0[1+g] = s.Erad[g]
    end
    return u0
end

@inline _ode_erad(uf, ::CouplingState{NG,M,T}) where {NG,M,T} =
    SVec{NG,T}(ntuple(g -> @inbounds(uf[1+g]), Val(NG))...)

# --- backward Euler, from the library -----------------------------------------

"""
    ImplicitEulerStepper()

Take [`DirectCoupling`](@ref)'s implicit step with OrdinaryDiffEq's
`ImplicitEuler`, run with `adaptive = false` over a single step of length `dt`.
That is backward Euler exactly -- one implicit stage, no error control, no
substepping -- so it is the same discretisation [`NewtonStepper`](@ref) applies
by hand, arrived at by a completely separate route. The default.

Being separate is what it is *for*. The two steppers share
[`coupling_rhs`](@ref) and [`coupling_jacobian`](@ref) and nothing else: not the
residual, not the opening iterate, not the convergence test, not the linear
algebra. An error in either iteration therefore shows up as a disagreement
between them, which a second copy of the same loop could never do.

Its nonlinear solve is tuned rather than left at its defaults, and each change is
there to make it solve the *same* problem the hand-written stepper solves:

  - `max_iter = max_iterations` -- the default 10 is not enough for a stiff cell.
    At `tau = 1e6` this iteration needs seven, and the default gives up at ten
    with the cell barely moved.
  - `always_new = true` -- re-form the Jacobian every iteration. The default
    reuses it while convergence looks fast enough, which is the right economy
    when a Jacobian is expensive; here it is a closed-form expression in `1 + NG`
    unknowns, and `newton_solve` re-forms it too.
  - `check_div = false` -- do not abandon the iteration on a step that grows.
    The first Newton step out of an optically thick cell is large by construction
    and trips the divergence test; with it off, a genuinely diverging cell reaches
    `max_iter` and is reported unconverged, which is exactly what the hand-written
    stepper does with the same cell.
  - `κ = 1e-4` -- the increment test, tightened from `1e-2` so that the answer is
    the root and not merely near it.

**It allocates**, unavoidably: `solve` builds an integrator, a nonlinear solver,
their caches and a solution object, per cell, for a problem in `1 + NG`
unknowns. Measured, 8.9 kB at one group, 10.1 at four, 12.6 at eight, and 2.4 to
120 times the wall time of [`NewtonStepper`](@ref), which allocates nothing. That
is the trade the default makes -- a library implementation of the step, in
exchange for the cost of getting to it -- and the other stepper is one keyword
away. Both figures are asserted in `test_radiation_multigroup.jl` as well as
quoted here, so neither can drift silently.

`linsolve` is not consulted on this path; `ImplicitEuler` factorises the full
matrix itself. See [`AbstractBackwardEuler`](@ref).
"""
struct ImplicitEulerStepper <: AbstractBackwardEuler end

# `NLNewton` and `ImplicitEuler` come from OrdinaryDiffEq's sub-packages rather
# than from the meta-package; see the note in `Hydro3D.jl` and in Project.toml.
_implicit_euler(max_iterations::Int) =
    ImplicitEuler(; nlsolve=NLNewton(; κ=1e-4, max_iter=max_iterations,
                                     always_new=true, check_div=false))

function backward_euler(::ImplicitEulerStepper, m::DirectCoupling,
                        s::CouplingState{NG,M,T}, rtol::T, atol::T) where {NG,M,T}
    prob = ODEProblem{true}(_ode_function(), _ode_initial(s), (zero(T), s.dt), s)
    # One step of length `dt` across a span of length `dt`, non-adaptively: the
    # integrator has exactly one decision left to make and it is the implicit
    # one. `reltol` and `abstol` are both handed over because the increment test
    # weighs by `abstol + reltol * |u|`; passing only the absolute number would
    # make the test mean different things in CGS and in code units.
    sol = solve(prob, _implicit_euler(m.max_iterations); dt=s.dt, adaptive=false,
                reltol=rtol, abstol=atol, save_everystep=false, dense=false)
    uf = sol.u[end]
    # Newton iterations, where `newton_solve` reports Newton iterations: both
    # answer "how much work did this cell cost".
    return (@inbounds(uf[1]), _ode_erad(uf, s), sol.stats.nnonliniter,
            successful_retcode(sol))
end

# --- the error-controlled reference solver ------------------------------------

"""
    AdaptiveCoupling(; reltol = 1e-2, abstol = 1e-10, alg = nothing)

Solve the matter-radiation coupling with an adaptive, error-controlled stiff ODE
integrator rather than one implicit step.

This is a **reference solver**, not a production one. Its job is to say what the
coupling ODE's answer actually is, so that [`QuokkaCoupling`](@ref) and
[`DirectCoupling`](@ref) -- which take a single backward-Euler step and are first
order in `dt` -- can be checked against something whose error is controlled
rather than assumed. An adaptive integrator takes tens of internal steps per cell
per timestep and allocates its own caches: measured against
[`NewtonStepper`](@ref) it is 2 to 70 times slower, the ratio growing with the
stiffness because that is what it is refusing to take in one step, and it costs 8
to 13 kB a cell where `NewtonStepper` costs nothing. Use it to verify, not to run.

  - `reltol` : relative tolerance of the integrator. The default `1e-2` is
    deliberately loose: the point of the comparison is whether the one-step
    solvers land in the right place, and a tolerance far below the discretisation
    error of the scheme they are embedded in buys nothing but wall time.
  - `abstol` : absolute tolerance, quoted as a **fraction of the cell's total
    energy**. The one-step solvers scale their residual test the same way, and for
    the same reason: an absolute energy means one thing in CGS and another in
    code units, and a group holding `1e-30` of the spectrum must not drive the
    step size to zero on its own relative error.
  - `alg`    : the integrator, or `nothing` for the default (`Rosenbrock23`).
    Any stiff `OrdinaryDiffEq` algorithm works -- `Rodas5P()` and `FBDF()` come
    with the package, and anything from a sub-package you load yourself does too
    (`TRBDF2()`, `KenCarp4()`). An explicit method will stall: `chat rho kappa dt`
    reaches `1e8` in an optically thick cell. Whichever you choose, it is handed
    the coupling's analytic Jacobian and an exact `df/dt`, so no algorithm here
    ever differentiates the right-hand side numerically.

```julia
rad = RadiationPhysics(; c = 1.0, a_rad = 1.0, opacity = ConstantOpacity(1e5),
                       coupling = AdaptiveCoupling(reltol = 1e-4))
```
"""
struct AdaptiveCoupling{T<:AbstractFloat,A} <: AbstractCoupling
    reltol::T
    abstol::T
    alg::A
end

AdaptiveCoupling(; reltol=1e-2, abstol=1e-10, alg=nothing) =
    AdaptiveCoupling(promote(float(reltol), float(abstol))..., alg)

# Rosenbrock23 is Shampine's stiff workhorse: L-stable, second order, one
# Jacobian and one factorisation per step. For `1 + NG` unknowns that
# factorisation is free, so the cost is the number of steps, and a Rosenbrock
# method needs no Newton iteration to converge before it can take one.
#
# What the analytic Jacobian buys, measured rather than assumed. Rosenbrock23 is
# a Rosenbrock-*W* method, so it keeps its order with an approximate Jacobian by
# construction -- fitting the fixed-step error gives 1.99 with the analytic
# Jacobian, 1.99 with finite differences, and 1.99 with a Jacobian deliberately
# halved. It does not need an exact one to be second order. What it gets instead:
#
#   * wall time. A finite-difference Jacobian costs `1 + NG` extra right-hand
#     side evaluations every time it is formed. Same step counts, 18% faster at
#     one group, 32% at four, 49% at eight.
#   * accuracy where it is stiffest. At `tau ~ 10` the fixed-step error floor
#     drops from 1e-10 to 3e-13, and at a step so large the finite-difference
#     version diverged outright this one still returned a finite answer.
_default_alg() = Rosenbrock23()

function solve_coupling(m::AdaptiveCoupling, s::CouplingState{NG,M,T}) where {NG,M,T}
    # The energy the cell has to play with, and hence what `abstol` is a
    # fraction of. Same scale the one-step solvers measure their residual against.
    scale = max(abs(s.Egas + dot(gas_scale(s), s.Erad + s.src)), eps(T))

    # Start on the absorption time, `1 / (chat rho kappa_P)`, capped at the step
    # itself. The integrator's own initial-step heuristic is a norm ratio, and at
    # tau = 1e9 it lands near machine epsilon, warns, and falls back to a guess
    # three orders too large; the physical timescale is right there in the state,
    # so hand it over instead. Zero opacity gives `Inf` and the cap takes `dt`.
    kappaP = first(coupling_opacity(s, gas_temperature_of(s, s.Egas)))
    dt0 = min(s.dt, one(T) / (maxabs(s.chat) * s.rho * maxabs(kappaP)))

    prob = ODEProblem{true}(_ode_function(), _ode_initial(s), (zero(T), s.dt), s)
    sol = solve(prob, m.alg === nothing ? _default_alg() : m.alg;
                dt=dt0, reltol=T(m.reltol), abstol=T(m.abstol) * scale,
                save_everystep=false, dense=false)

    uf = sol.u[end]
    Egas, Erad = apply_energy_floors(s, @inbounds(uf[1]), _ode_erad(uf, s))

    # Report the integrator's step count where the Newton solvers report their
    # iteration count: both answer "how much work did this cell cost".
    steps = sol.stats.naccept + sol.stats.nreject
    return CouplingResult{NG,T}(Egas, Erad, steps, successful_retcode(sol))
end
