"""
    Hydro3DOrdinaryDiffEqExt

The error-controlled matter-radiation coupling solver, loaded automatically when
OrdinaryDiffEq is available -- including via `using DifferentialEquations`,
which loads OrdinaryDiffEq itself.

Provides the [`Hydro3D.solve_coupling`](@ref) method for
[`Hydro3D.AdaptiveCoupling`](@ref).
"""
module Hydro3DOrdinaryDiffEqExt

using Hydro3D
using Hydro3D: AdaptiveCoupling, CouplingState, CouplingResult, SVec,
               apply_energy_floors, coupling_rhs, maxabs, sumvec
using OrdinaryDiffEq

# Rosenbrock23 is Shampine's stiff workhorse: L-stable, second order, one
# Jacobian and one factorisation per step. For `1 + NG` unknowns that
# factorisation is free, so the cost is the number of steps, and a Rosenbrock
# method needs no Newton iteration to converge before it can take one.
#
# `AutoFiniteDiff` is not a preference. `planck_fractions` returns an
# `SVec{NG,T}` whose element type is fixed by the group structure, and
# `coupling_rhs` demands the same `SVec{NG,T}` for its radiation argument, so a
# dual number cannot flow through the right-hand side at all: forward-mode AD
# would not merely be slower, it would not compile. Finite differences cost
# accuracy in the Jacobian only, which a Rosenbrock method absorbs into its
# step-size control.
_default_alg() = Rosenbrock23(autodiff=AutoFiniteDiff())

# The coupling ODE, written in place so that the right-hand side allocates
# nothing however many times the integrator calls it.
#
# The state is a plain `Vector{T}` of length `1 + NG`, not an `SVector`: a
# static vector would mean a StaticArrays dependency, which CLAUDE.md's
# dependency policy already declined once for `SVec`, and it would buy nothing
# here because `solve` allocates its own caches regardless. That allocation is
# inherent to an adaptive integrator and is the price this solver charges for
# controlling its error; it is why `AdaptiveCoupling` is a reference solver.
#
# `src` and `work` are increments over the *whole* step, held constant, which is
# exactly what the two backward-Euler solvers add to their residuals. Dividing
# them by `dt` here and integrating over `[0, dt]` returns them unchanged, so
# all three solvers demonstrably integrate the same problem.
function _coupling_ode!(du, u, s::CouplingState{NG,M,T}, _) where {NG,M,T}
    Erad = SVec{NG,T}(ntuple(g -> @inbounds(u[1+g]), Val(NG))...)
    dEgas, dErad = coupling_rhs(s, @inbounds(u[1]), Erad)
    invdt = one(T) / s.dt
    @inbounds du[1] = dEgas - (s.c / s.chat) * sumvec(s.work) * invdt
    @inbounds for g in 1:NG
        du[1+g] = dErad[g] + (s.src[g] + s.work[g]) * invdt
    end
    return nothing
end

function Hydro3D.solve_coupling(m::AdaptiveCoupling, s::CouplingState{NG,M,T}) where {NG,M,T}
    u0 = Vector{T}(undef, NG + 1)
    @inbounds u0[1] = s.Egas
    @inbounds for g in 1:NG
        u0[1+g] = s.Erad[g]
    end

    # The energy the cell has to play with, and hence what `abstol` is a
    # fraction of. Same scale the in-tree solvers measure their residual against.
    scale = max(abs(s.Egas + (s.c / s.chat) * (sumvec(s.Erad) + sumvec(s.src))), eps(T))

    # Start on the absorption time, `1 / (chat rho kappa_P)`, capped at the step
    # itself. The integrator's own initial-step heuristic is a norm ratio, and at
    # tau = 1e9 it lands near machine epsilon, warns, and falls back to a guess
    # three orders too large; the physical timescale is right there in the state,
    # so hand it over instead. Zero opacity gives `Inf` and the cap takes `dt`.
    dt0 = min(s.dt, one(T) / (s.chat * s.rho * maxabs(s.kappaP)))

    prob = ODEProblem{true}(_coupling_ode!, u0, (zero(T), s.dt), s)
    sol = solve(prob, m.alg === nothing ? _default_alg() : m.alg;
                dt=dt0, reltol=T(m.reltol), abstol=T(m.abstol) * scale,
                save_everystep=false, dense=false)

    uf = sol.u[end]
    Erad = SVec{NG,T}(ntuple(g -> @inbounds(uf[1+g]), Val(NG))...)
    Egas, Erad = apply_energy_floors(s, @inbounds(uf[1]), Erad)

    # Report the integrator's step count where the Newton solvers report their
    # iteration count: both answer "how much work did this cell cost".
    steps = sol.stats.naccept + sol.stats.nreject
    return CouplingResult{NG,T}(Egas, Erad, steps, successful_retcode(sol))
end

end # module Hydro3DOrdinaryDiffEqExt
