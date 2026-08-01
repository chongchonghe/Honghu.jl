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
               apply_energy_floors, coupling_jacobian, coupling_opacity, coupling_rhs,
               gas_temperature_of, maxabs, sumvec
using OrdinaryDiffEq

# Rosenbrock23 is Shampine's stiff workhorse: L-stable, second order, one
# Jacobian and one factorisation per step. For `1 + NG` unknowns that
# factorisation is free, so the cost is the number of steps, and a Rosenbrock
# method needs no Newton iteration to converge before it can take one.
#
# Both derivatives it needs -- the Jacobian and `df/dt` -- are supplied
# analytically below, so it needs no `autodiff` specifier, and none would work
# anyway: `planck_fractions` returns an `SVec{NG,T}` whose element type is fixed
# by the group structure, so a dual number cannot flow through the right-hand
# side at all.
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

# The exact Jacobian, the same one `DirectCoupling` builds its Newton matrix
# from. Dense `(1+NG)^2`, because that is what `Rosenbrock23` factorises and NG
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
# opacities are frozen -- so it is exactly zero, and saying so keeps the
# integrator from trying to differentiate the right-hand side with respect to
# `t` by any other means.
function _coupling_tgrad!(dT, u, s::CouplingState{NG,M,T}, _) where {NG,M,T}
    fill!(dT, zero(T))
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
    kappaP = first(coupling_opacity(s, gas_temperature_of(s, s.Egas)))
    dt0 = min(s.dt, one(T) / (s.chat * s.rho * maxabs(kappaP)))

    f = ODEFunction{true}(_coupling_ode!; jac=_coupling_jac!, tgrad=_coupling_tgrad!)
    prob = ODEProblem{true}(f, u0, (zero(T), s.dt), s)
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
