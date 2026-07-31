# The third coupling solver: hand the stiff ODE to an error-controlled
# integrator instead of taking one backward-Euler step.
#
# The type and its docstring live here, in the package proper, so that
# `AdaptiveCoupling()` can be constructed, stored in a `RadiationPhysics` and
# documented whether or not an ODE stack is installed. The `solve_coupling`
# method lives in `ext/Hydro3DOrdinaryDiffEqExt.jl` and appears the moment
# OrdinaryDiffEq does -- the same split, and for the same reason, as
# `plotting.jl` and the Plots extension.

const _ODE_HINT = "run `using OrdinaryDiffEq` first (`using DifferentialEquations` \
                   also works, it loads OrdinaryDiffEq); AdaptiveCoupling's solver \
                   lives in a package extension"

"""
    AdaptiveCoupling(; reltol = 1e-2, abstol = 1e-10, alg = nothing)

Solve the matter-radiation coupling with an adaptive, error-controlled stiff ODE
integrator rather than one implicit step.

This is a **reference solver**, not a production one. Its job is to say what the
coupling ODE's answer actually is, so that [`QuokkaCoupling`](@ref) and
[`DirectCoupling`](@ref) -- which take a single backward-Euler step and are first
order in `dt` -- can be checked against something whose error is controlled
rather than assumed. An adaptive integrator takes tens of internal steps per cell
per timestep and allocates its own caches: measured against the other two it is
10 to 100 times slower and costs a few kilobytes a cell where they cost nothing.
Use it to verify, not to run.

  - `reltol` : relative tolerance of the integrator. The default `1e-2` is
    deliberately loose: the point of the comparison is whether the one-step
    solvers land in the right place, and a tolerance far below the discretisation
    error of the scheme they are embedded in buys nothing but wall time.
  - `abstol` : absolute tolerance, quoted as a **fraction of the cell's total
    energy**. The in-tree solvers scale their residual test the same way, and for
    the same reason: an absolute energy means one thing in CGS and another in
    code units, and a group holding `1e-30` of the spectrum must not drive the
    step size to zero on its own relative error.
  - `alg`    : the integrator, or `nothing` for the default (`Rosenbrock23`).
    Any stiff `OrdinaryDiffEq` algorithm works -- `TRBDF2()`, `KenCarp4()`,
    `Rodas5P()`. An explicit method will stall: `chat rho kappa dt` reaches
    `1e8` in an optically thick cell.

**Requires `using OrdinaryDiffEq`** (or `using DifferentialEquations`, which
loads it). OrdinaryDiffEq is a weak dependency, so nothing is downloaded and
nothing is loaded unless you ask for this solver. Note that it also exports
`step!`, so with both packages in scope that one name is ambiguous and the
simulation stepper has to be spelled `Hydro3D.step!`; `run!` is unaffected.

```julia
using OrdinaryDiffEq, Hydro3D
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

# Informative fallback. The extension's method takes a `CouplingState` and is
# therefore more specific, so it wins whenever OrdinaryDiffEq is loaded.
solve_coupling(::AdaptiveCoupling, ::Any) = error("solve_coupling: ", _ODE_HINT)
