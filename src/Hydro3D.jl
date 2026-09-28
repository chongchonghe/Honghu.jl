"""
    Hydro3D

A three-dimensional radiation-hydrodynamics code in pure Julia.

The Euler equations are solved with an unsplit finite-volume method-of-lines
scheme: piecewise-constant, -linear or -parabolic reconstruction of the
primitive variables, a choice of Riemann solver (HLLC or LLF), and a
strong-stability-preserving Runge-Kutta integrator.

Radiation is treated with the two-moment (M1) approximation, grey or resolved
into `NG` frequency groups (see `RadiationGroups`). Transport uses
an HLL solver built on the M1 closure, and the stiff matter-radiation coupling
is solved implicitly, the two being combined by the second-order,
asymptotic-preserving IMEX PD-ARS scheme. Radiation subcycles inside each hydro
step at the reduced speed of light.

The algorithms follow Quokka (Wibking & Krumholz 2022; He, Wibking & Krumholz
2024).

# Getting started

```julia
using Hydro3D

sim = Simulation(n = (256, 1, 1), boundaries = Boundaries(Outflow()),
                 eos = IdealGas(gamma = 1.4))
set_initial_conditions!(sim) do x, y, z
    x < 0.5 ? (rho = 1.0, p = 1.0) : (rho = 0.125, p = 0.1)
end
run!(sim, tmax = 0.2)
```

or start from one of the built-in problems:

```julia
sim = Hydro3D.Problems.kelvin_helmholtz(n = 256)
run!(sim, tmax = 2.0, outdir = "kh", dtout = 0.02)
```
"""
module Hydro3D

using Printf
using Dates          # provenance timestamps only; a stdlib, not a package
# Two names, not the package: `DenseSolve` factorises the coupling's Newton
# matrix with them (see `newton.jl`). Imported selectively because LinearAlgebra
# exports `dot` and `norm`, which `staticvec.jl` defines for `SVec` as its own --
# a blanket `using` would make those definitions an error.
using LinearAlgebra: lu!, ldiv!

# Five names, not the package, and for the same reason: OrdinaryDiffEq exports
# `step!`, which this package also exports, so a blanket `using` would leave that
# one name ambiguous for anyone with both in scope -- `step!(sim)` would have to
# be spelled `Hydro3D.step!`. Naming what is used keeps `step!` this package's.
#
# `AdaptiveCoupling` in `coupling_ode.jl` is the only thing that needs them:
# `Rosenbrock23` is its default algorithm and the rest state the problem and read
# the answer. All five come from the meta-package; the two sub-package
# dependencies this file used to name went with `ImplicitEulerStepper`.
using OrdinaryDiffEq: ODEProblem, ODEFunction, solve, successful_retcode, Rosenbrock23

# Three names for `BracketedCoupling`, which hands its one scalar equation to a
# bracketing root-finder through SciMLBase's interface, and for `ZBrent`, the
# default one, which implements that interface in `rootfind.jl`. `solve` is
# `CommonSolve.solve`, the same function the line above imports, so it is not
# imported twice; `ZBrent` adds a method to it.
using SciMLBase: SciMLBase, IntervalNonlinearProblem, ReturnCode

# --- core types and kernels --------------------------------------------------
include("staticvec.jl")
include("newton.jl")     # Newton on a bordered system; knows no physics
include("rootfind.jl")   # Brent's method for a bracketed scalar root; knows no physics
include("constants.jl")
include("planck.jl")     # RadiationGroups, needed by the multigroup opacity below
include("eos.jl")
include("opacity.jl")
include("dust.jl")            # the dust-gas thermal coupling model
include("rsla.jl")             # the variable reduced speed of light, pointwise
include("coupling.jl")            # per-cell matter-radiation coupling interface
include("coupling_solvers.jl")    # ... and the two in-tree solvers
include("coupling_ode.jl")        # ... and the error-controlled reference solve
include("grid.jl")
include("variables.jl")
include("boundary.jl")
include("limiters.jl")
include("reconstruction.jl")
include("riemann/state.jl")
include("riemann/hllc.jl")
include("riemann/llf.jl")
include("riemann_radiation.jl")
include("hydro_system.jl")
include("radiation_system.jl")
include("source_terms.jl")
include("integrators.jl")
include("simulation.jl")
include("output.jl")
include("plotting.jl")
include("metrics.jl")
include("compare.jl")
include("problems/Problems.jl")

# --- public API --------------------------------------------------------------

# grid and state
export CartesianGrid, Simulation, set_initial_conditions!, run!, step!, timestep
export conserved_totals, primitives, gas_temperature, radiation_temperature
export Diagnostics, reset_diagnostics!, nrepaired
export SourceOrder, SourceForward, SourceReverse

# physics
export IdealGas, cv_from_mu, RadiationPhysics, Floors
export ConstantOpacity, PowerLawOpacity, UserOpacity, FrequencyOpacity
export LevermoreClosure, EddingtonClosure

# multigroup radiation
export RadiationGroups, ngroups, planck_integral, planck_fractions
export AbstractCoupling, BracketedCoupling, QuokkaCoupling, AdaptiveCoupling
export AbstractDustModel, NoDust, BateKetoDust
export AbstractChatLimiter, ConstantChat, StabilityBound, OpticalDepthFit, per_group
export AbstractJacobianSolve, DenseSolve, ArrowheadSolve

# numerics
export PCM, PLM, PPM
export HLLC, LLF, HydroState, AbstractRiemannSolver, riemann_flux
export MinMod, MonotonizedCentral, VanLeer, Sweby
export ForwardEuler, SSPRK2, SSPRK3

# boundaries
export Boundaries, Periodic, Outflow, Reflecting, Fixed, FunctionBC

# output
export write_vtk, write_line, save_checkpoint, load_checkpoint!

# metrics and comparison
export relative_l1, relative_l2, default_metrics
export compare, ComparisonTable, write_comparison, read_comparison, convergence_rate
export plotcompare

# fields and plotting (plotting methods require `using Plots`)
export field, available_fields, coordinates
export plotstate, plotpanels, animate_run

end # module Hydro3D
