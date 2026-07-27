"""
    Hydro3D

A three-dimensional radiation-hydrodynamics code in pure Julia.

The Euler equations are solved with an unsplit finite-volume method-of-lines
scheme: piecewise-constant, -linear or -parabolic reconstruction of the
primitive variables, an HLLC Riemann solver, and a strong-stability-preserving
Runge-Kutta integrator.

Radiation is treated with the grey two-moment (M1) approximation. Transport uses
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

# --- core types and kernels --------------------------------------------------
include("staticvec.jl")
include("constants.jl")
include("eos.jl")
include("opacity.jl")
include("grid.jl")
include("variables.jl")
include("boundary.jl")
include("limiters.jl")
include("reconstruction.jl")
include("riemann_hydro.jl")
include("riemann_radiation.jl")
include("hydro_system.jl")
include("radiation_system.jl")
include("source_terms.jl")
include("integrators.jl")
include("simulation.jl")
include("output.jl")
include("plotting.jl")
include("problems/Problems.jl")

# --- public API --------------------------------------------------------------

# grid and state
export CartesianGrid, Simulation, set_initial_conditions!, run!, step!, timestep
export conserved_totals, primitives, gas_temperature, radiation_temperature

# physics
export IdealGas, cv_from_mu, RadiationPhysics, Floors
export ConstantOpacity, PowerLawOpacity, UserOpacity

# numerics
export PCM, PLM, PPM
export MinMod, MonotonizedCentral, VanLeer, Sweby
export ForwardEuler, SSPRK2, SSPRK3

# boundaries
export Boundaries, Periodic, Outflow, Reflecting, Fixed

# output
export write_vtk, write_line, save_checkpoint, load_checkpoint!

# fields and plotting (plotting methods require `using Plots`)
export field, available_fields, coordinates
export plotstate, plotpanels, animate_run

end # module Hydro3D
