# Changelog

All notable changes to Honghu.jl are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/). `RELEASING.md` describes how a release is made.

## [Unreleased]

### Added

- Passive mass scalars: `Simulation(nscalars = n)` stores the conserved densities $\rho X_s$ as extra rows of `sim.U`, `set_initial_conditions!` reads their mass fractions from the key `X`, and `conserved_totals(sim).scalars` is the mass of each. A `Reflecting` wall flips only the normal momentum, never a scalar row, and a `Fixed` or `FunctionBC` hydro boundary must give `NHYDRO + nscalars` values.
- Mass scalars are advected: each $\rho X_s$ is reconstructed with the gas and carried by the Riemann mass flux with the upwind mass fraction, so a uniform $X_s$ stays uniform to round-off and a contact discontinuity stays monotone. The density floor keeps the mass fraction of the cell, and `enforce_scalar_floors!` clips a negative $\rho X_s$ to zero and counts it in `sim.diag.floors_scalar`, which `nrepaired(sim)` includes. `cons2prim!`, `hydro_fluxes!` and `enforce_floors!` take an optional trailing `nscalars`; a run without scalars is bit for bit unchanged.
- Mass scalars in the output: `available_fields` lists `:X_1`, ..., `:X_ns`, and `field(sim, :X_s)` returns the mass fraction $\rho X_s/\rho$; `write_vtk` writes it as `mass_fraction_s`. Checkpoints are format 2: the magic is `HONGHU\x00\x02` and the row counts of `U` and `R` follow the radiation flag, so `load_checkpoint!` refuses a file whose number of hydro or radiation variables differs from the simulation's. Format 1 and Hydro3D files still load, into a simulation with `nscalars == 0` only.

### Documentation

- The Zenodo concept DOI, `10.5281/zenodo.23569081`, in `CITATION.cff` and as a badge in the README.

## [0.1.0] - 2026-10-23

The first public release.

### Hydrodynamics

- The compressible Euler equations in one, two or three dimensions on a uniform Cartesian grid, with an ideal-gas equation of state and an interface for others.
- Method-of-lines finite-volume scheme: `PCM`, `PLM` (four limiters) or `PPM` reconstruction; the `HLLC` Riemann solver with Batten wave speeds, Rider's correction and the Minoshima & Miyoshi carbuncle cure, or `LLF`; SSP Runge-Kutta integrators up to third order.
- Boundary conditions: periodic, outflow, reflecting, fixed and user-defined functions.

### Radiation hydrodynamics

- Two-moment (M1) radiation transport with the Levermore closure, grey or in frequency groups, with an HLL Riemann solver; the Eddington closure for P1 reference problems.
- The IMEX PD-ARS integrator: explicit transport, implicit matter-radiation coupling, second order and asymptotic-preserving.
- Three multigroup coupling solvers: `BracketedCoupling` (the default; one scalar equation per cell, solved with BracketingNonlinearSolve's `Brent`), `QuokkaCoupling` (Quokka's Newton formulation) and `AdaptiveCoupling` (an error-controlled reference through OrdinaryDiffEq).
- Dust-gas thermal coupling (`BateKetoDust`), solved as one equation in the dust temperature.
- The reduced speed of light: the standard RSLA (`ConstantChat`) and the adaptive reduced speed of light approximation (`StabilityBound`), with a stability bound exact for the scheme.
- Grey and multigroup opacity laws, and `O(v/c)` terms in the coupling.

### Problems, verification and output

- Fourteen built-in problems in `Honghu.Problems`, from the Sod shock tube and the Sedov blast to Marshak waves and the Lowrie & Edwards radiative shocks, with exact or reference solutions where they exist.
- A test suite of over 1400 tests, run in CI on Julia 1.10 and 1.12.
- `compare` and `ComparisonTable` for scheme-comparison studies; relative L1 and L2 metrics.
- VTK output for ParaView, text line-outs, and checkpoints that can be restarted from.
- Plotting through a package extension, loaded only with Plots: `plotstate`, `plotpanels`, `animate_run`, `plotcompare`.
- A documentation site built from the docstrings, the README and the technical notes, with every page also published as Markdown.
