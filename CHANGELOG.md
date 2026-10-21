# Changelog

All notable changes to Honghu.jl are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/). `RELEASING.md` describes how a release is made.

## [0.1.0] - Unreleased

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
