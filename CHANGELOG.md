# Changelog

All notable changes to Honghu.jl are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/). `RELEASING.md` describes how a release is made.

## [Unreleased]

### Added

- Passive mass scalars: `Simulation(nscalars = n)` stores the conserved densities $\rho X_s$ as extra rows of `sim.U`, `set_initial_conditions!` reads their mass fractions from the key `X`, and `conserved_totals(sim).scalars` is the mass of each. A `Reflecting` wall flips only the normal momentum, never a scalar row, and a `Fixed` or `FunctionBC` hydro boundary must give `NHYDRO + nscalars` values.
- Mass scalars are advected: each $\rho X_s$ is reconstructed with the gas and carried by the Riemann mass flux with the upwind mass fraction, so a uniform $X_s$ stays uniform to round-off and a contact discontinuity stays monotone. The density floor keeps the mass fraction of the cell, and `enforce_scalar_floors!` clips a negative $\rho X_s$ to zero and counts it in `sim.diag.floors_scalar`, which `nrepaired(sim)` includes. `cons2prim!`, `hydro_fluxes!` and `enforce_floors!` take an optional trailing `nscalars`; a run without scalars is bit for bit unchanged.
- Mass scalars in the output: `available_fields` lists `:X_1`, ..., `:X_ns`, and `field(sim, :X_s)` returns the mass fraction $\rho X_s/\rho$; `write_vtk` writes it as `mass_fraction_s`. Checkpoints are format 2: the magic is `HONGHU\x00\x02` and the row counts of `U` and `R` follow the radiation flag, so `load_checkpoint!` refuses a file whose number of hydro or radiation variables differs from the simulation's. Format 1 and Hydro3D files still load, into a simulation with `nscalars == 0` only.
- A solver-agnostic chemistry interface, `AbstractNetwork`: a network gives the species, the radiation bands it reads, the right-hand side of the per-cell ODE in $u = [n_1 \ldots n_{N_s}, e, N_1 \ldots N_{N_b}]$, its exact Jacobian and one scale per unknown. Two integrators solve it, `RosenbrockChem` (the default; an adaptive OrdinaryDiffEq integrator, created once per workspace and re-aimed at each cell, with `alg`, `reltol` and an `abstol` that is a fraction of the scales) and `BackwardEulerChem` (one Newton-solved backward-Euler step that allocates nothing and reports non-convergence). Both return a `ChemStats`. `RosenbrockChem` switches OrdinaryDiffEq's own warnings off, because a failing cell is reported through its `ChemStats`. `OrdinaryDiffEq`'s `init`, `solve!`, `reinit!` and `SciMLLogging` are imported for it, still by name and not as a package. Its internal step is limited by the cell's `dt` alone: `dtmax` would otherwise default to the length of the placeholder time span the integrator is built on, one time unit, and a cell with $\Delta t = 10^{10}$ s would never finish.
- `HydrogenOTSA`, the photoionization network of pure hydrogen in the on-the-spot approximation, after Aubert & Teyssier (2008) and Quokka's `photoionization_H_caseB_handwritten`: the species $({\rm H^0}, {\rm H^+})$, one band of ionizing photons with its cross-section, photon energy and heating energy as fields, case B recombination with its coefficient and exponent as fields, collisional ionization (Krumholz et al. 2007) and recombination, free-free and Lyman $\alpha$ cooling (Cen 1992) behind switches, and an isothermal mode that drops the energy equation. The gas side of the photoionization rate uses $c$ and the photon equation uses $\hat c$, so the conserved budget is $n_{\rm H^+} + N/\phi$ and the equilibrium does not depend on $\hat c$. The default heating energy, $\varepsilon_b - 13.6$ eV as in Quokka, overheats ionized gas compared with the optically thin value. `CGS.eV` is added.

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
