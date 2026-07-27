# Development log

A record of what was asked for and what was built, written at the end of the session that produced this package.

## The request

> Rewrite Hydro.jl into a complete, usable 3D radiation hydrodynamics code. Write the new code in a new folder hydro3d.jl
>
> Use the best practice Julia code to redesign the code structure.
>
> Use the quokka code as a reference:
> - Hydro: use HLLC
> - Radiation: two-moment radiation transport with M1 closure; HLL; IMEX for time stepping.
> - method of lines split of time and space
> - Support piecewise constant, piecewise linear, and piecewise parabolic.

Two follow-ups arrived mid-session: *"Commit your changes to track the progress"* and *"don't forget to include README.md and CLAUDE.md"*. Both were done.

## Starting point

`Hydro.jl` was a 1D/2D pure-hydrodynamics teaching code: about 2100 lines, a mutable `Grid`/`Grid2d` pair carrying every work array as a field, hand-written HLL and a partly broken HLLC, minmod-only reconstruction, and `Plots.jl` in the dependency list. There was no radiation, no third dimension, and no threading. Rather than extend it, the whole thing was rewritten; only the problem set-ups (Sod, Kelvin-Helmholtz) and the idea of a Shu-Osher coefficient table survived in spirit.

`quokka/` (C++/AMReX) served as the algorithmic reference. The files that mattered were `src/hydro/HLLC.hpp`, `src/hyperbolic_system.hpp` (PLM/PPM), `src/radiation/radiation_system.hpp` (M1 closure, HLL, wave speeds), `src/radiation/source_terms_single_group.hpp` (the implicit coupling) and `src/QuokkaSimulation.hpp` (the IMEX PD-ARS stage structure and the radiation subcycle).

## Constraint discovered early: no dependencies

The machine had no Julia package registry and an empty depot (`~/.julia/packages` did not exist). Any `Pkg.add` would have made the package unrunnable. So `Hydro3D` depends on `Printf` and nothing else. (A registry appeared later in the session; see the follow-up section below. The core stayed dependency-free by choice rather than necessity.) Two things that would normally be a dependency are written in-tree instead:

- `src/staticvec.jl` -- a 58-line `SVec{N,T}`, deliberately *not* an `AbstractArray` so a stray broadcast cannot silently heap-allocate in a hot loop. It exists so the Riemann solvers can be written in the vector form the papers use.
- `src/output.jl` -- a VTK ImageData writer emitting raw appended binary. Validated by parsing the XML header and unpacking every binary block from Python.

## What was built

23 source files totalling about 3500 lines, plus a 126-line plotting extension, 1050 lines of tests, 6 examples and 2 figure scripts.

### Physics

| | |
|:---|:---|
| Hydrodynamics | 3D compressible Euler, ideal gas, 5 conserved variables |
| Radiation | grey two-moment (M1), 4 conserved variables, Levermore (1984) closure |
| Coupling | absorption/emission, radiation force, $\mathbf{v}\cdot\mathbf{F}$ work term to $O(v/c)$ |
| Opacity | Planck, energy and flux means distinguished; constant, power-law and user-supplied laws |

### Numerics

| Component | Choice |
|:---|:---|
| Reconstruction | `PCM`, `PLM` (MinMod / MC / VanLeer / Sweby), `PPM` (Colella-Woodward with the Mignone/Athena++ interface constraint) |
| Hydro Riemann solver | `HLLC` with Batten wave speeds, Rider's nonlinear correction, and Minoshima & Miyoshi's carbuncle cure and low-dissipation pressure |
| Radiation Riemann solver | `HLL` on the M1 system, frozen-Eddington-tensor wave speeds (Skinner & Ostriker 2013; Balsara 1999) |
| Hydro integration | SSP-RK, orders 1-3, in Shu-Osher form |
| Radiation integration | IMEX PD-ARS, three stages, second order, asymptotic-preserving |
| Coupling solve | $2\times2$ Newton-Raphson per cell in $(E_{\rm int}, R)$, analytic flux/momentum update, lagged outer iteration for the work term |

Space and time are separated by the method of lines throughout: a spatial operator $L(U) = -(F_{i+1/2}-F_{i-1/2})/\Delta x$ is assembled once per stage and the stages are pure ODE updates. Radiation subcycles inside each hydro step at the reduced speed of light, as in Quokka.

## Design decisions

**Array layout `(nvar, i, j, k)`.** Variable index first, so a cell's components are contiguous in Julia's column-major order, which is what the Riemann solvers want. This follows Trixi.jl rather than AMReX's `(i,j,k,n)`.

**Degenerate directions are free.** A direction with `n[d] == 1` carries zero ghost cells, is skipped by the sweeps, and needs no boundary condition. A 1D problem is `n = (nx, 1, 1)` and costs what a 1D problem should. This is why `hydro_fluxes!` guards its transverse stencil accesses on `grid.active[d]`.

**Directional sweeps take `Val(D)`** so the offsets fold at compile time, and are dispatched from an explicit `active[1] && sweep!(..., Val(1))` chain rather than `for D in 1:3`, which would create a runtime `Val`.

**Interface convention follows Quokka**: face `f` along direction `D` is the *low-side* face of cell `f`, so `qL[f]` comes from cell `f-1` and `qR[f]` from cell `f`. Minimum ghost width is 2 even for PCM, so the reconstruction sweep can write `qL` at cell `hi+1` without leaving the array.

**One ghost-fill implementation serves both systems.** The momentum component for direction `d` sits at index `d+1` in both the hydro and the radiation state vector, so reflecting boundaries flip the same slot in both.

**Closure-based initial conditions** rather than an `AbstractProblem` hierarchy: `set_initial_conditions!(sim) do x, y, z; (rho=..., p=...); end`, with a `NamedTuple` whose optional keys have physical defaults. Extension happens through the small dispatch interfaces (`AbstractEOS`, `AbstractOpacity`, `AbstractLimiter`, `AbstractReconstruction`, `AbstractIntegrator`, `BoundaryCondition`) instead.

**Unit-agnostic.** $c$, $a_r$, $c_v$ and the opacities are all supplied by the caller. `Hydro3D.CGS` holds constants for those who want them; the built-in radiation problems set $c = a_r = c_v = 1$.

## Two bugs found and fixed

**1. The HLLC star pressure went negative.** On Toro's "123" problem (two strong rarefactions opening a near vacuum) the linearised star-pressure estimate evaluates to $-1.096$ where the exact answer is $+0.0019$, because the Batten wave-speed estimates stop being valid there. Quokka carries the same formula. The fix is a clamp at zero, which is inactive for any well-resolved state. Verified that all five of Toro's shock tubes now run with strictly positive density and pressure and zero floor repairs.

**2. An inner closure boxed the Newton-loop variables.** An `ntuple(Val(3)) do a ... end` block in `radiation_source_step!` captured `kP`, `kE`, `kF`, `B`, `Erad` and friends. Because those are assigned in several places, Julia boxed them, so *every* assignment in the Newton iteration heap-allocated -- and the boxing is a lowering-time decision, so it applied even on the `beta_order = 0` branch where the closure never executes.

The cost was 406 MB and 200 ms per source step at $32^3$. Found with `Profile.Allocs`, which pointed at eleven separate lines all allocating bare `Float64`s -- the signature of `Core.Box`. Rewriting the block component by component made it allocation-free and 29x faster; radiation-hydrodynamics throughput went from 0.1 to 6.2 Mcell-updates/s at $64^3$ on 8 threads.

A second, related trap was avoided by construction: `reduce_max_jk` writes partial results to distinct slots keyed by the loop index rather than using `threadid()`, which is not stable under `Threads.@threads`'s default dynamic scheduling.

## Verification

The suite is 405 tests, about 8 s: `julia --project=. -t 4 test/runtests.jl`.

Measured against exact solutions:

| Test | Result |
|:---|:---|
| Convergence order, smooth advection | PCM 1.0, PLM 2.0, PPM 2.3 |
| Sod vs exact Riemann solution | $L_1(\rho) = 7.9\times10^{-4}$ at $n_x = 512$ |
| Toro's five shock tubes | all positive, `nrepaired == 0` |
| Sedov 3D | shock radius identical along $x$, $y$, $z$; mass to $3\times10^{-15}$ |
| Conservation, periodic | mass and energy to $10^{-14}$ relative |
| Free streaming | front at $\hat{c}t$ to 2%, $\lvert F\rvert \le cE$ everywhere |
| Static diffusion vs analytic Gaussian | 0.6%, 0.13%, 0.06% relative $L_1$ at 64, 128, 256 |
| Stiff coupling, $\kappa$ up to $10^9$ | equilibrium reached in one step; energy conserved to $10^{-13}$ |
| Radiation force vs analytic drag law | 1%, converging at second order |
| Shadow | $10^{12}$ contrast behind the clump, 66x at the far boundary |

PPM stopping at 2.3 rather than 3 is expected: the Colella-Woodward monotonicity constraint drops to a limited line at the two extrema of the sine wave. The same limiter behaves the same way in Athena++ and Quokka, which is why Quokka also ships an extremum-preserving variant.

Performance on Apple silicon, `Float64`, $64^3$:

| | 1 thread | 8 threads |
|:---|---:|---:|
| Hydrodynamics | 2.3 Mcell/s | 12.1 Mcell/s |
| Radiation-hydrodynamics | 1.1 Mcell/s | 6.2 Mcell/s |

## Issues flagged

**The first 3D CGS example was unphysical.** Its "radiative equilibrium" initial condition put $a_rT^4 = 7.6\times10^5$ against a gas internal energy of $2.1\times10^{-3}$ -- radiation-dominated by $3.6\times10^8$. The radiation force then accelerated the gas towards $c$, which these non-relativistic equations cannot represent, and the run collapsed onto the floors. The ratio scales as $T^3/\rho$, so a hot tenuous medium hits it easily: gas domination at $10^5$ K needs $\rho \gtrsim 4\times10^{-8}\ \mathrm{g\,cm^{-3}}$.

The example was rebuilt gas-dominated, it now prints both governing ratios before it starts, and the README gained a section on choosing radiation parameters. To confirm the collapse was the set-up and not the solver, the same problem was run in a closed reflecting box: mass conserved to $3\times10^{-13}$ and $E_{\rm gas} + (c/\hat{c})E_{\rm rad}$ to $1.7\times10^{-7}$, the latter being the per-cell Newton tolerance accumulating over 3000 source applications.

**A reduced speed of light rescales the diffusion coefficient** by $\hat{c}/c$. Free streaming and the equilibrium state are untouched, but the static diffusion limit is not, so `radiation_diffusion` is set up with $\hat{c} = c$ and the caveat is documented on `RadiationPhysics`.

**Floor hits are reported, not warned about.** A handful is normal wherever the solution genuinely reaches vacuum -- the deep shadow hits 8 in 88 million cell updates. A count comparable to the number of updates means the run is unphysical, and the message says so.

## Deliberately not included

No mesh refinement, MPI or GPU. Grey radiation only -- no multigroup, dust, chemistry or self-gravity. No dual-energy formalism (Quokka carries `Eint` as a separate conserved variable; here it is recovered as $E_{\rm tot} - E_{\rm kin}$, which is enough without AMR). No LLF/Rusanov fallback solver, since HLLC and HLL were what was asked for.

## Follow-up: plotting, figures and benchmarks

After the first pass the user installed a Julia registry and Plots, and asked what could be improved now that packages were available. Four things were agreed and implemented.

**Plotting as a package extension, not a hard dependency.** `Plots` is a `[weakdeps]` entry and the methods live in `ext/Hydro3DPlotsExt.jl`. `src/plotting.jl` holds `field` / `available_fields` / `coordinates` (which need nothing external and are useful on their own), the three function stubs with their docstrings, and fallbacks that raise "run `using Plots` first" rather than a bare `MethodError`. `plotstate` picks its representation from the dimensionality -- line in 1D, heat map in 2D, mid-plane slice in 3D -- and a `@recipe` makes plain `plot(sim)` work. The point of the extension is that solving still pulls in nothing: the ~100-package plotting stack stays out of the way on a compute node.

**Verification figures.** `docs/make_figures.jl` regenerates seven panels now embedded in the README: Sod against the exact Riemann solution, the convergence plot with measured slopes against reference lines, static diffusion against the analytic Gaussian, the Marshak wave at three times showing $T_{\rm rad}$ ahead of $T_{\rm gas}$, the shadow with the clump outlined and a log cut through it, Kelvin-Helmholtz at three times, and Sedov with every cell in the box collapsed onto one radial profile. The README previously asserted these numbers with nothing to look at.

**Animations.** `docs/make_animations.jl` writes `kh.gif`, `shadow.gif` and `sedov.gif`. Sizes were tuned down to about 3 MB, 0.8 MB and 0.8 MB; the GIF structure was verified by parsing the headers and counting frames, because the FFMPEG artifact on this machine is broken (`libatomic.1.dylib` missing) and cannot be used to inspect them.

**BenchmarkTools, as a test extra.** `test/benchmarks.jl` prints the per-step and per-kernel timing table that now backs the README's performance section, replacing hand-rolled timing loops. It shows the implicit coupling at 21.9 ns/cell dominating a radiation step, against 2.78 for the HLLC sweep and 2.71 for PPM reconstruction.

The more valuable addition was `test/test_performance.jl`, which is not about speed but about allocation. It asserts that every hot kernel stays under 8 bytes per interior cell and that allocation per step does not grow with the grid. Those two checks would have caught the boxing bug directly -- it was 12416 B/cell, against the 0.25 B/cell the implicit source term measures now -- and nothing in the physics suite noticed it, because the answers were right and only the run time was destroyed.

Three packages were considered and declined, for reasons recorded in `CLAUDE.md`: `StaticArrays` would replace 58 lines of already allocation-free `SVec`; `HDF5`/`WriteVTK` would replace a validated 200-line writer; a generic nonlinear solver would be slower than a hand-written $2\times2$ Newton whose Jacobian inverts analytically.

## Commits

```
effec5b chore: ignore .claude worktree directory
2fc1f79 docs: README, CLAUDE.md and runnable examples
94cf7b9 test: full test suite; fix HLLC star pressure and source-term boxing
5353be1 feat: complete Hydro3D solver, driver, I/O and test problems
6959e39 wip: core physics modules for Hydro3D
```

The repository was initialised in this session and has no remote, so there is no pull request; the work is on `main` in this directory. Development happened in a git worktree under `.claude/worktrees/` (background-session isolation) and was fast-forwarded onto `main` at the end.

## Layout

```
hydro3d.jl/
  Project.toml          only dependency: Printf
  README.md             what it solves, how to run it, verification figures
  CLAUDE.md             invariants, performance traps, where things live
  logs.md               this file
  ext/
    Hydro3DPlotsExt.jl  plotting methods, loaded only when Plots is present
  docs/
    make_figures.jl     regenerates the README figures
    make_animations.jl  regenerates the README GIFs
    figures/            the generated PNGs and GIFs
  src/
    Hydro3D.jl          module, include order, public API
    staticvec.jl        SVec{N,T}
    constants.jl        CGS submodule
    eos.jl              AbstractEOS, IdealGas
    opacity.jl          AbstractOpacity and three laws
    grid.jl             CartesianGrid, sweep ranges, threading helpers
    variables.jl        state layout, pointwise conversions
    boundary.jl         Periodic / Outflow / Reflecting / Fixed
    limiters.jl         MinMod / MC / VanLeer / Sweby
    reconstruction.jl   PCM / PLM / PPM
    riemann_hydro.jl    hllc
    riemann_radiation.jl M1 closure, Eddington tensor, hll_radiation
    hydro_system.jl     Floors, cons2prim!, HLLC sweep, hydro CFL
    radiation_system.jl RadiationPhysics, state repair, HLL sweep, radiation CFL
    source_terms.jl     implicit matter-radiation coupling
    integrators.jl      SSP-RK tableaux, IMEX PD-ARS constants
    simulation.jl       Simulation, workspaces, RHS, stage drivers, run!
    output.jl           VTK, text line-outs, checkpoints
    plotting.jl         field extraction, plotting stubs and fallbacks
    problems/           built-in problems and the exact Riemann solver
  test/                 405 tests in nine files, plus benchmarks.jl
  examples/             sod_1d, kelvin_helmholtz_2d, sedov_3d,
                        marshak_wave, radiation_shadow, radiating_blast_3d
```
