# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Overview

Hydro3D is a 3D radiation-hydrodynamics code in pure Julia. It solves the Euler equations with an HLLC method-of-lines scheme and grey two-moment (M1) radiation with an HLL solver and an IMEX PD-ARS integrator. The algorithms follow [Quokka](https://github.com/quokka-astro/quokka), whose source sits at `../quokka` and is the reference to check against when a numerical detail is in doubt.

## Commands

```bash
julia --project=. -t 4 test/runtests.jl             # full suite, ~9 s, 405 tests
julia --project=. -t auto test/runtests.jl --figures # ... and redraw docs/figures/
julia --project=. -t auto examples/sod_1d.jl        # run an example (writes a figure)
julia --project=. -t auto docs/make_figures.jl      # regenerate README figures
julia --project=. -t auto docs/make_animations.jl   # regenerate README GIFs
julia --project=. -t 8 test/benchmarks.jl           # timing table (BenchmarkTools)
julia --project=. -e 'include("src/Hydro3D.jl")'    # syntax check without precompiling
```

## Scripts and figures

Examples share `examples/common.jl`, which handles the flags (`--no-plot`, `--outdir=DIR`), leaves positional arguments in `OPTS`, sets `GKSwstype` for headless GR, and exposes `PLOTTING`, `outpath`, `saveplot` and `plotting_note`. A new example should `include` it, guard its plotting block with `PLOTTING`, route every output path through `outpath`, and end with `plotting_note()`.

Figures are **on by default** in examples: an example you have to post-process by hand is not much of an example. They are **off by default** in the test suite, because re-running every problem at production resolution costs a couple of minutes and writes files into the repo; `--figures` (or `HYDRO3D_FIGURES=1`) turns them on, and runs `docs/make_figures.jl` in a subprocess so its globals cannot collide with the test session.

Prefer building example figures out of `plotstate` / `plotpanels` rather than raw Plots calls. It keeps the examples short and doubles as documentation for the plotting API.

## Dependencies

The runtime dependency list is `Printf` and nothing else, and it should stay that way. That is a design property, not an accident: this code is meant to run on nodes where a large dependency tree is a liability. Two things that would normally be a package are written in-tree instead — `src/staticvec.jl` (a fixed-size vector) and the VTK writer in `src/output.jl`.

Anything that is not needed to *solve* goes in a weak dependency or a test extra:

- **`Plots`** is a `[weakdeps]` entry. The plotting methods live in `ext/Hydro3DPlotsExt.jl` and appear only when the user loads Plots. `src/plotting.jl` holds the function stubs, their docstrings, and fallbacks that raise a message telling the caller to load Plots. Add new plotting features to the extension, never to `src/`.
- **`BenchmarkTools`** is a test extra, used only by `test/benchmarks.jl`.

Before adding any dependency, ask whether it earns its place. `StaticArrays` would replace 58 lines of `SVec` that are already allocation-free; `HDF5`/`WriteVTK` would replace a validated 200-line writer; a generic nonlinear solver would be slower than the hand-written 2×2 Newton whose Jacobian inverts analytically. All three were considered and declined.

## Source layout

Include order in `src/Hydro3D.jl` matters, because later files use types defined earlier.

| File | Contents |
|:---|:---|
| `staticvec.jl` | `SVec{N,T}`, a stack-allocated fixed-size vector. Deliberately *not* an `AbstractArray`, so a stray broadcast cannot allocate. |
| `constants.jl` | `CGS` submodule. The code itself is unit-agnostic. |
| `eos.jl` | `AbstractEOS`, `IdealGas`. Energies are volumetric; `cv` is specific. |
| `opacity.jl` | `AbstractOpacity` and three concrete laws. Planck / energy / flux means are distinguished. |
| `grid.jl` | `CartesianGrid`, sweep index ranges, and the threading helpers `threaded_jk` / `reduce_max_jk`. |
| `variables.jl` | State-vector layout and pointwise conversions. |
| `boundary.jl` | `BoundaryCondition` subtypes and ghost filling. |
| `limiters.jl`, `reconstruction.jl` | Slope limiters; `PCM` / `PLM` / `PPM`. |
| `riemann_hydro.jl` | The pointwise `hllc` solver, in canonical (normal = x) form. |
| `riemann_radiation.jl` | M1 closure, Eddington tensor, `hll_radiation`. |
| `hydro_system.jl` | `Floors`, `cons2prim!`, the HLLC flux sweep, flux divergence, hydro CFL. |
| `radiation_system.jl` | `RadiationPhysics`, state repair, the HLL flux sweep, radiation CFL. |
| `source_terms.jl` | The implicit matter-radiation coupling — the hardest file. |
| `integrators.jl` | SSP-RK tableaux and the IMEX PD-ARS constants. |
| `simulation.jl` | `Simulation`, workspaces, right-hand sides, stage drivers, `run!`. |
| `output.jl` | VTK, text line-outs, checkpoints. |
| `plotting.jl` | `field` / `available_fields` / `coordinates`, plus the stubs and no-Plots fallbacks for the three plotting entry points. |
| `problems/` | Built-in test problems and the exact Riemann solver. |

`ext/Hydro3DPlotsExt.jl` sits outside `src/` and is loaded by Julia's extension mechanism when Plots is present.

## Invariants to preserve

**Interface convention.** Face `f` along direction `D` is the *low-side* face of cell `f`. So `qL[f]` is extrapolated from cell `f-1`, `qR[f]` from cell `f`, and the interior cells `lo:hi` are bounded by faces `lo:hi+1`. This is Quokka's convention; do not flip it.

**Array layout.** All state arrays are `(nvar, i, j, k)`. Variable index first means a cell's components are contiguous in Julia's column-major order, which is what the Riemann solvers want. Loop `for i in ir, n in 1:nvar` so `n` varies fastest.

**Degenerate directions.** `n[d] == 1` means direction `d` is inactive: zero ghost cells, skipped sweeps, no boundary condition needed. Anything that indexes `j±1` must check `grid.active[d]` first, or it will read out of bounds.

**Ghost width.** `nghost = required_ghosts(reconstruction)`, minimum 2. The minimum-2 rule exists so the reconstruction sweep can write `qL` at cell `hi+1` without leaving the array; PCM does not need the extra cell but gets it anyway.

**Momentum index.** The momentum component for direction `d` is at index `d+1` in *both* the hydro and the radiation state vector. `fill_ghosts!` relies on this to serve both systems with one implementation.

**Direction permutation.** For a sweep along `D`, the canonical transverse directions are `_tdir1(Val(D))`, `_tdir2(Val(D))` in cyclic order (1→2,3; 2→3,1; 3→1,2). These are written out as literal methods rather than computed, so they constant-fold.

## Performance rules

The inner loops must stay allocation-free. Two traps have already bitten this code:

1. **Closures inside the cell loop box their captured variables.** An `ntuple(Val(3)) do a … end` in `source_terms.jl` captured Newton-loop variables that were assigned in several places, so Julia boxed them; every assignment then heap-allocated. That cost 406 MB and 200 ms per source step at 32³. Write such kernels out component by component instead.
2. **Threaded reductions must not use `threadid()`.** `Threads.@threads` defaults to dynamic scheduling, so `threadid()` is not stable across a chunk. `reduce_max_jk` writes partial results to distinct slots keyed by the loop index instead.

`test/test_performance.jl` guards both: it asserts that every hot kernel allocates under 8 bytes per interior cell, and that allocation per step does not grow with the grid. Those are the tests that would have caught the boxing bug, so extend them whenever you add a kernel.

When changing a hot kernel, check with

```julia
a = @allocated some_kernel!(args...)   # after one warm-up call
```

and profile with `Profile.Allocs.@profile sample_rate=0.02` if it is non-zero. `test/benchmarks.jl` gives the per-kernel timing breakdown.

Directional kernels take `Val(D)` so `D` is a compile-time constant. Call them from an explicit `grid.active[1] && sweep!(…, Val(1))` chain, never `for D in 1:3` with a runtime `Val(D)`.

## The implicit source term

`radiation_source_step!` solves, per cell, a 2×2 Newton-Raphson system in `(Eint, Rex)` where `Rex` is the energy transferred from gas to radiation over the step. Using `Rex` rather than `Erad` as the second unknown is what keeps the iteration conditioned from optically thin all the way into equilibrium.

Structure, and why each piece is there:

- The **outer loop** exists only for the lagged `v·F` work term; with `beta_order = 0` or hydro disabled it runs once.
- `tau = dt * rho * kappa_P * chat` is the optical depth across `chat*dt`. When `tau` is negligible the residual can never be balanced, so there is an explicit optically-thin shortcut. Without it the iteration burns all 100 steps in transparent cells.
- The Jacobian holds `kappa_P/kappa_E` fixed. That affects the convergence rate only, not the converged answer.
- `J11 = -big` (a large finite number) rather than `-Inf` when `tau <= 0`: `Inf` arithmetic in the 2×2 solve produces `NaN`.
- Floors are applied at the end by moving energy between `Eint` and `Erad`, so total energy is still conserved.

Any change here must keep the `matter_coupling` test passing at `kappa = 1e9`, which takes one step a hundred million times longer than the coupling time.

## Style

- Every exported name has a docstring saying what it is *for*, not just what it does.
- Comments explain *why*. Reference the paper and equation when a formula is not obvious (`Minoshima & Miyoshi (2021) eq. 23`).
- Concrete types in struct fields; `where {T}` on kernels; `@inbounds` only in loops whose ranges are provably safe.
- Match the surrounding code. Do not reformat or refactor adjacent code that the task does not touch.

## Testing changes

Physics changes need a test that would fail without them. The suite already provides the shape:

- Analytic comparison where one exists (`sod_exact`, `diffusion_exact`, the radiation drag law).
- Order-of-accuracy checks rather than fixed error thresholds, where the answer is only asymptotically right.
- Invariants that must hold exactly: conservation on periodic boundaries, direction isotropy, `|F| <= cE`, positivity without floor repairs (`sim.nrepaired == 0`).

Prefer the last kind. They catch real bugs and do not need retuning when the scheme improves.
