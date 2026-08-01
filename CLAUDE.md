# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Overview

Hydro3D is a 3D radiation-hydrodynamics code in pure Julia. It solves the Euler equations with a method-of-lines scheme (HLLC or LLF) and grey two-moment (M1) radiation with an HLL solver and an IMEX PD-ARS integrator. It exists as a bench for prototyping algorithms and test problems destined for Quokka, so structural fidelity to Quokka is a design goal: a kernel written here should transliterate across, and back. The algorithms follow [Quokka](https://github.com/quokka-astro/quokka), whose source sits at `../quokka` and is the reference to check against when a numerical detail is in doubt.

## Commands

```bash
julia --project=. -t 4 test/runtests.jl             # full suite, ~25 s, 688 tests
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

**Use a package when it earns its place.** There is no prohibition here, and no
credit for hand-writing something a library does better. Justify the choice, put
it in the right tier, and move on.

The tiers, strongest preference first:

| Tier | What goes there | Cost to a user who doesn't use it |
|:---|:---|:---|
| `[weakdeps]` + `ext/` | anything not needed to *solve* -- plotting, ODE solvers, IO formats, analysis | nothing |
| `[extras]` / test target | anything used only by the test suite | nothing |
| `[deps]` | genuinely required to integrate the equations | paid by everyone, every run |

Prefer a weak dependency whenever the feature is optional, because the extension
mechanism makes that free: the methods appear when the user loads the package and
are absent otherwise. `src/plotting.jl` is the pattern -- stubs and docstrings in
`src/`, implementations in `ext/`, and a fallback that tells the caller which
package to load rather than throwing `MethodError`. Two worked examples:

- **`Plots`** — the four plotting entry points, in `ext/Hydro3DPlotsExt.jl`.
- **`OrdinaryDiffEq`** — `AdaptiveCoupling`, the error-controlled
  matter-radiation coupling. The two hand-written coupling solvers beside it
  need no packages, so the solver still starts on a machine with neither.

The `[deps]` list is currently `Printf` and `Dates`, both stdlibs. Keeping the
*solver* that light is still worth something — this code is meant to run on
compute nodes where a large tree is a liability, and a hard dependency is the one
tier every user pays for. So the bar for `[deps]` is real, but it is a bar, not a
wall: if integrating the equations genuinely needs a package, add it.

Three things were considered and written in-tree instead, each on merit rather
than principle, and each is worth revisiting if the calculus changes:

- `StaticArrays` — would replace `src/staticvec.jl`, 58 lines that are already
  allocation-free and deliberately *not* an `AbstractArray` so a stray broadcast
  cannot allocate.
- `HDF5` / `WriteVTK` — would replace a validated ~200-line writer in
  `src/output.jl`.
- a generic nonlinear solver — slower than the hand-written 2×2 Newton whose
  Jacobian inverts analytically, and slower than the arrowhead elimination in
  `src/coupling_solvers.jl`, which is `O(NG)` and forms no matrix.

What has not changed: keep hot kernels allocation-free, and don't let a
dependency into an inner loop without measuring what it costs there.

## Source layout

Include order in `src/Hydro3D.jl` matters, because later files use types defined earlier.

| File | Contents |
|:---|:---|
| `staticvec.jl` | `SVec{N,T}`, a stack-allocated fixed-size vector. Deliberately *not* an `AbstractArray`, so a stray broadcast cannot allocate. |
| `constants.jl` | `CGS` submodule. The code itself is unit-agnostic. |
| `eos.jl` | `AbstractEOS`, `IdealGas`. Energies are volumetric; `cv` is specific. |
| `opacity.jl` | `AbstractOpacity` and four concrete laws. Planck / energy / flux means are distinguished; `dkappa_dT` is the optional temperature derivative the coupling's exact Jacobian needs. |
| `grid.jl` | `CartesianGrid`, sweep index ranges, and the threading helpers `threaded_jk` / `reduce_max_jk`. |
| `variables.jl` | State-vector layout and pointwise conversions. |
| `boundary.jl` | `BoundaryCondition` subtypes and ghost filling. |
| `limiters.jl`, `reconstruction.jl` | Slope limiters; `PCM` / `PLM` / `PPM`. |
| `riemann/state.jl` | `HydroState` (field names match `quokka::HydroState`), `AbstractRiemannSolver`, the `riemann_flux` generic, and the direction permutation helpers. |
| `riemann/hllc.jl`, `riemann/llf.jl` | One solver per file, mirroring Quokka's `HLLC.hpp` / `LLF.hpp`. Pointwise, canonical (normal = x) form. |
| `riemann_radiation.jl` | M1 closure, Eddington tensor, `hll_radiation`. |
| `hydro_system.jl` | `Floors`, `cons2prim!` / `enforce_floors!`, the flux sweep, flux divergence, hydro CFL. |
| `radiation_system.jl` | `RadiationPhysics`, state repair, the HLL flux sweep, radiation CFL. |
| `source_terms.jl` | The implicit matter-radiation coupling — the hardest file. |
| `integrators.jl` | SSP-RK tableaux and the IMEX PD-ARS constants. |
| `simulation.jl` | `Simulation`, workspaces, right-hand sides, stage drivers, `run!`. |
| `output.jl` | VTK, text line-outs, checkpoints. |
| `plotting.jl` | `field` / `available_fields` / `coordinates`, plus the stubs and no-Plots fallbacks for the four plotting entry points. |
| `metrics.jl` | `relative_l1` / `relative_l2` and the `default_metrics` generic. |
| `compare.jl` | `compare`, `ComparisonTable`, and the table reader/writer. |
| `problems/` | Built-in test problems and the exact Riemann solver. |

`ext/Hydro3DPlotsExt.jl` sits outside `src/` and is loaded by Julia's extension mechanism when Plots is present.

## Invariants to preserve

**Interface convention.** Face `f` along direction `D` is the *low-side* face of cell `f`. So `qL[f]` is extrapolated from cell `f-1`, `qR[f]` from cell `f`, and the interior cells `lo:hi` are bounded by faces `lo:hi+1`. This is Quokka's convention; do not flip it.

**Array layout.** All state arrays are `(nvar, i, j, k)`. Variable index first means a cell's components are contiguous in Julia's column-major order, which is what the Riemann solvers want. Loop `for i in ir, n in 1:nvar` so `n` varies fastest.

**Degenerate directions.** `n[d] == 1` means direction `d` is inactive: zero ghost cells, skipped sweeps, no boundary condition needed. Anything that indexes `j±1` must check `grid.active[d]` first, or it will read out of bounds.

**Ghost width.** `nghost = max(required_ghosts(reconstruction), required_ghosts(radiation_reconstruction))`, minimum 2. The minimum-2 rule exists so the reconstruction sweep can write `qL` at cell `hi+1` without leaving the array; PCM does not need the extra cell but gets it anyway.

**Momentum index.** The momentum component for direction `d` is at index `d+1` in *both* the hydro and the radiation state vector. `fill_ghosts!` relies on this to serve both systems with one implementation.

**Conversion kernels are read-only.** `cons2prim!` and `rad_cons2prim!` only read the conserved state; projecting onto the admissible set is a separate operation, `enforce_floors!` / `enforce_rad_floors!` / `amend_rad_state!`. This mirrors Quokka's split between `EnforceLimits` and its `const`-qualified `ComputePrimVars`, and it is not stylistic: the two used to be fused, so *reading* a diagnostic field repaired the state being read, uncounted, and two runs differing only in plotting cadence could disagree. Cost of the split is ~3% of a step. `test_hydro.jl`'s "diagnostics do not perturb the state" is the guard. Never floor inside a conversion, an accessor, or `timestep`.

**The state is admissible between steps.** `step!` and `set_initial_conditions!` both end with `enforce_state!`, so every diagnostic, every output writer and `timestep` can assume `U` and `R` are physical. Intermediate Runge-Kutta stages are *not* covered — that is why the right-hand sides still floor on entry. `set_initial_conditions!` must fill ghosts before enforcing; ghost cells are uninitialised memory until the boundary conditions have run once.

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
- The Jacobian of *this* driver holds `kappa_P/kappa_E` fixed — it drops `d(kappa_P/kappa_E)/dT`. The residual does not: the opacity is re-evaluated at the current iterate's temperature every Newton step, as in Quokka. Dropping the derivative affects the convergence rate only, not the converged answer, because Newton still stops on the residual of the true equations.
- `J11 = -big` (a large finite number) rather than `-Inf` when `tau <= 0`: `Inf` arithmetic in the 2×2 solve produces `NaN`.
- Floors are applied at the end by moving energy between `Eint` and `Erad`, so total energy is still conserved.

Any change here must keep two tests passing: `matter_coupling` at `kappa = 1e9`, which takes one step a hundred million times longer than the coupling time, and `radhydro_shock`, the Lowrie & Edwards Mach 3 radiative shock. The second is the sensitive one -- its Zel'dovich spike only appears if the velocity-dependent coupling terms, the implicit solve and the shock capturing are all correct together, so it is the test that will catch a subtle regression here.

### The multigroup coupling solvers

`NG >= 2` does not use the driver above. It packs each cell into a `CouplingState` and hands it to an `AbstractCoupling` (`src/coupling.jl`, `src/coupling_solvers.jl`, and the `AdaptiveCoupling` extension). Those solvers **do** carry the `dkappa/dT` term, so their Jacobians are exact:

- `coupling_rhs` and `coupling_jacobian` are the single source of truth. `DirectCoupling`'s residual is `u - u0 - dt f(u)` and its Newton matrix `I - dt J`, generated, not hand-derived; `AdaptiveCoupling` hands `J` to `ODEFunction`.
- `QuokkaCoupling` is a separate formulation and stays hand-derived. `R_g` is the integrated transfer, not a state variable, so there is no `dR_g/dt` to differentiate — it is a different discretisation, not a change of variables. It shares `coupling_rhs` (its opening guess is one explicit step of it) and nothing else.
- The **opacity law travels on the state**, not a frozen pair of numbers, so `coupling_rhs` evaluates `kappa(T)` at whatever temperature it is called with. Without that the `dkappa/dT` term in the Jacobian would be the derivative of something the residual cannot see, and a Rosenbrock method would be worse off with it than without.
- An opacity supplies its derivative through `dkappa_dT` / `group_opacity_dT`, defaulting to zero. Zero costs iterations, not correctness. Measured on a `PowerLawOpacity`: the exact term cuts mean Newton iterations by 25–40% in the intermediate regime, and by nothing at all deep in equilibrium, where the correction is proportional to `B_g - E_g`.
- The optically-thin shortcut, the equilibrium-overshoot guard and `apply_energy_floors` are **not** ODE terms and are deliberately outside `coupling_rhs`. The header of `src/coupling.jl` says where each lives.

## Source terms

Two hooks, and they are not interchangeable.

`source` is a *radiation* energy source, consumed inside `radiation_source_step!` so that a stiff emission term is handled implicitly along with the rest of the coupling. `strang_sources` is the general operator-split hook, `f(sim, t, dt, order)`, applied at `dt/2` either side of the advance — the transliteration of Quokka's `addStrangSplitSources`, called from the same two places in the driver and with the same halved step.

Nothing is built on `strang_sources`. It is there so that gravity, cooling or driving can be added without editing `step!`, which is what the review of this code identified as the ceiling on "algorithm experimentation".

Two properties the tests pin, and that a change here must keep:

- The two half-steps are the trapezoid rule in time, so a source with a linear `a(t)` is integrated *exactly* and anything else is second order. A single full-step kick would be first order and the convergence test would catch it.
- `apply_strang_sources!` calls `enforce_state!` after the hook. A user source can push a cell out of the admissible set, and with hydrodynamics disabled the radiation coupling reads `U` before any right-hand side would floor it.

## Closures

`RadiationPhysics` carries a `closure`. `LevermoreClosure()` is the M1 default. `EddingtonClosure()` fixes `chi = 1/3`, cannot free-stream, and exists only so that reference solutions derived under the P1 approximation can be reproduced -- `radhydro_shock` needs it, exactly as Quokka overrides `ComputeEddingtonFactor` for the same problem. Do not make it the default, and do not use it for anything with a directional radiation field.

## Style

- Every exported name has a docstring saying what it is *for*, not just what it does.
- Comments explain *why*. Reference the paper and equation when a formula is not obvious (`Minoshima & Miyoshi (2021) eq. 23`).
- Concrete types in struct fields; `where {T}` on kernels; `@inbounds` only in loops whose ranges are provably safe.
- Match the surrounding code. Do not reformat or refactor adjacent code that the task does not touch.

## Testing changes

Physics changes need a test that would fail without them. The suite already provides the shape:

- Analytic comparison where one exists (`sod_exact`, `diffusion_exact`, the radiation drag law).
- Order-of-accuracy checks rather than fixed error thresholds, where the answer is only asymptotically right.
- Invariants that must hold exactly: conservation on periodic boundaries, direction isotropy, `|F| <= cE`, positivity without floor repairs (`nrepaired(sim) == 0`), and state that a read cannot perturb (`sim.U == U0` after every accessor).

Prefer the last kind. They catch real bugs and do not need retuning when the scheme improves.
