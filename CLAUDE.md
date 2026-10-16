# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Overview

Hydro3D is a 3D radiation-hydrodynamics code in pure Julia. It solves the Euler equations with a method-of-lines scheme (HLLC or LLF) and two-moment (M1) radiation — grey, or in `NG` frequency groups — with an HLL solver and an IMEX PD-ARS integrator. It exists as a bench for prototyping algorithms and test problems destined for Quokka, so structural fidelity to Quokka is a design goal: a kernel written here should transliterate across, and back. The algorithms follow [Quokka](https://github.com/quokka-astro/quokka), whose source sits at `../quokka` and is the reference to check against when a numerical detail is in doubt.

## Commands

```bash
julia --project=. -t 4 test/runtests.jl             # full suite, ~80 s
julia --project=. -t auto test/runtests.jl --figures # ... and redraw docs/figures/
julia --project=. -t auto examples/sod_1d.jl        # run an example (writes a figure)
julia --project=. -t 4 examples/irsla_paper.jl --outdir=DIR [section ...]  # every number of the ARSA paper's Sec. 5
julia --project=. -t auto docs/make_figures.jl      # regenerate README figures
julia --project=. -t auto docs/make_animations.jl   # regenerate README GIFs
julia --project=. docs/make.jl                      # build the docs site into docs/_site (needs quarto)
julia --project=. -t 8 test/benchmarks.jl           # timing table (BenchmarkTools)
julia --project=. -e 'include("src/Hydro3D.jl")'    # syntax check without precompiling
```

## Scripts and figures

Examples share `examples/common.jl`, which handles the flags (`--no-plot`, `--outdir=DIR`), leaves positional arguments in `OPTS`, sets `GKSwstype` for headless GR, and exposes `PLOTTING`, `outpath`, `saveplot` and `plotting_note`. A new example should `include` it, guard its plotting block with `PLOTTING`, route every output path through `outpath`, and end with `plotting_note()`.

Figures are **on by default** in examples: an example you have to post-process by hand is not much of an example. They are **off by default** in the test suite, because re-running every problem at production resolution costs a couple of minutes and writes files into the repo; `--figures` (or `HYDRO3D_FIGURES=1`) turns them on, and runs `docs/make_figures.jl` in a subprocess so its globals cannot collide with the test session.

Prefer building example figures out of `plotstate` / `plotpanels` rather than raw Plots calls. It keeps the examples short and doubles as documentation for the plotting API.

## Documentation site

`docs/make.jl` writes one reference page per group of source files (its `PAGES` table) from the docstrings, uses the README as the front page and four notes in `docs/` as further pages, renders them with Quarto, and publishes every page twice: as HTML and as its own source, `<page>.md`, with `hydro3d-docs.md` bundling them all. CI deploys it to GitHub Pages from `main`. The build is strict, and three things fail it: a `[`name`](@ref)` whose target has no docstring, a file in `src/` with a docstring but no entry in `PAGES`, and any relative link or anchor in the output that does not resolve. So a new source file needs a line in `PAGES`, and a cross-reference to an undocumented name needs the docstring, not the brackets removed. `docs/documentation.md` (also a page on the site) explains how the system works and how to update each kind of page.

## Dependencies

**Use a package when it earns its place.** There is no prohibition here, and no
credit for hand-writing something a library does better. Justify the choice, put
it in the right tier, and move on.

The tiers, strongest preference first:

| Tier | What goes there | Cost to a user who doesn't use it |
|:---|:---|:---|
| `[weakdeps]` + `ext/` | a feature a user can plausibly not want at all — plotting, IO formats, analysis | nothing |
| `[extras]` / test target | anything used only by the test suite | nothing |
| `[deps]` | anything the package itself calls, on any path a user might take | load and precompile time, paid by everyone, every run |

Prefer a weak dependency when the feature is genuinely optional, because the
extension mechanism makes that free: the methods appear when the user loads the
package and are absent otherwise. `src/plotting.jl` is the pattern — stubs and
docstrings in `src/`, implementations in `ext/`, and a fallback that tells the
caller which package to load rather than throwing `MethodError`. **`Plots`** is
the worked example, and now the only one: the four plotting entry points live in
`ext/Hydro3DPlotsExt.jl`, and a compute node never pays for the ~100-package
plotting stack.

`OrdinaryDiffEq` used to be the second, and is not any more. It is a hard
dependency, and the reasoning is worth keeping because it is what the tiers
actually mean:

- One solver needs it — `AdaptiveCoupling`, the error-controlled reference, and
  it is the only coupling path that *can* need it: the other two eliminate the
  group block algebraically, which is possible after an implicit step is taken
  and not along an ODE trajectory. A solver behind an extension boundary is one
  that most checkouts never run: OrdinaryDiffEq was simply not installed here,
  `test_coupling_adaptive.jl` skipped itself, and the suite reported a clean 1027
  passes with a third of the coupling code unexercised. A dependency that is
  *required for a code path* does not belong in `[weakdeps]`, whatever it costs.
- What it costs, measured: `using Hydro3D` goes from 0.035 s to 1.53 s, and a
  fresh install precompiles ~190 packages instead of none. Wall time of a run is
  unaffected. That is real and it is the price of the decision.
- One `[deps]` entry, not three. It used to be three: OrdinaryDiffEq 7 is a slim
  meta-package that exports only Tsit5, the Verner methods, `Rosenbrock23`,
  `Rodas5P` and `FBDF`, so `ImplicitEuler` had to come from `OrdinaryDiffEqSDIRK`
  and `NLNewton` from `OrdinaryDiffEqNonlinearSolve`. Both went with
  `ImplicitEulerStepper`. Removing them changed the manifest not at all, since
  they were already inside OrdinaryDiffEq's closure — but naming a package in
  `[deps]` says this code calls it, and it no longer does.
- **Import selectively.** `using OrdinaryDiffEq: ODEProblem, ODEFunction, …` and
  not `using OrdinaryDiffEq`, because it exports `step!` and so does this
  package; a blanket import would make `step!(sim)` ambiguous for every user with
  both in scope. `using LinearAlgebra: lu!, ldiv!` is there for the same reason
  (`dot` and `norm`), and the two comments in `src/Hydro3D.jl` say so.

`SciMLBase` is the other package: `BracketedCoupling` hands its one scalar
equation, per cell, per step, through SciMLBase's `IntervalNonlinearProblem`
interface, and `ZBrent` in `src/rootfind.jl` implements that interface. It was
already inside OrdinaryDiffEq's closure and already loaded, so it cost nothing.

`BracketingNonlinearSolve` is the worked example of a library measured in an
inner loop and then *not* kept there. It was adopted first, after a gate
measured 0 bytes per solve and no failures over the 240-cell sweep. Its `Brent`
then turned out to be a transcription of textbook pseudocode rather than of
Brent's algorithm: no minimum step, and `f(c)` recomputed every iteration. On
the coupling's equation that meant 20.1 evaluations a cell and 70 in the worst,
against 7.7 and 12 for an in-tree `zbrent` of about 60 lines of code. So `ZBrent` is written in-tree, and
BracketingNonlinearSolve is test-only (`[extras]`): the suite checks `ZBrent`
against its `ModAB`, `ITP` and `Bisection`. `[extras]` is not on the load path
when the suite is run directly, so `test/runtests.jl` loads it by UUID -- it is in
the manifest anyway, through OrdinaryDiffEq. Two things about bracketing
root-finders that are easy to get wrong, both handled in
`solve_coupling(::BracketedCoupling, …)`:

- **Every method stops on bracket width, never on the residual.** Where the
  residual is what you promise — here it is the cell's energy-conservation
  error — the tolerance has to be converted by the function's derivative, or the
  promise silently loosens by that factor.
- **A function argument that is only passed through is not specialised on.**
  `_energy_bracket(G::F, s) where {F}` rather than `(G, s)`: without the type
  parameter the same code allocated 2.8 kB a cell.

The `[deps]` list is `Printf`, `Dates`, `LinearAlgebra` (all stdlibs),
`OrdinaryDiffEq` and `SciMLBase`. The bar for `[deps]` is still real — this code is
meant to run on compute nodes where a large tree is a liability — but it is a
bar, not a wall, and "the package calls it" clears it.

Three things were considered and written in-tree instead, each on merit rather
than principle, and each is worth revisiting if the calculus changes:

- `StaticArrays` — would replace `src/staticvec.jl`, 58 lines that are already
  allocation-free and deliberately *not* an `AbstractArray` so a stray broadcast
  cannot allocate. Note that this one is now *load-bearing* for the ODE solvers:
  an `SVec`'s element type is fixed, so no dual number can flow through
  `coupling_rhs`, which is why they are handed an analytic Jacobian and `tgrad`.
- `HDF5` / `WriteVTK` — would replace a validated ~200-line writer in
  `src/output.jl`.
- a generic nonlinear solver — the arrowhead elimination in `src/newton.jl` is
  `O(NG)` and forms no matrix, and `QuokkaCoupling` around it is 0 bytes and
  ~1 μs a cell. `ImplicitEuler` was in the tree for a while as the measurement of
  that claim rather than an assertion of it — at four groups it was 10 kB and 8×
  the wall time for the same answer to the same tolerance — and it is now gone
  with `DirectCoupling`; the numbers are in `docs/coupling-findings.md` §7.

What has not changed: keep hot kernels allocation-free, and don't let a
dependency into an inner loop without measuring what it costs there. The package
default, `BracketedCoupling`, allocates nothing per cell -- it reduces the step
to one scalar unknown, so it does not even reach `src/newton.jl`, and the
root-finder it calls was measured at zero bytes before it was let in.

## Source layout

Include order in `src/Hydro3D.jl` matters, because later files use types defined earlier.

| File | Contents |
|:---|:---|
| `staticvec.jl` | `SVec{N,T}`, a stack-allocated fixed-size vector. Deliberately *not* an `AbstractArray`, so a stray broadcast cannot allocate. |
| `rootfind.jl` | `ZBrent`: Brent's method for a bracketed scalar root, as Brent (1973) and Numerical Recipes' `zbrent` write it -- with the minimum step and one evaluation per iteration. Implements SciMLBase's `IntervalNonlinearProblem` interface. No physics. |
| `newton.jl` | The linear solve of one Newton step on a bordered (arrowhead) system — `ArrowheadSolve`, `DenseSolve`. No physics: the caller supplies the four Jacobian blocks. `QuokkaCoupling` is the only caller left. |
| `constants.jl` | `CGS` submodule. The code itself is unit-agnostic. |
| `planck.jl` | The normalised incomplete Planck integral (two series, no table), `RadiationGroups`, the group centres and the Planck fractions. Included before `opacity.jl`, which needs the group type. |
| `eos.jl` | `AbstractEOS`, `IdealGas`. Energies are volumetric; `cv` is specific. |
| `opacity.jl` | `AbstractOpacity` and four concrete laws. Planck / energy / flux means are distinguished; `dkappa_dT` is the optional temperature derivative the coupling's exact Jacobian needs. |
| `dust.jl` | `AbstractDustModel`, `NoDust`, `BateKetoDust`: whether the radiation couples to the gas or to dust in instantaneous equilibrium, and the Keto & Field collision rate between dust and gas. The model only; the solve is in the coupling files. |
| `rsla.jl` | The rule that makes the reduced speed of light local: `AbstractChatLimiter` (`ConstantChat`, `StabilityBound`, `OpticalDepthFit`), the `LocalCell` it reads, the pointwise `chat_factor`, and the `per_group` / `flux_form` traits. The array drivers that apply it are in `radiation_system.jl`, as `source_terms.jl` holds the loop over cells for `coupling.jl`'s physics. |
| `coupling.jl` | `AbstractCoupling`, `CouplingState`, and the coupling physics in three layers: `coupling_coefficients` (the per-group emission rate and absorption coefficient — the single source of truth), `coupling_rhs` / `coupling_jacobian` built from it for the ODE, and `group_energies` / `energy_balance` built from it for the scalar reduction. |
| `coupling_solvers.jl` | The two one-step solvers: `BracketedCoupling` (the default; its scalar equation goes to a bracketing root-finder, `ZBrent` by default, in `E_gas` without dust and in `T_d` with it) and `QuokkaCoupling`. |
| `coupling_ode.jl` | `AdaptiveCoupling`, the error-controlled reference solve, and the `ODEFunction` it is written against. |
| `grid.jl` | `CartesianGrid`, sweep index ranges, and the threading helpers `threaded_jk` / `reduce_max_jk`. |
| `variables.jl` | State-vector layout and pointwise conversions. |
| `boundary.jl` | `BoundaryCondition` subtypes and ghost filling. |
| `limiters.jl`, `reconstruction.jl` | Slope limiters; `PCM` / `PLM` / `PPM`. |
| `riemann/state.jl` | `HydroState` (field names match `quokka::HydroState`), `AbstractRiemannSolver`, the `riemann_flux` generic, and the direction permutation helpers. |
| `riemann/hllc.jl`, `riemann/llf.jl` | One solver per file, mirroring Quokka's `HLLC.hpp` / `LLF.hpp`. Pointwise, canonical (normal = x) form. |
| `riemann_radiation.jl` | M1 closure, Eddington tensor, `hll_radiation` (its `epsE` is the optional wave-speed correction). |
| `hydro_system.jl` | `Floors`, `cons2prim!` / `enforce_floors!`, the flux sweep, flux divergence, hydro CFL. |
| `radiation_system.jl` | `RadiationPhysics`, the group layout (`ngroups`, `nradvars`, `gbase`), state repair, the HLL flux sweep, radiation CFL, the two drivers of the `chat` limiter (`update_chat!`, `energy_exchange_dt`), `scale_by_phi!`, and `floor_rad_energy!`, the only repair allowed before a coupling solve. |
| `source_terms.jl` | The implicit matter-radiation coupling — the hardest file. Two drivers, chosen by dispatch on `NG`: the grey 2×2 Newton written out inline, and `multigroup_source_step!`, which hands each cell to an `AbstractCoupling`. |
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

`NG >= 2` does not use the driver above. It packs each cell into a `CouplingState` and hands it to an `AbstractCoupling` (`src/coupling.jl`, `src/coupling_solvers.jl`, `src/coupling_ode.jl`). Those solvers **do** carry the `dkappa/dT` term, so their Jacobians are exact:

- **`coupling_coefficients` is the single source of truth**, not `coupling_rhs`. It returns the per-group emission rate `rho kappa_P B` and absorption coefficient `rho kappa_E`, with `coupling_coefficients_dT` for their temperature derivatives, and everything else is built from it: `coupling_rhs` / `coupling_jacobian` for `AdaptiveCoupling`, and `group_energies` / `energy_balance` for the two one-step solvers. It is stated as that pair rather than as a rate because the group block of a backward-Euler step is linear in `E_g` *precisely when* the exchange has the form `emission - absorption * E_g`, and a solver that eliminates the block needs to see that, not a rate.
- **`BracketedCoupling` is the default.** It takes `E_gas` as its *only* unknown: `group_energy_values` solves the group block in closed form, `energy_balance` is the scalar equation left, and that equation is handed as a lambda to a bracketing root-finder from BracketingNonlinearSolve (`root_finder`, `ZBrent()` by default; BracketingNonlinearSolve's `ModAB()` and `Bisection()` plug in unchanged). No `1 + NG` system, no Jacobian assembly, no linear solve, zero bytes.
- **Three things the solve adds to a bare root-finder call, and each has a test.** The bracket is *marched outward from the old gas energy* by factors of two, because with a steeply temperature-dependent opacity the step can have several roots and a bracketing method returns one without saying which — on the wide bracket `[cv Tfloor, E_tot]` one sweep cell returned a gas energy 23 times the physical one, depending on the method. The *tolerance is converted*, because the library stops on bracket width in `E_gas` while the promise is on `G`, the energy-conservation error, which differs by `dG/dE_gas`. And `iterations` counts the march plus one per root-finder call, because the library does not report evaluations and counting them through the lambda allocates.
- **Why it is the default, and this is the thing to know before changing it.** The unreduced `(1 + NG)` Newton does not merely overshoot with a steeply temperature-dependent opacity — its Newton *direction* goes wrong, because it linearises about a state where the group block is unsatisfied and `E_g` is far from equilibrium with `T`. Over 240 cells spanning three opacity laws and `tau_cell` from 1e-4 to 1e12, `BracketedCoupling` fails none and `QuokkaCoupling` fails 75. `docs/coupling-findings.md` derives it in §5 and `docs/coupling-experiments/coupling_study.jl` reproduces every number.
- The reduction also makes the convergence test unconditional: `G` is a difference of energies with no `tau (B - E)` left in it to cancel, so nothing amplifies round-off by `tau_cell`. That is the failure Quokka's replaced base variables exist to avoid, avoided a different way — see §2 and §3 of the findings.
- `linsolve` belongs to `QuokkaCoupling` alone now; it is the only solver that forms a Newton system at all.
- Every distinct `CouplingState{NG,M,T,O}` makes OrdinaryDiffEq specialise its whole integrator stack again, ~2–3 s of compilation each. It is why the test suite defines its opacity helpers once at file scope, and why a run pays this once rather than per cell.
- `QuokkaCoupling` is a separate formulation and stays hand-derived. `R_g` is the integrated transfer, not a state variable, so there is no `dR_g/dt` to differentiate — it is a different discretisation, not a change of variables. It shares `coupling_rhs` (its opening guess is one explicit step of it) and nothing else. It is kept because Quokka runs it and structural fidelity to Quokka is a design goal here, and because it and `BracketedCoupling` agreeing is worth more than either alone: one unknown against `1 + NG`, sharing no residual, no iteration and no linear solve.
- The **opacity law travels on the state**, not a frozen pair of numbers, so `coupling_rhs` evaluates `kappa(T)` at whatever temperature it is called with. Without that the `dkappa/dT` term in the Jacobian would be the derivative of something the residual cannot see, and a Rosenbrock method would be worse off with it than without.
- An opacity supplies its derivative through `dkappa_dT` / `group_opacity_dT`, defaulting to zero. Zero costs iterations, not correctness. Measured on a `PowerLawOpacity`: the exact term cuts mean Newton iterations by 25–40% in the intermediate regime, and by nothing at all deep in equilibrium, where the correction is proportional to `B_g - E_g`.
- The optically-thin shortcut, the equilibrium-overshoot guard and `apply_energy_floors` are **not** ODE terms and are deliberately outside `coupling_rhs`. The header of `src/coupling.jl` says where each lives.
- The arrowhead elimination is an **assumption, not a truth**: it is exact only while groups exchange energy solely through the gas, which is a property of the terms in `coupling_rhs` today, not of radiation. `QuokkaCoupling` takes a `linsolve` (`AbstractJacobianSolve`) and **defaults to `ArrowheadSolve()`** — `O(NG)`, no matrix formed, zero bytes per cell. Defaulting to an assumption is acceptable only because this one is *checked*: `DenseSolve` solves the same system without it, and `test/test_radiation_multigroup.jl` asserts the two agree block by block and end to end. That cross-check is what licenses the default, and it exists because the failure would be quiet — Newton stops on the residual of the true equations, so a damaged matrix cannot converge to a wrong answer, only fail to converge and bump `newton_failures`. Add a term that couples groups directly (Compton redistribution, line overlap) and switch to `DenseSolve`, which needs only to write a block where it writes a diagonal; the arrowhead has nowhere to put the entries at all. **`group_energies` makes the same assumption and is worse off**: its closed form is a per-group scalar division, and a non-diagonal group block would leave no closed form at all, so `BracketedCoupling` would have to go rather than be adjusted.

### Dust-gas thermal coupling

`RadiationPhysics(; dust = BateKetoDust(coefficient = …))` makes the radiation couple to dust at its own temperature `T_d`, and the dust to the gas by collisions; the opacity law is then the dust's and the gas has none. `docs/coupling-new-method-dust.md` derives the method and holds the measurements. Four things to know before changing it:

- **The unknown is `T_d`, not `E_gas`.** Given `T_d`, `_eliminate_groups` gives the group energies unchanged and `energy_balance = 0` gives the gas energy, so `dust_energy_balance` leaves one equation, the gas equation. Total energy is conserved to round-off at every iterate, not only at the root. There is no "decoupled" regime and no threshold: Quokka's method divides by the collision coefficient, this one only multiplies by it, and `K = 0` is an ordinary case.
- **Convergence is tested on the state, not the residual.** The residual's collision term amplifies round-off in `T - T_d` by `dt K sqrt(T)`; with the residual test the sweep in `test_dust.jl` reports 127 of 432 cells unconverged, all correct. The test is instead how far `E_gas` differs across the root-finder's final bracket, which bounds the error of the returned state.
- **`BracketedCoupling` only.** Every other solver's `solve_coupling` is written for `CouplingState{…,NoDust}`, a fallback throws for a dust state, and `RadiationPhysics` refuses the combination. A grey run with dust goes through `multigroup_source_step!`, because the grey driver's inline Newton has no dust; the grey driver's dispatch is `NoDust` only, so grey runs without dust are untouched.
- **The flux update uses `res.Tdust`.** The flux-mean opacity and `B` in the `v/c` terms are evaluated at the dust temperature, as in Quokka. `CouplingResult.Tdust` is `NaN` without dust.

## The variable reduced speed of light

`chat` is a field, not a number: `sim.rw.chat[g, i, j, k]`, one value per cell and per group. `RadiationPhysics` carries a `chat_limiter` that sets it. `ConstantChat()`, the default, is the standard RSLA and a measured no-op, down to the ghost fills. `StabilityBound()` is the adaptive reduced speed of light approximation (ARSA) of He, Krumholz & Wibking, "An Adaptive Reduced Speed of Light Approximation" (J. Comput. Phys., in preparation), and this code is its reference implementation; the code and the example names still call it IRSLA. Every cell and group runs at the largest `phi = chat/c` that the PD-ARS step supports stably at the time step of the standard RSLA, `phi = min(1, max(phi0, beta / (c dt Lambda)))`, with `Lambda` the `stability_denominator`. `OpticalDepthFit` is the earlier two-parameter fitting form; it carries no stability bound and is kept to reproduce old results.

`examples/irsla_paper.jl` produces every number and data file of the paper's Sec. 5, one section at a time, and `examples/linear_waves_rhd.jl` the linear-wave test; `test/test_rsla.jl` pins each point below. `docs/rsla-findings.md` holds the measurements in two parts: *Standard RSLA* (S1–S4) for the global reduction on its own, and *ARSA* (A1–A9), where A1 is the method as `b287f03` left it and A2–A9 are the findings, measured with the earlier limiter, that led there. `git log irsla-variable-chat-stage2` has the reasoning commit by commit.

Things about it that are not obvious from the code:

- **The bound is exact, and only for this scheme.** A von Neumann analysis of one PD-ARS substep with HLL transport finds that the grid-scale checkerboard is always the worst mode, and there the coupling splits into an energy block and a momentum block, each with a closed-form boundary `phi c dt [sum_d sigma_d/dx_d - chi/(2 + K)] <= 1`: `chi_E` and `K = 4 a T^3 kappa_P c dt / c_v` for the energy block, `chi_F` and `K` dropped (smaller by `(a/c)^2`) for the momentum block. A checkerboard grows at 1.1 times the bound and decays at 0.9 times it, in one, two and three dimensions. The optical depth enters as `chi / sum_d(1/dx_d)`; the earlier `chi dx_min` was `D` times too generous and grew a checkerboard by 1.06 to 1.88 per substep in 2D and 3D. Change the transport flux, the integrator or the coupling, and the bound has to be derived again; the checkerboard test is what will notice.
- **`phi` multiplies the time derivative, not the flux divergence.** With the default `flux_form = false` the sweep computes the unreduced HLL flux (`rad_fluxes!` is handed `chat = nothing`), and `scale_by_phi!` multiplies each cell's update by its own `phi`. Steady states are then exact for any `phi(x)`, the flux differences telescope in `E/phi`, and no interface value is needed. With `phi` inside the divergence the steady state solves `div F + F . grad(ln phi) = S` instead; on the Mach 5 shock a 5% ripple in `phi` raised the L1 error from 0.19% to 2.3%, worse than the RSLA. `flux_form = true` keeps that form, with the minimum of the two cells at each interface, for comparison only. For a uniform `phi` the two forms are the same scheme, and `ConstantChat` keeps the old path through the `chat` array so that it stays bit for bit.
- **The signal speed is the closure's, per direction.** `sigma_d = max(0.1, sqrt(T_dd))`, the speed `m1_flux` hands the Riemann solver, summed by `_signal_sum`. Under M1 it reaches one where the radiation free-streams, so a bound written with the P1 value `1/sqrt(3)` is `sqrt(3)` too generous in exactly the cells with no damping to spare; it made the Vaytet Marshak wave 130% wrong (findings A3).
- **The time step comes from the global `phi0`, never from the local `phi`.** Cells at `phi = 1` are the ones thick enough that the implicit exchange stabilises them, and shrinking `dt` for them would forfeit the point. No step restriction is needed for stability either: the standard RSLA is stable at its own step for any `P_rad/P_gas`, which is why `radiation_pressure_dt` was removed. Its replacement, `StabilityBound(energy_step = true)` through `energy_exchange_dt`, is opt-in and buys accuracy, not stability: in thick, radiation-rich cells the heat-capacity channel caps `phi` near `phi0/(sigma C_r)`, and a shorter step lifts the cap. It never goes below the CFL step of the unreduced scheme, where `phi = 1` is stable everywhere; without that floor `radhydro_pulse_dyn` asked for 9 times the substeps of running at `chat = c`. The limiter reads the `dt` actually taken, so the shortened last step of `run!` shows a larger `phi` than the run had; report `phi` from the last full step.
- **One `phi` per group, set once per IMEX substep.** `per_group(::StabilityBound)` is always true. What a local `phi` conserves, per cell or per group alike, is `e + sum_g E_g/phi_g`; the RSLA identity `Delta[e + sum_g E_g] = (1 - phi) Delta e` holds only for a globally uniform `phi`, so the per-cell default it used to justify bought nothing and was less accurate (Vaytet at `phi0 = 0.01`: 11.5% against 7.8%). `OpticalDepthFit(per_group = false)` keeps the old behaviour. `_stage_chat!` computes `phi` from the state at the start of the substep and both stages use it. That is what the bound assumes, and a `phi` changed between the stages would leave a residual `(E(2) - E^n)(1/phi_b - 1/phi_a)` in the conserved energy. Only the wave-speed correction's `chiF` is refreshed per stage.
- **Before the coupling solve, repair nothing the solve can absorb.** This holds for every limiter, the standard RSLA included. A thick cell at `phi = 1` runs at a local CFL number far above one, and its transport stage leaves `|F| >> cE` and sometimes `E < 0`. The solve divides the flux by `1 + chat rho kappa_F dt`, which turns it into the diffusion flux, and it returns a positive temperature whenever `e + sum_g E_g/phi_g > 0`. So `floor_rad_energy!` floors `E` only where that budget is not positive, and `amend_rad_state!` does the full repair after the solve. Clipping `|F|` first raised the error on the thick hot spot from 0.0003% to 0.80%; flooring `E` everywhere made the 64-cell Ensman shock 17 times worse than the RSLA. `Diagnostics` counts the two as `floors_presolve` and `repairs_postsolve`.
- **The wave-speed correction is for the standard RSLA only.** `RadiationPhysics(wavespeed_correction = true)` (Skinner et al. 2019, optional in Quokka) throws with any limiter but `ConstantChat`, because the bound does not hold for the corrected scheme: a thick cell at `phi = 1` beside a thin one grew by about `1e43` in 200 steps, and in a uniform medium modes near a quarter of the grid frequency grow where the radiation's inertia matters, which the checkerboard bound does not see.

`radhydro_shock_super`, the Mach 5 Lowrie & Edwards shock in the form of Quokka's `RadhydroShockSuper`, is the problem that shows the method (0.17% L1 against 3.39% for the RSLA); `radhydro_pulse_dyn` is the radiation-dominated one that exercises `energy_step`.

## Source terms

Two hooks, and they are not interchangeable.

`source` is a *radiation* energy source, consumed inside `radiation_source_step!` so that a stiff emission term is handled implicitly along with the rest of the coupling. `strang_sources` is the general operator-split hook, `f(sim, t, dt, order)`, applied at `dt/2` either side of the advance — the transliteration of Quokka's `addStrangSplitSources`, called from the same two places in the driver and with the same halved step.

Nothing is built on `strang_sources`. It is there so that gravity, cooling or driving can be added without editing `step!`, which is what the review of this code identified as the ceiling on "algorithm experimentation".

Two properties the tests pin, and that a change here must keep:

- The two half-steps are the trapezoid rule in time, so a source with a linear `a(t)` is integrated *exactly* and anything else is second order. A single full-step kick would be first order and the convergence test would catch it.
- `apply_strang_sources!` calls `enforce_state!` after the hook. A user source can push a cell out of the admissible set, and with hydrodynamics disabled the radiation coupling reads `U` before any right-hand side would floor it.

## Closures

`RadiationPhysics` carries a `closure`. `LevermoreClosure()` is the M1 default. `EddingtonClosure()` fixes `chi = 1/3`, cannot free-stream, and exists only so that reference solutions derived under the P1 approximation can be reproduced -- `radhydro_shock` and `radhydro_shock_super` need it, exactly as Quokka overrides `ComputeEddingtonFactor` for the same problem. Do not make it the default, and do not use it for anything with a directional radiation field.

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
