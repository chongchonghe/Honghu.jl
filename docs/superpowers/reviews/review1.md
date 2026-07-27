# Architecture review 1 — usability and extensibility

**Date:** 2026-07-28
**Scope:** `src/` (23 files, ~3 000 lines), `ext/Hydro3DPlotsExt.jl`, `test/`, `examples/`, `docs/make_figures.jl`
**Baseline:** `julia --project=. -t 4 test/runtests.jl` → 454 pass, 0 fail, 11.3 s
**Lens:** the stated goal — a 3D radiation-hydrodynamics code for *algorithm experimentation* and *self-learning of the methods*

---

## 1. Summary

The code is in good shape. It is small, readable, well documented, dependency-free, and it passes a verification suite built out of the right kind of test (analytic comparison, order of accuracy, exact invariants). Nothing here is a rescue operation.

The friction is concentrated in one place, and it is worth naming up front:

> **Every numerical choice in this code is a dispatched type reachable from the `Simulation` constructor — except the Riemann solver, which is the one you most want to swap.**

Reconstruction, limiter, integrator, closure, opacity, EOS and boundary condition are all `AbstractX` hierarchies you can extend from outside the package. The Riemann solver is a literal function call in the middle of a flux sweep (`src/hydro_system.jl:169`, `src/radiation_system.jl:211`). For a code whose purpose is experimenting with algorithms, that is the single largest gap between what the architecture promises and what it delivers.

The second theme is **observability**. The hardest part of the code — the implicit matter–radiation coupling — computes a full diagnostic record and then throws it away (§3.2). And several functions that look like read-only accessors silently mutate the conserved state (§3.3, demonstrated below). Both matter more for a learning code than they would for a production one: the whole point is being able to see what the scheme is doing.

The third theme is **structural duplication between the two subsystems** (§3.5), which is what makes adding a *third* one expensive. This one deserves a measured response rather than an abstraction spree — see the recommendation.

Findings are ranked by impact on the stated goal. Each carries evidence, candidate solutions with trade-offs, and a recommendation. §5 sequences them.

---

## 2. What the architecture gets right

Worth stating, because it constrains what should change.

- **Compile-time direction dispatch.** `Val(D)` threaded through every kernel, with `_tdir1`/`_tdir2`/`_tshift1`/`_tshift2` written out as literal methods (`src/riemann_hydro.jl:99-110`) so they constant-fold. The explicit `g.active[1] && sweep!(…, Val(1))` chain instead of `for D in 1:3` is exactly right and the reason it stays allocation-free.
- **Degenerate directions as a first-class concept.** `n[d] == 1` ⇒ no ghost cells, no sweep, no boundary condition (`src/grid.jl:59-63`). A 1D problem genuinely costs 1D. This is a better design than most codes of this size manage.
- **Ghost width derived from the numerics.** `nghost = required_ghosts(reconstruction)` (`src/simulation.jl:135`) means a new reconstruction scheme declares its own stencil and everything downstream adapts. This is the extension mechanism working as intended.
- **The allocation regression tests.** `test/test_performance.jl` asserts bytes-per-*cell*, not bytes, and separately asserts that per-step allocation does not scale with grid size. That is the right invariant, it caught a real 406 MB bug, and it will catch the next one. Any refactor proposed below must keep it passing — I flag where that is a live risk.
- **Plotting as a weak dependency**, with stubs and informative fallbacks in `src/plotting.jl:157-159` rather than `MethodError`s. Clean separation, correctly motivated.
- **Docstring discipline.** Nearly every exported name says what it is *for*. The comments explain *why* and cite equations. `src/source_terms.jl` and `src/integrators.jl:99-108` in particular explain non-obvious choices (why $R_{\rm ex}$ rather than $E_{\rm rad}$; why the internal energy is recombined rather than the total) that a reader would otherwise have to reverse-engineer.

---

## 3. Findings

### 3.1 The Riemann solver is not an extension point — HIGH

**Evidence.** `src/hydro_system.jl:169` calls `hllc(...)` directly inside the flux sweep. `src/radiation_system.jl:211` calls `hll_radiation(...)` the same way. Neither is reachable from `Simulation(...)`; swapping either means editing package source.

The entanglement runs deeper than the call site. `hydro_fluxes!` computes the carbuncle inputs `du` and `dw` (`src/hydro_system.jl:151-167`) — sixteen array reads per face that *only* HLLC consumes — and then unpacks the result into HLLC's canonical `(normal, transverse1, transverse2)` permutation. A different solver inherits work it does not need and a convention it did not ask for.

**Why it matters here.** "Compare HLL against HLLC against Rusanov on the same shock tube" is the most common experiment in this field and the most instructive one for a learner: it makes numerical dissipation visible in a single plot. Right now it is the one experiment the architecture does not support.

**Candidate solutions**

| | Approach | Cost | Trade-off |
|:--|:--|:--|:--|
| A | `AbstractRiemannSolver` + `riemann_flux(solver, …)` dispatched inside the existing sweep; `Simulation(; riemann = HLLC())` | ~60 lines + 2 new solvers | Mirrors every other extension point. Needs care to keep the pointwise signature flat |
| B | Pass the pointwise flux function as a plain field/closure on `Simulation` | ~30 lines | Less discoverable, no place to hang per-solver docs, awkward to add solver parameters |
| C | Leave hard-wired, document that swapping means editing one function | 2 lines of docs | Honest, but abandons the stated goal for its most common use case |

**Recommendation: A**, with two constraints.

1. **Keep the pointwise signature scalar-flat.** `hllc` currently takes 15 scalars and returns an `SVec{5}`; that is why it inlines and why `test_performance.jl` passes. Wrapping the arguments in a struct is the obvious tidy-up and it is the one thing likely to break the allocation bound. Dispatch on the solver type, keep the arguments as they are.
2. **Ship two extra solvers immediately** so the extension point is load-bearing rather than speculative. Rusanov/local Lax–Friedrichs and HLLE are ~10 lines each, need none of the carbuncle machinery, and have direct teaching value. An extension point with one implementation is an untested guess.

For the `du`/`dw` asymmetry: gate their computation on a solver trait (`needs_transverse_gradients(::HLLC) = true`, default `false`) so cheaper solvers skip the sixteen reads. This also makes the carbuncle cure something a learner can toggle and *see*.

The radiation side deserves the same treatment for the same reason, though the payoff is smaller — HLL on the M1 system with frozen Eddington wave speeds has fewer interesting competitors than the Euler equations do.

---

### 3.2 `SourceStats` is computed and discarded — HIGH

**Evidence.** `radiation_source_step!` builds a `SourceStats` (cells, total Newton iterations, worst-case iterations, failures) and returns it. `imex_substep!` returns it. `advance_radiation!` sums it across substeps and returns it (`src/simulation.jl:376`). `step!` then drops the value on the floor:

```julia
has_radiation(sim) && advance_radiation!(sim, dtT)     # src/simulation.jl:415
```

`grep -rn "SourceStats" src/ test/ examples/` finds no consumer anywhere in the package.

**Why it matters here.** `CLAUDE.md` calls `source_terms.jl` "the hardest file", and it is. Its convergence behaviour — how many Newton iterations a cell actually takes, whether any cell hit `max_iterations` without converging, whether the optically-thin shortcut is firing — is precisely what a learner needs to see and what tells an experimenter that a run is in trouble. At the moment the only visible signal is a positivity-floor count printed once at the end.

There is a related weakness in that counter. `sim.nrepaired` accumulates across every RHS evaluation and is never reset, so it is a cumulative total across stages and steps rather than a per-step figure. Worse, identical floor repairs applied inside `timestep()` (`src/simulation.jl:392`) and `primitives()` (`src/simulation.jl:541`) discard their return value, so the count under-reports by an amount that depends on how often you called a diagnostic. As a quantity you assert `== 0` on it is fine; as a quantity you *read* it is not trustworthy.

**Candidate solutions**

- **A — Store and report.** Add `srcstats::SourceStats` to `Simulation`, accumulate in `step!`, and extend `run!`'s summary with mean/max Newton iterations and a loud line if `failures > 0`. Add `reset_diagnostics!(sim)`. ~15 lines.
- **B — A general diagnostics record.** A small mutable `Diagnostics` struct that both subsystems write into (floor repairs by subsystem, Newton statistics, substep counts, wall time per phase), with one `show` method. ~60 lines, and the right shape if a third subsystem ever arrives.
- **C — Return from `step!`.** Cheapest, but callbacks receive `sim`, not `step!`'s return value, so the data would not reach the place users actually write analysis code.

**Recommendation: A now, shaped so B is a later widening rather than a rewrite.** Put the counters in one struct from the start even if radiation is its only contributor today. While there, fix `nrepaired`: count per step, expose the cumulative total separately, and stop dropping counts in the diagnostic paths (which §3.3 removes anyway).

---

### 3.3 Diagnostic accessors mutate the conserved state — HIGH

**Evidence.** Demonstrated on an 8-cell run with `floors = Floors(rho=1e-3, pressure=1e-3)`:

```
U changed by a *read* of :density  -> true
  rho[interior] before: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0e-8, 1.0e-8]
  rho[interior] after : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.001,  0.001 ]
  nrepaired counter   : 0    (the floor hits from the read are not counted)
```

The path is `field` → `primitives` (`src/simulation.jl:541`) → `cons2prim!`, which writes floors into `U` in place (`src/hydro_system.jl:41-60`) and returns a repair count that `primitives` discards.

Everything downstream inherits this: `field`, `plotstate`, `plotpanels`, `animate_run`, `write_vtk`, `write_line`, `output_fields`, and `timestep` itself.

**Why it matters here.** Two distinct problems.

1. **Results depend on how often you looked.** A user who plots every 10 steps and a user who does not can get different answers, because floors are applied at different times. That is a genuinely nasty class of bug to chase, and it is invisible.
2. **It undermines the test philosophy.** The suite asserts `sim.nrepaired == 0` in three places as a positivity invariant (`test/test_hydro.jl:63,108`, `test/test_radiation.jl:213`). A read path that repairs silently and *without counting* is a hole in exactly that invariant.

**Candidate solutions**

- **A — Split the kernel.** A flooring `cons2prim!` for the RHS and a non-mutating one for diagnostics. Clear, but duplicates a hot kernel — the sort of duplication that drifts.
- **B — Parameterize on a `Val{FLOOR}`.** One kernel, one branch, constant-folded at compile time; diagnostics call `cons2prim!(…, Val(false))`. ~10 lines changed, no hot-path cost, allocation tests unaffected.
- **C — Snapshot and floor a copy in `primitives()`.** Simplest to write, but allocates a full state array on every diagnostic call, including inside `animate_run`'s per-frame loop.

**Recommendation: B.** Smallest diff, single kernel, no performance question. Then apply the same treatment to `timestep()`, which has no need to floor `U` — the RHS floors it moments later anyway.

**Add the regression test this is missing:** reading any field must leave `sim.U` bit-identical. It is three lines and it would have caught this.

---

### 3.4 `Simulation` is a god-object — MEDIUM

**Evidence.** 18 fields, 9 type parameters, a 14-keyword constructor (`src/simulation.jl:95-144`). Radiation presence is encoded twice — once in `RP`, once in `RW` — although `RW` is fully determined by `RP` and could be dropped today.

Adding one new piece of physics currently means touching six places: the struct, the constructor, a workspace type, `step!`, `timestep`, and the field/output plumbing (`output_fields`, `field`, `available_fields`).

**Why it matters here.** This is the ceiling on "algorithm experimentation". The code supports experimenting *within* its two subsystems well; it does not support adding a third. Whether that ceiling is a problem depends entirely on whether you want self-gravity, passive scalars, MHD or multigroup — which is a question about your goals, not about the code.

**Candidate solutions**

- **A — Do nothing structural; document the ceiling.** Genuinely defensible. Two subsystems do not justify a subsystem protocol, and `CLAUDE.md` is right to warn against speculative flexibility. But then say so in the README, so the boundary is a stated design decision rather than a surprise.
- **B — Group the fields into three plain structs.** `Physics` (eos, radiation, source, floors), `Numerics` (reconstruction, integrator, cfls, substep cap), `State` (U, R, t, dt, step, counters), leaving `Simulation` with about five fields. Changes no dispatch and no performance; it shrinks the constructor and `show`, and makes "configuration vs. evolving state" visible at a glance — which is worth something in a code meant to be read. Keyword-forwarding constructors keep `Simulation(; n=…, eos=…)` working unchanged.
- **C — A real `AbstractSubsystem` protocol.** `nvars`, `state`, `workspace`, `cons2prim!`, `fluxes!`, `timestep`, `sources!`, with `Simulation` holding a tuple of subsystems. This is the version that makes a new subsystem a *user* extension. It is also a rewrite of `simulation.jl`, and the tuple-of-subsystems pattern needs care to stay type-stable — Trixi.jl demonstrates it can be done allocation-free, but the allocation tests are the thing most at risk.

**Recommendation: B now; C only when a third subsystem is actually wanted, not before.** B is a couple of hours and pays for itself in readability. C is worth roughly a day and should be paid for by a concrete need. If self-gravity is on the roadmap, do §3.7 first and see whether B plus a source hook is already enough — for gravity specifically, it is.

---

### 3.5 The two subsystems duplicate the entire sweep machinery — MEDIUM

**Evidence.** These pairs are structurally identical modulo the variable count and the pointwise flux function:

| Hydro | Radiation |
|:--|:--|
| `HydroWorkspace` (`simulation.jl:19`) | `RadiationWorkspace` (`simulation.jl:38`) |
| `_hydro_sweep!` (`simulation.jl:252`) | `_rad_sweep!` (`simulation.jl:279`) |
| `hydro_rhs!` (`simulation.jl:268`) | `rad_rhs!` (`simulation.jl:295`) |
| `cons2prim!` (`hydro_system.jl:31`) | `rad_cons2prim!` (`radiation_system.jl:103`) |
| `hydro_fluxes!` (`hydro_system.jl:101`) | `rad_fluxes!` (`radiation_system.jl:168`) |
| `hydro_timestep` (`hydro_system.jl:211`) | `radiation_timestep` (`radiation_system.jl:229`) |

Roughly 150 lines of structural duplication. A third subsystem means a third copy of all of it.

**Recommendation — the intermediate step, not the abstraction.** Two changes capture most of the value at a fraction of the risk of §3.4C:

1. **One `Workspace(g, nvar)` type.** The two differ only in which extra scratch slot they carry (`U0` vs `Rtmp`/`Utmp`). A single parameterized type with the union of slots, or a small tuple of named scratch arrays, removes the duplicate definitions.
2. **One `_sweep!(dst, prim, qL, qR, flux, fluxkernel, recon, Val(D), g, nvar)`.** Both callers then differ only in which flux kernel they pass — which is the same generalization §3.1 needs. Doing §3.5 first makes §3.1 nearly free.

That is the honest ordering: this is not a refactor worth doing for its own sake, but it is a genuine prerequisite that makes two other improvements cheaper. Do it as the first step of the Riemann-solver work, not as a standalone cleanup.

---

### 3.6 Reconstruction is one global choice shared by both subsystems — MEDIUM

**Evidence.** `_rad_sweep!` uses `sim.reconstruction` — the same object as hydro (`src/simulation.jl:282`). Ghost width is derived from it once for the whole grid (`src/simulation.jl:135`).

**Why it matters here.** "PPM for hydro, PLM for radiation" is a standard configuration, not an exotic one: the M1 admissibility constraint $|F| < cE$ makes a more diffusive radiation reconstruction the pragmatic choice, and both flux sweeps *already* implement a per-face fallback to piecewise-constant for exactly this reason (`hydro_system.jl:123`, `radiation_system.jl:200`). The code has the concept; it just does not let you ask for it globally.

**Recommendation.** Add `radiation_reconstruction = reconstruction` as a keyword, and set the ghost width from `max(required_ghosts(reconstruction), required_ghosts(radiation_reconstruction))`. About five lines, no risk, and it roughly doubles the usefulness of the reconstruction extension point. This is the best value-per-line item in the report after §3.11.

---

### 3.7 No hydro source-term hook — MEDIUM

**Evidence.** The `source` keyword is documented as "a callable `source(x, y, z, t)` returning a *radiation* energy source density" (`src/simulation.jl:71-72`) and is consumed only inside `radiation_source_step!` (`src/source_terms.jl:96`). There is no mechanism for gravity, an external driving force, optically-thin cooling, or a geometric source term.

**Why it matters here.** Gravity is the standard next algorithm in any hydrodynamics course, and the split-versus-unsplit treatment of source terms — and the well-balancing problem that comes with it — is itself worth experimenting with. Right now adding gravity means editing `hydro_rhs!`.

**Candidate solutions**

- **A — A concrete `gravity` field** taking `g(x,y,z,t) -> (gx,gy,gz)`, applied as an unsplit term in `hydro_rhs!` (momentum and energy). ~30 lines, immediately useful, teaches the well-balancing problem.
- **B — A general `hydro_source!(dU, U, prim, sim, t)` user callback** added to `dU` after the sweeps. One hook, no physics baked in, maximum flexibility. The caveat must be documented: user code inside a per-cell loop is exactly where the boxing trap in `CLAUDE.md` bites, so the docstring needs to say "write this out component by component" and point at the existing note in `source_terms.jl:196-201`.
- **C — Both:** B as the mechanism, with `ConstantGravity` and `PointMassGravity` shipped as worked examples built on it.

**Recommendation: C.** The shipped examples are what make the hook discoverable and give a learner something to copy. While doing this, rename the existing `source` to `radiation_source` — with two source kinds in play the bare name becomes ambiguous. That is a breaking change to a keyword argument; it affects `Simulation` callers only and no built-in problem passes it, so the cost is a line in the README.

---

### 3.8 `NRAD = 4` blocks multigroup by construction — MEDIUM (as a boundary to name)

**Evidence.** `const NRAD = 4` with fixed literal indices `IERAD, IFRX, IFRY, IFRZ` (`src/variables.jl:18,32-35`), used at roughly forty sites across `radiation_system.jl`, `source_terms.jl`, `simulation.jl`, `output.jl` and `plotting.jl`.

**Why it matters here.** Multigroup is the obvious next radiation algorithm and Quokka — the stated reference — has it. This is a "you cannot get there from here" boundary, and it is worth naming even if you never cross it.

**Recommendation — prepare the boundary; do not cross it speculatively.**

The cheap half is worth doing now: replace the bare literals with accessor functions (`ierad(gi) = 4*(gi-1) + 1`, `ifr(gi, d) = 4*(gi-1) + 1 + d`) that today constant-fold to exactly the current numbers. It is numerically a no-op, costs nothing at run time, and turns a future multigroup port from "edit forty call sites" into "change two functions".

The expensive half — parameterizing `RadiationPhysics` on a group count — should wait for a real need, and the reason is instructive. Transport generalizes almost for free: each group advects independently, so the HLL sweep and the state repair extend by looping. The *coupling* does not. The 2×2 Newton system in $(E_{\rm int}, R_{\rm ex})$ becomes $(1+N_g) \times (1+N_g)$, the analytic 2×2 inverse that makes the current solve fast disappears, and the conditioning argument that motivated using $R_{\rm ex}$ as the second unknown has to be re-derived. That is the actual research content and it is where all the work is. Do not let an index refactor create the illusion that it is nearly done.

---

### 3.9 The hydro↔radiation operator split is first order and undocumented — LOW (documentation), MEDIUM (as an exercise)

**Evidence.** `step!` (`src/simulation.jl:412-420`) applies hydro then radiation — Lie splitting. SSPRK3 is third-order accurate and the IMEX PD-ARS scheme is second-order, but the composition is first order in $\Delta t$ whenever both subsystems are active.

This matches Quokka, so it is a deliberate inheritance rather than a defect. But the README's "Limitations" section lists four items and this is not among them, while the module docstring and README both advertise "second order and asymptotic-preserving". A learner reading those claims will not know the split caps the coupled accuracy at first order.

**Recommendation.**

- **Document it** in the README limitations and the `step!` docstring: one sentence, and it is the sentence that keeps someone from spending a day wondering why their coupled convergence test shows slope 1.
- **Optionally offer `splitting = Strang()`** — half a hydro step, a full radiation subcycle, half a hydro step. About fifteen lines, and it turns the limitation into a self-contained experiment: measure the observed order on a coupled problem with each splitting and see the difference. That is a good exercise for a code with this purpose.

---

### 3.10 No `Problem` abstraction; problem, reference solution and figure are wired together three times — LOW/MEDIUM

**Evidence.** `Problems.*` functions return bare `(sim, tmax)` tuples. The pattern "run it, extract the line-out, call the exact solution, compare or plot" is then re-implemented independently in `test/test_hydro.jl:30`, `docs/make_figures.jl:33` and `examples/sod_1d.jl:29` — and again for diffusion (`test_radiation.jl:33`, `make_figures.jl:85`) and for the radiative shock (`test_radiation.jl:191`, `make_figures.jl:120`).

Adding a new verification problem therefore means touching `src/problems/`, `test/`, `docs/make_figures.jl`, and possibly `examples/`.

**Candidate solutions**

- **A — A `Problem` record and registry**: name, setup, default `tmax`, default resolution, optional `reference(sim)`, fields worth plotting. Tests and `make_figures.jl` then loop over the registry. Real payoff; moderate churn across `test/` and `docs/`.
- **B — Keep `(sim, tmax)`, add a uniform `reference_solution` dispatch** so at least the exact-solution wiring is shared. Much smaller change, captures most of the value, and A can grow out of it later.
- **C — Leave it.** Defensible: the duplication totals maybe forty lines and each copy reads clearly on its own.

**Recommendation: B**, and only when you next add a problem — let the third instance of the pattern justify the abstraction rather than doing it speculatively. C is a reasonable answer if new problems are rare.

---

### 3.11 Extension points have no tests, so they can rot — HIGH value, LOW cost

**Evidence.** `grep -rn "<: Abstract" test/` returns nothing. Every `AbstractX` hierarchy is exercised exclusively through its built-in subtypes.

**Why it matters here.** The README advertises six extension points and shows a worked `KramersOpacity` example. Nothing verifies that a *user-defined* subtype actually works end to end — and one of them is already subtly broken (§3.12), which is exactly the failure mode this predicts.

**Recommendation.** Add `test/test_extensions.jl`: define a minimal toy subtype of each of `AbstractEOS`, `AbstractOpacity`, `AbstractLimiter`, `AbstractReconstruction`, `AbstractIntegrator` and `BoundaryCondition`, and run a three-step simulation with each. About eighty lines.

Two payoffs beyond regression safety: it pins the *real* required method set for each interface (which is how §3.12 gets caught automatically), and it doubles as the copy-paste documentation a learner actually wants. **This is the highest value-per-line item in the report.**

---

### 3.12 The `AbstractEOS` contract is under-specified — a user-defined EOS fails — MEDIUM

**Evidence.** `src/eos.jl:11-21` documents six required methods and states that "only the first three are needed for pure hydrodynamics". But `hydro_fluxes!` calls `adiabatic_index(eos)` on every flux sweep (`src/hydro_system.jl:111`), and `adiabatic_index` appears in neither list. A user who follows the docstring gets a `MethodError` on the first step of a *pure hydro* run — the case the docstring explicitly says needs only three methods.

There is a deeper point behind the missing entry. `hllc` takes a scalar `gamma` and uses it both in the Roe-averaged sound speed (`riemann_hydro.jl:42`) and in Rider's nonlinear wave-speed correction (`riemann_hydro.jl:47`). The solver is *structurally* a gamma-law solver. Defining `adiabatic_index` for a non-ideal EOS would silence the `MethodError` and then silently use an effective constant gamma inside the Riemann solve — which is worse than failing.

**Candidate solutions**

- **A — Fix the contract and state the constraint.** Add `adiabatic_index` to the documented interface, and say plainly in the `AbstractEOS` docstring that the built-in HLLC assumes a gamma-law EOS. One line of code, two sentences of docs.
- **B — Pass an effective $\Gamma_1 = (\partial \ln p / \partial \ln \rho)_s$** supplied by the EOS, so a general EOS degrades gracefully rather than silently.
- **C — A general-EOS HLLC.** Real work, and only worth it once a non-ideal EOS exists.

**Recommendation: A now.** It is a two-minute fix for a documented interface that does not match the code. B is the right shape if a non-ideal EOS is ever added; note that §3.1's pluggable solvers make B easier, since a general-EOS solver could then coexist with the gamma-law HLLC rather than replacing it.

---

### 3.13 Smaller items

| Item | Evidence | Suggested response |
|:--|:--|:--|
| `Fixed` boundary ignores the `t` it is handed; no built-in space- or time-dependent BC | `src/boundary.jl:157-169` — `t` is a parameter, never read | Add `FunctionBC(f)` calling `f(x,y,z,t)`, ~15 lines. Covers inflow profiles and time-dependent drivers, and makes the `t` parameter earn its place |
| `RW` type parameter is redundant | `src/simulation.jl:96` — fully determined by `RP` | Drop it; one less parameter in every signature |
| 1D problems get no threading at all | `threaded_jk(1:1, 1:1)` ⇒ `ntot == 1` ⇒ serial (`src/grid.jl:170`) | Document rather than fix. Splitting the `i` loop would complicate every kernel for little gain, but a reader benchmarking the 1D Sod tube should know why `-t 8` changes nothing |
| Checkpoints carry no physics metadata | `src/output.jl:152-201` — grid and state only | Acceptable for the scope, and the mismatch errors are good. One README sentence: restart requires reconstructing the `Simulation` yourself |
| `field` recomputes `primitives(sim)` per call | `src/plotting.jl:52` — `plotpanels` with 6 vars does 6 full conversions | Low priority; only matters inside `animate_run`. §3.3's fix is the prerequisite for caching it safely |

---

## 4. Cross-cutting observation

Three of the higher-ranked findings — §3.1 (pluggable Riemann solvers), §3.5 (unify the sweeps), §3.7 (source hook) — all want the same underlying change: **the flux sweep should be parameterized by what it does at a face and what it adds to the RHS, rather than hard-coding both.**

That is a single, well-bounded refactor of `_hydro_sweep!`/`_rad_sweep!` into one generic sweep, and it unlocks all three. It is worth treating as one piece of work rather than three, and it is the only structural change in this report I would argue for on its own merits.

Everything else is either a small fix (§3.2, §3.3, §3.6, §3.12), a test (§3.11), a documentation correction (§3.9), or a decision to defer (§3.4C, §3.8, §3.10).

---

## 5. Suggested sequencing

**Tier 1 — no architectural change, high value.** Roughly a day.

1. §3.11 — `test/test_extensions.jl`. Do this first; it defines the contracts everything else must not break.
2. §3.12 — add `adiabatic_index` to the `AbstractEOS` docs; state the gamma-law constraint. (Tier 1.1 will have caught it.)
3. §3.3 — `Val{FLOOR}` on `cons2prim!`; make diagnostics read-only; add the "reading does not mutate" test.
4. §3.2 — surface `SourceStats` in `run!`'s summary; fix `nrepaired` accounting.
5. §3.6 — separate `radiation_reconstruction`.
6. §3.9 — document the first-order operator split in the README limitations and `step!`.

**Tier 2 — the enabling refactor.** Roughly two to three days.

7. §3.5 — unify the workspaces and the sweep into one generic function.
8. §3.1 — `AbstractRiemannSolver` on top of 7, shipping Rusanov/LLF and HLLE. Verify `test_performance.jl` still passes at each step.
9. §3.7 — `hydro_source!` hook plus a shipped `ConstantGravity`; rename `source` → `radiation_source`.
10. §3.8 (cheap half) — replace radiation index literals with accessor functions.

**Tier 3 — only against a concrete need.**

11. §3.4B — group `Simulation`'s fields into `Physics`/`Numerics`/`State`.
12. §3.10B — uniform `reference_solution` dispatch, when the next problem is added.
13. §3.4C / §3.8 (expensive half) — subsystem protocol; multigroup. Both are real projects; neither should be started to make the architecture tidier.

Tier 1 is worth doing regardless. Tier 2 is what converts "a well-built radiation-hydro code" into "a platform for algorithm experimentation", which is the stated goal. Tier 3 should stay on the shelf until something concrete asks for it.

---

## 6. One thing not to change

The no-dependency constraint, the `Val(D)` dispatch discipline, and the allocation regression tests are load-bearing and should survive every item above. Where a proposed change puts one of them at risk, I have said so — §3.1 (keep the pointwise signature flat) and §3.4C (tuple-of-subsystems type stability) are the two live cases. `test/test_performance.jl` is the guard; run it after each Tier 2 step, not at the end.
