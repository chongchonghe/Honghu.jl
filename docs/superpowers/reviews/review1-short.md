# Architecture review — five key findings

**Date:** 2026-07-28 · **Baseline:** 454 tests pass in 11.3 s
**Lens:** the stated goal — a radiation-hydro code for *algorithm experimentation* and *self-learning*

The code is in good shape: small, readable, dependency-free, verified against analytic solutions and order-of-accuracy checks. The `Val(D)` direction dispatch, the degenerate-direction handling, and the per-cell allocation regression tests are all load-bearing and should survive everything below.

Five findings, ranked. The long version with the remaining eight is in `review1.md`.

---

## 1. The Riemann solver is the one numerical choice you cannot swap — HIGH

Reconstruction, limiter, integrator, closure, opacity, EOS and boundary condition are all `AbstractX` types reachable from `Simulation(...)`. The Riemann solver is a literal call inside the flux sweep: `hllc(...)` at `src/hydro_system.jl:169`, `hll_radiation(...)` at `src/radiation_system.jl:211`. Swapping either means editing package source.

The entanglement is deeper than the call site. `hydro_fluxes!` computes the carbuncle inputs `du`/`dw` (`hydro_system.jl:151-167`) — sixteen array reads per face that only HLLC consumes — then unpacks into HLLC's canonical permutation. Another solver inherits work it does not need.

**Why it matters:** "compare HLL vs HLLC vs Rusanov on the same shock tube" is the most common experiment in this field and the most instructive one — it makes numerical dissipation visible in one plot. It is the one experiment the architecture does not support.

**Candidates**

| | Approach | Cost | Trade-off |
|:--|:--|:--|:--|
| A | `AbstractRiemannSolver` + dispatch inside the existing sweep; `Simulation(; riemann = HLLC())` | ~60 lines + 2 solvers | Mirrors every other extension point |
| B | Pass the flux function as a plain field | ~30 lines | No place for per-solver docs or parameters |
| C | Leave it; document that swapping means editing one function | 2 lines | Abandons the goal for its main use case |

**Recommend A**, with two constraints:

- **Keep the pointwise signature scalar-flat.** `hllc` takes 15 scalars and returns an `SVec{5}`; that is why it inlines and why `test_performance.jl` passes. Bundling arguments into a struct is the obvious tidy-up and the one thing likely to break the allocation bound.
- **Ship Rusanov/LLF and HLLE immediately.** ~10 lines each, no carbuncle machinery, direct teaching value. An extension point with one implementation is an untested guess.

Gate `du`/`dw` on a solver trait so cheaper solvers skip the reads — this also makes the carbuncle cure something a learner can toggle and *see*.

**Prerequisite that pays for itself:** `_hydro_sweep!`/`_rad_sweep!`, `hydro_rhs!`/`rad_rhs!`, and the two workspace types (`simulation.jl:19,38,252,279`) are pairwise identical modulo variable count and flux kernel — ~150 lines of structural duplication. Collapsing them into one generic sweep parameterized by its face kernel *is* the change finding 1 needs, and it also unlocks finding 5. Treat it as one piece of work, not three.

---

## 2. Diagnostic accessors mutate the conserved state — HIGH

Verified on an 8-cell run with `floors = Floors(rho=1e-3, pressure=1e-3)`:

```
U changed by a *read* of :density  -> true
  rho before: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0e-8, 1.0e-8]
  rho after : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.001,  0.001 ]
  nrepaired : 0    (the floor hits from the read are not counted)
```

Path: `field` → `primitives` (`simulation.jl:541`) → `cons2prim!`, which writes floors into `U` in place (`hydro_system.jl:41-60`) and discards the repair count. Everything downstream inherits it: `field`, `plotstate`, `plotpanels`, `animate_run`, `write_vtk`, `write_line`, and `timestep`.

**Why it matters:** two distinct problems. Results depend on how often you looked — a user who plots every 10 steps and one who does not can get different answers, invisibly. And it holes the test philosophy: the suite asserts `sim.nrepaired == 0` as a positivity invariant in three places (`test_hydro.jl:63,108`, `test_radiation.jl:213`), while a read path repairs silently *and without counting*.

**Candidates:** (A) split into flooring and non-flooring kernels — duplicates a hot kernel that will drift; (B) parameterize on `Val{FLOOR}`, constant-folded, diagnostics pass `Val(false)`; (C) snapshot and floor a copy — simplest, but allocates a full state array per diagnostic call, including inside `animate_run`'s frame loop.

**Recommend B** — ~10 lines changed, one kernel, no hot-path cost. Then give `timestep()` the same treatment; it has no need to floor `U`, since the RHS floors it moments later.

**Add the missing test:** reading a field must leave `sim.U` bit-identical. Three lines, and it would have caught this.

---

## 3. The implicit solver's diagnostics are computed and thrown away — HIGH

`radiation_source_step!` builds a `SourceStats` (cells, Newton iterations, worst case, failures). `imex_substep!` returns it. `advance_radiation!` sums it across substeps and returns it (`simulation.jl:376`). Then `step!` drops it:

```julia
has_radiation(sim) && advance_radiation!(sim, dtT)     # simulation.jl:415
```

`grep -rn "SourceStats" src/ test/ examples/` finds no consumer anywhere.

**Why it matters:** `CLAUDE.md` calls `source_terms.jl` "the hardest file", and it is. Its convergence behaviour — iterations per cell, whether any cell hit `max_iterations`, whether the optically-thin shortcut is firing — is exactly what a learner needs to see and what tells an experimenter a run is in trouble. The only visible signal today is a floor count printed once at the end.

That counter is itself unreliable: `nrepaired` accumulates across stages and is never reset, and identical repairs inside `timestep()` (`simulation.jl:392`) and `primitives()` (`simulation.jl:541`) discard their return value. As a quantity you assert `== 0` on it is fine; as a quantity you *read* it under-reports by an amount depending on how often you called a diagnostic.

**Candidates:** (A) add a `srcstats` field, accumulate in `step!`, report in `run!`'s summary, ~15 lines; (B) a general `Diagnostics` record both subsystems write into, ~60 lines; (C) return it from `step!` — cheapest, but callbacks receive `sim`, not the return value, so the data never reaches user analysis code.

**Recommend A, shaped so B is a later widening rather than a rewrite** — put the counters in one struct now even though radiation is the only contributor. Fix `nrepaired` while there.

---

## 4. No extension point has a test, and one contract is already broken — HIGH value, LOW cost

`grep -rn "<: Abstract" test/` returns nothing. Every `AbstractX` hierarchy is exercised only through its built-in subtypes. The README advertises six extension points and shows a worked `KramersOpacity`; nothing verifies a user-defined subtype works end to end.

That is not hypothetical. `src/eos.jl:11-21` documents six required `AbstractEOS` methods and states that "only the first three are needed for pure hydrodynamics" — but `hydro_fluxes!` calls `adiabatic_index(eos)` on every flux sweep (`hydro_system.jl:111`), and it appears in neither list. **A user following the docstring gets a `MethodError` on step one of a pure-hydro run** — the exact case the docstring says needs only three methods.

The deeper point: `hllc` takes a scalar `gamma` and uses it in the Roe-averaged sound speed (`riemann_hydro.jl:42`) and Rider's correction (`riemann_hydro.jl:47`). It is *structurally* a gamma-law solver. Defining `adiabatic_index` on a non-ideal EOS would silence the error and then silently use an effective constant gamma — worse than failing.

**Recommend:**

- `test/test_extensions.jl` — a toy subtype of each of `AbstractEOS`, `AbstractOpacity`, `AbstractLimiter`, `AbstractReconstruction`, `AbstractIntegrator`, `BoundaryCondition`, each running a three-step simulation. ~80 lines. It pins the *real* required method set per interface, and doubles as the copy-paste documentation a learner wants. **Highest value-per-line item in this review.**
- Add `adiabatic_index` to the documented `AbstractEOS` interface and state plainly that the built-in HLLC assumes a gamma-law EOS. One line of code, two sentences of docs.

Finding 1's pluggable solvers make the honest fix easier later: a general-EOS solver could coexist with the gamma-law HLLC rather than replacing it.

---

## 5. Adding a third kind of physics means touching six places — MEDIUM

`Simulation` has 18 fields, 9 type parameters, and a 14-keyword constructor (`simulation.jl:95-144`); radiation presence is encoded twice (`RP` and `RW`, though `RW` is fully determined by `RP`). Adding one piece of physics means editing the struct, the constructor, a workspace type, `step!`, `timestep`, and the field/output plumbing.

The nearest concrete case is a source-term hook. The `source` keyword is documented and consumed only as a *radiation* energy source (`simulation.jl:71`, `source_terms.jl:96`). There is no mechanism for gravity, external driving, cooling, or geometric sources — so adding gravity, the standard next algorithm in any hydro course, means editing `hydro_rhs!`.

**Why it matters:** this is the ceiling on "algorithm experimentation". The code supports experimenting *within* its two subsystems well; it does not support adding a third.

**Candidates**

| | Approach | Cost | Trade-off |
|:--|:--|:--|:--|
| A | A `hydro_source!(dU, U, prim, sim, t)` hook, plus a shipped `ConstantGravity` built on it | ~40 lines | Covers gravity, driving and cooling without any new physics baked in. Doc must warn about the boxing trap (`source_terms.jl:196-201`) |
| B | Group the fields into `Physics` / `Numerics` / `State` sub-structs | ~2 hours | Changes no dispatch or performance; cuts the constructor and `show`, makes config-vs-state visible |
| C | A real `AbstractSubsystem` protocol; `Simulation` holds a tuple of subsystems | ~1 day | The version where a new subsystem is a *user* extension — and a rewrite of `simulation.jl`, with the allocation tests most at risk |

**Recommend A now, B when convenient, C only against a concrete need.** For gravity specifically, A alone is enough — try it before assuming C is required. Rename `source` → `radiation_source` alongside A, since two source kinds make the bare name ambiguous.

Do not start C to make the architecture tidier. Two subsystems do not justify a subsystem protocol, and `CLAUDE.md` is right to warn against speculative flexibility.

---

## Three smaller items worth doing anyway

| Item | Evidence | Fix |
|:--|:--|:--|
| Reconstruction is one global choice shared by both subsystems | `_rad_sweep!` uses `sim.reconstruction` (`simulation.jl:282`) | Add `radiation_reconstruction = reconstruction`; ghost width from the max. ~5 lines. "PPM hydro + PLM radiation" is standard, not exotic |
| The hydro↔radiation split is first order, and undocumented | `step!` is Lie splitting (`simulation.jl:412-420`) while the docs advertise "second order" | One sentence in the README limitations and the `step!` docstring. Optionally offer `Strang()` (~15 lines) — it turns the limitation into a measurable exercise |
| `Fixed` boundary ignores the `t` it is handed | `boundary.jl:157-169` | Add `FunctionBC(f)` calling `f(x,y,z,t)`. ~15 lines; covers inflow profiles and drivers |

---

## Sequencing

**First, no architecture change (~1 day):** finding 4's `test_extensions.jl` (it defines the contracts everything else must not break), then the `adiabatic_index` doc fix, finding 2's `Val{FLOOR}`, finding 3's stats reporting, the shared-reconstruction fix, the operator-split note.

**Then, the enabling refactor (~2–3 days):** collapse the duplicated sweeps, build `AbstractRiemannSolver` on top, add the source hook. Run `test/test_performance.jl` after *each* step, not at the end — it is the guard on the properties that make this code fast.

**Leave on the shelf:** the subsystem protocol, and multigroup radiation. On multigroup, one warning: `NRAD = 4` and the literal indices `IERAD/IFRX/...` (`variables.jl:18,32-35`) appear at ~40 sites, and replacing them with accessor functions that constant-fold to the same numbers is cheap and worth doing. But it would make the port *look* nearly done when it is not — transport generalizes for free, while the coupling turns the 2×2 Newton into $(1+N_g) \times (1+N_g)$, destroys the analytic inverse that makes it fast, and invalidates the conditioning argument for using $R_{\rm ex}$ as the second unknown. That is where all the work is.
