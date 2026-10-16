# Future work

Things worth doing, with enough detail to start. Nothing here is committed to — except the section immediately below, which is not a plan but a defect that exists now.

## Known bugs

### `QuokkaCoupling` fails to converge on a steeply falling opacity from a hot start

**What happens.** In code units (`rho = cv = c = chat = a_rad = k_B = 1`, `dt = 1e-3`), a cell opened at `Tgas = 1.3`, `Trad = 0.5` with `PowerLawOpacity(kappa0 = 1e9, n = -3)` runs the **default** coupling solver to its hundred-iteration cap and returns `Tgas = 0.46388`, below the radiation temperature it started from, having lost 63% of the cell's energy. The right answer is `0.84724` — the radiative equilibrium temperature of the cell, `T + T^4 = E_0 = 1.3625` — which `DirectCoupling` reaches in nine iterations and `AdaptiveCoupling` confirms to every digit printed.

It reports `converged = false`, so it lands in `newton_failures` and in the `run!` summary rather than passing silently. The cell is still wrong.

The blast radius is multigroup runs and anything calling `solve_coupling` directly: at one group `radiation_source_step!` dispatches to the grey driver and no `AbstractCoupling` is consulted at all.

**Reproduce.**

```julia
using Honghu
using Honghu: CouplingState, solve_coupling, sumvec, SVec

groups = RadiationGroups(0.1, 20.0, 4)
Trad, Tgas = 0.5, 1.3
Erad = Trad^4 * planck_fractions(groups, Trad, 1.0)
z = zero(SVec{4,Float64})
s = CouplingState{4,5,Float64,PowerLawOpacity{Float64}}(
        Tgas, Erad, z, z, 1.0, 1.0, PowerLawOpacity(kappa0 = 1e9, n = -3.0),
        1e-3, 1.0, 1.0, 1.0, 1.0, groups, 1e-10, 1e-30)

solve_coupling(QuokkaCoupling(), s)    # converged = false, 100 iterations, Egas = 0.46388
solve_coupling(DirectCoupling(), s)    # converged = true,    9 iterations, Egas = 0.84724
```

`cell_state` in `test/test_radiation_multigroup.jl` builds the same thing, and the comment above `ref_cell` in `test/test_coupling_adaptive.jl` records that its `Tgas = 1.0, Trad = 0.3` is chosen to stay clear of this: from there the same configuration converges.

**What it is not**, each checked:

- Not multigroup. `NG = 1` and `NG = 4` fail identically, to every digit.
- Not the arrowhead elimination. `QuokkaCoupling(linsolve = DenseSolve())` fails identically, to every digit.
- Not slow convergence. `max_iterations = 20000` does not converge either, and the iterate drifts *away* (`0.46388` at 100 iterations, `0.45972` at 20000). Loosening `tolerance` to `1e-3` changes nothing, so it is stuck far from the root rather than fussing over the last digits.
- Not stiffness on its own. `ConstantOpacity(1e9)` from the same start converges in six.
- Not the formulation. The grey driver in `src/source_terms.jl` is the same backward Euler in the same replaced base variables, and on the same cell it converges in six iterations to `0.8472403`.

**What it takes.** Three things at once, measured by sweeping one at a time from `Tgas = 1.3`, `Trad = 0.5`, `kappa0 = 1e9`, `n = -3`:

- a steeply falling opacity — `n = -2` converges, `n = -3` and `n = -3.5` do not;
- a large optical depth — `kappa0 = 1e3` converges, `1e5` and above do not;
- a gas hotter than the radiation by enough — at `Trad = 0.5` it converges up to `Tgas = 1.2` and fails from `1.3`; at `Tgas = 1.3` it converges again once `Trad >= 0.7`. The energy lost grows with the mismatch: 63% at `Tgas = 1.3`, 88% at `Tgas = 2`.

**The leading hypothesis, and the experiment that supports it.** The two solvers that converge — `DirectCoupling` and the grey driver — differ from `QuokkaCoupling` in exactly one place that these three conditions all point at: the `dkappa/dT` term in its hand-derived `dF_g/dE_gas`,

```
Jg0 = (kappa_P/kappa_E * dB_g/dT + (dkappa_P B_g - dkappa_E E_g) / kappa_E) / cv
```

For a power law with `kappa_P = kappa_E` the second term is `n (B_g - E_g) / T`, of the *opposite sign* to the first whenever `n < 0` and the gas is hotter than the radiation. At `T = 1.3`, summed over the groups, the two are `4aT^3 = 8.79` against `n(aT^4 - E)/T = -6.45`: the exact term cancels three quarters of the emission term, the gas row of the Newton matrix goes nearly singular, and the step it produces is large enough to trip the equilibrium-overshoot guard, which discards the step and snaps the gas to `cv * Trad` instead. That is a cycle, not progress, which is what "drifts away over 20000 iterations" looks like. The cancellation becomes total near `n ≈ -4`, and the grey driver never sees it because it deliberately holds `kappa_P/kappa_E` fixed and carries only `4 a T^3`.

The experiment: run the *same* law with the derivative withheld, so that `dkappa_dT` falls through to its zero default and nothing else changes.

```julia
UserOpacity(planck = (rho, T) -> 1e9 * max(T, 0.0)^-3.0)   # no dplanck_dT
```

It converges in six iterations to `0.8472403`. So the trigger is that term, in this formulation — not the term itself, which `DirectCoupling` also carries through `coupling_jacobian` and converges with.

**Where to start.** Instrument the guard first: count how often `dEgas / cv > max(Tgas, Trad)` trips across the hundred iterations, and print the iterate. If it trips most of them the diagnosis above is right, and the fix is a globalisation that makes progress — a damped or line-searched step — rather than a jump that throws the Newton step away. Do not simply drop the `dkappa/dT` term: it is exact, it is what `coupling_jacobian` supplies to the other two solvers, and CLAUDE.md's 25–40% iteration saving is measured. Whatever the fix, `DirectCoupling` and `AdaptiveCoupling` already answer this cell correctly and are the oracle for it.

The regression test to add is a `cell_state` at these temperatures. Note that "one group reproduces the grey driver" in `test/test_radiation_multigroup.jl` sweeps the power law only to `kappa0 = 1e3` — one decade below where this starts — and does it on a Marshak-wave state rather than a hot start, which is why the suite is green with this in the tree.

---

## GPU acceleration

### Packages

| | |
|:---|:---|
| [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) | CUDA programming in Julia. NVIDIA only. |
| [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) | Heterogeneous programming: one kernel, backends for CUDA, ROCm, oneAPI, Metal and plain CPU. |
| [Juliana.jl](https://github.com/artecs-group/Juliana.jl) | Converts CUDA.jl code to multi-backend KernelAbstractions.jl code. |

**Go straight to KernelAbstractions.** Juliana translates *existing* CUDA.jl code, and there is none here, so it would only help if the port went via CUDA.jl first — which means writing the NVIDIA-specific version in order to have something to convert. The one argument for that detour is that CUDA.jl's profiling and debugging tooling is better than any backend-agnostic equivalent; the argument against is that it doubles the work and nothing here requires NVIDIA hardware.

A KernelAbstractions port is also testable on this machine: the Metal backend runs on Apple silicon, so correctness can be checked locally and only performance work needs a real GPU.

### What already fits

More than expected, mostly as a side effect of decisions made for other reasons.

**Every type that would cross to the device is already `isbits`** — checked, not assumed:

```
SVec{5,Float64}      HydroState{Float64}   Floors{Float64}
IdealGas{Float64}    ConstantOpacity       PowerLawOpacity
RadiationPhysics     CartesianGrid{Float64}
```

including `UserOpacity` holding a closure that captures a `Float64`. So the physics parameters, the grid and the interface states pass into a kernel as plain arguments — no adaptor, no `Adapt.@adapt_structure`. `SVec` being deliberately *not* an `AbstractArray` (`src/staticvec.jl`) pays off a second time here: it is a plain immutable struct, which is what a GPU kernel wants.

**The allocation discipline is already enforced.** GPU kernels cannot allocate at all, and Julia's GPU compiler rejects code that tries. `test/test_performance.jl` has been holding every kernel under 8 bytes per cell for unrelated reasons, so they should compile for the device largely as written.

**`Val(D)` direction dispatch survives.** It constant-folds on the device as it does on the host.

**Per-cell branchy code is fine.** The 2x2 Newton iteration in `radiation_source_step!` has a `while` loop with a data-dependent trip count, which costs warp divergence but not correctness. Quokka runs the same algorithm on GPU.

### What has to change

**1. The parallel decomposition is wrong, and this is the real work.**

`threaded_jk` (`src/grid.jl:157`) flattens `(j,k)` into one task index, and each task then runs the `i` loop serially:

```julia
threaded_jk(jr, kr) do j, k
    @inbounds for i in ir
        ...
    end
end
```

The parallel unit is a *pencil* along `i`, so available parallelism is `ny*nz` — right for 8 CPU threads, three orders of magnitude short for a GPU, which wants `nx*ny*nz` work items. Every kernel body must be rewritten from "given `(j,k)`, loop over `i`" to "given `(i,j,k)`, do one cell". Mechanical, but it touches roughly a dozen kernels and cannot be deferred or done partially.

The right shape is probably a `@kernel` plus a thin dispatch layer, so the CPU path keeps its pencil loop — cache-friendly, and slower as one-cell-per-task — while the GPU path gets one work item per cell.

**2. The array layout may need to flip.**

State arrays are `(nvar, i, j, k)` (`CLAUDE.md`, "Array layout"). Variable-index-first makes a cell's components contiguous, which is what the CPU Riemann solvers want. On a GPU, with consecutive threads handling consecutive `i`, those threads read addresses `nvar` apart — uncoalesced, and the bandwidth loss is the usual factor of several.

AMReX, and so Quokka, uses `(i,j,k,n)` for exactly this reason. Since structural fidelity to Quokka is a design goal, flipping would align two things at once — but it touches every kernel and every `IRHO`/`IMX`-style index, and it makes the CPU path slower. Measure before deciding. A layout type parameter is a third option and probably a bad one.

**3. Atomics and reductions.**

The `Threads.Atomic{Int}` counters in `enforce_floors!` (`src/hydro_system.jl:42`), `enforce_rad_floors!` (`src/radiation_system.jl:178`) and both source drivers (`src/source_terms.jl:95-97` and `394-396`) need device atomics or a proper reduction. `reduce_max_jk`, used by `hydro_timestep`, needs a real GPU reduction: per-block max, then a second pass. KernelAbstractions does not provide reductions; `GPUArrays`' `mapreduce` does, and would cover `hydro_timestep` and `conserved_totals` directly.

**4. Dependency policy.**

KernelAbstractions would be the first non-stdlib runtime dependency, which breaks the rule in `CLAUDE.md`. The precedent is `Plots`: a `[weakdeps]` entry with the device kernels in an extension, so a CPU-only user still installs nothing, and backend packages (`CUDA`, `Metal`, `AMDGPU`) stay the user's choice and never enter `Project.toml`. Whether KernelAbstractions can live in a weak dependency when the *kernels themselves* are written against it needs checking — it may have to be a hard dependency, in which case the rule needs an explicit amendment rather than a quiet exception.

### How to know it works

`compare` already provides the verification harness. A backend is a variant like any other:

```julia
compare(P.radhydro_shock, (cpu = (backend = CPU(),), gpu = (backend = GPUBackend(),));
        sizes = (128, 256, 512))
```

One table, both backends, with `wall_s` sitting next to the error columns so speedup and correctness are read off together. Bitwise agreement is not the target — reduction order differs — but the error metrics should match to several digits and `nrepaired` must stay zero on both.

The stronger check is the existing suite, which asserts invariants that hold exactly: conservation on periodic boundaries, direction isotropy, `|F| <= cE`, positivity without floor repairs. Those must hold on the device too.

### Rough order

1. Lift the `i` loop into the parallel index on the **CPU** path, still using `Threads.@threads`. No GPU involved, no behaviour change, suite stays green. This is the risky refactor; do it alone and measure the CPU cost.
2. Port one kernel end to end — `enforce_floors!` is simplest, no stencil — and get the extension plumbing right.
3. Reconstruction, then the flux sweeps. Stencil kernels; check the ghost indexing.
4. `radiation_source_step!`, and `multigroup_source_step!` beside it. Self-contained per cell, so the easiest of the hard ones — though the multigroup one calls out to an `AbstractCoupling`, and the two OrdinaryDiffEq-backed solvers will not follow it onto a device.
5. Reductions: `hydro_timestep`, `conserved_totals`, the repair counters.
6. Measure. Only then decide about the array layout.

---

## Deferred from the architecture review

Considered and consciously postponed, with the reason:

| | Why it was deferred |
|:---|:---|
| Gravity, self-gravity, cooling | The `strang_sources` hook exists so these can be added without touching `step!`. Nothing needs them yet. |
| A subsystem protocol (`AbstractSubsystem`) | Two subsystems do not justify one, and a Julia tuple-of-subsystems has no AMReX counterpart, so it works against the transliteration goal. |
| A pluggable radiation Riemann solver | Quokka has none either, and there is no second M1 solver to ship, so the extension point would have exactly one implementation. |
| Strang splitting of hydro against radiation | Currently Lie, and documented as such. Source terms are already Strang-split. |
| Renaming `source` to `radiation_source` | Clearer now that there are two source kinds, but it breaks callers to fix a documentation problem. |

**Multigroup radiation** was on that list and has since been built (`src/planck.jl`, `src/coupling*.jl`, and the README's *Multigroup radiation* section). The estimate held exactly: transport generalised for free, and every difficulty was in the coupling — which is why there are now three interchangeable solvers for it, two backward-Euler steppers inside one of them, and a pluggable linear solve inside two. What the entry got wrong is that the conditioning argument for the replaced base variables survived intact; it is `DirectCoupling`, in the natural variables, whose achievable tolerance degrades with the optical depth, which is the measurement of what the replacement buys.
