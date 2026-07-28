# Future work

Things worth doing, with enough detail to start. Nothing here is committed to.

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

The `Threads.Atomic{Int}` counters in `enforce_floors!` (`src/hydro_system.jl:42`), `enforce_rad_floors!` (`src/radiation_system.jl:129`) and `radiation_source_step!` (`src/source_terms.jl:82-84`) need device atomics or a proper reduction. `reduce_max_jk`, used by `hydro_timestep`, needs a real GPU reduction: per-block max, then a second pass. KernelAbstractions does not provide reductions; `GPUArrays`' `mapreduce` does, and would cover `hydro_timestep` and `conserved_totals` directly.

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
4. `radiation_source_step!`. Self-contained per cell, so the easiest of the hard ones.
5. Reductions: `hydro_timestep`, `conserved_totals`, the repair counters.
6. Measure. Only then decide about the array layout.

---

## Deferred from the architecture review

Considered and consciously postponed, with the reason:

| | Why it was deferred |
|:---|:---|
| Gravity, self-gravity, cooling | The `strang_sources` hook exists so these can be added without touching `step!`. Nothing needs them yet. |
| A subsystem protocol (`AbstractSubsystem`) | Two subsystems do not justify one, and a Julia tuple-of-subsystems has no AMReX counterpart, so it works against the transliteration goal. |
| Multigroup radiation | Transport generalises for free; the coupling turns the 2x2 Newton into `(1+N_g)x(1+N_g)`, destroys the analytic inverse that makes it fast, and invalidates the conditioning argument for using `R_ex` as the second unknown. That is where all the work is. Replacing the literal `IERAD`/`IFRX` indices with accessors that constant-fold is cheap and worth doing first, but it would make the port *look* nearly done when it is not. |
| A pluggable radiation Riemann solver | Quokka has none either, and there is no second M1 solver to ship, so the extension point would have exactly one implementation. |
| Strang splitting of hydro against radiation | Currently Lie, and documented as such. Source terms are already Strang-split. |
| Renaming `source` to `radiation_source` | Clearer now that there are two source kinds, but it breaks callers to fix a documentation problem. |
