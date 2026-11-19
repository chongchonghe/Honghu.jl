# Shared-memory parallelism: what the measurements say

How Honghu uses the cores of one shared-memory node, what was changed to make it use them better, and what each change bought. Every number here is a step of the 3D Sedov blast (`Problems.sedov_blast`: PPM, HLLC, SSPRK3, outflow boundaries), timed by `test/benchmarks_parallel.jl`.

## Machine and method

- AMD Ryzen Threadripper PRO 7975WX: 32 cores, 64 hardware threads, one socket, one NUMA node. 1 MiB L2 per core, 128 MiB L3 in four 32 MiB slices (one per 8-core CCD). 124 GB RAM. Julia 1.13.4, Linux 6.12.
- `julia --project=. -t N test/benchmarks_parallel.jl n [--passes]`. The script takes three warm-up steps, then reports the minimum over 7 steps at a fixed `dt`. Threads are unpinned unless a row says otherwise.
- The node is shared, with a load average of about 4 from other users during these measurements. Differences under about 5% are within the noise and are not claimed.
- Every change is checked bit for bit against the state before any change, after 10 steps of seven problems: Sedov at 48³ and 37³ (PPM and PLM), a 2D Kelvin–Helmholtz, a 1D Sod, a 3D run with two mass scalars, and a 3D radiation-hydrodynamics run. The check compares `U`, `R`, `t` and the floor counters with `==`.

## Baseline

| threads | 128³ ms/step | speedup | 256³ ms/step | speedup |
|---:|---:|---:|---:|---:|
| 1 | 1373 | 1.0 | 10892 | 1.0 |
| 2 | 717 | 1.9 | | |
| 4 | 407 | 3.4 | | |
| 8 | 250 | 5.5 | 1994 | 5.5 |
| 16 | 188 | 7.3 | | |
| 32 | 137 | 10.0 | 1037 | 10.5 |
| 64 | 139 | 9.9 | 1027 | 10.6 |

Per pass, 128³ (each step runs three right-hand sides, each with one sweep per direction):

| pass | 1 thread | 32 threads | speedup |
|:---|---:|---:|---:|
| sweep (reconstruct + HLLC + divergence), each | 145 ms | 11.0 ms | 13 |
| of which HLLC | 89 ms | 5.0 ms | 18 |
| of which PPM | 48 ms | 3.9 ms | 12 |
| of which divergence | 6.5 ms | 1.7 ms | 4 |
| `cons2prim!` | 4.7 ms | 1.7 ms | 2.8 |
| `enforce_floors!` | 3.6 ms | 0.45 ms | 8 |
| `fill!(dU, 0)` (serial) | 2.6 ms | 2.5 ms | 1 |
| `copyto!(U0, U)` (serial) | 4.4 ms | 4.4 ms | 1 |
| `combine!` | 8.6 ms | 2.5 ms | 3.4 |
| `hydro_rhs!` total | 447 ms | 40 ms | 11 |

The arithmetic scales: HLLC reaches 18× on 32 cores. The passes that only move memory do not, and two of them do not run in parallel at all. Each sweep writes the full-grid interface states `qL` and `qR` and the full-grid `flux`, 84 MB each at 128³, and reads them back in the next pass, so past about 8 threads a step is bound by memory traffic and serial code, not by arithmetic. The 64 hardware threads give nothing over 32 cores.

## Changes, in the order they were made

### 1. Thread the serial whole-array copies

`hydro_rhs!` zeroed `dU` with `fill!`, `advance_hydro!` saved the stage-0 state with `copyto!`, and the radiation path did the same with `dR` and `Utmp`. All four were serial, and a single core cannot saturate the memory bus. `threaded_fill!` and `threaded_copyto!` (`src/grid.jl`) split the array into one contiguous chunk per thread through `threaded_chunks`.

| pass, 128³, 32 threads | before | after |
|:---|---:|---:|
| `copyto!(U0, U)` | 4.4 ms | 1.6 ms |
| `fill!(dU, 0)` | 2.5 ms | 0.8 ms |
| step | 137 ms | 131–133 ms |

That is about 8 ms of serial work a step, 4% at 32 threads. The 1-thread step is unchanged (1380 ms). Results are bit-identical.

### 2. Ghost filling: not threaded, deliberately

The plan was to split each built-in boundary face over threads. Measured first, the whole `fill_ghosts!` of the gas state costs 0.47 ms at 128³ on one thread, and it runs once per right-hand side, so at most 1.4 ms of a 131 ms step (1%) is there to be won. Ghost work scales with the surface (n²), not the volume (n³), so the fraction falls as the grid grows. That is within the noise of these measurements, and it is not worth complicating `fill_face!`, which users extend. It stays serial.

### 3. Fold the zeroing of `dU` into the first sweep

`flux_divergence!` takes `overwrite = true` for the first active direction, which stores `-ΔF/Δx` instead of subtracting it, so `hydro_rhs!` and `rad_rhs!` no longer zero `dU` and `dR` first. This removes one whole-array write per right-hand side, three per step. Only the sign of an exact zero can differ (`-0.0` against `0.0`), which `==` does not see, and nothing reads the ghost cells of `dU`. Measured: 131 → 130 ms at 32 threads, 1380 → 1366 ms at 1 thread. That is within the noise, as expected after change 1 had already made the zeroing 0.8 ms. It is kept because it costs nothing and removes a pass.

### 4. Fusing the floors into the conversion: not done

The plan was to fold `enforce_floors!` and `enforce_scalar_floors!` into `cons2prim!` in a single pass. The pass this would remove costs 0.45 ms at 32 threads (the scalar floors return at once without scalars), so at most 1.4 ms of a 130 ms step is there to be won. That is below the noise, and it would touch the split between projection and read-only conversion, which CLAUDE.md asks to preserve. Not done.

### 5. Static, contiguous scheduling: no gain, reverted

`Threads.@threads` defaults to dynamic scheduling. A variant of `threaded_jk` that gives each thread one contiguous block of `(j, k)` rows, so neighbouring rows share a core's cache, was tried on every hydrodynamic kernel (reconstruction, Riemann fluxes, divergence, floors, conversion, combine):

| threads | dynamic | static |
|---:|---:|---:|
| 1 | 1366 ms | 1376 ms |
| 8 | 250 ms | 241 ms |
| 32 | 130 ms | 134 ms |
| 64 | 139 ms | 129 ms |

All within the noise. The passes are bound by memory traffic, not by the scheduler or by cache reuse between neighbouring rows: a row of 128 cells is 5 kB per variable array, and the stencil of a `y` or `z` sweep reaches rows a whole plane away. Reverted.

### 6. Thread pinning: 10–16%, for free

`JULIA_EXCLUSIVE=1` pins Julia's thread `i` to CPU `i`. On this node CPUs 0–31 are the 32 physical cores and 32–63 their hyperthread siblings (`lscpu -e`), so `-t 32` gets one thread per core. Unpinned, the OS migrates threads between cores, and each thread loses the L2 contents its rows left behind. Measured twice at 32 threads:

| threads | unpinned | `JULIA_EXCLUSIVE=1` | gain |
|---:|---:|---:|---:|
| 16 | 194 ms | 164 ms | 16% |
| 32 | 132, 131 ms | 119, 119 ms | 10% |
| 64 | 129–139 ms | 123 ms | |

No code change. The README recommends it. Two caveats: it assumes the job has the whole node, or at least CPUs `0:N-1`, because a batch system that hands out other CPUs needs its own binding (`taskset`, or `srun --cpu-bind=cores`); and on a node with more than one NUMA domain the first-touch placement of the arrays would also matter, which this single-domain node cannot measure.
