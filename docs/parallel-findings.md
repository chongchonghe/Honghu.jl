# Shared-memory parallelism: what the measurements say

How Honghu uses the cores of one shared-memory node, what was changed to make it use them better, and what each change bought. Every number here is a step of the 3D Sedov blast (`Problems.sedov_blast`: PPM, HLLC, SSPRK3, outflow boundaries), timed by `test/benchmarks_parallel.jl`.

## Summary

The 3D Sedov blast at 128³, the same machine, the code before and after, run back to back:

| | before | after | gain |
|:---|---:|---:|---:|
| 1 thread | 1381 ms | 1201 ms | 1.15× |
| 32 threads | 141 ms | 93 ms | 1.52× |
| 32 threads, pinned | 136 ms | 82 ms | 1.66× |
| speedup over 1 thread (32, pinned) | 10.2× | 14.6× | |

The answers are the same to the bit, on every problem checked, at every thread count. Three changes carry it. **Tiling** the hydrodynamic sweep (section 7) moves the interface states and fluxes out of whole-grid arrays and into cache. **Forming the Runge–Kutta stage in the last sweep** (section 8) removes a four-array pass per stage. **Splitting each kernel's body into a row function** that the tile driver shares made one thread 13% faster on its own (section 7). Pinning the threads is worth another 10% at 32 threads, but only when the job uses every core (section 6). The goal set at the outset, 70 ms at 32 threads, was not reached. The measurement in section 7 says why it could not have been: with every core busy, each core runs 1.47× slower than one core alone, so the ideal is about 56 ms, and the sweeps are now within 10–15% of their share of it. Section 9 lists what remains.

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

### 7. The tiled hydrodynamic sweep

`src/tiling.jl`. The sweep along `D` no longer makes three passes over whole-grid arrays. It goes box by box: reconstruct, solve the Riemann problems and take the divergence for one box, with the box's interface states and fluxes in a per-task scratch array that stays in cache. Only `dU` is written to the whole grid. This is AMReX's tiling, and Quokka's: box-local temporaries around the same pointwise kernels. Here the kernels are literally the same. Each kernel's per-row body became an `@inline` function (`_reconstruct_row!`, `_hydro_flux_row!`, `_divergence_row!`) that the whole-grid kernel and the tile driver both call, and the scratch is a `TileArray`, indexed with global cell indices like AMReX's `Array4`. Results are bit-identical, with any tile, to the untiled sweep and to the code before any change. `test/test_parallel.jl` checks this on 3D, 2D and 1D problems, with PCM, PLM and PPM, with and without mass scalars, and on grids that the tile does not divide.

**Tile shape.** A box spans the whole interior along the sweep, so no face is computed twice, and blocks of `tile[e]` cells along each other direction `e`. Pinned, 32 threads:

| tile | 64³ | 128³ | 256³ |
|:---|---:|---:|---:|
| untiled | 26.6 | 121 | 976 |
| (8, 8, 8) | | 94 | 801 |
| (4, 4, 4) | 17.2 | 92 | 691 |
| (8, 2, 2) | 16.8 | 87–89 | 606–662 |
| (16, 1, 1) | | 85 | 651 |
| (32, 1, 1) | | 85–86 | 631–678 |
| (128, 1, 1) | | 86 | 755 |

Two things set the optimum. The scratch of a box is `3 nvar (n_D + 3) b b'` numbers, and it has to fit in a core's 1 MiB L2: (128, 1, 1) is as good as any at 128³, but at 256³ its 4 MB scratch spills and it loses. Short rows cost loop overhead: (2, 2, 2) is 112 ms. `DEFAULT_TILE = (32, 1, 1)` is among the best at both sizes, with a scratch of 0.5 MB at 128³ and 1 MB at 256³.

**Not on one core.** On one thread every tile was 5–10% slower than untiled, repeatably (1276–1335 ms against 1211 ms at 128³). That holds in each direction, the `x` sweep included, whose rows are full length, so it is not row length. Wrapping the scratch as a plain `Array` made it worse (`x` sweep 182 ms against 131 ms). A single core has no contention on the memory bus for tiling to relieve. The crossover, untiled against `(32, 1, 1)`:

| threads | 2 | 4 | 8 | 16 | 32 |
|:---|---:|---:|---:|---:|---:|
| untiled | 663 | 375 | 230 | 159 | 121 |
| tiled | 693 | 382 | 219 | 147 | 85 |

So `Simulation(tile = :auto)`, the default, tiles from `TILE_MIN_THREADS = 8` threads and not below. A tuple forces a tile, and `nothing` forces the untiled sweep.

**Not on a grid with too few boxes.** The tile was tuned on 3D Sedov, but a sweep's tasks are its boxes, and a 2D run's transverse sweep has few of them: `ceil(n1/32)` with `(32, 1, 1)`. The final review measured Kelvin–Helmholtz at 32 threads. At 256² the step was unchanged by tiling, at 512² (16 boxes, half the threads idle) it was 4% slower, and at 1024² it was 18% faster (51.4 → 42.1 ms). So `:auto` also requires at least one box per thread in every active direction. That covers every 3D grid from about 32³ on, and 2D grids from 1024² at 32 threads.

**A bug worth recording.** The first version picked up each task's scratch in the spawning loop, in the same function as the one-task branch, which assigned the same local names. The loop's assignments then rebound variables the spawned closures had captured (the boxing trap of CLAUDE.md's performance rules), so tasks shared one scratch and the result depended on timing. The allocation test could not see it, since it costs bytes per task and not per cell, and on one thread it cannot happen. The bit-identity tests at `-t 4` caught it at once. Each task now takes its scratch from its chunk number inside the task.

**Where it stands.** At 128³, 32 threads: unpinned 97 ms, pinned 85.5 ms (from 137 and 121 before tiling). A sweep takes 6.4–8 ms, against 11 ms untiled. One more measurement puts that in proportion. The same HLLC row kernel on a cache-resident grid, one private copy per task, runs 1.47× slower per core with 32 tasks than with one (1.32× with 8, 2.40× per hardware thread with 64). Nothing is shared there, so this is the clock dropping from its single-core boost, plus the other users of the node, and it is a ceiling no software change moves. Scaled by it, the ideal 32-core step is 1217 × 1.47 / 32 ≈ 56 ms, and the sweeps, about 60 ms a step, are close to their share of it. What is left above that is the passes that still stream whole arrays: about 20 ms a step.

### 8. The Runge–Kutta stage formed in the last sweep

`combine!` formed each stage, `U = a U0 + b U + c Δt dU`, as a pass of its own over four whole arrays: 2.3 ms at 32 threads, three times a step. Now `hydro_rhs!(dU, U, sim; update = (a, U0, b, c Δt))` hands the stage to the last active sweep, whose divergence row forms `U` directly from the fluxes, with the same arithmetic as `combine!`, and never stores the final `dU`. This is safe because the sweeps read `prim`, never `U`. It is also how Quokka forms its stages: from the fluxes, with no stored right-hand side. Bit-identical, checked against `hydro_rhs!` followed by `combine!` in 3D, 2D and 1D, tiled and untiled. Pinned, 32 threads: 85.5 → 82.0 ms. Unpinned: 97 → 93.5 ms. 64 threads pinned: 78.4 ms.

### 9. Final numbers, and what is left

Sedov, final code, one run each (128³ before and after are also in the summary, run back to back):

| threads | 128³ unpinned | 128³ pinned | 256³ unpinned | 256³ pinned |
|---:|---:|---:|---:|---:|
| 1 | 1201 | 1249 | 9526 | 9864 |
| 2 | 628 | 657 | | |
| 4 | 354 | 363 | | |
| 8 | 198 | 216 | 1625 | 1747 |
| 16 | 115 | 140 | 881 | 1064 |
| 32 | 93 | 82 | 690 | 670 |
| 64 | 90 | 78 | 634 | 640 |

At 256³ the step went from 1037 ms to 690 ms at 32 threads (1.50×), and from 10.9 s to 9.5 s on one thread.

**Pinning, revised.** Section 6 measured pinning before tiling, when it won at 16 threads as well. Measured again on the final code, it wins only when every core is used. `JULIA_EXCLUSIVE=1` puts thread `i` on CPU `i`, so a 16-thread run sits on CPUs 0–15: two of the four 8-core CCDs, with two of the four L3 slices and memory links. That made it 20% slower than unpinned at 16 threads (140 against 115 ms) and 9% slower at 8. Pin a run that has the whole node. Leave a smaller one unpinned, or bind it to cores spread over the CCDs.

**Where a 32-thread step goes now** (pinned, 128³, 82 ms): three right-hand sides of 24.4 ms each. Each is three sweeps of 6.4–7.0 ms, plus the floors, the conversion and the ghost fill (2.5 ms together) and task start-up. The rest of the step is the stage copy (1.6 ms), the CFL scan, which converts the whole grid to primitives first (2.3 ms), and the end-of-step floors (0.3 ms).

What is left, largest first:

- **The kernels on one core.** The sweeps are within 10–15% of their clock-limited share, so further gains in them have to come from making each core faster, not from more parallelism. The HLLC row and the PPM row are scalar loops over faces. Vectorising them across `i` (SIMD, 8 doubles with AVX-512 on this CPU) is the largest remaining lever, and it is a separate piece of work.
- **The remaining whole-grid passes**, about 10 ms a step: the floor pass, the ghost fill, the conversion, the stage copy and the CFL scan. Fusing the floors into the conversion, threading the ghost fill and reusing the conversion in the CFL scan would save perhaps 5 ms (6%). Each was set aside as too small to measure at the time (sections 2 and 4). Together they are no longer below the noise.
- **The radiation sweep** is untouched. It has the same structure, reconstruct then HLL then divergence, so the same tiling would apply, but the implicit coupling dominates a radiation step.
- **NUMA.** This node has one NUMA domain, so first-touch placement of the arrays, which matters on a two-socket node, could not be measured.
