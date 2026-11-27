# Performance optimization notes (single core)

Branch `perf/single-core`, from `docs/readme-performance` (`3b89a4b`).
Machine: AMD Ryzen Threadripper PRO 7975WX (Zen 4, AVX-512, up to 5.3 GHz), Julia 1.13.4.

Target: Athena++ quotes **2.9 Mcell-updates/s per core** for MHD + PLM + HLLD (VL2,
two stages per step, like SSPRK2 here) on a Skylake-SP Gold 6148, a slower core
than this one.

## Baseline

`julia --project=. -t 1 test/benchmarks.jl` (Sedov blast, 128^3, PLM, SSPRK2, 10 steps):

| | Hydro | RHD | MHD |
|:---|---:|---:|---:|
| Mcell-updates/(s CPU) | 3.51 | 1.57 | 1.40 |
| ns CPU / cell-update | 285 | 637 | 714 |
| reconstruction (PLM) | 33.0 | 33.3 | 61.7 |
| Riemann solver | 203.6 | 201.6 | 453.7 |
| flux divergence and stage update | 19.3 | 18.2 | 18.4 |
| EMF and induction | -- | -- | 103.0 |
| conversions and floors | 11.9 | 26.8 | 27.3 |
| ghost cells | 1.0 | 1.0 | 16.2 |
| time step | 3.4 | 3.4 | 13.3 |
| other | 12.5 | 24.1 | 20.1 |

The Riemann solver row of MHD is 63% of the step: six HLLD solves per cell per
step (3 directions x 2 stages), about 75 ns each including loads and stores.

Athena++'s whole step is 1/2.9e6 = 345 ns per cell on the slower CPU. Matching it
means the Riemann row must fall to about 100 ns, i.e. ~15 ns per HLLD solve.

## Scripts

In `docs/performance-experiments/`, each with its usage in its header:
`bench.jl` (steps per second of one configuration), `vec.jl` (does a kernel's row
loop vectorise, and with how many gathers), `micro.jl` (the MHD Riemann solvers
alone), `athena_blast.jl` (Athena++'s MHD blast input in Honghu).

## Parameter changes (no code change)

| n | steps | MHD Mcell/s | Hydro Mcell/s |
|---:|---:|---:|---:|
| 32 | 610 | 0.94 | 2.98 |
| 64 | 76 | 0.85 | 3.49 |
| 128 | 10 | 1.42 | -- |

- **Smaller grids are slower for MHD**, not faster: the MHD sweep solves Riemann
  problems `mhd_riemann_ghosts = 3` rows (PLM) into the ghosts across each sweep, so
  at 32^3 a sweep does (38/32)^2 = 1.41 times the interior's faces; at 128^3, 1.10.
  Athena++'s meshblocks pay the same kind of overhead, so its quoted number is for
  a large block.
- **64^3 over 76 steps is slower per cell than 128^3 over 10.** That turned out to
  be subnormal numbers, not the grid size (next section). At 64^3 over 10 steps:
  1.28 Mcell/s.

## Finding 1: subnormal arithmetic (fixed)

As the blast spreads, PLM puts values like `|m_x| ~ 1e-200` ahead of the shock:
physically zero, perfectly normal floats. Their products inside the Riemann
solvers (`u^2`, `b u`, ...) underflow into the subnormal range, which x86 handles
by a microcode assist of ~100+ cycles. The stored state holds no subnormals at
all; only the intermediates do, which is why it was not obvious.

| 64^3 MHD, 60 steps | 1 thread | 4 threads |
|:---|---:|---:|
| default (no flush) | 0.88 | 2.79 |
| subnormals flushed to zero | 1.19-1.22 | 3.53 |

Athena++ built with the Intel compiler runs with flush-to-zero by default (`-ftz`
is implied by `-O2` and above), so its numbers include this.

Fix: `Simulation(flush_subnormals = true)` (the default). `step!` sets FTZ on the
calling thread and restores the caller's mode on return; the threading helpers
(`threaded_jk`, `threaded_chunks`, the tiled sweep's spawn) copy the caller's mode
into each task through `_with_ftz` and restore the worker's afterwards, because
MXCSR is per OS thread and a task can land on any thread. Results change only
where a result would have been subnormal.

Caveat: `KABackend` launches through KernelAbstractions' own tasks, which do not
copy the mode.

## Finding 2: HLLD is divider-bound (partly fixed)

In isolation (random states, scalar loop) HLLD costs 50 ns per solve against 15
for HLL-MHD and 13 for LLF. Its LLVM IR had 26 `fdiv` and 10 `sqrt`, six of the
square roots with a branch to `throw_complex_domainerror`. Zen 4's scalar divider
takes ~5 cycles per `vdivsd` and more per `vsqrtsd`, so 26 divisions and 10 square
roots alone are ~200 cycles, i.e. the whole 50 ns.

Change: reciprocals formed once and multiplied by, as Athena++'s `hlld.cpp` does
(`1/(gamma-1)`, `1/sm_denom` shared by `S_M` and `p*`, `1/rho`, `1/den` in the
star state, `1/sqrt(rho*) = sqrt(rho*)/rho*`, `1/(S_R - S_L)` for the HLL blend);
`_sqrt`, the IEEE square root without Julia's domain check (C's `std::sqrt`); one
`sqrt` of the larger `c_u^2` instead of two.

Result: 26 -> 14 `fdiv`, microbenchmark 50 -> 44 ns, MHD benchmark at 128^3
1.42 -> 1.54 Mcell/s.

## Finding 3: nothing in the flux loops was SIMD-vectorised (fixed)

Athena++ runs its Riemann solvers as `#pragma omp simd` loops over a pencil in
`i`, eight cells per AVX-512 instruction. Honghu's kernels are written for one
cell and `launch!` loops over `i`, which is the same structure, but LLVM vectorised
none of the flux kernels. Checking `code_llvm` of the row loop (`vec.jl`), four things stood in the way:

1. **Possible aliasing.** The kernel's arrays may alias as far as LLVM knows, so
   it would not vectorise even a clean body. `launch!`'s contract (calls are
   independent, any order, possibly concurrent) is exactly `ivdep`, so the row
   loop is now `@simd ivdep`, in `CPUThreads` and `CPUSerial`. Each lane does the
   scalar loop's IEEE arithmetic, so results do not change.
2. **The passive-scalar loop inside the flux kernel.** LLVM vectorises innermost
   loops only, and vectorised the zero-trip `for s in 1:nscalars` instead of the
   loop over faces. The scalar fluxes are now `_scalar_flux_cell!`, a kernel of
   their own launched only when `nscalars > 0`, with the same arithmetic.
3. **The fallback branch** loaded from `prim` conditionally. Now both candidates
   are loaded and an `ifelse` picks: same values, straight-line code.
4. **`throw(DomainError)` branches in `sqrt`** (`soundspeed`,
   `fast_magnetosonic_speed`, HLLC, HLLD). Now `_sqrt`, the bare IEEE square root,
   in `src/staticvec.jl`.

| 128^3, 10 steps, 1 thread | MHD | Hydro |
|:---|---:|---:|
| baseline | 1.42 | 3.51 |
| + HLLD reciprocals | 1.54 | -- |
| + vectorised flux loops (1-4) | 2.36 | 5.54 |

## Finding 4: run-time strides make every vector load a gather (fixed)

Once vectorised, the MHD flux loop had 141 `masked.gather`s, the EMF 88. In
`A[n, i, j, k]` the stride along `i` is `size(A, 1)`, known only at run time,
even for the face arrays whose first dimension is 1, so LLVM can only gather
eight separate addresses per load.

`launch!` on the CPU now wraps every `Array{T,4}` (and the tiled sweep's
`TileArray`) with at most 16 rows in a `StaticRows{R}`: the same memory and the
same index arithmetic, with `R = size(A, 1)` a type parameter. One dynamic dispatch
per launch; the kernels are unchanged. With the stride a constant the face arrays
load contiguously and the 5-row states with wide loads and shuffles.

| 128^3, 10 steps, 1 thread | MHD | Hydro |
|:---|---:|---:|
| vectorised flux loops | 2.36 | 5.54 |
| + `StaticRows` | 3.01 | 6.72 |

## Finding 5: loops over variables inside a cell kernel (fixed)

`_reconstruct_cell!` and `_divergence_cell!` loop `for n in 1:nvar` with `nvar` a
run-time `Int`, so LLVM vectorised that loop (5 trips, mostly remainder) and left
the loop over `i` scalar. The launch sites now pass `Val(nvar)`; the kernels loop
over `1:_count(nvar)`, which is a constant, so the variable loop is fully unrolled
and the cell loop vectorises. Kernels still accept a plain `Int`.

| 128^3, 10 steps, 1 thread | MHD | Hydro |
|:---|---:|---:|
| `StaticRows` | 3.01 | 6.72 |
| + `Val(nvar)` | 3.32 | 7.47 |

## Finding 6: the radiation flux sweep was not a cell kernel (fixed)

`rad_fluxes!` was a hand-written `threaded_jk` loop over `i` and groups, with the
fallback as a branch and checked `sqrt`s in the closure. It is now
`_rad_flux_cell!` launched through `launch!`, with the group loop over `Val(NG)`,
the fallback a select and `_sqrt` in `eddington_factor`, `eddington_tensor` and
`m1_flux`. RHD at 128^3: 2.06 -> 2.75 Mcell/s.

## Breakdown after findings 1-5

`test/benchmarks.jl` (before finding 6):

| | Hydro | RHD | MHD |
|:---|---:|---:|---:|
| Mcell-updates/(s CPU) | 7.34 | 2.06 | 3.29 |
| ns CPU / cell-update | 136 | 485 | 304 |
| reconstruction (PLM) | 25.2 | 26.7 | 39.3 |
| Riemann solver | 74.0 | 71.8 | 134.6 |
| flux divergence and stage update | 14.6 | 13.4 | 13.4 |
| EMF and induction | -- | -- | 46.5 |
| radiation transport (HLL) | -- | 219.2 | -- |
| matter-radiation coupling | -- | 104.1 | -- |
| conversions and floors | 11.6 | 27.1 | 27.0 |
| ghost cells | 1.0 | 0.9 | 16.2 |
| time step | 2.9 | 2.6 | 7.2 |
| other | 7.1 | 19.6 | 20.2 |

Against the baseline, MHD went from 714 to 304 ns per cell-update (2.35x),
hydro from 285 to 136 (2.1x).

## Finding 7: the tiled sweep must build `StaticRows` type-stably

`StaticRows` costs one dynamic dispatch per launch. That is nothing for a
whole-grid launch, but the tiled sweep launches three kernels per tile box, and
`test_performance.jl` caught 87 bytes per cell (limit 8) in it. Now
`_sweep_chunk!` receives `Val(nvar)`, wraps `dU`, `prim`, the update tuple and the
three tile buffers once with that row count (`_static_row(A, Val(R))`,
`_static_tile`), and `launch!(::CPUSerial)` passes its arguments through untouched.
Two traps on the way, both 500-800 bytes per tile box:

- `_static_row(A, Val(R))` returning the bare array on a row mismatch made its
  result a `Union`, which was boxed. It throws instead.
- handing `_sweep_box!` the row count as an `Int` (`NV`) and calling `Val(nvar)`
  there is a dynamic `Val` per box. `_val(nvar)` passes a `Val` through.

The tiled sweep is now 5.7 kB per call on 24^3 (0.4 B per cell), less than the
untiled one.

## Like for like: Athena++ on this machine

Athena++ `1395e2f5`, built from a copy of the source tree (`configure.py --prob=blast -b --flux=hlld --cxx=g++`), its own
`inputs/mhd/athinput.blast` at 128^3 (PLM, HLLD, VL2, CFL 0.3, periodic), 20
cycles, one core:

| build | meshblock 128^3 | meshblock 64^3 |
|:---|---:|---:|
| `g++ -O3` (configure's default, SSE2) | 1.91 | 1.84 |
| `g++ -O3 -march=native -fopenmp-simd` | 2.08 | 1.97 |

Honghu on the same problem (`athena_blast.jl`: same domain,
states, field and CFL; SSPRK2 instead of VL2, also two stages per step):

| Honghu | Mcell-updates/s |
|:---|---:|
| `docs/readme-performance` (`3b89a4b`) | 1.42 |
| this branch | 3.36 |

So on this core Honghu went from 0.7x Athena++ (g++, AVX-512) to 1.6x. That ratio is
mostly a compiler artefact, as the next section shows. The schemes are not identical either: Honghu's HLLD
carries the MM21 low-Mach and carbuncle terms, and its CT is Q26 + LD04 rather
than Athena++'s Gardiner & Stone CT, so this is a comparison of codes, not of
implementations of one algorithm.

## Why Athena++ gets 2.1 here and 2.9 in its paper

The paper's number (Stone et al. 2020, Sec. III.6.1, Table 2: 2.924 MZone-cycles/s, MHD
PLM HLLD) is for the blast problem as one 64^3 MeshBlock, VL2, on one core of a Skylake-SP
Gold 6148. The paper does not name the compiler. Athena++'s `configure.py` for `icpc` uses
`-O3 -ipo -xhost -inline-forceinline -qopenmp-simd`, and icpc's defaults add FTZ and
`-fp-model fast=1`.

Athena++ on this core, the same 64^3 single-MeshBlock blast, no output, 40 cycles:

| g++ 14 build | MZone-cycles/s |
|:---|---:|
| `-O3` (SSE2) | 1.69 |
| `-O3 -march=native -fopenmp-simd` | 1.41 |
| ... `-mdaz-ftz` (flush subnormals) | 1.76 |
| ... `-fno-math-errno -flto` (≈ icpc's `-ipo`) | 2.00 |
| `-Ofast -march=native -fopenmp-simd -flto` | 2.09 |
| ... `-mrecip=all` | 2.09 |

`-fopt-info-vec` on the LTO link shows why the flags stop at 2.1: g++ does not vectorise
the HLLD `#pragma omp simd` loop (`hlld.cpp:54`, "Analysis failed"), nor PLM's main
loops, nor the cons-to-prim and CT loops. Only `calculate_fluxes`, `calculate_corner_e`
and one PLM loop vectorise. The paper says about 85% of the MHD code vectorises with
icpc. So this build runs Athena++'s dominant kernel scalar, which is the gap to its paper.

The CPUs also differ less than their clocks suggest. A Gold 6148 core has two 512-bit FMA
units (32 double flops a cycle) at about 3 GHz under AVX-512 load. Zen 4 executes AVX-512
as two 256-bit halves (16 flops a cycle) at up to 5.3 GHz. Peak vector throughput per core
is about 100 against 85 GFLOP/s, so for vectorised code the Skylake core is not the weaker
one. Zen 4's advantage is in scalar code, which is what g++ produced.

Not measured: icpx (oneAPI) is not installed here, and neither is clang.

## Summary, 128^3 Sedov blast, one core

| | Hydro | RHD | MHD |
|:---|---:|---:|---:|
| baseline (`3b89a4b`) | 3.51 | 1.57 | 1.40 |
| this branch | 7.36 | 2.74 | 3.27 |
| speed-up | 2.1x | 1.75x | 2.3x |

Plus whatever subnormals cost a given run (1.4x on the 64^3 MHD blast over 60
steps), which the 10-step benchmark does not show.

## Threads

Not the target here, but the changes must not cost the threaded runs. 16 threads,
128^3, 20 steps, `tile = :auto`:

| | Hydro | MHD |
|:---|---:|---:|
| `3b89a4b` | 27.3 | 8.9 |
| this branch | 39.2 | 10.1 |

Hydro tiles at 16 threads, and `DEFAULT_TILE = (32, 1, 1)` is still as good as any
tried with the vectorised kernels: (128, 4, 1) 39.5, (128, 8, 1) 37.7, (64, 4, 1)
36.5, untiled 21.7. On one thread the default `:auto` does not tile, and (32, 1, 1)
would now cost 20% (5.9 against 7.4): a 32-cell row is four vector iterations.
(128, 8, 1) gains 5% on one thread (7.75).

MHD gains least because its sweep is untiled: once the arithmetic is cheap, its
whole-grid passes are bound by memory bandwidth.

## Not done, and what to try next

Measured costs, MHD at 128^3 on one core, are in the breakdown above; the
recommendations are in order of expected gain.

1. **Tile the MHD sweep.** It is untiled because its faces reach `mhd_riemann_ghosts`
   rows across the sweep and the EMF reads two directions of face arrays. At 16
   threads the hydro sweep is 1.8 times faster tiled than untiled, and MHD now
   scales worst of the three. The face arrays an edge reads (`fspd`, `ustar`) are
   small (3 rows), so a tile can keep its interface states in cache and write only
   those.
2. **Fuse reconstruction into the flux kernel**, as Athena++'s pencils do: `qL` and
   `qR` are written to whole-grid arrays and read back, two streams of `nvar` rows
   per cell per direction per stage. A flux kernel that reconstructs its two
   states from `prim` itself would read `prim` only. This also removes a launch.
3. **The remaining 48 gathers in the MHD flux kernel.** Likely the fallback loads
   from `prim`/`bcc` at `im` and the transverse-jump stencil; check which with
   `DUMP=1 KERN=mhd vec.jl`. Perhaps 10-20% of the Riemann row.
4. **A structure-of-arrays layout** (`(i, j, k, n)`, AMReX's `Array4`, Quokka's
   layout): unit-stride loads with no shuffles, for every kernel. It is the
   invariant "Array layout" in `CLAUDE.md`, and changing it touches every file, so
   it needs its own measurement first: the shuffles of `StaticRows{5}` may already
   cost little.
5. **Vectorise the remaining hand-written loops.** `cons2prim!`,
   `enforce_floors!`, the ghost fills, `mhd_ghost_energy!`, `rad_cons2prim!`,
   `amend_rad_state!` and the radiation coupling are `threaded_jk` loops, not
   `launch!` kernels, so they get none of `ivdep`, `StaticRows` or `Val(nvar)`.
   Conversions and floors are 27 ns of MHD's 304, ghost cells 16, "other" 20.
6. **HLLC's divisions**: the star flux divides an `SVec` by `S_L - S_star` (five
   divisions, and both branches are computed once vectorised). One reciprocal, as in
   HLLD. HLLC agrees with Quokka only through the hydro tests' tolerances, so check
   them.
7. **The grey coupling** (104 ns per cell in RHD) is a per-cell Newton iteration
   with a data-dependent trip count; it will not vectorise as written. Iterating a
   fixed number of times with a convergence mask would, at the cost of iterations.
8. **`DEFAULT_TILE` on one thread**: (128, 8, 1) is 5% faster than untiled; `:auto`
   could tile on one thread with a wider tile. Measure on Apple Silicon too, where
   tiling has been slower.
9. **Float32** is accepted by `Simulation(T = Float32)` and would double the SIMD
   width; `bench.jl`'s initial conditions pass `Float64` values and fail, so it was
   not measured.
10. **Julia `-O3`** was not measured: it recompiles every dependency into a second
   cache, which exceeded the disk quota of the home directory on this machine.

Caveats of what was done:

- `flush_subnormals` does not reach `KABackend` launches, whose tasks are
  KernelAbstractions'.
- The `@simd ivdep` in `launch!` is a promise for every kernel ever launched: a
  kernel that reads what another cell of the same launch writes is now wrong in a
  new way (it always broke the contract, which says calls may run concurrently).
- `StaticRows` costs one dynamic dispatch, and a few hundred bytes, per
  `launch!(::CPUThreads)`; the tiled sweep builds its wrappers type-stably instead
  (finding 7). A caller of `launch!(::CPUSerial)` in a hot loop must do the same.
- The README's performance table was measured on an Apple M5 Max and is not
  updated here.
