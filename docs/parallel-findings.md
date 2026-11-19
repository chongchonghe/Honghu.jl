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
