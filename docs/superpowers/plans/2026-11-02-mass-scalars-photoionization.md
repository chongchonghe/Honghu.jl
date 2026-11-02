# Mass Scalars and Hydrogen Photoionization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add passive mass scalars (conserved $\rho X_s$) to Honghu, and on top of them a solver-agnostic photoionization chemistry with a hydrogen OTSA network, two integrators, and Iliev et al. (2006) Tests 1 and 2.

**Architecture:** Scalars are extra rows `NHYDRO+1 : NHYDRO+ns` of `sim.U`, reconstructed as $\rho X_s$ and fluxed as mass flux times upwind $X_s$ inside `hydro_fluxes!`. Chemistry follows the `coupling.jl` pattern: a network object supplies `chem_rhs!` / `chem_jacobian!`, an `AbstractChemSolver` integrates one cell, and a driver `chemistry_step!` runs once per radiation substep after `imex_substep!`. The ionizing photons are one existing radiation group with zero thermal opacity.

**Tech Stack:** Julia 1.10+, OrdinaryDiffEq (already a dependency: `Rosenbrock23`, `init` / `reinit!` / `solve!`), Honghu's own `SVec`, `threaded_jk`, `Test`.

**Spec:** `docs/superpowers/specs/2026-11-02-mass-scalars-photoionization-design.md`. Read it before any task. Quokka reference: `/Users/u1149259/softwares/quokka/quokka-local` (`src/networks/photoionization_H_caseB_handwritten/actual_rhs.H`, `src/radiation/photochemistry.hpp`, `src/problems/StromgrenSphere/testStromgrenSphere.cpp`).

## Global Constraints

- No new `[deps]`. Add `init`, `solve!`, `reinit!` to the existing selective `using OrdinaryDiffEq: ...` line in `src/Honghu.jl`; never `using OrdinaryDiffEq` wholesale (it exports `step!`).
- A run with no scalars and no chemistry stays bit for bit identical (checked against `docs/superpowers/plans/baseline.txt`, Task 1).
- Hot kernels allocate under 8 bytes per interior cell (`LIMIT = 8.0` in `test/test_performance.jl`). `BackwardEulerChem` allocates 0 bytes per call. The `RosenbrockChem` path may allocate (OrdinaryDiffEq).
- Never floor inside `cons2prim!`, an accessor, or `timestep` (CLAUDE.md invariant).
- Inside `threaded_jk` closures, do not reassign a variable captured from outside the closure (boxing trap, CLAUDE.md). Never key anything on `Threads.threadid()`.
- Chemistry runs require: radiation enabled, `chat_limiter isa ConstantChat`, `beta_order == 0`, zero thermal opacity in each band group, CGS units.
- Defaults: `RosenbrockChem(alg = Rosenbrock23(), reltol = 1e-2, abstol = 1e-8)`, `BackwardEulerChem(tol = 1e-10, max_iterations = 50)`.
- Reduced speed of light (spec 4.3): gas-side photoionization rate $c\sigma n_{\rm H^0}N$; radiation-side loss $\hat c\sigma n_{\rm H^0}N$; emission adds $\phi\,q\,\Delta t$ to $N$ with $\phi=\hat c/c$; conserved budget $n_{\rm H^+}+N/\phi$.
- Every exported name has a docstring saying what it is for. Every new `src/` file gets a line in `docs/make.jl` `PAGES` in the task that creates it. Every user-visible change gets a line in `CHANGELOG.md` under `## [Unreleased]` in the task that makes it.
- Markdown written for the docs: one paragraph per line, LaTeX math (`$\geq$`, not Unicode symbols).
- New test files must qualify internal names (`Honghu.scalar_index`, `Honghu.IRHO`, ...), so they run alone. Run one file with `julia --project=. -t 4 -e 'using Test, Honghu; const P = Honghu.Problems; include("test/<file>.jl")'`. Full suite: `julia --project=. -t 4 test/runtests.jl` (about 80 s today; the new tests may add about 30 s).
- Commit after each task with a Conventional Commit message.

## Review Focus

- A `Fixed` or `FunctionBC` hydro boundary whose state has 5 entries while `nscalars > 0`: expect a `DimensionMismatch` at the first ghost fill, not a silent fill. Test in Task 1.
- A dark, fully neutral, cold cell ($N = 0$, $x = 0$, $T = 100$ K, cooling on): both integrators return converged, no NaN, and the state unchanged. Test in Task 5.
- A checkpoint written with a different number of scalars: loading throws, it does not misread the bytes. Test in Task 3.
- Reading diagnostics (`field`, `gas_temperature`, `primitives`, `conserved_totals`) with scalars and with chemistry leaves `sim.U` bit for bit unchanged. Tests in Task 3 and Task 6.
- Chemistry with hydro on: the hydrogen mass is conserved and $X_{\rm H^0}+X_{\rm H^+}=1$ per cell. Test in Task 6.

## File Structure

| File | Responsibility | Task |
|:---|:---|:---|
| `src/variables.jl` | `scalar_index(s)` | 1 |
| `src/boundary.jl` | `fill_ghosts!(...; momentum_blocks)`, Reflecting fix | 1 |
| `src/simulation.jl` | `nscalars` / `chemistry` plumbing, IC keys, totals, diagnostics, chemistry hook in `advance_radiation!` | 1, 2, 6 |
| `src/hydro_system.jl` | scalar copy in `cons2prim!`, scalar flux in `hydro_fluxes!`, scalar floors | 2 |
| `src/output.jl`, `src/plotting.jl` | mass-fraction fields, checkpoint format v2 | 3 |
| `src/chemistry.jl` (new) | network interface, `ChemCell`, solver interface | 4 |
| `src/chemistry_solvers.jl` (new) | `RosenbrockChem`, `BackwardEulerChem` | 4 |
| `src/chemistry_hydrogen.jl` (new) | `HydrogenOTSA` network and rates | 5 |
| `src/chemistry_step.jl` (new) | `Photochemistry`, per-row workspaces, `chemistry_step!` | 6 |
| `src/problems/stromgren.jl` (new) | Iliev Tests 1 and 2, radius references | 7 |
| `test/test_scalars.jl`, `test/test_chemistry.jl`, `test/test_stromgren.jl` (new) | tests | 1-7 |
| `examples/stromgren_sphere.jl`, `docs/photoionization-findings.md`, `docs/photoionization-experiments/` (new) | example, measurements, findings note | 8 |

---

## Part A: mass scalars

### Task 1: Scalar storage, initial conditions, totals and the Reflecting boundary

**Files:**
- Create: `docs/superpowers/plans/baseline.jl` and `baseline.txt` (scratch: never `git add` them), `test/test_scalars.jl`
- Modify: `src/variables.jl`, `src/boundary.jl:136-148,180-198`, `src/simulation.jl:100-150,231-265,337-393,414,489,498,510,857-881`, `test/runtests.jl`, `CHANGELOG.md`

**Interfaces:**
- Produces: `scalar_index(s::Integer) -> Int` (`NHYDRO + s`); `Simulation(; ..., nscalars::Integer = 0)` with a `const nscalars::Int` field; `nhydrovars(sim) -> Int` (`NHYDRO + sim.nscalars`); `HydroWorkspace(g, nvar::Integer = NHYDRO)`; `RadiationWorkspace(g, nrad, ng, nvar_gas::Integer = NHYDRO)`; `fill_ghosts!(A, b, g, t = zero(T); momentum_blocks::Integer = size(A, 1) ÷ NRAD)`; IC key `X` (tuple or vector of `nscalars` mass fractions, default all zero); `conserved_totals(sim).scalars::Vector{T}`.

- [ ] **Step 1: Record the baseline before touching any code**

Write `docs/superpowers/plans/baseline.jl`:

```julia
using Honghu
const P = Honghu.Problems
function fingerprint()
    out = String[]
    s, _ = P.sod_shock_tube(n=64); run!(s, tmax=0.05, verbose=false)
    push!(out, "sod $(hash(s.U))")
    s, _ = P.kelvin_helmholtz(n=32); run!(s, tmax=1.0, maxsteps=20, verbose=false)
    push!(out, "kh $(hash(s.U))")
    s = Simulation(n=(16, 16, 1), boundaries=Boundaries(Reflecting()))
    set_initial_conditions!((x, y, z) -> (rho=1.0, p=(x - 0.5)^2 + (y - 0.5)^2 < 0.02 ? 10.0 : 0.1), s)
    run!(s, tmax=1.0, maxsteps=20, verbose=false)
    push!(out, "blast-reflecting $(hash(s.U))")
    s, _ = P.radhydro_shock(n=64); run!(s, tmax=1.0, maxsteps=20, verbose=false)
    push!(out, "radshock $(hash(s.U)) $(hash(s.R))")
    s, _ = P.marshak_vaytet(n=32); run!(s, tmax=1.0, maxsteps=10, verbose=false)
    push!(out, "vaytet $(hash(s.U)) $(hash(s.R))")
    return out
end
foreach(println, fingerprint())
```

Run: `julia --project=. -t 4 docs/superpowers/plans/baseline.jl > docs/superpowers/plans/baseline.txt`
Expected: five lines in `baseline.txt`.

- [ ] **Step 2: Write the failing test**

Create `test/test_scalars.jl` and add `include("test_scalars.jl")` to `test/runtests.jl` right after `include("test_hydro.jl")`.

```julia
@testset "mass scalars: storage and boundaries" begin
    mk(; kw...) = Simulation(; n=(8, 1, 1), nscalars=2, kw...)
    sim = mk()
    @test size(sim.U, 1) == Honghu.NHYDRO + 2
    @test Honghu.nhydrovars(sim) == 7
    @test Honghu.scalar_index(1) == 6
    @test size(Simulation(n=(8, 1, 1)).U, 1) == Honghu.NHYDRO
    @test_throws ArgumentError Simulation(n=(8, 1, 1), nscalars=-1)

    set_initial_conditions!((x, y, z) -> (rho=2.0, p=1.0, X=(0.25, 0.75)), sim)
    i = sim.grid.lo[1]
    @test sim.U[6, i, 1, 1] == 0.5
    @test sim.U[7, i, 1, 1] == 1.5
    @test conserved_totals(sim).scalars ≈ [0.5, 1.5]
    @test_throws ArgumentError set_initial_conditions!((x, y, z) -> (rho=1.0, p=1.0, X=(1.0,)), mk())

    # Reflecting flips the normal momentum only, never a scalar row. Eight rows
    # on purpose: 8 ÷ NRAD == 2 is what the old stride rule would have flipped.
    g = sim.grid
    A = Honghu.allocate(g, 8)
    for ii in g.lo[1]:g.hi[1], n in 1:8
        A[n, ii, 1, 1] = 10n + ii
    end
    Honghu.fill_ghosts!(A, Boundaries(Reflecting()), g; momentum_blocks=1)
    @test A[2, g.lo[1]-1, 1, 1] == -A[2, g.lo[1], 1, 1]
    @test A[6, g.lo[1]-1, 1, 1] == A[6, g.lo[1], 1, 1]

    # a Fixed hydro boundary has to cover the scalars too
    bad = Simulation(n=(8, 1, 1), nscalars=1,
                     boundaries=Boundaries(Fixed((1.0, 0.0, 0.0, 0.0, 1.0))))
    @test_throws DimensionMismatch set_initial_conditions!((x, y, z) -> (rho=1.0, p=1.0, X=(1.0,)), bad)

    # the IMEX stages never touch the scalar rows
    rad = RadiationPhysics(c=1.0, a_rad=1.0, opacity=ConstantOpacity(1.0))
    rs = Simulation(n=(8, 1, 1), nscalars=1, hydro=false, radiation=rad)
    set_initial_conditions!((x, y, z) -> (rho=1.0, T=1.0, Trad=0.5, X=(0.3,)), rs)
    S0 = rs.U[6, :, :, :]
    step!(rs)
    @test rs.U[6, :, :, :] == S0
end
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `julia --project=. -t 4 -e 'using Test, Honghu; const P = Honghu.Problems; include("test/test_scalars.jl")'`
Expected: FAIL with `MethodError` / unsupported keyword `nscalars`.

- [ ] **Step 4: Implement**

- `src/variables.jl`: `@inline scalar_index(s::Integer) = NHYDRO + s`, with a docstring (Quokka's `scalar0_index`). Extend the header comment with the scalar rows.
- `src/simulation.jl`:
  - `Simulation` gets the keyword `nscalars::Integer = 0` (throw `ArgumentError` if negative) and a `const nscalars::Int` field.
  - `U = allocate(grid, NHYDRO + nscalars)`, `HydroWorkspace(grid, NHYDRO + nscalars)`, `RadiationWorkspace(grid, nrad, ng, NHYDRO + nscalars)` (its `Utmp` is filled by `copyto!` from `U` in `imex_substep!`).
  - Add `nhydrovars(sim)` with a docstring.
  - `set_initial_conditions!` reads the key `X` with `_field(s, Val(:X), nothing)`: missing means zeros; a length different from `nscalars` throws `ArgumentError`. It writes `U[scalar_index(s)] = rho * X[s]` and documents the key in the table.
  - `conserved_totals` adds `scalars = [Σ ρX_s dV for s in 1:nscalars]`.
  - Every internal `fill_ghosts!` on a hydro array (lines 388, 414, 489, 498, 510) passes `momentum_blocks = 1`.
- `src/boundary.jl`:
  - `fill_ghosts!` takes `momentum_blocks::Integer = size(A, 1) ÷ NRAD` and calls an internal `_fill_face!(A, bc, g, dir, side, t, momentum_blocks)`. That function forwards to the public `fill_face!` for every BC except `Reflecting`, which calls `_reflect!(A, g, dir, side, momentum_blocks)` and flips rows `dir + 1 + NRAD*(m-1)` for `m in 1:momentum_blocks`.
  - The public `fill_face!(A, ::Reflecting, ...)` keeps working, as `_reflect!` with the default block count, so user code and the existing tests are unchanged.
  - Update the `Fixed` / `FunctionBC` docstrings: "`NHYDRO + nscalars` values on a hydro boundary".

- [ ] **Step 5: Run the test to verify it passes**

Run the command of Step 3. Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `julia --project=. -t 4 test/runtests.jl`
Expected: all pass, including `test_grid.jl`'s Reflecting checks and `test_radiation_multigroup.jl:651-658`, which call `fill_ghosts!` without the keyword.

- [ ] **Step 7: Commit**

```bash
git add src/variables.jl src/boundary.jl src/simulation.jl test/test_scalars.jl test/runtests.jl CHANGELOG.md
git commit -m "feat: storage, initial conditions and boundaries for mass scalars"
```

### Task 2: Advection of mass scalars

**Files:**
- Modify: `src/hydro_system.jl:22-110,124-215`, `src/simulation.jl` (`Diagnostics`, `reset_diagnostics!`, `_print_summary`, `_hydro_sweep!`, `hydro_rhs!`, `advance_hydro!`, `enforce_state!`, `primitives`), `test/test_scalars.jl`, `test/test_performance.jl`, `CHANGELOG.md`

**Interfaces:**
- Consumes: `scalar_index`, `nhydrovars`, `sim.nscalars` (Task 1).
- Produces: `cons2prim!(prim, U, eos, g, nscalars::Integer = 0)`, which copies the $\rho X_s$ rows unchanged; `hydro_fluxes!(flux, qL, qR, prim, eos, solver, Val(D), g, nscalars::Integer = 0)`; `enforce_floors!(U, eos, g, floors, nscalars::Integer = 0) -> Int` (density reset keeps $X_s$); `enforce_scalar_floors!(U, g, nscalars) -> Int`; `Diagnostics.floors_scalar::Int`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_scalars.jl`:

```julia
@testset "mass scalars: advection" begin
    for riemann in (HLLC(), LLF()), recon in (PLM(), PPM())
        sim = Simulation(n=(64, 1, 1), nscalars=2, riemann=riemann, reconstruction=recon,
                         boundaries=Boundaries(Periodic()))
        set_initial_conditions!((x, y, z) -> (rho=1 + 0.5sinpi(2x), vx=1.0, p=1.0,
                                              X=(0.25, 0.75)), sim)
        m0 = conserved_totals(sim).scalars
        run!(sim, tmax=0.5, verbose=false)
        ir = sim.grid.lo[1]:sim.grid.hi[1]
        X1 = sim.U[6, ir, 1, 1] ./ sim.U[1, ir, 1, 1]
        X2 = sim.U[7, ir, 1, 1] ./ sim.U[1, ir, 1, 1]
        @test maximum(abs.(X1 .- 0.25)) < 1e-12          # uniform stays uniform
        @test maximum(abs.(X1 .+ X2 .- 1)) < 1e-12
        @test conserved_totals(sim).scalars ≈ m0 rtol = 1e-13

        sim = Simulation(n=(64, 1, 1), nscalars=1, riemann=riemann, reconstruction=recon,
                         boundaries=Boundaries(Periodic()))
        set_initial_conditions!((x, y, z) -> (rho=1.0, vx=1.0, p=1.0,
                                              X=(0.25 < x < 0.5 ? 1.0 : 0.0,)), sim)
        run!(sim, tmax=0.3, verbose=false)
        X = sim.U[6, ir, 1, 1] ./ sim.U[1, ir, 1, 1]
        @test minimum(X) >= -1e-14                         # monotone contact
        @test maximum(X) <= 1 + 1e-14
        @test sim.diag.floors_scalar == 0
    end

    # scalars are passive: the hydro rows are identical with and without them,
    # with reflecting walls and three scalars (eight rows) on purpose
    a = Simulation(n=(32, 32, 1), boundaries=Boundaries(Reflecting()))
    b = Simulation(n=(32, 32, 1), nscalars=3, boundaries=Boundaries(Reflecting()))
    w(x, y) = (rho=1 + 0.2sinpi(2x) * cospi(2y), vx=0.3, vy=-0.2, p=1.0)
    set_initial_conditions!((x, y, z) -> w(x, y), a)
    set_initial_conditions!((x, y, z) -> (; w(x, y)..., X=(0.2, 0.3, 0.5)), b)
    run!(a, tmax=1.0, maxsteps=10, verbose=false)
    run!(b, tmax=1.0, maxsteps=10, verbose=false)
    @test a.U == b.U[1:Honghu.NHYDRO, :, :, :]

    # the density floor keeps the mass fraction; a negative scalar is clipped and counted
    s = Simulation(n=(4, 1, 1), nscalars=1)
    set_initial_conditions!((x, y, z) -> (rho=1.0, p=1.0, X=(0.4,)), s)
    i = s.grid.lo[1]
    s.U[1, i, 1, 1] = 1e-6
    s.U[6, i, 1, 1] = 0.4e-6
    Honghu.enforce_floors!(s.U, s.eos, s.grid, Floors(rho=1e-3), 1)
    @test s.U[1, i, 1, 1] == 1e-3
    @test s.U[6, i, 1, 1] / s.U[1, i, 1, 1] ≈ 0.4
    s.U[6, i+1, 1, 1] = -1e-3
    @test Honghu.enforce_scalar_floors!(s.U, s.grid, 1) == 1
    @test s.U[6, i+1, 1, 1] == 0.0
end
```

Add to `test/test_performance.jl`, after the "hot kernels do not allocate per cell" testset:

```julia
@testset "scalar kernels do not allocate per cell" begin
    LIMIT = 8.0
    sim = Simulation(n=(24, 24, 24), nscalars=2, boundaries=Boundaries(Periodic()))
    set_initial_conditions!((x, y, z) -> (rho=1 + 0.2sinpi(2x), vx=0.3, p=1.0, X=(0.4, 0.6)), sim)
    hw, g, nv = sim.hw, sim.grid, Honghu.nhydrovars(sim)
    Honghu.cons2prim!(hw.prim, sim.U, sim.eos, g, 2)
    Honghu.reconstruct!(hw.qL, hw.qR, hw.prim, PPM(), Val(1), g, nv)
    @test bytes_per_cell(() -> Honghu.cons2prim!(hw.prim, sim.U, sim.eos, g, 2), sim) < LIMIT
    @test bytes_per_cell(() -> Honghu.hydro_fluxes!(hw.flux, hw.qL, hw.qR, hw.prim, sim.eos,
                                                    sim.riemann, Val(1), g, 2), sim) < LIMIT
    @test bytes_per_cell(() -> Honghu.enforce_floors!(sim.U, sim.eos, g, sim.floors, 2), sim) < LIMIT
    @test bytes_per_cell(() -> Honghu.enforce_scalar_floors!(sim.U, g, 2), sim) < LIMIT
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. -t 4 -e 'using Test, Honghu; const P = Honghu.Problems; include("test/test_scalars.jl")'`
Expected: FAIL. The scalar rows are not fluxed, so $X$ drifts, and `floors_scalar` / `enforce_scalar_floors!` are undefined.

- [ ] **Step 3: Implement**

- `cons2prim!`: copy rows `scalar_index(1):scalar_index(nscalars)` unchanged. Quokka reconstructs $\rho X$ (`hydro_system.hpp:323-326`).
- `hydro_fluxes!`:
  - Keep a Bool per side recording whether that side fell back to the cell-centred state.
  - After `F = riemann_flux(...)`, for each `s` take `XL = (fallbackL ? prim[idx, im, jm, km] : qL[idx, i, j, k]) / rhoL`, and `XR` likewise, with `idx = scalar_index(s)`.
  - Set `flux[idx, i, j, k] = F[1] * (F[1] >= 0 ? XL : XR)`.
  - `riemann_flux` and `HydroState` do not change.
- `enforce_floors!`:
  - In the density-reset branch, first compute `X_s = (rho_old > 0 && isfinite(rho_old)) ? U[idx]/rho_old : 0`, then write `U[idx] = X_s * floors.rho`.
  - The return value still counts hydro repairs only.
  - This departs from Quokka on purpose: Quokka scales by `rho/rho_new` and renormalises. Here $X_s$ is kept.
- `enforce_scalar_floors!(U, g, nscalars)`: over the whole array, ghosts included, set any negative or NaN $\rho X_s$ to zero, and return the number of cells changed. No renormalisation.
- `src/simulation.jl`:
  - `_hydro_sweep!` passes `nhydrovars(sim)` to `reconstruct!` and `flux_divergence!`, and `sim.nscalars` to `hydro_fluxes!`.
  - `hydro_rhs!` and `enforce_state!` pass `sim.nscalars` to `enforce_floors!` and add `enforce_scalar_floors!` to `diag.floors_scalar`. `enforce_state!` does this even when `hydro_enabled` is false, whenever `nscalars > 0`.
  - `advance_hydro!` combines `nhydrovars(sim)` rows.
  - `primitives` passes `sim.nscalars`, and its docstring says the scalar rows hold $\rho X_s$.
  - `Diagnostics` gets `floors_scalar`. Update `Diagnostics()`, `reset_diagnostics!`, the docstring table and `_print_summary`.

- [ ] **Step 4: Run the tests to verify they pass**

Run the command of Step 2, then `julia --project=. -t 4 -e 'using Test, Honghu; const P = Honghu.Problems; include("test/test_performance.jl")'`. (`bytes_per_cell` is defined at the top of `test_performance.jl`, so that file runs alone.)
Expected: PASS. If the uniform-$X$ check fails by more than round-off with `PPM`, look first at a near-tie in a `ppm_edges` comparison before loosening anything.

- [ ] **Step 5: Commit**

```bash
git add src/hydro_system.jl src/simulation.jl test/test_scalars.jl test/test_performance.jl CHANGELOG.md
git commit -m "feat: advect mass scalars with the Riemann mass flux"
```

### Task 3: Output, fields and checkpoints with scalars

**Files:**
- Modify: `src/output.jl:35-66,186-244`, `src/plotting.jl:30-150`, `test/test_scalars.jl`, `CHANGELOG.md`, `CLAUDE.md` (the "Momentum index" invariant)

**Interfaces:**
- Consumes: Task 1, Task 2.
- Produces:
  - Field names `:X_1 ... :X_ns`, the mass fraction $\rho X_s/\rho$, with label `"mass fraction s"`.
  - VTK arrays `"mass_fraction_s"`.
  - Checkpoint magic `b"HONGHU\x00\x02"`, which writes `Int64(size(U, 1))` and `Int64(size(R, 1))` after the radiation flag. It still reads `b"HONGHU\x00\x01"` and the Hydro3D magic, but only into a simulation with `nscalars == 0`.

- [ ] **Step 1: Write the failing test**

```julia
@testset "mass scalars: output, fields, checkpoints" begin
    sim = Simulation(n=(8, 4, 1), nscalars=2)
    set_initial_conditions!((x, y, z) -> (rho=1 + x, p=1.0, X=(x, 1 - x)), sim)
    U0 = copy(sim.U)
    @test (:X_1, :X_2) ⊆ available_fields(sim)
    X1, lbl = field(sim, :X_1)
    @test lbl == "mass fraction 1"
    @test X1[3, 2, 1] ≈ coordinates(sim.grid, 1)[3]
    @test size(primitives(sim), 1) == Honghu.NHYDRO + 2
    gas_temperature(sim); conserved_totals(sim); field(sim, :density)
    @test sim.U == U0                                   # reading does not perturb

    mktempdir() do dir
        fn = write_vtk(sim, joinpath(dir, "s"))
        @test occursin("Name=\"mass_fraction_2\"", read(fn, String))

        cp = save_checkpoint(sim, joinpath(dir, "c"))
        sim.U .= 0
        load_checkpoint!(sim, cp)
        @test sim.U == U0
        other = Simulation(n=(8, 4, 1), nscalars=1)
        @test_throws ErrorException load_checkpoint!(other, cp)

        # a version-1 file (no variable counts) still loads into a scalar-free simulation
        plain = Simulation(n=(8, 4, 1))
        set_initial_conditions!((x, y, z) -> (rho=1.0, p=1.0), plain)
        v1 = joinpath(dir, "old.honghu")
        open(v1, "w") do f
            write(f, b"HONGHU\x00\x01", Int64(8))
            foreach(d -> write(f, Int64(plain.grid.n[d])), 1:3)
            foreach(d -> write(f, Int64(plain.grid.ng[d])), 1:3)
            write(f, 0.0, 0.0, Int64(0), Int64(0), plain.U)
        end
        Uplain = copy(plain.U); plain.U .= 0
        load_checkpoint!(plain, v1)
        @test plain.U == Uplain
        @test_throws ErrorException load_checkpoint!(sim, v1)
    end
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. -t 4 -e 'using Test, Honghu; const P = Honghu.Problems; include("test/test_scalars.jl")'`
Expected: FAIL (`field` does not know `:X_1`).

- [ ] **Step 3: Implement**

- `available_fields`: append `Symbol(:X_, s)` for each scalar.
- `field`: in the `_group_suffix` branch, handle `base === :X` with `1 <= s <= nscalars` (otherwise `ArgumentError`) and return `U[scalar_index(s), interior] ./ U[IRHO, interior]`.
- `output_fields`: push `("mass_fraction_$s", 1, ...)` after `gas_temperature`.
- Checkpoints:
  - Bump the magic to version 2 and write both row counts.
  - Accept v1 and Hydro3D files only when `sim.nscalars == 0`.
  - On a count mismatch, `error("checkpoint has $n hydro variables, the simulation has $m")`.
- `CLAUDE.md`, "Momentum index" invariant: add that hydro arrays now carry scalar rows, so internal ghost fills on them pass `momentum_blocks = 1`.

- [ ] **Step 4: Run the test, then the bit-for-bit baseline**

Run the command of Step 2. Expected: PASS.
Run: `julia --project=. -t 4 docs/superpowers/plans/baseline.jl | diff - docs/superpowers/plans/baseline.txt`
Expected: no output (identical).

- [ ] **Step 5: Full suite and commit**

Run: `julia --project=. -t 4 test/runtests.jl`. Expected: all pass.

In `CHANGELOG.md`, under `## [Unreleased]`, add a `### Added` line about mass scalars covering `Simulation(nscalars = n)`, the `X` initial-condition key, `:X_s` fields and checkpoint format 2.

```bash
git add src/output.jl src/plotting.jl test/test_scalars.jl CHANGELOG.md CLAUDE.md
git commit -m "feat: mass-scalar fields, VTK output and checkpoint format 2"
```

---

## Part B: photoionization chemistry

### Task 4: Network interface and the two integrators

**Files:**
- Create: `src/chemistry.jl`, `src/chemistry_solvers.jl`, `test/test_chemistry.jl`
- Modify: `src/Honghu.jl` (includes after `source_terms.jl`, the OrdinaryDiffEq import line, exports), `docs/make.jl` (`PAGES`), `test/runtests.jl` (include `test_chemistry.jl` after `test_coupling_adaptive.jl`), `CHANGELOG.md`

**Interfaces:**
- Produces, in `src/chemistry.jl`, all with docstrings:
  - `abstract type AbstractNetwork end`. A network implements `nspecies(net)::Int`, `nbands(net)::Int`, `species_mass(net, s)` (g), `band_group(net, b)::Int`, `band_photon_energy(net, b)` (erg), `chem_rhs!(du, u, net, cell)`, `chem_jacobian!(J, u, net, cell)` (exact, dense), and `chem_scales!(scale, u, net, cell)` (one absolute scale per unknown).
  - Methods with defaults: `network_temperature(net, n::AbstractVector, eint)` (reads `n[1:nspecies]`); `network_eint(net, n::AbstractVector, T)`; `chem_internal_energy(net, u)` (default `u[ienergy(net)]`); `initial_mass_fractions(net, nt::NamedTuple)` (default throws `ArgumentError` asking for `X`).
  - The layout of the unknowns is `u = [n_1 … n_NS, e, N_1 … N_NB]`, with `nunknowns(net) = nspecies + 1 + nbands`, `ienergy(net) = nspecies + 1` and `iband(net, b) = nspecies + 1 + b`. Number densities are in cm$^{-3}$, `e` is the internal energy density, and `N` is the photon number density.
  - `struct ChemCell{T}; rho::T; c::T; chat::T; dt::T; end`.
  - `abstract type AbstractChemSolver end`; `struct ChemStats; steps::Int; converged::Bool; end`.
  - `chem_workspace(solver, net, ::Type{T})` returns one workspace per thread-safe user.
  - `solve_chemistry!(work, solver, net, cell, u::Vector{T}) -> ChemStats`: on entry `u` is the initial state, on exit the answer.
- Produces, in `src/chemistry_solvers.jl`:
  - `RosenbrockChem(; alg = Rosenbrock23(), reltol = 1e-2, abstol = 1e-8)`. `abstol` is a fraction of `chem_scales!`.
  - `BackwardEulerChem(; tol = 1e-10, max_iterations = 50)`.
  - Exported: `AbstractNetwork, AbstractChemSolver, RosenbrockChem, BackwardEulerChem`.

- [ ] **Step 1: Write the failing test**

Create `test/test_chemistry.jl`:

```julia
# A -> B -> C, first order: an exact solution, three species, no energy, no bands.
struct ChainNet <: Honghu.AbstractNetwork
    k1::Float64
    k2::Float64
end
Honghu.nspecies(::ChainNet) = 3
Honghu.nbands(::ChainNet) = 0
Honghu.species_mass(::ChainNet, s) = 1.0
function Honghu.chem_rhs!(du, u, n::ChainNet, cell)
    du[1] = -n.k1 * u[1]; du[2] = n.k1 * u[1] - n.k2 * u[2]; du[3] = n.k2 * u[2]; du[4] = 0.0
    return nothing
end
function Honghu.chem_jacobian!(J, u, n::ChainNet, cell)
    fill!(J, 0.0); J[1, 1] = -n.k1; J[2, 1] = n.k1; J[2, 2] = -n.k2; J[3, 2] = n.k2
    return nothing
end
Honghu.chem_scales!(s, u, ::ChainNet, cell) = (fill!(s, 1.0); nothing)
chain_exact(t, k1, k2) = (exp(-k1 * t), k1 / (k2 - k1) * (exp(-k1 * t) - exp(-k2 * t)))

function be_bytes(w, s, net, cell, u)
    Honghu.solve_chemistry!(w, s, net, cell, u)
    @allocated Honghu.solve_chemistry!(w, s, net, cell, u)
end

@testset "chemistry: integrators on an exact network" begin
    net = ChainNet(1.0, 10.0)
    cell(dt) = Honghu.ChemCell(1.0, 1.0, 1.0, dt)
    u0 = [1.0, 0.0, 0.0, 0.0]
    @test Honghu.nunknowns(net) == 4
    @test Honghu.ienergy(net) == 4

    for (rtol, bound) in ((1e-2, 2e-2), (1e-6, 1e-5))
        s = RosenbrockChem(reltol=rtol, abstol=1e-12)
        w = Honghu.chem_workspace(s, net, Float64)
        @test w.integ.opts.abstol === w.abstol       # per-cell abstol is mutated in place
        u = copy(u0)
        st = Honghu.solve_chemistry!(w, s, net, cell(1.0), u)
        @test st.converged && st.steps > 0
        @test abs(u[1] - chain_exact(1.0, 1.0, 10.0)[1]) < bound
        @test abs(u[2] - chain_exact(1.0, 1.0, 10.0)[2]) < bound
        @test sum(u[1:3]) ≈ 1 atol = 1e-12
        v = copy(u0)                                  # a reused workspace gives the same answer
        Honghu.solve_chemistry!(w, s, net, cell(1.0), v)
        @test v == u
    end

    s = BackwardEulerChem()
    w = Honghu.chem_workspace(s, net, Float64)
    function be_error(n)
        u = copy(u0)
        for _ in 1:n
            @test Honghu.solve_chemistry!(w, s, net, cell(0.5 / n), u).converged
        end
        abs(u[2] - chain_exact(0.5, 1.0, 10.0)[2])
    end
    @test 1.8 < be_error(50) / be_error(100) < 2.2     # first order
    @test be_bytes(w, s, net, cell(0.01), copy(u0)) == 0
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. -t 4 -e 'using Test, Honghu; const P = Honghu.Problems; include("test/test_chemistry.jl")'`
Expected: FAIL (`AbstractNetwork` not defined).

- [ ] **Step 3: Implement `src/chemistry.jl`**

Write the interface exactly as listed under Interfaces. The header comment states the split, as `coupling.jl` does: physics in the network, numerics in the solver. It also lists the three things that are not ODE terms and live in the driver (emission, the flux rescaling by $N_{\rm end}/N_{\rm start}$, and positivity clamping of the written-back state).

- [ ] **Step 4: Implement `src/chemistry_solvers.jl`**

`RosenbrockChem`:
- `mutable struct ChemParams{N,T}; net::N; cell::ChemCell{T}; end`.
- `_chem_ode!(du, u, p, t) = chem_rhs!(du, u, p.net, p.cell)`, with `_chem_jac!` likewise and `_chem_tgrad!` filling zeros.
- `chem_workspace` returns `RosenbrockWork(integ, p, scale::Vector{T}, abstol::Vector{T})`. `integ` is `init(ODEProblem{true}(ODEFunction{true}(_chem_ode!; jac=_chem_jac!, tgrad=_chem_tgrad!), ones(T, n), (zero(T), one(T)), p), s.alg; reltol=T(s.reltol), abstol=work.abstol, save_everystep=false, dense=false)`.
- `solve_chemistry!` then:
  - sets `p.cell = cell` and calls `chem_scales!(w.scale, u, net, cell)`;
  - sets `w.abstol .= s.abstol .* w.scale`;
  - records `n0 = integ.stats.naccept + integ.stats.nreject` (the stats are cumulative across `reinit!`);
  - calls `reinit!(integ, u; t0 = 0, tf = cell.dt)` and `solve!(integ)`, then `u .= integ.u`;
  - returns `ChemStats(steps - n0, successful_retcode(integ.sol.retcode))`.

`BackwardEulerChem`:
- The workspace holds preallocated `u0, f, du, scale::Vector{T}` and `J::Matrix{T}`.
- Newton on $G(u)=u-u_0-\Delta t\,f(u)$ with matrix $I-\Delta t\,J$, solved by an in-place Gaussian elimination with partial pivoting, `_gauss_solve!(A, b)`. Do not use `lu!`, which allocates its pivot vector.
- After each update, clamp the species and band unknowns at zero.
- Stop when `maximum(abs(du[i]) / scale[i]) <= tol`.
- Return `ChemStats(iterations, converged)`.

- [ ] **Step 5: Wire up and run the test**

- `src/Honghu.jl`:
  - Change the import line to `using OrdinaryDiffEq: ODEProblem, ODEFunction, solve, successful_retcode, Rosenbrock23, init, solve!, reinit!`, and update its comment.
  - Add `include("chemistry.jl")` and `include("chemistry_solvers.jl")` after `include("source_terms.jl")`.
  - Export the four names.
- `docs/make.jl` `PAGES`: add `"chemistry" => ("Photoionization chemistry", ["chemistry.jl", "chemistry_solvers.jl"])` after `"coupling"`.

Run the command of Step 2. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/chemistry.jl src/chemistry_solvers.jl src/Honghu.jl docs/make.jl test/test_chemistry.jl test/runtests.jl CHANGELOG.md
git commit -m "feat: solver-agnostic chemistry network interface with Rosenbrock and backward-Euler integrators"
```

### Task 5: The hydrogen OTSA network

**Files:**
- Create: `src/chemistry_hydrogen.jl`
- Modify: `src/constants.jl` (add `eV = 1.602176634e-12`), `src/Honghu.jl` (include after `chemistry.jl`, export `HydrogenOTSA`), `docs/make.jl` (`PAGES` chemistry list), `test/test_chemistry.jl`, `CHANGELOG.md`

**Interfaces:**
- Consumes: Task 4 interface.
- Produces:
  - `HydrogenOTSA(; group = 1, sigma, photon_energy, heating_energy = photon_energy - CHI_H, alpha0 = 2.63e-13, alpha_exp = -0.7, collisional_ionization = true, cooling = true, isothermal = nothing, Tmin = 1.0)`.
  - Species order `(H⁰, H⁺)`, both of mass `CGS.m_H`; one band. Internal `const CHI_H = 13.6 * CGS.eV`.
  - `initial_mass_fractions(net::HydrogenOTSA, nt)` reads the key `x_HII` (default 0) and returns `(1 - x, x)`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_chemistry.jl`:

```julia
const CGS = Honghu.CGS
hstate(nH, x, T, N) = [nH * (1 - x), nH * x, 1.5 * CGS.k_B * nH * (1 + x) * T, N]
function fd_jacobian(net, u, cell)
    n = length(u); J = zeros(n, n); fp = zeros(n); fm = zeros(n)
    for j in 1:n
        h = 1e-6 * abs(u[j]); up = copy(u); um = copy(u); up[j] += h; um[j] -= h
        Honghu.chem_rhs!(fp, up, net, cell); Honghu.chem_rhs!(fm, um, net, cell)
        J[:, j] = (fp - fm) / 2h
    end
    J
end
invariants(u, phi, eps) = (u[1] + u[2], u[2] + u[4] / phi,
                           u[3] + Honghu.CHI_H * u[2] + eps * u[4] / phi)

@testset "chemistry: hydrogen network" begin
    c = CGS.c_light
    eps = 29.65 * CGS.eV
    net = HydrogenOTSA(sigma=1.63e-18, photon_energy=eps)
    cell = Honghu.ChemCell(1e-3 * CGS.m_H, c, 1e-3c, 1e12)

    # exact Jacobian, row by row, at three states (150 K, not 100 K: the
    # recombination cooling switches on at 100 K and a difference across it is noise)
    for u in (hstate(1e-3, 0.01, 150.0, 1e-4), hstate(1e-3, 0.5, 8e3, 1e-5),
              hstate(1e-3, 0.999, 2e4, 1e-6))
        J = zeros(4, 4); Honghu.chem_jacobian!(J, u, net, cell)
        Jfd = fd_jacobian(net, u, cell)
        for i in 1:4
            @test isapprox(J[i, :], Jfd[i, :]; rtol=1e-5)
        end
    end

    # photoionization alone conserves three linear invariants exactly
    pure = HydrogenOTSA(sigma=1.63e-18, photon_energy=eps, alpha0=0.0,
                        collisional_ionization=false, cooling=false)
    for s in (RosenbrockChem(), BackwardEulerChem())
        w = Honghu.chem_workspace(s, pure, Float64)
        u = hstate(1e-3, 0.01, 100.0, 1e-4); I0 = invariants(u, 1e-3, eps)
        @test Honghu.solve_chemistry!(w, s, pure, Honghu.ChemCell(1e-3 * CGS.m_H, c, 1e-3c, 1e13), u).converged
        @test all(isapprox.(invariants(u, 1e-3, eps), I0; rtol=1e-10))
        @test u[2] > 1e-5                                  # it did ionize
    end

    # a dark, fully neutral, cold cell does not move
    for s in (RosenbrockChem(), BackwardEulerChem())
        w = Honghu.chem_workspace(s, net, Float64)
        u = hstate(1e-3, 0.0, 100.0, 0.0); u0 = copy(u)
        @test Honghu.solve_chemistry!(w, s, net, cell, u).converged
        @test all(isfinite, u)
        @test u ≈ u0 rtol = 1e-12
    end

    # a photon density at the radiation floor stays non-negative
    for s in (RosenbrockChem(), BackwardEulerChem())
        w = Honghu.chem_workspace(s, net, Float64)
        u = hstate(1e-3, 1e-6, 100.0, 1e-40)
        Honghu.solve_chemistry!(w, s, net, cell, u)
        @test u[2] >= 0 && u[4] >= 0
    end

    # isothermal mode: e is not evolved; the energy written back follows n_tot
    iso = HydrogenOTSA(sigma=6.3e-18, photon_energy=13.6 * CGS.eV, isothermal=1e4)
    w = Honghu.chem_workspace(RosenbrockChem(), iso, Float64)
    u = hstate(1e-3, 0.01, 1e4, 1e-4); e0 = u[3]
    Honghu.solve_chemistry!(w, RosenbrockChem(), iso, cell, u)
    @test u[3] ≈ e0 rtol = 1e-12
    @test Honghu.chem_internal_energy(iso, u) ≈ 1.5 * CGS.k_B * (u[1] + 2u[2]) * 1e4

    # ionization equilibrium x^2/(1-x) = Gamma/(alpha n) = 0.5 gives x = 0.5. A tiny
    # chat keeps N fixed while the gas sees c sigma N.
    eq = HydrogenOTSA(sigma=6.3e-18, photon_energy=13.6 * CGS.eV, isothermal=1e4,
                      alpha0=2.59e-13, alpha_exp=0.0, collisional_ionization=false)
    N = 0.5 * 2.59e-13 * 1.0 / (c * 6.3e-18)
    ceq = Honghu.ChemCell(CGS.m_H, c, 1e-14c, 1e4 / 2.59e-13)
    for s in (RosenbrockChem(reltol=1e-6), BackwardEulerChem())
        w = Honghu.chem_workspace(s, eq, Float64)
        u = hstate(1.0, 0.01, 1e4, N)
        Honghu.solve_chemistry!(w, s, eq, ceq, u)
        @test u[2] / (u[1] + u[2]) ≈ 0.5 rtol = 1e-3
    end

    # the two integrators agree at a small step
    small = Honghu.ChemCell(1e-3 * CGS.m_H, c, 1e-3c, 1e9)
    ua = hstate(1e-3, 0.01, 100.0, 1e-4); ub = copy(ua)
    Honghu.solve_chemistry!(Honghu.chem_workspace(RosenbrockChem(reltol=1e-8), net, Float64),
                            RosenbrockChem(reltol=1e-8), net, small, ua)
    Honghu.solve_chemistry!(Honghu.chem_workspace(BackwardEulerChem(), net, Float64),
                            BackwardEulerChem(), net, small, ub)
    @test ua[2] ≈ ub[2] rtol = 1e-3
    @test ua[3] ≈ ub[3] rtol = 1e-3
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. -t 4 -e 'using Test, Honghu; const P = Honghu.Problems; include("test/test_chemistry.jl")'`
Expected: FAIL (`HydrogenOTSA` not defined).

- [ ] **Step 3: Implement `src/chemistry_hydrogen.jl`**

The unknowns are `u = [n_H0, n_HII, e, N]`. Use clamped values $a^+=\max(a,0)$ for $n_{\rm H^0}$, $n_{\rm H^+}$ and $N$ inside every rate, as Quokka clamps $N$, and zero the Jacobian column of a clamped variable. Then:

- $n_e = n_{\rm H^+}^+$.
- $T = T_{\rm iso}$ if isothermal, else $\max\big(e / (1.5 k_B (n_{\rm H^0}^+ + 2 n_{\rm H^+}^+)), T_{\min}\big)$. The derivative through the `Tmin` clamp is zero when it is active.
- $\Gamma = c\,\sigma\,N^+$ (`cell.c`, the gas side).
- $\alpha_B = \alpha_0 (T/10^4)^{\alpha_{\rm exp}}$.
- $\beta = 5.84\times10^{-11}\sqrt T\,\exp(-2.18\times10^{-11}/k_B T)$ if `collisional_ionization`, else 0. This is Krumholz et al. (2007), as in Quokka's `get_collisional_ionization_coefficient`.
- ${\rm ion} = \Gamma n_{\rm H^0}^+ + \beta n_e n_{\rm H^0}^+$ and ${\rm rec} = \alpha_B n_e n_{\rm H^+}^+$.
- `du[1] = rec - ion`, `du[2] = ion - rec`.
- `du[3] = 0` if isothermal. Otherwise `du[3]` is ε_heat·Γ·n_H0⁺ − CHI_H·β·n_e·n_H0⁺, minus three further terms only when `cooling` is on:
  - $\Lambda_{\rm rec}\,n_e n_{\rm H^+}^+$, with $\Lambda_{\rm rec}=6.1\times10^{-10}k_B T^{0.11}$ for $T\geq100$ K and 0 below (Quokka);
  - $\Lambda_{\rm ff}\,n_e n_{\rm H^+}^+$, with $\Lambda_{\rm ff}=1.3\times1.427\times10^{-27}\sqrt T$. This is only the free-free part of Quokka's `get_ion_ff_cooling_coefficient`, without `get_cle_term`;
  - $\Lambda_{\rm ce}\,n_e n_{\rm H^0}^+$, with $\Lambda_{\rm ce}=7.5\times10^{-19}\exp(-118348/T)/(1+\sqrt{T/10^5})$ (Cen 1992; check the coefficient against the paper and cite it in the docstring).

  The CHI_H·β term belongs to collisional ionization, not to the `cooling` switch: it is the energy that process costs.
- `du[4] = -cell.chat * sigma * n_H0⁺ * N⁺`.

The other required methods:
- `chem_scales!`: species `n_H = u[1] + u[2]`; energy `max(abs(u[3]), 1.5 k_B n_H Tmin)`; photons `(cell.chat / cell.c) * n_H`.
- `network_eint(net, n, T) = 1.5 k_B (n[1] + 2n[2]) T`.
- `chem_internal_energy`: isothermal gives `network_eint(net, u, Tiso)`, otherwise `u[3]`.

Docstring: cite Aubert & Teyssier (2008) and the Quokka file, list every rate with its source, and say that the default `heating_energy` overheats ionized gas compared with the optically thin value (spec 4.3).

- [ ] **Step 4: Run the tests to verify they pass**

Run the command of Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/chemistry_hydrogen.jl src/constants.jl src/Honghu.jl docs/make.jl test/test_chemistry.jl CHANGELOG.md
git commit -m "feat: hydrogen OTSA photoionization network (Aubert & Teyssier 2008)"
```

### Task 6: The chemistry step inside a simulation

**Files:**
- Create: `src/chemistry_step.jl`
- Modify: `src/simulation.jl` (struct, constructor, `advance_radiation!`, `Diagnostics`, `_print_summary`, `set_initial_conditions!`, `gas_temperature`), `src/Honghu.jl` (include after `chemistry_hydrogen.jl`, export `Photochemistry`), `docs/make.jl`, `test/test_chemistry.jl`, `test/test_performance.jl`, `CHANGELOG.md`

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces:
  - `Photochemistry(network::AbstractNetwork; solver::AbstractChemSolver = RosenbrockChem(), emission = nothing)`. `emission(x, y, z, t)` gives photons cm$^{-3}$ s$^{-1}$ into band 1.
  - `Simulation(; ..., chemistry = nothing)`, with fields `const chemistry::CH` and `const cw::CW` (a `Vector` of per-row workspaces, one per interior `(j, k)` row, built once).
  - `chemistry_step!(sim, dt::T, t::T) -> ChemStepStats` and `struct ChemStepStats; cells; steps; max_steps; failures; end` with `+`.
  - `Diagnostics` fields `chem_cells, chem_steps, chem_max, chem_failures`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_chemistry.jl`:

```julia
function chem_box(; nH=1e-3, phi=1e-2, side=1e21, n=4, net, emission=nothing, hydro=false,
                  solver=RosenbrockChem(), ic)
    rad = RadiationPhysics(c=CGS.c_light, chat=phi * CGS.c_light, a_rad=CGS.a_rad,
                           k_B=CGS.k_B, opacity=ConstantOpacity(0.0), beta_order=0)
    sim = Simulation(n=(n, n, n), xhi=(side, side, side), hydro=hydro, radiation=rad,
                     boundaries=Boundaries(Periodic()), eos=IdealGas(cv=cv_from_mu(1.0, 5 / 3)),
                     chemistry=Photochemistry(net; solver, emission))
    set_initial_conditions!(ic, sim)
    sim
end
cellsum(sim, f) = sum(f(i, j, k) for k in sim.grid.lo[3]:sim.grid.hi[3],
                      j in sim.grid.lo[2]:sim.grid.hi[2], i in sim.grid.lo[1]:sim.grid.hi[1])

@testset "photochemistry in a simulation" begin
    eps = 29.65 * CGS.eV
    net = HydrogenOTSA(sigma=1.63e-18, photon_energy=eps)
    rad(; kw...) = RadiationPhysics(; c=CGS.c_light, a_rad=CGS.a_rad, k_B=CGS.k_B,
                                    opacity=ConstantOpacity(0.0), beta_order=0, kw...)
    pc = Photochemistry(net)
    @test_throws ArgumentError Simulation(n=(4, 1, 1), chemistry=pc)                    # no radiation
    @test_throws ArgumentError Simulation(n=(4, 1, 1), radiation=rad(chat_limiter=StabilityBound()), chemistry=pc)
    @test_throws ArgumentError Simulation(n=(4, 1, 1), radiation=rad(beta_order=1), chemistry=pc)
    @test_throws ArgumentError Simulation(n=(4, 1, 1), chemistry=pc, nscalars=3, radiation=rad())
    @test_throws ArgumentError Simulation(n=(4, 1, 1), chemistry=pc,
        radiation=RadiationPhysics(c=CGS.c_light, a_rad=CGS.a_rad, opacity=ConstantOpacity(1.0), beta_order=0))

    # photon budget and energy invariant on a periodic box with uniform emission
    pure = HydrogenOTSA(sigma=1.63e-18, photon_energy=eps, alpha0=0.0,
                        collisional_ionization=false, cooling=false)
    q, phi = 1e-17, 1e-2
    sim = chem_box(net=pure, phi=phi, emission=(x, y, z, t) -> q,
                   ic=(x, y, z) -> (rho=1e-3 * CGS.m_H, T=100.0, x_HII=0.01, Erad=0.0))
    U, R, mH = sim.U, sim.R, CGS.m_H
    b0 = cellsum(sim, (i, j, k) -> U[7, i, j, k] / mH + R[1, i, j, k] / eps / phi)
    e0 = cellsum(sim, (i, j, k) -> U[5, i, j, k] + Honghu.CHI_H * U[7, i, j, k] / mH + R[1, i, j, k] / phi)
    run!(sim, tmax=1e20, maxsteps=5, verbose=false)
    ncell = prod(sim.grid.n)
    b1 = cellsum(sim, (i, j, k) -> U[7, i, j, k] / mH + R[1, i, j, k] / eps / phi)
    e1 = cellsum(sim, (i, j, k) -> U[5, i, j, k] + Honghu.CHI_H * U[7, i, j, k] / mH + R[1, i, j, k] / phi)
    @test b1 - b0 ≈ q * sim.t * ncell rtol = 1e-8
    @test e1 - e0 ≈ eps * q * sim.t * ncell rtol = 1e-8
    @test sim.diag.chem_cells > 0 && sim.diag.chem_failures == 0

    # the equilibrium does not depend on chat: q = alpha x^2 n^2 gives x = 0.5
    iso = HydrogenOTSA(sigma=6.3e-18, photon_energy=13.6 * CGS.eV, isothermal=1e4,
                       alpha0=2.59e-13, alpha_exp=0.0, collisional_ionization=false)
    trec = 1 / 2.59e-13
    xs = map((1e-2, 1e-3)) do f
        s = chem_box(net=iso, nH=1.0, phi=f, side=2.3e21, emission=(x, y, z, t) -> 0.25 * 2.59e-13,
                     ic=(x, y, z) -> (rho=CGS.m_H, T=1e4, x_HII=0.01, Erad=0.0))
        run!(s, tmax=10trec, verbose=false)
        i = s.grid.lo[1]
        (s.U[7, i, i, i] / s.U[1, i, i, i], s.R[1, i, i, i])
    end
    @test xs[1][1] ≈ 0.5 rtol = 1e-3
    @test xs[2][1] ≈ 0.5 rtol = 1e-3
    @test xs[1][2] ≈ xs[2][2] rtol = 1e-3                    # same physical photon density

    # hydro on: hydrogen mass conserved, X_H0 + X_HII = 1 per cell
    hs = chem_box(net=net, n=8, side=8e18, hydro=true,
                  ic=(x, y, z) -> (rho=1e-3 * CGS.m_H * (1 + 0.5sinpi(2x / 8e18)), vx=1e6,
                                   T=100.0, x_HII=0.5 + 0.4sinpi(2x / 8e18), Erad=0.0))
    m0 = sum(conserved_totals(hs).scalars)
    run!(hs, tmax=1e20, maxsteps=10, verbose=false)
    @test sum(conserved_totals(hs).scalars) ≈ m0 rtol = 1e-12
    ir, jr, kr = hs.grid.lo[1]:hs.grid.hi[1], hs.grid.lo[2]:hs.grid.hi[2], hs.grid.lo[3]:hs.grid.hi[3]
    @test maximum(abs.((hs.U[6, ir, jr, kr] .+ hs.U[7, ir, jr, kr]) ./ hs.U[1, ir, jr, kr] .- 1)) < 1e-12

    # reading does not perturb, and the temperature is the network's
    U0 = copy(hs.U)
    T = gas_temperature(hs); field(hs, :temperature); field(hs, :X_2); primitives(hs)
    @test hs.U == U0
    i = hs.grid.lo[1]
    n = [hs.U[6, i, i, i], hs.U[7, i, i, i]] ./ CGS.m_H
    @test T[1, 1, 1] ≈ Honghu.network_temperature(net, n, hs.U[5, i, i, i] -
                       hs.U[2, i, i, i]^2 / (2hs.U[1, i, i, i]))
end
```

Add to `test/test_performance.jl`:

```julia
@testset "chemistry step does not allocate per cell (backward Euler)" begin
    net = HydrogenOTSA(sigma=1.63e-18, photon_energy=29.65 * Honghu.CGS.eV)
    rad = RadiationPhysics(c=Honghu.CGS.c_light, chat=1e-3 * Honghu.CGS.c_light,
                           a_rad=Honghu.CGS.a_rad, k_B=Honghu.CGS.k_B,
                           opacity=ConstantOpacity(0.0), beta_order=0)
    sim = Simulation(n=(16, 16, 16), xhi=(1e21, 1e21, 1e21), hydro=false, radiation=rad,
                     chemistry=Photochemistry(net; solver=BackwardEulerChem()))
    set_initial_conditions!((x, y, z) -> (rho=1e-3 * Honghu.CGS.m_H, T=100.0, x_HII=0.01,
                                          Erad=1e-12), sim)
    @test bytes_per_cell(() -> Honghu.chemistry_step!(sim, 1e10, 0.0), sim) < 8.0
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. -t 4 -e 'using Test, Honghu; const P = Honghu.Problems; include("test/test_chemistry.jl")'`
Expected: FAIL (`Photochemistry` not defined).

- [ ] **Step 3: Implement `src/chemistry_step.jl` and the simulation hooks**

`chemistry_step!(sim, dt, t)`: use `threaded_jk` over the interior rows, with workspace `sim.cw[(j - g.lo[2]) + (k - g.lo[3]) * g.n[2] + 1]`. Each row workspace also holds a `u::Vector{T}` scratch buffer. For each cell:
1. Read $\rho$; $n_s = U[{\rm scalar\_index}(s)]/m_s$; $e = U[{\rm IEGAS}] - E_{\rm kin}$.
2. For each band, with group `gr = band_group(net, b)`: $\phi_b = $ `sim.rw.chat[gr, i, j, k] / c`, and $N_{\rm start} = R[{\rm gbase}(gr)+{\rm IERAD}]/\varepsilon_b$. For band 1, add $\phi_1 \cdot$`emission(x, y, z, t)`$\cdot\Delta t$ when there is an emission function.
3. Set `cell = ChemCell(rho, c, chat_of_group_1, dt)` (or `phys.chat` when `nbands == 0`), pack `u`, call `solve_chemistry!` and accumulate the stats.
4. Write back:
   - `U[scalar_index(s)] = max(u[s], 0) * m_s`;
   - `U[IEGAS] += chem_internal_energy(net, u) - e`;
   - per band, $N_{\rm end}=\max(u_{\rm band},0)$, `R[E] = N_end * eps_b`, and multiply the three flux components by `N_start > 0 ? N_end / N_start : 0`.

Simulation hooks:
- Constructor with `chemistry`:
  - `nscalars` must be 0 or `nspecies` (it is then set to `nspecies`);
  - throw `ArgumentError` when radiation is missing, when the limiter is not `ConstantChat`, when `beta_order != 0`, when a band group is out of range, or when `group_opacities(radiation.opacity, radiation.groups, one(T), T(1e4))[band_group] != 0`;
  - build `cw`.
- `advance_radiation!`: after `stats += imex_substep!(sim, dts, tt)` and before `tt += dts`, when there is chemistry, record `chemistry_step!(sim, dts, tt)` into `Diagnostics`, then add `amend_rad_state!(sim.R, sim.radiation, sim.grid, sim.floors)` to `diag.repairs_postsolve`.
- `set_initial_conditions!`:
  - with chemistry and no `X` key, use `initial_mass_fractions(net, s)`;
  - with a `T` key, compute `eint = network_eint(net, n, T)` and `p = pressure(eos, rho, eint)`;
  - say in the docstring that `Erad` defaults to $a_r T^4$, which is wrong for photoionization problems, so they must pass `Erad`.
- `gas_temperature` with chemistry uses `network_temperature`.
- `_print_summary` prints the mean and worst chemistry steps per cell, and a `WARNING` line when `chem_failures > 0`.

- [ ] **Step 4: Run the tests to verify they pass**

Run the command of Step 2 and the `test_performance.jl` command of Task 2 Step 4. Expected: PASS. If the photon-budget test is off, check first that the grey coupling with `ConstantOpacity(0.0)` leaves `E` unchanged.

- [ ] **Step 5: Baseline and full suite**

Run: `julia --project=. -t 4 docs/superpowers/plans/baseline.jl | diff - docs/superpowers/plans/baseline.txt`, expecting no output. Then `julia --project=. -t 4 test/runtests.jl`, expecting all pass.

- [ ] **Step 6: Commit**

```bash
git add src/chemistry_step.jl src/simulation.jl src/Honghu.jl docs/make.jl test/test_chemistry.jl test/test_performance.jl CHANGELOG.md
git commit -m "feat: photochemistry step in the radiation substep"
```

### Task 7: Iliev Tests 1 and 2

**Files:**
- Create: `src/problems/stromgren.jl`, `test/test_stromgren.jl`
- Modify: `src/problems/Problems.jl` (include and exports), `test/runtests.jl` (include `test_stromgren.jl` after `test_chemistry.jl`), `CHANGELOG.md`

**Interfaces:**
- Consumes: `Photochemistry`, `HydrogenOTSA`, `RosenbrockChem`.
- Produces, all exported from `Honghu.Problems`:
  - `stromgren_iliev(; test::Integer = 1, n::Integer = 32, centered::Bool = false, chat_factor = 1e-3, solver = RosenbrockChem(), kwargs...) -> (sim, tmax)`, with `tmax` = 500 Myr;
  - `ionized_radius(sim; centered = false, x0 = 0.0)`;
  - `stromgren_radius_analytic(t; Ndot = 5e48, n = 1e-3, alpha_B = 2.59e-13)`;
  - `stromgren_radius_finite_c(t; chat, Ndot = 5e48, n = 1e-3, alpha_B = 2.59e-13)`.

- [ ] **Step 1: Write the failing tests**

```julia
const PC = Honghu.CGS
_median(v) = (s = sort(vec(v)); s[cld(length(s), 2)])

@testset "Stromgren sphere: Iliev Test 1" begin
    trec = 1 / (2.59e-13 * 1e-3)
    @test P.stromgren_radius_finite_c(2trec; chat=PC.c_light) ≈ P.stromgren_radius_analytic(2trec) rtol = 1e-3
    @test P.stromgren_radius_analytic(500e6 * PC.year) / (1e3 * PC.parsec) ≈ 5.36 rtol = 2e-3

    sim, _ = P.stromgren_iliev(test=1, n=16, chat_factor=3e-4)
    for tt in (0.5trec, 1.0trec, 2.0trec)
        run!(sim, tmax=tt, verbose=false)
        r = P.ionized_radius(sim; x0=1.2e-3)
        @test abs(r / P.stromgren_radius_finite_c(tt; chat=3e-4 * PC.c_light) - 1) < 0.10
    end
    @test sim.diag.chem_failures == 0

    # behind the front the neutral fraction does not depend on chat
    xHI = map((3e-4, 1e-3)) do f
        s, _ = P.stromgren_iliev(test=1, n=16, chat_factor=f)
        run!(s, tmax=0.5trec, verbose=false)
        field(s, :X_1)[1]
    end
    inside = (xHI[1] .< 0.1) .& (xHI[2] .< 0.1)
    @test count(inside) > 20
    @test abs(_median(xHI[1][inside] ./ xHI[2][inside]) - 1) < 0.1
end

@testset "Stromgren sphere: Iliev Test 2 (qualitative)" begin
    trec = 1 / (2.59e-13 * 1e-3)
    sim, _ = P.stromgren_iliev(test=2, n=16, chat_factor=3e-4)
    rprev = 0.0
    for tt in (0.25trec, 0.5trec, 1.0trec)
        run!(sim, tmax=tt, verbose=false)
        r = P.ionized_radius(sim)
        @test r > rprev
        rprev = r
    end
    T = gas_temperature(sim)
    x = field(sim, :X_2)[1]
    @test _median(T[x .> 0.9]) > 1e4                  # hot behind the front
    @test _median(T[x .< 0.01]) < 2e3                 # cold ahead of it
    @test sim.diag.chem_failures == 0
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `julia --project=. -t 4 -e 'using Test, Honghu; const P = Honghu.Problems; include("test/test_stromgren.jl")'`
Expected: FAIL (`stromgren_iliev` not defined).

- [ ] **Step 3: Implement `src/problems/stromgren.jl`**

Constants (Iliev et al. 2006, checked against the paper): $L = 6.6$ kpc ($6.6\times10^3$ `CGS.parsec`), $n_{\rm H} = 10^{-3}$, $\dot N = 5\times10^{48}$ s$^{-1}$.

| | Test 1 | Test 2 |
|:---|:---|:---|
| network | `isothermal = 1e4`, `alpha0 = 2.59e-13`, `alpha_exp = 0`, `collisional_ionization = false` (my call: keeps the background at $x_0$), `sigma = 6.3e-18`, `photon_energy = 13.6 eV`, `heating_energy = 0` | defaults, `sigma = 1.63e-18`, `photon_energy = 29.65 eV` |
| initial state | $T = 10^4$ K, $x_0 = 1.2\times10^{-3}$ | $T = 100$ K, $x_0 = 0$ |

Setup:
- Radiation: `RadiationPhysics(c=CGS.c_light, chat=chat_factor*CGS.c_light, a_rad=CGS.a_rad, k_B=CGS.k_B, opacity=ConstantOpacity(0.0), beta_order=0)`, one group.
- Simulation: `hydro=false`, `eos=IdealGas(gamma=5/3, cv=cv_from_mu(1.0, 5/3))`.
- Initial condition: `rho = n_H * CGS.m_H`, `T`, `x_HII`, and `Erad = 0.0`. This is required: the default $a_r T^4$ would be about $10^{12}$ photons cm$^{-3}$.
- Geometry:
  - octant: `xlo = (0,0,0)`, `xhi = (L,L,L)`, `Boundaries((Reflecting(), Outflow()), (Reflecting(), Outflow()), (Reflecting(), Outflow()))` for both systems;
  - `centered = true`: `xlo = (-L,-L,-L)`, `xhi = (L,L,L)`, `Outflow()` everywhere.
- Emission: a Gaussian at the origin of width $s = 1.5\Delta x$, truncated at $r < 2.5s$, normalised so that the sum over interior cells of rate × cell volume is exactly $\dot N/8$ (octant) or $\dot N$ (centred). Precompute the normaliser at construction.

The helper functions:
- `ionized_radius`: $V = \sum \max(x - x_0, 0)/(1 - x_0)\,dV$ with $x = U[{\rm scalar\_index}(2)]/U[{\rm IRHO}]$; $V_{\rm full} = 8V$ for the octant; $r = (3V_{\rm full}/4\pi)^{1/3}$.
- `stromgren_radius_finite_c`: RK4 in $R$ from 0 on $dR/dt = (\dot N - \tfrac{4\pi}{3}R^3\alpha_B n^2)/(\dot N/\hat c + 4\pi R^2 n)$ (Shapiro et al. 2006). Start with 256 steps and double until the relative change is below $10^{-6}$, as Quokka's `integrate_radius` does.

Docstrings cite Iliev et al. (2006) and say which numbers are from the paper and which are choices.

- [ ] **Step 4: Run the tests, then tighten**

Run the command of Step 2. Expected: PASS.
Then record the measured worst relative radius error of Test 1 in a comment above the assertion, and set the bound to about 1.5 times that value, but not below 0.02. Note the wall time of the file; if it exceeds 30 s with 4 threads, reduce `n` to 12 and say so in the comment.

- [ ] **Step 5: Commit**

```bash
git add src/problems/stromgren.jl src/problems/Problems.jl test/test_stromgren.jl test/runtests.jl CHANGELOG.md
git commit -m "feat: Iliev et al. (2006) Stromgren Tests 1 and 2"
```

### Task 8: Cost gate, example, solver comparison and documentation

**Files:**
- Create: `docs/photoionization-experiments/measure.jl`, `docs/photoionization-findings.md`, `examples/stromgren_sphere.jl`
- Modify: `docs/make.jl` (`NOTES`), `CLAUDE.md` (source-layout rows, and a short "Photoionization chemistry" section), `CHANGELOG.md`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the measurement script and run the cost gate**

`docs/photoionization-experiments/measure.jl` prints and writes `docs/photoionization-experiments/measure.txt` with three parts:

1. **Microseconds per cell of `chemistry_step!`.** Measure `RosenbrockChem()` and `BackwardEulerChem()` on a 32³ octant of each Iliev test. Run 0.25 $t_{\rm rec}$ first so that the front is in the box, then take the median of 5 calls on 1 thread and on 8 threads.
2. **Integrator comparison.** Run Test 2 at `n = 16` and `chat_factor = 3e-4` to 100 and 500 Myr with each integrator, and record `ionized_radius`, the spherically averaged $T$ and $x_{\rm HI}$ in 16 radial bins, and the wall time.
3. **$\hat c$ sweep.** Run Test 1 at `n = 16` with `chat_factor ∈ (1e-4, 3e-4, 1e-3, 3e-3)`, and record `ionized_radius` at 10, 30, 100 and 500 Myr against `stromgren_radius_finite_c` and `stromgren_radius_analytic`.

Run: `julia --project=. -t 8 docs/photoionization-experiments/measure.jl`

Gate: if `RosenbrockChem` costs more than 20 µs per cell on 1 thread, the centred 64³ example at `chat_factor = 3e-4` over 500 Myr would exceed about 30 minutes on 8 threads. If so, **stop here and report the numbers to the user** (spec section 5). Do not build a fast path or an in-tree integrator.

- [ ] **Step 2: Write the example**

`examples/stromgren_sphere.jl` follows `examples/common.jl` (include it, use `PLOTTING`, `outpath`, `saveplot`, `plotting_note`).
- Positional arguments in `OPTS`: test (`1` or `2`, default 1), `n` (default 64), and the word `octant` to use the corner geometry. The default is centred, as the author asked, with `chat_factor = 3e-4`.
- It runs to 500 Myr, printing the radius at Iliev's output times (10, 30, 100, 200, 500 Myr).
- It writes `stromgren_test<k>_radius.txt` with columns $t$, $r_I$, the finite-$\hat c$ reference and the analytic reference.
- Figures: `plotpanels(sim; vars = [:X_1, :temperature])` (the octant uses `slice = 1`), and a line plot of $r_I(t)$ against both references.

Run: `julia --project=. -t 8 examples/stromgren_sphere.jl 1 32 octant --no-plot --outdir=/tmp/stromgren`
Expected: finishes; the radius file has 5 rows; no chemistry failures in the summary.

- [ ] **Step 3: Write `docs/photoionization-findings.md`**

Markdown style per Global Constraints. Sections:
- What was measured, with the commands.
- The cost table from Step 1.1.
- The integrator comparison from Step 1.2. State how far backward Euler is from the Rosenbrock reference in $r_I$ and $T$, and at what cost.
- The $\hat c$ sweep from Step 1.3: the smallest $\hat c$ at which $r_I$ follows the finite-$\hat c$ reference, against the author's $\hat c > 3c_s$ expectation (spec 4.4).
- The expected Test 2 overheating (spec 4.3) and how large it came out.

Point to the measurement script and its output file rather than repeating raw numbers twice. Add `"photoionization-findings.md"` to `NOTES` in `docs/make.jl`.

- [ ] **Step 4: Update CLAUDE.md and the changelog**

`CLAUDE.md`:
- Add rows to the Source-layout table for the four chemistry files and `problems/stromgren.jl`, keeping the include order.
- Add a short "Photoionization chemistry" section with the five non-obvious points:
  - the reduced-speed-of-light convention (gas $c$, radiation $\hat c$, emission $\times\phi$, budget $n_{\rm H^+}+N/\phi$);
  - `ConstantChat` and `beta_order = 0` only;
  - the flux is scaled by $N_{\rm end}/N_{\rm start}$ instead of carried in the ODE;
  - `Erad` must be set explicitly in photoionization initial conditions;
  - the default heating per ionization overheats compared with the optically thin value.

`CHANGELOG.md`: `### Added` lines for the photochemistry, the hydrogen network, the two integrators, the Iliev problems and the example.

- [ ] **Step 5: Full suite and commit**

Run: `julia --project=. -t 4 test/runtests.jl`
Expected: all pass, in no more than about 30 s beyond the time measured before Task 1.

```bash
git add docs/photoionization-experiments/measure.jl docs/photoionization-experiments/measure.txt docs/photoionization-findings.md examples/stromgren_sphere.jl docs/make.jl CLAUDE.md CHANGELOG.md
git commit -m "docs: Stromgren example, integrator comparison and photoionization findings"
```
