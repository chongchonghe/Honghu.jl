# Hydro3D.jl

A three-dimensional radiation-hydrodynamics code in pure Julia, with no dependencies outside the standard library.

Hydro3D solves the Euler equations coupled to the grey two-moment (M1) radiation transport equations on a uniform Cartesian grid. It is a rewrite of [Hydro.jl](../Hydro.jl) following the algorithms of [Quokka](https://github.com/quokka-astro/quokka) (Wibking & Krumholz 2022; He, Wibking & Krumholz 2024).

## What it solves

**Hydrodynamics.** The compressible Euler equations,

$$\partial_t \rho + \nabla\cdot(\rho\mathbf{v}) = 0, \qquad \partial_t (\rho\mathbf{v}) + \nabla\cdot(\rho\mathbf{v}\mathbf{v} + P\mathbb{I}) = \mathbf{f}_\mathrm{rad}, \qquad \partial_t E + \nabla\cdot[(E+P)\mathbf{v}] = \dot{e}_\mathrm{rad},$$

with an ideal-gas equation of state.

**Radiation.** The first two angular moments of the radiative transfer equation, closed with the Levermore (1984) M1 closure,

$$\partial_t E_r + \frac{\hat{c}}{c}\nabla\cdot\mathbf{F}_r = S, \qquad \partial_t \mathbf{F}_r + \hat{c}\,c\,\nabla\cdot(\mathbb{T}E_r) = \mathbf{S}_F,$$

where $\mathbb{T}(\mathbf{f})$ is the Eddington tensor built from the reduced flux $\mathbf{f} = \mathbf{F}_r/(cE_r)$. M1 interpolates between the correct free-streaming limit ($\mathbb{T}\to\hat{n}\hat{n}$, radiation travels in straight lines and casts shadows) and the diffusion limit ($\mathbb{T}\to\mathbb{I}/3$), which is what distinguishes it from flux-limited diffusion.

## Numerical method

Space and time are separated by the **method of lines**: a spatial operator $L(U) = -(F_{i+1/2}-F_{i-1/2})/\Delta x$ is assembled once per stage, and the stages are pure ODE updates.

| Component | Choice |
|:---|:---|
| Reconstruction | piecewise constant (`PCM`), piecewise linear (`PLM`, four limiters), piecewise parabolic (`PPM`) |
| Hydro Riemann solver | `HLLC` with Batten wave speeds, Rider's nonlinear correction, and the Minoshima & Miyoshi (2021) carbuncle cure and low-dissipation pressure |
| Radiation Riemann solver | `HLL` on the M1 system with frozen-Eddington-tensor wave speeds (Skinner & Ostriker 2013; Balsara 1999) |
| Hydro time integration | SSP Runge-Kutta, first to third order (`ForwardEuler`, `SSPRK2`, `SSPRK3`) |
| Radiation time integration | **IMEX PD-ARS**: transport explicit, stiff matter coupling implicit; second order and asymptotic-preserving |
| Matter-radiation coupling | 2×2 Newton-Raphson per cell in $(E_\mathrm{int}, R)$, with an analytic flux/momentum update and a lagged outer iteration for the $\mathbf{v}\cdot\mathbf{F}$ work term |

Radiation subcycles inside each hydro step at the reduced speed of light $\hat{c}$. A degenerate direction (`n[d] == 1`) carries no ghost cells and is skipped entirely, so 1D and 2D problems cost what they should.

## Install and run

Requires Julia 1.9 or newer. There is nothing to install — the package depends only on `Printf`.

```julia
using Pkg; Pkg.develop(path = "hydro3d.jl")
using Hydro3D
```

A shock tube in five lines:

```julia
sim = Simulation(n = (256, 1, 1), eos = IdealGas(gamma = 1.4),
                 boundaries = Boundaries(Outflow()))
set_initial_conditions!(sim) do x, y, z
    x < 0.5 ? (rho = 1.0, p = 1.0) : (rho = 0.125, p = 0.1)
end
run!(sim, tmax = 0.2)
```

A 3D radiation-hydrodynamics run, writing snapshots for ParaView:

```julia
rad = RadiationPhysics(c = 2.998e10, chat = 2.998e8, a_rad = 7.5657e-15,
                       opacity = ConstantOpacity(0.4))
sim = Simulation(n = (128, 128, 128), xhi = (1e15, 1e15, 1e15),
                 eos = IdealGas(gamma = 5/3, cv = cv_from_mu(0.6, 5/3)),
                 radiation = rad, boundaries = Boundaries(Outflow()))
set_initial_conditions!(sim) do x, y, z
    (rho = 1e-20, T = 10.0, Erad = 7.5657e-15 * 10.0^4)
end
run!(sim, tmax = 1e5, outdir = "out", dtout = 1e4)
```

Run with threads: `julia -t auto script.jl`.

## Built-in problems

Each returns `(sim, tmax)`, ready to `run!`:

```julia
sim, tmax = Hydro3D.Problems.kelvin_helmholtz(n = 512)
run!(sim; tmax, outdir = "kh", dtout = 0.02)
```

| Problem | Tests |
|:---|:---|
| `sod_shock_tube` | shocks, contacts, rarefactions against `sod_exact` |
| `smooth_advection` | order of accuracy of the reconstruction |
| `kelvin_helmholtz` | 2D shear instability, exact conservation |
| `sedov_blast` | 3D strong explosion, spherical symmetry on a Cartesian grid |
| `radiation_streaming` | free-streaming limit: the front must move at exactly $\hat{c}$ |
| `radiation_diffusion` | static diffusion limit against an analytic Gaussian |
| `matter_coupling` | stiff energy exchange, equilibrium and exact energy conservation |
| `marshak_wave` | non-equilibrium radiation-driven heat wave |
| `radiation_shadow` | M1 casting a sharp shadow behind an opaque clump |

See `examples/` for scripts that run and post-process these.

## Verification

The test suite (`julia --project=. -t 4 test/runtests.jl`, 374 tests, ~10 s) checks, among other things:

- **Order of accuracy** on smooth advection: PCM 1, PLM 2, PPM 2.3. PPM stops short of 3 because the Colella-Woodward monotonicity constraint drops to a limited line at the two extrema of the sine wave; this matches the same limiter in Athena++ and Quokka.
- **Toro's five shock tubes**, including the "123" near-vacuum problem and both strong blasts, all staying positive with no help from the floors.
- **Conservation**: mass, momentum and total energy to $10^{-14}$ relative on periodic boundaries.
- **Directional isotropy**: the same problem run along $x$, $y$ and $z$ gives bitwise-comparable profiles.
- **Free streaming**: the radiation front arrives at $\hat{c}t$ to 2%, with $|\mathbf{F}| \le cE$ everywhere.
- **Static diffusion**: relative $L_1$ error against the analytic Gaussian of 0.6%, 0.13%, 0.06% at 64, 128, 256 cells — the scheme is asymptotic-preserving, so the answer does not depend on how many mean free paths a cell spans.
- **Stiff coupling**: a single step $10^{5}$ times longer than the coupling time still lands on radiative equilibrium, conserving $E_\mathrm{int} + (c/\hat{c})E_r$ to $10^{-13}$.
- **Radiation force**: momentum transfer matches $\Delta p = (\rho\kappa/c)F_0(1-e^{-\hat{c}\rho\kappa t})/(\hat{c}\rho\kappa)$ to 1%, converging at second order.

## Performance

Apple M-series, Julia 1.12, `Float64`, $64^3$ cells:

| | 1 thread | 8 threads |
|:---|---:|---:|
| Hydrodynamics | 2.3 Mcell/s | 12.1 Mcell/s |
| Radiation-hydrodynamics | 1.1 Mcell/s | 6.2 Mcell/s |

The inner loops are allocation-free; the few hundred KiB per step come from the threading runtime.

**Memory.** Buffers scale as roughly `72 * ncells * 8` bytes when radiation is enabled, `35 * ncells * 8` without, where `ncells` includes ghost cells. A $128^3$ radiation-hydrodynamics run needs about 1.4 GB; pure hydrodynamics at $128^3$ needs about 0.7 GB.

## Output

- `write_vtk(sim, path)` — a self-contained `.vti` image-data file with raw appended binary, written without any dependency. ParaView, VisIt and yt open it directly, and a numbered series (`data_00000.vti`, `data_00001.vti`, …) is read as a time series.
- `write_line(sim, path; dir, at)` — a plain-text table along one axis, for quick plotting of shock tubes and Marshak waves.
- `save_checkpoint` / `load_checkpoint!` — raw binary restart files.

`run!(sim; tmax, outdir, dtout)` writes VTK snapshots on a fixed cadence and trims the step so that it lands exactly on the output times.

## Extending

The code is built around small, dispatch-based interfaces. To add your own piece of physics, define a type and a few methods:

```julia
struct KramersOpacity <: Hydro3D.AbstractOpacity end
Hydro3D.planck_opacity(::KramersOpacity, rho, T) = 4e25 * rho * T^-3.5
Hydro3D.flux_mean_opacity(op::KramersOpacity, rho, T) = Hydro3D.planck_opacity(op, rho, T)
Hydro3D.energy_mean_opacity(op::KramersOpacity, rho, T) = Hydro3D.planck_opacity(op, rho, T)
```

The same pattern works for `AbstractEOS`, `AbstractLimiter`, `AbstractReconstruction`, `AbstractIntegrator` and `BoundaryCondition`. See `CLAUDE.md` for the layout of the source tree and the invariants each layer relies on.

## Units

Hydro3D is unit-agnostic: `c`, `a_rad`, `cv` and the opacities are all supplied by you. Use `Hydro3D.CGS` for CGS constants, or set `c = a_rad = cv = 1` and work in code units, as the built-in radiation problems do.

## Choosing radiation parameters

Two dimensionless ratios decide whether a radiation-hydrodynamics setup is well posed. Both are worth printing before a long run.

**Radiation-to-gas energy, $a_r T^4 / (\rho c_v T)$.** These equations are non-relativistic and keep only $O(v/c)$ terms in the coupling. A radiation energy density that dominates the gas by many orders of magnitude will accelerate the matter towards $c$, and the model stops describing anything. Because the ratio scales as $T^3/\rho$, a hot, tenuous medium is radiation-dominated very easily: at $10^5$ K, gas domination needs $\rho \gtrsim 4\times10^{-8}\ \mathrm{g\,cm^{-3}}$. Keep the ratio of order unity or below unless you know the gas stays slow.

**Reduced speed of light, $\hat{c} / \max(|v| + c_s)$.** The reduced speed of light must still outrun every other signal; a factor of about 20 is comfortable. Making $\hat{c}$ smaller buys a proportionally longer radiation step (`max_radiation_substeps` caps how many substeps one hydro step may take, shortening the hydro step instead of exceeding it), but it also rescales the radiation diffusion coefficient by $\hat{c}/c$ — so use $\hat{c} = c$ wherever the static diffusion limit matters quantitatively.

`examples/radiating_blast_3d.jl` checks both before it starts, and is a reasonable template.

## Limitations

- Uniform grid only — no mesh refinement, no MPI, no GPU.
- Grey (single-group) radiation. No multigroup, dust, chemistry, or self-gravity.
- Shared-memory threading only, so problem sizes are bounded by one node's memory.
- A reduced speed of light rescales the radiation diffusion coefficient by $\hat{c}/c$; use $\hat{c}=c$ when the static diffusion limit matters quantitatively.

## References

- Colella & Woodward (1984), *JCP* 54, 174 — PPM
- Toro (2009), *Riemann Solvers and Numerical Methods for Fluid Dynamics* — HLLC, exact Riemann solver
- Minoshima & Miyoshi (2021), *JCP* 446, 110639 — low-dissipation HLL/HLLC
- Levermore (1984), *JQSRT* 31, 149 — M1 closure
- Skinner & Ostriker (2013), *ApJS* 206, 21 — M1 wave speeds, reduced speed of light
- Chu et al. (2019), *JCP* 389, 62 — IMEX PD-ARS
- Wibking & Krumholz (2022), *MNRAS* 512, 1430; He, Wibking & Krumholz (2024), *MNRAS* 531, 3906 — Quokka

## Author

Rewritten from [Hydro.jl](https://github.com/chongchonghe/Hydro.jl) by Chong-Chong He (che1234@umd.edu).
