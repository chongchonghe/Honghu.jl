# Sphericity of the Stromgren HII region

**Question.** Is the ionized region of the Stromgren-sphere test smooth and spherical, and does the kernel that deposits the source photons on the grid change its shape?

**Answer.** It depends on the kernel. A compact or sharp-edged source makes an aspherical front with a cubic pattern, of $\pm 10\%$ to $\pm 20\%$ in radius. A Gaussian source of width $3\,\Delta x$ makes a front that is round to 0.3%. The default kernel, which is Quokka's (a Gaussian of width $1.5\,\Delta x$, cut at 2.5 widths), gives eight lobes with a spread of 4%.

## Setup

- Iliev et al. (2006) Test 1: hydrogen at $n_{\rm H} = 10^{-3}$ cm$^{-3}$, isothermal at $10^4$ K, $\dot N = 5\times10^{48}$ s$^{-1}$, $\alpha_B = 2.59\times10^{-13}$ cm$^3$ s$^{-1}$. The Stromgren radius is 5.39 kpc.
- The whole sphere in a centred box of side 13.2 kpc, $64^3$ cells ($\Delta x = 0.206$ kpc), `Outflow` boundaries. The source is at the origin, which is a cell corner.
- Hydrodynamics off. M1 radiation with one group, HLL fluxes, $\hat c = 3\times10^{-4}c$. Chemistry solver `BackwardEulerChem` (see the check with `RosenbrockChem` below).
- Every kernel is normalised over the grid, so the cells receive exactly $\dot N$ photons per second in each run. Only the shape of the source changes.

| Kernel | Weight of the cell centred at $\mathbf{r}$ |
|:---|:---|
| `cells8` | 1 in the 8 cells that touch the source, 0 elsewhere |
| `gauss` (default, Quokka's) | $\exp(-r^2/2\sigma^2)$, $\sigma = 1.5\,\Delta x$, zero for $r > 2.5\sigma$ |
| `tophat3` | the volume fraction of the cell inside the sphere $r < 3\,\Delta x$ |
| `gauss_nocut` | as `gauss`, cut at $5\sigma$ instead of $2.5\sigma$ |
| `gauss3` | $\sigma = 3\,\Delta x$, cut at $4\sigma$ |

**The measurement.** At 30, 100, 200 and 500 Myr, the script finds the radius $R$ where the ionization fraction $x_{\rm H^+}$ falls through 0.5, along 4000 directions spread evenly over the sphere. It interpolates $x_{\rm H^+}$ trilinearly between the cell centres. The **spread** is ${\rm std}(R)/\bar R$. A perfect sphere has a spread of 0.

## Reproduce

From a clone of Honghu.jl, on this branch:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'     # once

# the five runs: about 80 s each on 6 threads of an Apple M5 Max
julia --project=. -t auto docs/photoionization-experiments/sphericity.jl cells8 gauss tophat3 gauss_nocut gauss3

# the figures (needs Plots in the global environment)
julia --project=. docs/photoionization-experiments/sphericity_plots.jl cells8 gauss tophat3 gauss_nocut gauss3
```

Options of both scripts: `--n=64` (cells per side), `--solver=be` or `--solver=ros` (`BackwardEulerChem` or `RosenbrockChem`), `--outdir=DIR` (default: this folder). With no kernel names, the run script runs all of them.

The run script writes one table per kernel, `sphericity_<kernel>_<solver>.txt`, and the ionization fraction at each output time as raw `Float32` ($n^3$ values, $x$ fastest), `x_<kernel>_<solver>_<t>Myr.f32`. The raw files are not in git, because they are 1 MB each.

To try another kernel, pass any function `w(x, y, z, dx)` to `stromgren_iliev`:

```julia
using Honghu
sim, tmax = Honghu.Problems.stromgren_iliev(test = 1, n = 64, centered = true, chat_factor = 3e-4,
                                            kernel = (x, y, z, dx) -> exp(-(x^2 + y^2 + z^2) / (2 * (2dx)^2)))
```

## Results

$64^3$, 500 Myr unless stated. The finite-$\hat c$ reference radius (Shapiro et al. 2006) is 2.072 kpc at 30 Myr and 5.317 kpc at 500 Myr.

| Kernel | Spread, 100 Myr | Spread, 500 Myr | $R_{\min}$ (kpc) | $R_{\max}$ (kpc) | $r_I$, 30 Myr (kpc) | $r_I$, 500 Myr (kpc) |
|:---|---:|---:|---:|---:|---:|---:|
| `cells8` | 9.64% | 10.58% | 4.43 | $\geq 6.47$ | 2.079 | 5.463 |
| `gauss` (default) | 3.06% | 4.11% | 5.11 | 6.12 | 2.117 | 5.468 |
| `tophat3` | 1.86% | 2.63% | 5.28 | 6.03 | 2.111 | 5.467 |
| `gauss_nocut` | 1.36% | 1.86% | 5.23 | 5.75 | 2.128 | 5.466 |
| `gauss3` | 0.39% | 0.32% | 5.46 | 5.57 | 2.256 | 5.464 |

$r_I$ is the volume-equivalent radius of the ionized gas (`ionized_radius`). For `cells8` the front reaches the edge of the box along the axes, so its $R_{\max}$ is a lower limit.

![All kernels](sphericity_comparison_be.png)

*One row per kernel. Columns 1 and 2: $\log_{10}$ of the neutral fraction in the mid-plane $z = -\Delta x/2$, at 100 and 500 Myr. Column 3: the front radius in all directions at 500 Myr, $R/\bar R - 1$, against the azimuth $\phi$ and $\cos\theta$, on one colour scale for all rows.*

![Spread and anisotropy against time](sphericity_summary_be.png)

*The spread, and $(R_{\rm body} - R_{\rm axis})/\bar R$, where $R_{\rm body}$ is the mean radius along the 8 body diagonals and $R_{\rm axis}$ along the 6 axes. The second quantity misses lobes that lie between those directions, as for `gauss`, so the spread is the better measure.*

The single-kernel figures `sphericity_<kernel>_be.png` show the same panels with the front radius at 100 Myr too, each on its own colour scale.

**Solver check.** With `RosenbrockChem`, `gauss` gives a spread of 3.05% at 100 Myr and 4.18% at 500 Myr, against 3.06% and 4.11% with `BackwardEulerChem`, and `gauss3` gives 0.42% and 0.33%, against 0.39% and 0.32%. The chemistry solver does not change the shape. A Rosenbrock run takes about 16 minutes on 6 threads, against 80 s for backward Euler.

## What this means

1. The kernel sets the shape. All runs emit the same photons per second, and their $r_I$ agree to 0.8% at 100 Myr and to 0.1% at 500 Myr, but their spread at 500 Myr goes from 0.3% to 10.6%.
2. A source in a few cells emits more photons along the grid axes. `cells8` makes an octahedron: the front is up to 20% ahead along the axes and behind along the body diagonals. My reading, not tested: the M1 transport keeps the angular pattern that the radiation gets near the source and carries it out to the front, so the pattern does not smooth out with time.
3. A sharp edge in the kernel makes lobes. The cut of `gauss` at $2.5\sigma = 3.75\,\Delta x$ and the edge of `tophat3` both leave lobes. Removing the cut (`gauss_nocut`) halves the spread.
4. A smooth source over several cells gives a round front. `gauss3` is round to 0.3%, with dips of about 1% in the coordinate planes.
5. The cost of a wide source is the early radius. The `gauss3` source reaches 2.5 kpc, and at 30 Myr its $r_I$ is 8.9% above the reference, against 2.2% for `gauss`. At 100 Myr the difference is 0.3%, and at 500 Myr it is gone.

**For Quokka.** The default kernel here is the one of Quokka's `testStromgrenSphere.cpp`, and Honghu transliterates Quokka's M1 transport. So a Quokka run with that source is likely to show the same eight lobes (not verified in Quokka). A first test is a wider Gaussian without a sharp cut.

## Not yet known

- Whether the transport alone (no chemistry) makes the same pattern from the same sources. A test of the optically thin field $E(\mathbf{r})$ around each source would separate the two.
- How the spread scales with resolution, and whether the width that a round front needs is a number of cells or a physical size.
- The spread for widths between $1.5\,\Delta x$ and $3\,\Delta x$. The kernel `gauss2` ($\sigma = 2\,\Delta x$, cut at $4\sigma$) is in the script but was not run.
