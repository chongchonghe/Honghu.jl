# Photoionization: the chemistry integrators and the reduced speed of light, measured

This note reports what was measured on the Stromgren-sphere tests of Iliev et al. (2006), which Honghu runs through `Honghu.Problems.stromgren_iliev` with the hydrogen network `HydrogenOTSA` and the photon band of `Photochemistry`. It answers four questions: what the chemistry costs per cell, how far the single-step `BackwardEulerChem` is from the adaptive `RosenbrockChem` on a real problem, how small the reduced speed of light $\hat c$ can be before the result changes, and how much the default heating per ionization overheats Test 2. Every number below is produced by `docs/photoionization-experiments/measure.jl` and written to `docs/photoionization-experiments/measure.txt`, which also holds the radial profiles and the counts that this note does not repeat. The numbers come from one machine (Apple silicon, Julia 1.12.6, `-t 8`), so the timings are for that machine and the physics is not.

## What was measured

```bash
julia --project=. -t 8 docs/photoionization-experiments/measure.jl          # all three parts, about 3 minutes
julia --project=. -t 8 docs/photoionization-experiments/measure.jl cost     # one part (it only prints)
```

The one-thread timings need a second Julia process, because the threaded loops follow `Threads.nthreads()`. The script starts it itself.

| Part | Setup |
|:---|:---|
| 1. Cost | `chemistry_step!` on a $32^3$ octant of each test with $\hat c = 3\times10^{-4}c$, after $0.25\,t_{\rm rec}$ so that the front is in the box. The median of 5 calls that all start from the same state, on 1 and on 8 threads, with both integrators. |
| 2. Integrators | Test 2 at $n = 16$ and $\hat c = 3\times10^{-4}c$, to 100 and 500 Myr, with `RosenbrockChem()` (relative tolerance $10^{-2}$, the default), the same with $10^{-4}$ (a check on the reference) and `BackwardEulerChem()`. The ionized radius, the temperature and neutral fraction averaged in 16 radial bins, and the wall time. Also the same run with the heating per ionization set to 6.3 eV. |
| 3. Reduced speed of light | Test 1 at $n = 16$ with $\hat c/c = 10^{-4}$, $3\times10^{-4}$, $10^{-3}$ and $3\times10^{-3}$, the ionized radius at 10, 30, 100 and 500 Myr against `stromgren_radius_finite_c` and `stromgren_radius_analytic`. Also, at each time, where the front lies along an axis and along the body diagonal, and at 500 Myr the radius of the recombination volume. |

The radius $r_I$ is `ionized_radius`: the radius of a sphere with the volume of the ionized gas, $\sum (x - x_0)/(1 - x_0)\,dV$ over the cells, which counts a half-ionized cell as half a sphere. The recombination time is $t_{\rm rec} = 122.3$ Myr.

## Headlines

- The cost gate passed by a factor of about three. `RosenbrockChem` costs 5.6 $\mu$s per cell on one thread in Test 1 and 6.7 $\mu$s in Test 2, against the limit of 20 $\mu$s, and 0.76 and 0.97 $\mu$s on eight threads. No fast path and no in-tree integrator was needed.
- On Test 2 the single backward-Euler step reproduces the Rosenbrock reference's ionized radius to 0.02% (100 Myr) and 0.10% (500 Myr), and the temperature of the ionized gas to 0.06%. It differs where the profile is a few cells wide, at the front, by up to 29% in the bin-averaged temperature. It makes the whole run 4.6 times faster and had no failed cell in 4.6 million solves.
- The run follows the finite-$\hat c$ reference to within 5% at all four times for $\hat c \geq 10^{-3}c$, and for $\hat c = 3\times10^{-4}c$ from 100 Myr on (2.6% and 2.9%). It does not at $10^{-4}c$ (7.8% and 4.5% at 100 and 500 Myr). That is consistent with the expectation $\hat c > 3c_s$, but the sweep has no point between $10^{-4}c$ and $3\times10^{-4}c$ and cannot place the threshold. Part of the error does not depend on $\hat c$: at saturation $r_I$ is 2.7% to 2.9% above the reference for every $\hat c \geq 3\times10^{-4}c$, and the photon budget closes to 1%, so it is the way the front is counted.
- The default heating per ionization (16.05 eV) is 2.5 times the optically thin value (6.3 eV) and raises the temperature of the ionized gas by only 11% to 14%.

## 1. Cost of the chemistry step

| Test | Integrator | 1 thread ($\mu$s per cell) | 8 threads ($\mu$s per cell) | Solver steps per cell, mean / max |
|:---|:---|---:|---:|:---|
| 1 | `RosenbrockChem` | 5.58 | 0.76 | 15.0 / 20 |
| 1 | `BackwardEulerChem` | 0.088 | 0.016 | 1.03 / 3 |
| 2 | `RosenbrockChem` | 6.69 | 0.97 | 15.0 / 19 |
| 2 | `BackwardEulerChem` | 0.063 | 0.014 | 0.15 / 3 |

Eight threads speed `RosenbrockChem` up by a factor of 7.3 and 6.9. That needs the pure-Julia linear solver the integrator uses; with the OpenBLAS default the same step was slower on four threads than on one (the `RosenbrockChem` docstring has the numbers).

The Rosenbrock cost is nearly the same in all cells: a mean of 15 steps and a maximum of 20, with only 1.8% of the cells ionized. So most of the cost is the integrator taking about 15 steps in cells where little changes, and not the physics of the front. I did not look at the steps of a single quiet cell. A fast path for cells whose state does not change would cut the cost, and was not built.

The default example, the centred $64^3$ box at $\hat c = 3\times10^{-4}c$ over 500 Myr (`examples/stromgren_sphere.jl`), was run once on 8 threads, with figures: it took 8 min 40 s, with 0 failed chemistry solves in 585 million and 15.1 solver steps per cell on average. That is well inside the 30 minutes that the 20 $\mu$s gate stood for. Its $r_I$ at 100, 200 and 500 Myr (3.906, 4.830 and 5.463 kpc) is the same to 0.01% as the octant of the example at $n = 32$, which has the same cell size, so the whole sphere with outflow walls and the corner octant with reflecting walls agree. Both are outputs of the example and are not in `measure.txt`.

## 2. The two integrators on Test 2

| | `RosenbrockChem` | Rosenbrock, tolerance $10^{-4}$ | `BackwardEulerChem` |
|:---|---:|---:|---:|
| Wall time to 500 Myr, 8 threads | 6.0 s | 6.1 s | 1.3 s |
| Solver steps per cell, mean / max | 16.0 / 20 | 16.5 / 31 | 1.33 / 5 |
| Failed solves, of 4.57 million | 0 | 0 | 0 |

The substep of this run is 0.45 Myr, 0.4% of $t_{\rm rec}$. Against the default Rosenbrock result:

| | 100 Myr | 500 Myr |
|:---|---:|---:|
| Backward Euler, $r_I$ | $-0.018\%$ | $+0.098\%$ |
| Backward Euler, temperature of the cells with $x_{\rm H^+} > 0.9$ | $-0.055\%$ | $-0.056\%$ |
| Backward Euler, largest bin-averaged $T$ difference | 28.8% | 2.4% |
| Backward Euler, largest $x_{\rm HI}$ difference | $8.1\times10^{-3}$ | $8.3\times10^{-3}$ |
| Rosenbrock at $10^{-4}$, $r_I$ | $0.000\%$ | $0.000\%$ |
| Rosenbrock at $10^{-4}$, largest bin-averaged $T$ difference | 1.7% | 0.009% |

Behind the front backward Euler is as good as the reference allows: the ten bins inside the front agree to better than 0.5% at 100 Myr. The differences are in the bins that hold the front. At 100 Myr those are bins 11 to 13, where the mean temperature is 9401, 1538 and 120 K for backward Euler and 8961, 1195 and 108 K for the reference. At 500 Myr the front is in the last bin, 9545 against 9318 K. These are averages over the few cells that the front crosses, and one backward-Euler step is first order in time there. I did not test that, for instance by shortening the substep, so the cause is a guess.

The reference is itself good to the level it is used at: tightening the tolerance from $10^{-2}$ to $10^{-4}$ moves $r_I$ by less than $10^{-5}$ in relative terms and the front bins by up to 1.7%, which is 17 times less than the backward-Euler difference there.

The cost: backward Euler is 50 to 110 times cheaper per cell (section 1), and makes the whole run 4.6 times faster. The 4.7 s that the two wall times differ by is the chemistry, which is then about three quarters of the Rosenbrock run, if the rest of the step costs the same in both.

**Backward Euler failed nowhere in this run, and that does not carry over.** A stress sweep run when `BackwardEulerChem` was written (a script outside the repository), over a synthetic grid of 2592 cells with $n_{\rm H} = 10^{-4}$, $10^{-2}$ and $1$ cm$^{-3}$, $T$ from 30 K to $10^5$ K, $x_{\rm H^+}$ from 0 to 0.999, several photon densities and steps of $10^8$, $10^{11}$ and $10^{13}$ s, and not the states of this problem, left 269 cells (10.4%) unconverged even with the line search: 182 of 300 where collisional growth is stiff ($\beta n_{\rm H^0}\Delta t > 1$), 74 of 123 at the front with $\Gamma\Delta t \geq 10^2$, and 13 of 2169 (0.6%) elsewhere; at exactly 100 K the switch of the recombination cooling made the iteration cycle in 150 of 432 cells. `RosenbrockChem` converged in all 3888 cells of the three sweeps, which also include steps of $3\times10^{14}$ s. So backward Euler is a good answer on this problem at this substep, and it is the cheap choice, but only `RosenbrockChem` is safe where the gas is hot and neutral or the step is long. How the backward-Euler error depends on $\hat c$ and $n$, which set the substep through $\Delta t \propto \Delta x/\hat c$, was measured at one pair only.

## 3. The reduced speed of light on Test 1

The error of the ionized radius against the finite-$\hat c$ front equation, $r_I/r_{\rm ref} - 1$, at $n = 16$:

| $\hat c/c$ | $\hat c$ (km s$^{-1}$) | 10 Myr | 30 Myr | 100 Myr | 500 Myr |
|:---|---:|---:|---:|---:|---:|
| $10^{-4}$ | 30 | $+382\%$ | $+91\%$ | $+7.8\%$ | $+4.5\%$ |
| $3\times10^{-4}$ | 90 | $+76\%$ | $+6.7\%$ | $+2.6\%$ | $+2.9\%$ |
| $10^{-3}$ | 300 | $+5.0\%$ | $+1.3\%$ | $+1.4\%$ | $+2.7\%$ |
| $3\times10^{-3}$ | 900 | $+0.9\%$ | $+0.7\%$ | $+1.0\%$ | $+2.7\%$ |

The same radii against the $\hat c = c$ curve are in `measure.txt`.

**Where it follows.** At 500 Myr every run with $\hat c \geq 3\times10^{-4}c$ ends within 0.5% of the others ($5.47$, $5.50$ and $5.50$ kpc), as the equilibrium requires: the Stromgren radius does not depend on $\hat c$, and the $\hat c = 10^{-4}c$ run has not reached it yet (5.36 kpc, against a reference of 5.13 kpc). Before that, the run follows the reference to within 5% at every time shown from $10^{-3}c$ on, and from 100 Myr on at $3\times10^{-4}c$ (2.6%); at $10^{-4}c$ it is 7.8% and 4.5% off at 100 and 500 Myr.

**The early times are a resolution limit, not a $\hat c$ effect.** At 10 Myr the reference radius of the two smallest $\hat c$ is 0.31 and 0.87 kpc, which is less than one and about two cells (0.41 kpc at $n = 16$), and less than the extent of the source, a Gaussian of width $1.5\,\Delta x = 0.62$ kpc that is cut at 1.55 kpc. The radii the runs report there, 1.48 and 1.54 kpc, sit at that source size. A run at $n = 32$ ($3\times10^{-4}c$, the octant of the example) gives 1.10 kpc, which supports it but is not in `measure.txt`.

**Against the expectation $\hat c > 3c_s$.** With $c_s \approx 15$ km s$^{-1}$, $3c_s \approx 45$ km s$^{-1}$, between $10^{-4}c$ (30 km s$^{-1}$, which does not follow) and $3\times10^{-4}c$ (90 km s$^{-1}$, which does from 100 Myr on). So the sweep is consistent with the expectation, and it cannot say more: it has no point between the two. With hydrodynamics off the gas does not move, so $c_s$ does not enter this test. What matters is the speed of the front, $v_I = 70$, 30 and 9 km s$^{-1}$ at 10, 30 and 100 Myr, which $\hat c$ must exceed for the run to resemble the $\hat c = c$ solution. That is a stronger condition than following the finite-$\hat c$ reference: against the $\hat c = c$ curve at 100 Myr the radii are $-11.7\%$ at $3\times10^{-4}c$, $-2.6\%$ at $10^{-3}c$ and $-0.3\%$ at $3\times10^{-3}c$.

**The excess of 2.7% to 2.9% at 500 Myr is in the measurement of the front.** It is the same for all $\hat c \geq 3\times10^{-4}c$, so it is not a $\hat c$ effect. The radius $r_I$ counts ions, $\sum x\,dV$, while the recombinations that balance the photons go as $\sum x^2\,dV$. A front narrower than a cell counts a half-ionized cell as half a sphere in the first and a quarter in the second, so $r_I$ overstates the sphere. The radius of the recombination volume $8\sum x^2\,dV$ is $-0.9\%$, $-0.6\%$ and $-0.5\%$ from the analytic Stromgren radius at $3\times10^{-4}c$, $10^{-3}c$ and $3\times10^{-3}c$: the photon budget closes to 1%. The front is about as wide as the mean free path of an ionizing photon in neutral gas, $1/n\sigma = 51$ pc, which is 8 times narrower than a cell. This is consistent with the part of the excess that does not depend on $\hat c$ and is there at saturation. The part that grows as $\hat c$ falls (at 100 Myr: 1.0%, 1.4%, 2.6% and 7.8% from $3\times10^{-3}c$ to $10^{-4}c$) is not explained here. The candidates are that the reference counts the photons in flight as $\dot N R/\hat c$ and ignores their depletion by recombinations inside $R$, that the HLL scheme diffuses the photons in flight, and the source size above. None was tested.

**The front is not round.** The last line of each block of part 3 in `measure.txt` gives the position where $x_{\rm H^+}$ falls through 0.5 along the row of cells next to an axis and along the body diagonal. At $n = 16$ and $\hat c = 3\times10^{-4}c$ the diagonal is 9% behind the axis at 100 Myr (3.64 against 3.99 kpc) and 3% ahead at 500 Myr (5.58 against 5.41 kpc); at $10^{-3}c$ and $3\times10^{-3}c$ it is 1.5% and 4.3% behind at 100 Myr and 3.1% and 3.0% ahead at 500 Myr. The centred $64^3$ run shows it as lobes in the slice of the neutral fraction. A saturated Stromgren sphere should be round, so this is probably a grid imprint of the transport scheme at this resolution. It averages out of $r_I$ to first order, since $r_I$ is a volume, and I did not look for its cause.

## 4. The overheating of Test 2

The default heating per ionization of the network is $\varepsilon_b - 13.6$ eV $= 16.05$ eV for the 29.65 eV band of Test 2. The optically thin value for the $10^5$ K blackbody is about 6.3 eV, so the default is 2.5 times too large. The same run with `heating_energy = 6.3 eV` (part 2 of the script builds it itself; the replica reproduces the factory run exactly) gives:

| Heating per ionization | $T$ of the cells with $x_{\rm H^+} > 0.9$, 100 Myr | 500 Myr | $r_I$, 100 Myr | 500 Myr |
|:---|---:|---:|---:|---:|
| 16.05 eV (default) | 14756 K | 13704 K | 3.995 kpc | 5.932 kpc |
| 6.3 eV | 12988 K | 12317 K | 3.963 kpc | 5.806 kpc |

So 2.5 times the heating raises the temperature of the ionized gas by 13.6% and 11.3%, and the radius by 0.8% and 2.2%, through $\alpha_B(T)$. The temperature responds weakly because the cooling rises steeply with it, in particular the Lyman $\alpha$ term, which is exponential in $1/T$. That is my reading, and I did not isolate it. Near the source the default run reaches 23900 K in the innermost bin, against 9300 K in the front bin at 500 Myr. Neither run is the real spectrum: a single band has no hardening of the spectrum with radius, so these temperatures should not be compared with the paper's profile. The numbers say only how much the gas temperature moves with the heating energy when everything else is kept.

## What is not known

- Which part of the backward-Euler error at the front is the time step. The cheap test is the same run at a larger $\hat c$, which shortens the substep.
- The origin of the part of the radius excess that grows as $\hat c$ falls (section 3), and whether the finite-$\hat c$ reference or the scheme is closer to the truth there.
- The cause of the anisotropy of the front, and how it scales with $n$.
- Anything at $n > 16$ for parts 2 and 3. The octant example at $n = 32$ (`examples/stromgren_sphere.jl 1 32 octant`) gives $r_I = 5.463$ kpc at 500 Myr against 5.474 kpc at $n = 16$ for the same $\hat c$.
- The Test 2 profile against published data. The band is one cross-section and one photon energy, so only the shape is checked.
