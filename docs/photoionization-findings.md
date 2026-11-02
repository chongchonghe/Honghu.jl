# Photoionization: the chemistry integrators and the reduced speed of light, measured

This note reports what was measured on the Stromgren-sphere tests of Iliev et al. (2006), which Honghu runs through `Honghu.Problems.stromgren_iliev` with the hydrogen network `HydrogenOTSA` and the photon band of `Photochemistry`. It answers four questions: what the chemistry costs per cell, how far the single-step `BackwardEulerChem` is from the adaptive `RosenbrockChem` on a real problem, how small the reduced speed of light $\hat c$ can be before the result changes, and how much the default heating per ionization overheats Test 2. It also records two choices in the scheme that the results depend on (section 5).

Every number in the tables of parts 1 to 3 below is produced by `docs/photoionization-experiments/measure.jl` and written to `docs/photoionization-experiments/measure.txt`, which also holds the radial profiles and the counts that this note does not repeat. Other numbers have other sources, and each one says where it comes from: a single run of `examples/stromgren_sphere.jl` (the centred $64^3$ run and the $n = 32$ octant), the work on the two integrators and on the photochemistry step (named by commit, because the measuring scripts are not in the repository), or a formula (the front speed, $3c_s$). The numbers come from one machine (Apple silicon, Julia 1.12.6, `-t 8`), so the timings are for that machine and the physics is not.

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
| 3. Reduced speed of light | Test 1 at $n = 16$ with $\hat c/c = 10^{-4}$, $3\times10^{-4}$, $10^{-3}$ and $3\times10^{-3}$, the ionized radius at 10, 30, 100 and 500 Myr against `stromgren_radius_finite_c` and `stromgren_radius_analytic`. Also, at each time, where the front lies along an axis and along the body diagonal, and at 500 Myr the radius of the recombination volume and how the gap between $\sum x$ and $\sum x^2$ is made up. |

The radius $r_I$ is `ionized_radius`: the radius of a sphere with the volume of the ionized gas, $\sum (x - x_0)/(1 - x_0)\,dV$ over the cells. The recombination time is $t_{\rm rec} = 1/(\alpha_B n_{\rm H}) = 122.3$ Myr and the analytic Stromgren radius is $r_S = 5.39$ kpc, both from the formulas in the docstring of `stromgren_radius_analytic`.

## Headlines

- The cost gate passed with room to spare (section 1), so no fast path and no in-tree integrator was needed. A run of the default example takes minutes on 8 threads.
- On Test 2 the single backward-Euler step agrees with the Rosenbrock reference to a fraction of a percent in the ionized radius and in the temperature of the ionized gas, differs where the profile is a few cells wide, at the front, and makes the whole run about 4.5 times faster (section 2). That is one problem at one substep; section 2 and the `BackwardEulerChem` docstring say where backward Euler is known to fail.
- The run follows the finite-$\hat c$ reference for $\hat c \geq 10^{-3}c$ at all times and for $3\times10^{-4}c$ from 100 Myr on, and not at $10^{-4}c$ (section 3). That is consistent with the expectation $\hat c > 3c_s$, but the sweep cannot place the threshold. A $\hat c$-independent excess of the saturated radius is a feature of how the front is counted, and the photon budget closes.
- The default heating per ionization is 2.5 times the optically thin value and raises the temperature of the ionized gas by about a tenth (section 4).
- Two things in the scheme bias the result and are not removed: the emission is injected before the burn, and the species are renormalised, not advected conservatively (section 5).

## 1. Cost of the chemistry step

| Test | Integrator | 1 thread ($\mu$s per cell) | 8 threads ($\mu$s per cell) | Solver steps per cell, mean / max (Newton iterations for backward Euler) |
|:---|:---|---:|---:|:---|
| 1 | `RosenbrockChem` | 5.02 | 0.73 | 15.0 / 20 |
| 1 | `BackwardEulerChem` | 0.085 | 0.016 | 1.03 / 3 |
| 2 | `RosenbrockChem` | 6.49 | 0.94 | 15.0 / 19 |
| 2 | `BackwardEulerChem` | 0.064 | 0.014 | 0.15 / 3 |

The gate was 20 $\mu$s per cell on one thread for `RosenbrockChem`; it passes by a factor of 3 to 4. Eight threads speed it up by a factor of 6.9 in both tests.

That scaling needs the pure-Julia linear solver the integrator uses. With the integrator's default factorisation, which calls OpenBLAS, one $24^3$ chemistry step took 84.7 ms on one thread and 183.7 ms on four, because OpenBLAS takes a global lock to allocate its buffers on every call. With LinearSolve's `SimpleLUFactorization` the same step took 73.7 ms and 20.0 ms. Those two pairs were measured when the fix was made (commit `9668501`) and `measure.jl` does not reproduce them; its 8-thread scaling above is an independent confirmation on a different problem.

The Rosenbrock cost is nearly the same in all cells: a mean of 15 steps and a maximum of 20, with only 1.8% of the cells ionized. So most of the cost is the integrator taking about 15 steps in cells where little changes, and not the physics of the front. I did not look at the steps of a single quiet cell. A fast path for cells whose state does not change would cut the cost, and was not built.

The default example, the centred $64^3$ box at $\hat c = 3\times10^{-4}c$ over 500 Myr (`examples/stromgren_sphere.jl`), was run once on 8 threads, with figures: it took 8 min 40 s, with 0 failed chemistry solves in 585 million and 15.1 solver steps per cell on average. That is well inside the 30 minutes that the 20 $\mu$s gate stood for. Its $r_I$ at 100, 200 and 500 Myr (3.906, 4.830 and 5.463 kpc) is the same to 0.01% as that of the octant of the example at $n = 32$, which has the same cell size (`examples/stromgren_sphere.jl 1 32 octant`), so the whole sphere with outflow walls and the corner octant with reflecting walls agree. Both are outputs of the example and are not in `measure.txt`.

## 2. The two integrators on Test 2

| | `RosenbrockChem` | Rosenbrock, tolerance $10^{-4}$ | `BackwardEulerChem` |
|:---|---:|---:|---:|
| Wall time to 500 Myr, 8 threads | 5.9 s | 6.0 s | 1.3 s |
| Solver steps per cell, mean / max (Newton iterations for backward Euler) | 16.0 / 20 | 16.5 / 31 | 1.33 / 5 |
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

The cost: backward Euler is 45 to 100 times cheaper per cell (section 1), and makes the whole run 4.5 times faster. The 4.6 s that the two wall times differ by is the chemistry, which is then about three quarters of the Rosenbrock run, if the rest of the step costs the same in both.

**Backward Euler failed nowhere in this run, and that does not carry over.** A stress sweep run when `BackwardEulerChem` was written (final solver in commit `bba25f9`; the script is not in the repository, and its numbers are not reproduced by `measure.jl`) used a synthetic grid of 2592 cells with $n_{\rm H} = 10^{-4}$, $10^{-2}$ and $1$ cm$^{-3}$, $T$ from 30 K to $10^5$ K, $x_{\rm H^+}$ from 0 to 0.999, several photon densities and steps of $10^8$, $10^{11}$ and $10^{13}$ s, and not the states of this problem. It left 269 cells (10.4%) unconverged even with the line search: 182 of 300 where collisional growth is stiff ($\beta n_{\rm H^0}\Delta t > 1$), 74 of 123 at the front with $\Gamma\Delta t \geq 10^2$, and 13 of 2169 (0.6%) elsewhere. A second group at exactly 100 K left 150 of 432 cells unconverged: 130 of the 404 cells away from the front, where the recombination cooling switches on at 100 K and the energy iterate alternates across it, and 20 of the 28 front cells. `RosenbrockChem` converged in all 3888 cells of the three sweeps, which also include steps of $3\times10^{14}$ s. So backward Euler is a good answer on this problem at this substep, and it is the cheap choice, but only `RosenbrockChem` is safe where the gas is hot and neutral or the step is long. How the backward-Euler error depends on $\hat c$ and $n$, which set the substep through $\Delta t \propto \Delta x/\hat c$, was measured at one pair only.

## 3. The reduced speed of light on Test 1

The error of the ionized radius against the finite-$\hat c$ front equation, $r_I/r_{\rm ref} - 1$, at $n = 16$:

| $\hat c/c$ | $\hat c$ (km s$^{-1}$) | 10 Myr | 30 Myr | 100 Myr | 500 Myr |
|:---|---:|---:|---:|---:|---:|
| $10^{-4}$ | 30 | $+382.27\%$ | $+90.93\%$ | $+7.80\%$ | $+4.54\%$ |
| $3\times10^{-4}$ | 90 | $+76.29\%$ | $+6.67\%$ | $+2.63\%$ | $+2.94\%$ |
| $10^{-3}$ | 300 | $+4.95\%$ | $+1.32\%$ | $+1.41\%$ | $+2.70\%$ |
| $3\times10^{-3}$ | 900 | $+0.88\%$ | $+0.66\%$ | $+1.02\%$ | $+2.66\%$ |

The same radii against the $\hat c = c$ curve are in `measure.txt`.

**Where it follows.** At 500 Myr the three runs with $\hat c \geq 3\times10^{-4}c$ agree to 0.52% ($r_I = 5.474$, $5.496$ and $5.502$ kpc), as the equilibrium requires: the Stromgren radius does not depend on $\hat c$. The $\hat c = 10^{-4}c$ run has not saturated. The radius of its recombination volume (see below) is 5.22 kpc, 2.6% below the analytic radius at 500 Myr, 5.363 kpc, and its $r_I = 5.362$ kpc is 4.5% above the finite-$\hat c$ reference of 5.129 kpc. Before 500 Myr, the run follows the reference to within 5% at every time shown from $10^{-3}c$ on (4.95% at worst), and from 100 Myr on at $3\times10^{-4}c$ (2.6%). At $10^{-4}c$ it is 7.8% and 4.5% off at 100 and 500 Myr.

**The early-time errors at low $\hat c$ are large, and the source size may contribute to them.** At 10 Myr the reference radius of the two smallest $\hat c$ is 0.31 and 0.87 kpc, which is less than one and about two cells (0.41 kpc at $n = 16$), and less than the extent of the source, a Gaussian of width $1.5\,\Delta x = 0.62$ kpc that is cut at 1.55 kpc. For the two largest $\hat c$ the reference radius (1.75 and 2.12 kpc) is larger than the source and the errors are 4.95% and 0.88%. The radii the two smallest runs report, 1.48 and 1.54 kpc, sit near the source size, which suggests the source sets $r_I$ there. A run at $n = 32$ ($3\times10^{-4}c$, the octant of the example) gives 1.10 kpc: closer to the reference (0.87 kpc) but still above it, and above the source's cut radius at that resolution (0.77 kpc). So the error falls with resolution, and I did not establish how much of it the source extent explains. The $n = 32$ value is an output of the example and is not in `measure.txt`.

**Against the expectation $\hat c > 3c_s$.** The design specification takes $3c_s \approx 45$ km s$^{-1}$, between $10^{-4}c$ (30 km s$^{-1}$, which does not follow) and $3\times10^{-4}c$ (90 km s$^{-1}$, which does from 100 Myr on). So the sweep is consistent with the expectation, and it cannot say more: it has no point between the two. With hydrodynamics off the gas does not move, so $c_s$ does not enter this test. What matters is the speed of the front, $v_I = (r_S/3t_{\rm rec})\,e^{-t/t_{\rm rec}}(1 - e^{-t/t_{\rm rec}})^{-2/3}$ with $r_S/3t_{\rm rec} = 14.4$ km s$^{-1}$, which is about 70, 30 and 9 km s$^{-1}$ at 10, 30 and 100 Myr (computed from this formula), and which $\hat c$ must exceed for the run to resemble the $\hat c = c$ solution. That is a stronger condition than following the finite-$\hat c$ reference: against the $\hat c = c$ curve at 100 Myr the radii are $-11.7\%$ at $3\times10^{-4}c$, $-2.6\%$ at $10^{-3}c$ and $-0.3\%$ at $3\times10^{-3}c$.

**The excess of 2.7% to 2.9% at 500 Myr does not depend on $\hat c$,** for $\hat c \geq 3\times10^{-4}c$, so it is not a $\hat c$ effect. What was measured: $r_I$ counts ions, $\sum x\,dV$, while the recombinations that balance the photons go as $\sum x^2\,dV$. The radius of the recombination volume, $8\sum x^2\,dV$ with $x = x_{\rm H^+}$ in the octant, is 0.9%, 0.6% and 0.5% below the analytic radius at 500 Myr (5.363 kpc) for $3\times10^{-4}c$, $10^{-3}c$ and $3\times10^{-3}c$, so the photon budget closes to about 1%, and $r_I$ is high because $\sum x$ exceeds $\sum x^2$ by $\sum x(1-x)$. Two things contribute to that sum, and I did not separate them. One is the neutral fraction left behind the front: in the three saturated runs the mean over the cells with $x > 0.5$ is 0.07 to 0.08, and 29% to 30% of $\sum x(1-x)$ comes from the cells with $x > 0.9$. The other is the transition layer itself: 397 to 415 of the 4096 cells have $0.1 < x < 0.9$ in those runs, and the rest of the sum (70% to 71%) comes from the cells with $x \leq 0.9$, which include that layer. A front narrower than a cell is not the whole explanation: if it were, the excess should roughly halve when the cell does, and it does not; the $n = 32$ octant of the example gives $+2.75\%$ against the finite-$\hat c$ reference at 500 Myr, against $+2.94\%$ at $n = 16$ for the same $\hat c$ (an output of the example, not in `measure.txt`). So the excess is not set by the cell size alone, and the width of the transition layer, numerical or physical, was not examined.

The part of the error that grows as $\hat c$ falls (at 100 Myr: 1.02%, 1.41%, 2.63% and 7.80% from $3\times10^{-3}c$ to $10^{-4}c$) is not explained here. The candidates are that the HLL scheme diffuses the photons in flight, the source size above, and the reference itself. The reference of Shapiro et al. (2006), `stromgren_radius_finite_c`, counts the photons in flight as $\dot N R/\hat c$ and ignores their depletion by recombinations inside $R$. A sharp-front version of that term, $(\dot N - 4\pi R^3\alpha_B n^2/3)/\hat c$, moves the reference radius by +1.5% to +1.9% at $\hat c = 3\times10^{-4}c$ and +0.6% at $10^{-3}c$. That estimate was made when the reference was written (the wording of its docstring is from commit `d482aa3`), it is not in the code or in `measure.txt`, and it has the sign and the $\hat c$ trend of the unexplained part. I list it as a candidate, not a finding. None of the candidates was tested.

**The front is not round.** The last lines of each block of part 3 in `measure.txt` give the position where $x_{\rm H^+}$ falls through 0.5 along the row of cells next to an axis and along the body diagonal. At $n = 16$ and $\hat c = 3\times10^{-4}c$ the diagonal is 9% behind the axis at 100 Myr (3.64 against 3.99 kpc) and 3% ahead at 500 Myr (5.58 against 5.41 kpc); at $10^{-3}c$ and $3\times10^{-3}c$ it is 1.5% and 4.3% behind at 100 Myr and 3.1% and 3.0% ahead at 500 Myr. The centred $64^3$ run shows it as lobes in the slice of the neutral fraction. A saturated Stromgren sphere should be round, so this is probably a grid imprint of the transport scheme at this resolution. It averages out of $r_I$ to first order, since $r_I$ is a volume, and I did not look for its cause.

## 4. The overheating of Test 2

The default heating per ionization of the network is $\varepsilon_b - 13.6$ eV $= 16.05$ eV for the 29.65 eV band of Test 2. The optically thin value for the $10^5$ K blackbody is about 6.3 eV (design specification, section 4.3), so the default is 2.5 times too large. The same run with `heating_energy = 6.3 eV` (part 2 of the script builds it itself; the replica reproduces the factory run exactly) gives:

| Heating per ionization | $T$ of the cells with $x_{\rm H^+} > 0.9$, 100 Myr | 500 Myr | $r_I$, 100 Myr | 500 Myr |
|:---|---:|---:|---:|---:|
| 16.05 eV (default) | 14756 K | 13704 K | 3.995 kpc | 5.932 kpc |
| 6.3 eV | 12988 K | 12317 K | 3.963 kpc | 5.806 kpc |

So 2.5 times the heating raises the temperature of the ionized gas by 13.6% and 11.3%, and the radius by 0.8% and 2.2%, through $\alpha_B(T)$. The temperature responds weakly because the cooling rises steeply with it, in particular the Lyman $\alpha$ term, which is exponential in $1/T$. That is my reading, and I did not isolate it. Near the source the default run reaches 23900 K in the innermost bin, against 9300 K in the front bin at 500 Myr. Neither run is the real spectrum: a single band has no hardening of the spectrum with radius, so these temperatures should not be compared with the paper's profile. The numbers say only how much the gas temperature moves with the heating energy when everything else is kept.

## 5. Two choices in the scheme that the results depend on

Neither was measured by `measure.jl`. Both are recorded here because they bias what the sections above report, and because each has a follow-up that is the user's decision.

**The emission is injected all at once at the start of each substep.** The chemistry step adds $\phi\,\dot N\,\Delta t$ to the photon density before it integrates the cell, as Quokka does for its chemical bands. In an opaque cell the burn absorbs those photons within a small part of the step, so the ionization fraction at the end of the step sits below its equilibrium by about $\alpha n x^2\Delta t/2$: the scheme is first order in the substep. In the uniform opaque test box of `test/test_chemistry.jl`, measured when the step was written (commit `8e86f10`), the equilibrium $x = 0.5$ came out as 0.4939 at $\hat c = 10^{-2}c$ ($\Delta t = 0.05\,t_{\rm rec}$) and as 0.4450 at $\hat c = 10^{-3}c$ ($\Delta t = 0.5\,t_{\rm rec}$), and it converged to 0.5 linearly in $\Delta t$ when the substep was cut (0.49938 and 0.49993 for a substep 10 and 100 times smaller at $\hat c = 10^{-2}c$). The substep is set by $\hat c$, so a smaller $\hat c$ means a larger bias. I did not measure the bias in the Stromgren runs. Two follow-ups would remove it: injecting the emission with Strang splitting, half before and half after the burn, or putting the emission inside the ODE. The second is kept out of the ODE on purpose (`chemistry.jl`), so that the two integrators solve one problem, and it needs a decision.

**The species are renormalised to add up to $\rho$.** Each $\rho X_s$ is reconstructed and limited on its own, so the species fluxes do not add up to the mass flux. With PLM or PPM the per-cell sum $\sum_s X_s$ then drifted from 1: after 10 steps of a two-species run on an 8-cell sine, the largest deviation was $4\times10^{-16}$ with PCM, $1.1\times10^{-2}$ with PLM and $3.7\times10^{-2}$ with the default PPM (measured in the work behind commit `9668501`, not reproduced by `measure.jl`). With a network, `enforce_scalar_floors!` therefore rescales the species of each cell so that they add up to $\rho$, which is Quokka's cure in `EnforceLimits`. The rescaling does not conserve each species separately. The conservative alternative, CMA (Plewa & Mueller 1999), is not implemented. The Stromgren tests here have hydrodynamics off, so nothing advects the species in them.

## What is not known

- Which part of the backward-Euler error at the front is the time step. The cheap test is the same run at a larger $\hat c$, which shortens the substep.
- The origin of the part of the radius excess that grows as $\hat c$ falls (section 3), and whether the finite-$\hat c$ reference or the scheme is closer to the truth there; and the width of the transition layer behind the $\hat c$-independent excess.
- How much of the early-time error at low $\hat c$ the source extent explains.
- The cause of the anisotropy of the front, and how it scales with $n$.
- The size of the emission-injection bias in the Stromgren runs.
- Anything at $n > 16$ for parts 2 and 3. The octant example at $n = 32$ gives $r_I = 5.463$ kpc at 500 Myr against 5.474 kpc at $n = 16$ for the same $\hat c$.
- The Test 2 profile against published data. The band is one cross-section and one photon energy, so only the shape is checked.
