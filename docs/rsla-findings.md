# The reduced speed of light: what the measurements say

Findings about the reduced speed of light in Honghu, in two parts. **Standard RSLA** is the global reduction $\phi_0 = \hat{c}/c$, the default (`ConstantChat()`), and holds what is true of it on its own. **ARSA** is the adaptive reduced speed of light approximation of He, Krumholz & Wibking, "An Adaptive Reduced Speed of Light Approximation" (J. Comput. Phys., in preparation), for which Honghu is the reference implementation; the code calls it IRSLA and its limiter `StabilityBound()`. The implementation is `src/rsla.jl` plus the array drivers in `src/radiation_system.jl` (`update_chat!`, `scale_by_phi!`, `floor_rad_energy!`, `energy_exchange_dt`), and `test/test_rsla.jl` pins what follows.

Everything is scored as a relative $L_1$ error in the gas temperature against a $\hat{c} = c$ reference at the same resolution, and — unless a row says otherwise — at the same number of steps as the standard RSLA it is compared with, so every comparison is at equal cost.

**Which numbers are current.** The ARSA took its present form in commit `b287f03`, the squash of branch `irsla-variable-chat-stage2`. Section A1 says what that commit changed and quotes its measurements; `examples/irsla_paper.jl` reproduces them and every other number of the paper's Sec. 5. Sections A2 to A9 are the findings that led there. They were measured before `b287f03`, with the earlier limiter, which took the optical depth as $\chi\,\Delta x_\mathrm{min}$, used one $\phi$ per cell and put $\phi$ inside the flux divergence. `examples/rsla_limiter.jl` produced them, but it now runs the current limiter, so its limiter columns no longer reproduce those tables. Its RSLA columns and the transport-only table of S3 do not depend on the limiter, but `b287f03` also changed what is repaired before the coupling solve, for every limiter, so they too may shift slightly if re-run. Where a finding was later reversed, its section says so.

## Standard RSLA

### S1. Its error is a modelling error, and it does not refine away

The standard RSLA multiplies the radiation source terms by $\phi_0$ and leaves the matter source terms alone, so radiation and gas equilibrate at the wrong temperature. The error is largest at high optical depth, which is where no reduction is needed for stability, because the implicit coupling damps the radiation there. It grows as $\phi_0$ falls: on the Lowrie & Edwards Mach 3 radiative shock at $n = 256$ it is 0.11% at $\phi_0 = 0.023$ and 2.5% at $\phi_0 = 0.0023$, a factor of 22 (table in A2). And since it comes from the modified equations, not from the discretisation, refining the grid does not remove it: on the same shock at $\phi_0 = 0.0069$ it stays between 0.54% and 0.67% from $n = 64$ to $n = 512$ (table in A6).

### S2. With a uniform $\phi$, the energy error is set by the gas energy change alone

Summing the source terms of the gas and the radiation groups,

$$\frac{d}{dt}\left[e + \sum_g E_g\right] = c\,\rho \sum_g \left(\phi_g - 1\right)\left(\kappa_P B_g - \kappa_E E_g\right) .$$

With $\phi$ uniform over groups and cells, the sum collapses onto the gas equation and leaves the exact identity

$$\Delta\left[e + \sum_g E_g\right] = \left(1 - \phi\right) \Delta e .$$

Nobody claims the RSLA conserves energy. What this says is stronger and is worth stating: the RSLA's non-conservation is *determined by how much energy the gas gained or lost, and by nothing else*. It does not depend on the spectrum, on how many absorption and re-emission cycles the run performed, or on which groups they passed through. Measured in a single cell with no transport at all, relaxing over 400 coupling steps, the identity holds to five decimals at $\phi = 0.1$ and at $\phi = 0.01$.

The identity is a property of a *global* reduction. Any local $\phi$, per cell or per group, loses it; A5 measures the loss and says what a local $\phi$ conserves instead.

### S3. Its time step: the closure's signal speed, and no radiation-pressure limit

The step at which the transport is stable is set by the closure's own signal speed, not by the P1 speed $c/\sqrt{3}$. Where M1 free-streams, the Eddington factor rises to one and the HLL wave speed estimates become $\pm\hat{c}$. The table below runs the Vaytet multigroup Marshak wave ($n = 64$, four groups) at $\hat{c} = c$ with no limiter, so that nothing but the transport step is under test, and varies only the radiation CFL number $\nu$:

| reconstruction | $\nu = 0.8$ | $\nu = 1.2$ | $\nu = 1.5$ | $\nu = \sqrt{3}$ |
|:---|---:|---:|---:|---:|
| PCM | 2.51% | 2.61% | 32.6% | 79.8% |
| PLM | 0.39% | 0.93% | 139.6% | overflow |
| PPM | 0.05% | 0.65% | 128.7% | overflow |

The scheme fails between $\nu = 1.2$ and $\nu = 1.5$ for *every* reconstruction, first-order upwind included, so this is not a reconstruction-order effect. It is the closure. $\nu = \sqrt{3}$, a Courant number of one for the P1 speed, overflows to infinity for PLM and PPM. A3 shows what this does to the ARSA's bound. For the RSLA itself it fixes the largest stable step: the paper runs in `examples/irsla_paper.jl` take $\beta = 0.9$ of it, $C_r = 0.9\sqrt{3}$ for P1 and $C_r = 0.9\sqrt{D}$ for M1, in Honghu's convention $C_r = D\,\nu_0$.

The standard RSLA needs no step limit from the radiation pressure, at any $P_\mathrm{rad}/P_\mathrm{gas}$. In the exact bound of A1, at $\phi = \phi_0$ the left-hand side is at most the CFL number times $\sigma$, whatever the heat capacities, and the checkerboard test in `test/test_rsla.jl` confirms that the RSLA's checkerboard decays at its own step for $\mathcal{P} = 1$ and $100$ in one and two dimensions. Honghu used to have such a limit, `radiation_pressure_dt`, from the radiation-force constraint (eq. 33) of the earlier paper draft; `b287f03` removed it. Before that, an earlier finding was that it should at least not be applied when the gas cannot move, since the feedback loop behind it runs through the gas velocity: on the Vaytet wave, a static problem with hot radiation over cold gas, applying it cost 5.4 times the steps for no change in the answer.

### S4. The wave-speed correction belongs to the standard RSLA only

`RadiationPhysics(wavespeed_correction = true)` scales the HLL dissipation of the radiation energy at each face by $\epsilon = \min(1, 1/\tau)$, with $\tau$ the harmonic mean of the two cells' optical depths (Skinner et al. 2019; optional in Quokka). In optically thick cells the HLL dissipation is a numerical diffusion far faster than the physical one, $c/(3\tau)$, and the correction removes it. Honghu allows it with the standard RSLA only; A1 says why.

## ARSA

### A1. The method as implemented, and what `b287f03` changed

Every cell and frequency group runs at the largest reduced speed of light that the IMEX PD-ARS step supports stably at the time step of the standard RSLA,

$$\phi = \min\left\{1,\ \max\left[\phi_0,\ \frac{\beta}{c\,\Delta t\,\Lambda}\right]\right\},$$

with $\Lambda$ the stability denominator below (`stability_denominator`) and $\beta = 0.9$ by default. Compared with the earlier limiter of A2 to A9:

- **The bound is the exact von Neumann boundary of the scheme**, in any number of dimensions. The grid-scale checkerboard is always the worst mode, and there the coupling splits into an energy block and a momentum block, each with boundary $$\phi\,c\,\Delta t\left[\sum_d \frac{\sigma_d}{\Delta x_d} - \frac{\chi}{2 + K}\right] \leq 1 ,$$ with $\chi = \rho\kappa_E$ and $K = 4 a_R T^3 \kappa_P\, c\,\Delta t / c_v$ for the energy block, and $\chi = \rho\kappa_F$ for the momentum block, whose $K$ is smaller by $(a/c)^2$ and dropped; $\Lambda$ is the larger of the two brackets. A checkerboard decays at 0.9 times the bound and grows at 1.1 times it, and at $\beta = 1$ it sits on $|g| = 1$ to $10^{-3}$, in one, two and three dimensions. The earlier limiter took the optical depth as $\chi\,\Delta x_\mathrm{min}$ instead of $\chi / \sum_d (1/\Delta x_d)$, which is $D$ times too generous: it grew a checkerboard by a factor of 1.06 to 1.88 per substep in two and three dimensions. Where radiation matters energetically the channel that binds is the heat-capacity one, through $K$, not the radiation force. The signal speed is taken per direction (A3).
- **$\phi$ multiplies the time derivative, not the flux divergence.** Each cell's update is $\phi_i$ times the difference of the unreduced HLL fluxes. Steady states are then exact for any $\phi(x)$, the flux differences telescope in $E/\phi$, which is the radiation energy the coupling conserves, and no interface value is needed. With $\phi$ inside the divergence, as the earlier drafts had it, the steady state solves $\nabla\cdot\boldsymbol{F} + \boldsymbol{F}\cdot\nabla\ln\phi = S$ instead; on the Mach 5 supercritical shock a 5% ripple in $\phi$ raised the $L_1$ error from 0.19% to 2.3%, worse than the standard RSLA. For a uniform $\phi$ the two forms are the same scheme. `flux_form = true` keeps the old placement for comparison.
- **One $\phi$ per group, always** (A5).
- **$\phi$ is set once per IMEX substep**, from the state at its start, and held through both stages, as the bound assumes. A $\phi$ that changed between the stages would leave a residual $(E^{(2)} - E^n)(1/\phi_b - 1/\phi_a)$ in the conserved energy.
- **Nothing the coupling solve can absorb is repaired before it.** A thick cell at $\phi = 1$ runs at a local CFL number far above one, and its transport stage leaves $|\boldsymbol{F}| \gg cE$, and at a steep jump in $\phi$ sometimes $E < 0$. The solve divides the flux by $1 + \hat{c}\rho\kappa_F\Delta t$, which turns it into the diffusion flux, and returns a positive temperature whenever $e + \sum_g E_g/\phi_g > 0$. Clipping $|\boldsymbol{F}|$ to $cE$ before the solve threw the diffusion flux away: on the thick hot spot the error rose from 0.0003% to 0.80%, against 0.37% for the standard RSLA. Flooring $E$ everywhere before the solve added energy every substep: on Ensman's supercritical shock at 64 cells it made the ARSA error 17 times the RSLA's. Now $E$ is floored before the solve only where $e + \sum_g E_g/\phi_g$ is not positive, and the full repair runs after the solve. `Diagnostics` counts the two as `floors_presolve` and `repairs_postsolve`. This applies to every limiter, the standard RSLA included.
- **The energy-exchange step is opt-in.** Where radiation carries a large share of the heat capacity, the damping $\chi_E/(2 + K)$ saturates once $K \gg 2$, and the energy channel caps $\phi$ near $\phi_0/(\sigma C_r)$ at the RSLA step. `StabilityBound(energy_step = true)` takes a shorter step (`energy_exchange_dt`, $\xi = 0.5$) at which thick cells reach $\phi = 1$. It buys accuracy, not stability (S3), and it never goes below the CFL step of the unreduced scheme, where $\phi = 1$ is stable everywhere: without that floor it asked for 9 times the substeps of running at $\hat{c} = c$ on `radhydro_pulse_dyn`, which is radiation-dominated at its centre.
- **The wave-speed correction of S4 is refused with a local limiter.** `RadiationPhysics` throws for it with any limiter but `ConstantChat`, because the bound does not hold for the corrected scheme. A face's correction is set by both cells beside it, so a thick cell bounded with its own correction ran at $\phi = 1$ behind an almost undamped face beside a thin one, and grew by about $10^{43}$ in 200 steps. In a uniform medium the lowered dissipation lets modes near a quarter of the grid frequency grow wherever the radiation's inertia matters, which the checkerboard bound does not see.

Two results quoted in `b287f03`. On `radhydro_shock_super`, the Mach 5 Lowrie & Edwards shock in the form of Quokka's `RadhydroShockSuper`, the ARSA scores 0.17% against 3.39% for the standard RSLA. And `examples/linear_waves_rhd.jl` seeds exact eigenmodes of the linearised radiation-hydrodynamic system and measures their complex frequencies: $\hat{c} = c$, the RSLA and the ARSA all agree with linear theory to about 1% or better from $\tau = 0.1$ to $10^3$.

### A2. The headline: a local $\phi$ removes the $\hat{c}$ dependence of the shock

Lowrie & Edwards Mach 3 radiative shock, $n = 256$, three shock-crossing times. The IRSLA column is the earlier limiter.

| $\phi_0$ | RSLA | IRSLA | gain | steps |
|---:|---:|---:|---:|---:|
| 0.0231 | 0.1129% | 0.1321% | 0.85$\times$ | 2726 / 2726 |
| 0.0069 | 0.6066% | 0.1307% | 4.64$\times$ | 2724 / 2726 |
| 0.0023 | 2.5124% | 0.1309% | 19.19$\times$ | 2721 / 2726 |

The interesting column is the third one, and what it shows is stronger than a gain factor: **the IRSLA error does not depend on $\phi_0$ at all**. It is 0.131% at every reduction, while the RSLA error grows by a factor of 22 across the same range. This is what the method is for, and it is a cleaner statement of the result than "19$\times$ better", which only says how far down the RSLA had fallen.

The first row is worth keeping in view. At $\phi_0 = 0.023$ the RSLA is *already* accurate — 0.113% — and the IRSLA is very slightly worse. There is a floor below which the limiter cannot help, set by the rest of the scheme, and above that floor there is nothing to win.

![The Mach 3 radiative shock at $\phi_0 = 0.0069$: the improved RSLA sits on the $\hat{c} = c$ reference through the precursor and the Zel'dovich spike, where the standard RSLA does not. The lower panel is the reduction factor the limiter chose. Drawn with the earlier limiter.](figures/rsla_limiter.png)

The multigroup Marshak wave of Vaytet et al., $n = 64$, four groups, is the harder case and behaves as the authors report.

| $\phi_0$ | RSLA | IRSLA | gain |
|---:|---:|---:|---:|
| 0.03 | 5.276% | 3.172% | 1.66$\times$ |
| 0.01 | 13.303% | 4.903% | 2.71$\times$ |
| 0.003 | 28.620% | 22.894% | 1.25$\times$ |

Roughly a factor of two, and the $\phi_0$ dependence survives. The authors' own explanation holds: the groups carrying the front's energy sit at $\tau_\mathrm{cell} \lesssim 1$, where the stability bound genuinely caps $\phi$, so no local limiter can make this problem $\hat{c}$-independent.

### A3. The bound was written with the wrong signal speed

This is the finding that matters most, because without it the method does not work at all on the Marshak wave.

The paper draft's transport-versus-damping channel is derived by balancing the amplification $2\nu_\phi/\sqrt{3} - 1$ of one upwind transport substep against the damping $1/(1 + \nu_\phi \tau_\mathrm{cell})$ of the implicit source substep that follows it, giving

$$\nu_\phi\left(\frac{2}{\sqrt{3}} - \tau_\mathrm{cell}\right) \leq 2 .$$

The $2/\sqrt{3}$ is twice the P1 signal speed $c/\sqrt{3}$. That is correct under the Eddington closure, and correct under M1 wherever the radiation field is isotropic. It is **not** correct wherever M1 free-streams: there the Eddington factor rises to one, the HLL wave speed estimates become $\pm\hat{c}$ rather than $\pm\hat{c}/\sqrt{3}$, and the bound is too generous by exactly $\sqrt{3}$. This is the worst possible place to be too generous, because a free-streaming cell is an optically thin one and has no source damping to fall back on.

The consequence is not subtle. With the published bound, the Vaytet wave is 130% wrong at every $\phi_0$, and the slab — heated by a 1000 K wall — reaches 2158 K. The transport-only table in S3 confirms the mechanism independently: at $\hat{c} = c$ with no limiter the scheme fails between $\nu = 1.2$ and $1.5$, and $\nu = \sqrt{3}$, which the published bound permits as $\tau_\mathrm{cell} \to 0$, overflows.

The fix is to carry the closure's own signal speed into the bound. Writing $\sigma$ for the fastest signal as a fraction of $c$,

$$\nu_\phi\left(2\sigma - \tau_\mathrm{cell}\right) \leq 2 ,
\qquad
\sigma = \sqrt{\chi\!\left(|\boldsymbol{F}|/cE\right)} ,$$

evaluated per cell through the same `eddington_factor` the Riemann solver uses. It reduces to the published form exactly where the radiation is isotropic, and tightens by $\sqrt{3}$ where it beams. With this in place the Marshak wave goes from 130% to 3.2%, and every number in A2 is measured with it.

Two side effects of the correction are worth naming. The cell optical depth beyond which no reduction is needed becomes $\tau_\mathrm{cell} \geq 2\sigma$ rather than $2/\sqrt{3} \approx 1.15$, so it is $2$ in a free-streaming cell and $1.15$ in an isotropic one. And the transparent-cell limit becomes $\phi \leq \phi_0/(\sigma\, C_r)$, which is the ordinary Courant condition at the true signal speed rather than at the P1 one.

The current bound (A1) keeps the correction and goes one step further: it takes $\sigma_d = \max(0.1, \sqrt{T_{dd}})$ per direction, the speed the Riemann solver is handed along $d$, and sums $\sigma_d/\Delta x_d$ (`_signal_sum`). A cell that beams along one axis in two or three dimensions is then not bounded as if it beamed along all of them. In one dimension, while $K \ll 2$, A1's energy block reduces to the form above.

### A4. Accuracy is not monotonic in $\phi$ — here, but not in Quokka

**Cross-check, 2026-08-19: this section does not reproduce in Quokka and should not be carried into the paper.** The same scan run on Quokka's Mach 5 supercritical shock is cleanly monotone — $L_1$ errors of 1.29%, 0.50%, 0.23%, 0.12% and 0.068% at $\beta = 0.3, 0.5, 0.7, 0.9, 1.0$, then 172% at $\beta = 1.2$ — so the premise holds there and the stability cliff sits exactly at $\beta = 1$ where the derivation puts it. What follows is a property of this code, or of the Mach 3 problem, and the mechanism proposed below is a conjecture that the Quokka result argues against. It is kept because the difference between the two codes is itself worth understanding.

The paper's argument for taking the largest stable $\phi$ rests on accuracy increasing monotonically with $\phi$. On the shock, in Honghu with the earlier limiter, it does not. Scanning the safety factor $\beta$, which scales both stability channels:

| $\beta$ | shock $L_1$ | Vaytet $L_1$ | Vaytet max $T$ |
|---:|---:|---:|---:|
| 0.3 | 0.1233% | 13.301% | 932 K |
| 0.6 | **0.0590%** | 12.346% | 934 K |
| 0.9 | 0.1307% | 4.903% | 937 K |
| 1.0 | 0.2165% | **3.663%** | 936 K |
| 1.5 | 290.5% | 412.1% | 2947 K |
| 2.0 | 301.5% | 460.3% | 3272 K |

On the Marshak wave accuracy does improve monotonically up to $\beta = 1$, as the argument predicts. On the shock it has a clear minimum at $\beta = 0.6$, and $\beta = 1$ is three and a half times worse than that.

A plausible mechanism is that the HLL flux carries a numerical diffusion proportional to the signal speed, so raising $\phi$ in a transparent cell raises the numerical diffusion there. The accuracy argument counts only the physical error the RSLA introduces and does not see this. If that is right, the optimal $\phi$ is a balance between two errors rather than the stability bound itself, and the balance point is problem-dependent — which would make $\beta$ a real tuning parameter rather than a safety margin, and would weaken the "no tunable physics parameters" claim.

The cliff between $\beta = 1$ and $\beta = 1.5$ is sharp and is where the derivation says it should be: $\beta = 1$ sits exactly on the derived bound. That part of the theory is confirmed. Note also that $\beta \geq 1.5$ on the shock costs 9$\times$ the steps as well as being wrong, so it fails loudly there; on the Marshak wave it runs to completion at the same cost and is silently 412% wrong, which is the failure mode the stability-bound formulation exists to exclude.

### A5. A local $\phi$ loses the RSLA's identity, and one $\phi$ per group is the better choice

The paper gives each group its own $\phi$, "which poses no difficulty because the groups form independent hyperbolic subsystems coupled only through the common matter state". The groups are independent in transport. They are not independent in the coupling: the identity of S2 needs $\phi$ uniform over the groups, and with $\phi$ varying between groups the sum does not collapse. Measured in a single cell with no transport at all, relaxing over 400 coupling steps: at $\phi = (1, 0.1)$ the identity predicts no drift whatever — the gas equation is unmodified and $\phi_1 = 1$ — and the measured drift is $-44.5\%$ of the cell's initial energy. At $\phi = (1, 0.01)$ it is $-56.7\%$. Exchanging which group is the fast one changes the answer again.

This was first read as an argument for one $\phi$ per cell, and the earlier limiter defaulted to that, taken from the most restrictive group, on the grounds that a conservation property one can write down is worth a few per cent of $L_1$ error. **That reasoning was reversed in `b287f03`.** The identity needs $\phi$ uniform over cells as well, so any local limiter loses it, per cell or per group. What a local $\phi$ keeps, in the time-derivative form of A1, is exact conservation of $e + \sum_g E_g/\phi_g$: the coupling conserves it cell by cell, and the flux differences telescope in $E_g/\phi_g$. That holds for a per-group $\phi$ exactly as for a per-cell one, and the von Neumann analysis finds both stable for the coupled multigroup system. The choice therefore comes down to accuracy, and on the Vaytet wave per-group is the more accurate, with both limiters:

| $\phi_0$ | earlier limiter, per cell | earlier limiter, per group | current limiter, per cell | current limiter, per group |
|---:|---:|---:|---:|---:|
| 0.03 | 3.172% | 3.975% | 4.3% | 2.7% |
| 0.01 | 4.903% | 2.830% | 11.5% | 7.8% |
| 0.003 | 22.894% | 18.579% | 26.0% | 21.2% |

The current-limiter columns are those of the `per_group` docstring in `src/rsla.jl`. With the earlier limiter per-group was better at two of the three reductions; with the current one it is better at all three, at the same cost. So `StabilityBound` now always uses one $\phi$ per group, and only `OpticalDepthFit` keeps the `per_group` switch, to reproduce earlier results. This is also what Quokka does (A7).

### A6. The advantage grows under refinement

The open question in the project's handoff notes is what happens as the grid is refined, since $\tau_\mathrm{cell} \propto \Delta x$ moves cells from the thick side of the bound to the thin side and leaves the limiter less room. The expectation is that the advantage shrinks. On the shock, at $\phi_0 = 0.0069$ with the earlier limiter, it does the opposite.

| $n$ | RSLA | IRSLA | gain |
|---:|---:|---:|---:|
| 64 | 0.6491% | 0.2164% | 3.00$\times$ |
| 128 | 0.5379% | 0.1755% | 3.07$\times$ |
| 256 | 0.6066% | 0.1307% | 4.64$\times$ |
| 512 | 0.6742% | 0.0624% | 10.81$\times$ |

The IRSLA error falls by a factor of 3.5 from $n = 64$ to $n = 512$ while the RSLA error is flat, or slightly rising. That is the more important statement: **the IRSLA converges under refinement and the RSLA does not.** The RSLA's error is a modelling error, not a discretisation error (S1), so refining the grid cannot remove it; the IRSLA reduces it to something that refines away. The gain factor grows because the denominator shrinks, not because the numerator grows.

### A7. What happened when these were checked against Quokka

All of A3 and A5 were confirmed by reading Quokka's `ComputeReducedSpeedOfLightFactor` and its multigroup source term, and the A3 correction was implemented and measured there. Quokka was then at `0bf3741b` on `chong/iRSLA-framework-202607-improved`, with `radiation.variable_chat_legacy` restoring the published behaviour bit by bit.

Confirmed and fixed in Quokka:

| | effect on Quokka's own tests |
|:---|:---|
| the $P_1$ signal speed in the bound (A3) | supercritical shock 0.159% $\to$ 0.116%; Vaytet at $\phi_0 = 0.01$, 8.30% $\to$ 3.65% |
| the bound evaluated at the nominal rather than the actual CFL number (A8, second point) | Vaytet 3.65% $\to$ 1.97% with $\sigma$ also corrected; **57%** with $\sigma$ left alone |
| $\beta$ multiplying an absent bound, capping $\phi$ at 0.9 in thick cells | no measurable effect; makes the paper's conservation claim true |
| the interface reading a cell the face does not touch (new, found in Quokka) | $\leq 0.2\%$ on the Vaytet wave, zero on the shock |

The second row is the important one. Quokka's Vaytet runs stepped at $C_r = 0.4$ while evaluating the bound at $0.3$, so every $\phi$ exceeded the derivation by a third. Correcting that alone — unambiguously the right thing to do in isolation — takes the error from 8.3% to 57% and drives the slab to 1118 K under a 1000 K wall. The two errors were partly cancelling, which is why the published campaign looked healthy, and why they have to be fixed together.

Found and confirmed, but **not** acted on in Quokka: the per-group $\phi$ of A5. The identity fails there exactly as it does here — Quokka's multigroup source term carries `cscale[g] = 1/phi_g` — but per-group is *slightly more accurate* on the Vaytet wave, so it stayed a decision rather than a bug. Honghu has since made the same decision (A5).

Not reproduced: the non-monotonicity of A4, as noted there.

### A8. Smaller points

**The paper draft contradicted itself on the interface value.** Its Section 5 transport paragraph defined $\phi_{i+1/2} = \min(\phi_i, \phi_{i+1})$ and argued for it at length — an optically thick cell must not impose a CFL-violating signal speed on a transparent neighbour that has no source damping to control the amplification. The algorithm summary at the end of the same section said $\hat{c}_{i+1/2} = \max(\phi_i, \phi_{i+1})c$. The argument is right and the summary was a typo, and Honghu implemented the minimum. Since `b287f03` the question no longer arises in the default form: with $\phi$ on the time derivative each cell scales its own update and no interface value is needed (A1). The minimum remains in the flux form (`flux_form = true`) and under the standard RSLA, where both cells hold the same value, and `test/test_rsla.jl` asserts it there.

**Evaluate the bound at the step actually being taken.** The paper writes the bound in terms of the code's nominal radiation CFL number $C_r$, on the assumption that $\Delta t = C_r \Delta x/(\phi_0 c)$. When the step is shorter than that — because hydrodynamics limited it, or because subcycling divided it unevenly — the nominal form understates how much $\phi$ is available. Honghu forms the bound from the actual $\Delta t$ (`update_chat!` is passed the substep), which reduces to the nominal form whenever the radiation CFL is what binds and is never less conservative than it. One consequence: `run!` shortens the last step to land on `tmax`, so $\phi$ at the end of a run is larger than it was during the run; `examples/irsla_paper.jl` reports $\phi$ from the last full step.

The radiation-force step constraint, which the earlier version of this section discussed, is in S3: `b287f03` removed it.

### A9. What would be worth doing next

The non-monotonicity in A4 is the loose end, with the caveat that Quokka does not show it and it was measured with the earlier limiter. If the mechanism really is the HLL numerical diffusion, then the accuracy argument that justifies "take the largest stable $\phi$" needs a second term, and the right local $\phi$ is where the two errors balance. That balance is computable: the RSLA error scales as $(1-\phi)$ times something the coupling already knows, and the HLL diffusion scales as $\phi\, c\, \Delta x$ times a gradient the transport step already forms. A limiter built on the balance rather than on the bound alone would be a genuinely different rule, and A4's table says there is roughly a factor of two in it on the shock.

$\sigma$ is evaluated from the state at the start of the IMEX substep and held through both stages, as the bound assumes. In a cell whose radiation field is beaming up rapidly, the state at the end is what the step has to survive.

An item listed here before is done. The earlier closed-form bound missed the damping from the energy exchange, and the paper draft noted that the true boundary reached $\phi = 1$ at $\tau_\mathrm{cell} \approx 0.8$ rather than at $2\sigma$. The bound of A1 is the exact boundary of the scheme, so there is no gap left between the closed form and the stability limit.
