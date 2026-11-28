# Standard and adaptive reduced speed of light approximations

This note records measurements of the reduced speed of light in Honghu in two parts. **Standard RSLA** covers the global reduction $\phi_0 = \hat{c}/c$, which is the default (`ConstantChat()`). **ARSA** covers the adaptive reduced speed of light approximation of He, Krumholz & Wibking, "An Adaptive Reduced Speed of Light Approximation" ([arXiv:2611.21353](https://scixplorer.org/abs/2026arXiv261121353H/abstract)), for which Honghu is the reference implementation (named IRSLA in the code, with limiter `StabilityBound()`). The implementation lives in `src/rsla.jl` and the array drivers of `src/radiation_system.jl` (`update_chat!`, `scale_by_phi!`, `floor_rad_energy!`, `energy_exchange_dt`), tested in `test/test_rsla.jl`.

Errors are reported as the relative $L_1$ error in gas temperature against a $\hat{c} = c$ reference at the same resolution and (unless noted otherwise) at the same step count as the standard RSLA, so every comparison is at equal cost.

The ARSA took its present form in commit `b287f03` (the squash of branch `irsla-variable-chat-stage2`, archived at tag `archive/irsla-variable-chat-stage2`). Section A1 summarises what that commit changed and quotes its measurements; `examples/irsla_paper.jl` reproduces them along with all numbers in Section 5 of the paper. Sections A2 to A9 record the earlier experiments that led to `b287f03`, measured with the prior limiter (which defined the cell optical depth as $\chi\,\Delta x_\mathrm{min}$, used a single $\phi$ per cell, and placed $\phi$ inside the flux divergence). Because `examples/rsla_limiter.jl` now runs the current limiter and `b287f03` also updated pre-solve floor repairs for all limiters, re-running that script today produces the current limiter's numbers rather than the historical tables in A2 to A9. Where a design choice from those earlier sections was later reversed, the text notes it explicitly.

## Standard RSLA

### S1. Its error is a modelling error that does not refine away

The standard RSLA multiplies the radiation source terms by $\phi_0$ while leaving the matter source terms unchanged, shifting the intermediate equilibrium temperature between gas and radiation. The error is largest at high optical depth, where implicit coupling already damps the stiff terms and no speed-of-light reduction is needed for stability. As $\phi_0$ decreases, the error grows: on the Lowrie & Edwards Mach 3 radiative shock at $n = 256$, it rises from 0.11% at $\phi_0 = 0.023$ to 2.5% at $\phi_0 = 0.0023$, a factor of 22 (table in A2). Because this error comes from the modified equations rather than the spatial discretisation, refining the grid does not reduce it: on the same shock at $\phi_0 = 0.0069$, it remains between 0.54% and 0.67% from $n = 64$ to $n = 512$ (table in A6).

### S2. With a uniform $\phi$, the energy error depends only on the net gas energy change

Summing the source terms of the gas and the radiation groups gives

$$\frac{d}{dt}\left[e + \sum_g E_g\right] = c\,\rho \sum_g \left(\phi_g - 1\right)\left(\kappa_P B_g - \kappa_E E_g\right) .$$

When $\phi$ is uniform across groups and cells, the sum collapses onto the gas energy equation, leaving the exact identity

$$\Delta\left[e + \sum_g E_g\right] = \left(1 - \phi\right) \Delta e .$$

Thus the total energy drift over a step or a run depends solely on the net change in gas energy $\Delta e$, regardless of the spectrum, the number of absorption and re-emission cycles, or which frequency groups carried the energy. In a single-cell relaxation test over 400 coupling steps with no transport, this identity holds to five decimal places at both $\phi = 0.1$ and $\phi = 0.01$.

The identity requires a globally uniform reduction. When $\phi$ varies across cells or groups, the sum no longer collapses; A5 measures the resulting drift and states the modified invariant that a local $\phi$ conserves instead.

### S3. Time step: closure signal speed and removal of the radiation-pressure limit

The stable transport step is set by the closure's actual signal speed rather than the P1 speed $c/\sqrt{3}$. Where M1 free-streams, the Eddington factor approaches 1 and the HLL wave speeds reach $\pm\hat{c}$. The table below runs the Vaytet multigroup Marshak wave ($n = 64$, four groups) at $\hat{c} = c$ without a limiter, isolating the explicit transport step across four values of the radiation CFL number $\nu$:

| reconstruction | $\nu = 0.8$ | $\nu = 1.2$ | $\nu = 1.5$ | $\nu = \sqrt{3}$ |
|:---|---:|---:|---:|---:|
| PCM | 2.51% | 2.61% | 32.6% | 79.8% |
| PLM | 0.39% | 0.93% | 139.6% | overflow |
| PPM | 0.05% | 0.65% | 128.7% | overflow |

The scheme goes unstable between $\nu = 1.2$ and $\nu = 1.5$ for all three reconstructions, including first-order PCM, showing that the limit comes from the M1 closure's wave speed rather than the reconstruction order. At $\nu = \sqrt{3}$ (unit Courant number for the P1 wave speed), PLM and PPM overflow. Section A3 shows how this affects the ARSA stability bound. For the standard RSLA, it sets the maximum stable Courant number: the paper runs in `examples/irsla_paper.jl` use a safety factor $\beta = 0.9$, giving $C_r = 0.9\sqrt{3}$ for P1 and $C_r = 0.9\sqrt{D}$ for M1 (in Honghu's convention $C_r = D\,\nu_0$).

The standard RSLA does not require an additional timestep restriction from radiation pressure at any $P_\mathrm{rad}/P_\mathrm{gas}$. In the exact stability bound of A1 evaluated at $\phi = \phi_0$, the left-hand side is bounded by the CFL number times $\sigma$ regardless of the heat capacities, and the checkerboard test in `test/test_rsla.jl` confirms that checkerboard modes decay at the standard RSLA step for $\mathcal{P} = 1$ and $100$ in 1D and 2D. Commit `b287f03` therefore removed the earlier `radiation_pressure_dt` constraint (from eq. 33 of the initial paper draft), which had previously increased the step count on the static Vaytet Marshak wave by $5.4\times$ without changing the solution.

### S4. The wave-speed correction applies only to the standard RSLA

`RadiationPhysics(wavespeed_correction = true)` scales the HLL numerical dissipation of the radiation energy at each face by $\epsilon = \min(1, 1/\tau)$, where $\tau$ is the harmonic mean of the two adjacent cells' optical depths (Skinner et al. 2019; optional in Quokka). In optically thick cells, unmodified HLL dissipation introduces numerical diffusion that exceeds the physical diffusion rate $c/(3\tau)$, and the $\epsilon$ factor suppresses it. Honghu restricts this correction to `ConstantChat`; section A1 explains why it is incompatible with a local limiter.

## ARSA

### A1. The current formulation (`b287f03`)

Every cell and frequency group runs at the largest reduced speed of light that the IMEX PD-ARS step supports stably at the timestep of the standard RSLA:

$$\phi = \min\left\{1,\ \max\left[\phi_0,\ \frac{\beta}{c\,\Delta t\,\Lambda}\right]\right\},$$

where $\Lambda$ is `stability_denominator` and $\beta = 0.9$ by default. Relative to the earlier limiter used in A2 through A9, `b287f03` changed the following:

- The stability bound is the exact von Neumann boundary in $D$ dimensions. The grid-scale checkerboard is always the most unstable mode, where the linearised update decouples into an energy block and a momentum block, each bounded by $$\phi\,c\,\Delta t\left[\sum_d \frac{\sigma_d}{\Delta x_d} - \frac{\chi}{2 + K}\right] \leq 1 ,$$ with $\chi = \rho\kappa_E$ and $K = 4 a_R T^3 \kappa_P\, c\,\Delta t / c_v$ for the energy block, and $\chi = \rho\kappa_F$ (with $K$ smaller by $(a/c)^2$ and omitted) for the momentum block; $\Lambda$ is the larger of the two brackets. In 1D, 2D, and 3D, a checkerboard mode decays at $0.9\times$ the bound, grows at $1.1\times$ the bound, and sits at $|g| = 1$ to within $10^{-3}$ at $\beta = 1$. The earlier limiter defined the cell optical depth as $\chi\,\Delta x_\mathrm{min}$ rather than $\chi / \sum_d (1/\Delta x_d)$, which is $D$ times too permissive in $D$ dimensions and caused checkerboard modes to grow by $1.06\times$ to $1.88\times$ per substep in 2D and 3D. When radiation energy is significant, the heat-capacity term $K$ in the energy block is the active constraint. Signal speeds $\sigma_d$ are evaluated per direction (A3).
- $\phi$ multiplies the time derivative rather than the flux divergence: each cell's transport update multiplies the unreduced HLL flux difference by the cell's own $\phi_i$. Steady states then satisfy $\nabla\cdot\boldsymbol{F} = S$ for any spatial profile $\phi(x)$, the flux differences telescope in the conserved quantity $E/\phi$, and no face-interpolated $\phi_{i+1/2}$ is required. Placing $\phi$ inside the divergence (as in earlier drafts) adds a spurious $\boldsymbol{F}\cdot\nabla\ln\phi$ term to the steady state; on the Mach 5 supercritical shock, a 5% ripple in $\phi$ raised the $L_1$ error from 0.19% to 2.3%, worse than the standard RSLA. For uniform $\phi$ the two forms are identical; `flux_form = true` retains the divergence form for comparison.
- `StabilityBound` always uses one $\phi$ per group (`per_group` is true; see A5), computed once at the start of each IMEX substep and held fixed across both stages to match the von Neumann analysis. Recomputing $\phi$ between stages would leave an uncancelled $(E^{(2)} - E^n)(1/\phi_b - 1/\phi_a)$ conservation residual.
- Pre-solve state repair is restricted to non-positive total energy budgets. An optically thick cell with $\phi = 1$ steps at a local transport Courant number well above unity, so its explicit transport stage can produce $|\boldsymbol{F}| \gg cE$ or (at steep jumps in $\phi$) $E < 0$. The implicit coupling solve divides the flux by $1 + \hat{c}\rho\kappa_F\Delta t$, recovering the diffusion flux, and yields a positive temperature whenever $e + \sum_g E_g/\phi_g > 0$. Clipping $|\boldsymbol{F}|$ to $cE$ before the solve discarded the diffusion flux and raised the thick hot-spot error from 0.0003% to 0.80% (versus 0.37% for the standard RSLA). Flooring $E$ unconditionally before the solve injected spurious energy every substep, making the 64-cell Ensman supercritical shock $17\times$ less accurate than the standard RSLA. Honghu now floors $E$ before the solve only when $e + \sum_g E_g/\phi_g \le 0$, deferring full flux and energy repair (`amend_rad_state!`) until after the solve. `Diagnostics` tracks the two counts separately as `floors_presolve` and `repairs_postsolve`.
- The energy-exchange timestep (`energy_step = true`) is opt-in. When radiation dominates the heat capacity ($K \gg 2$), the energy-block damping $\chi_E/(2 + K)$ saturates and caps $\phi$ near $\phi_0/(\sigma C_r)$ at the standard RSLA step. Passing `StabilityBound(energy_step = true)` enables `energy_exchange_dt` ($\xi = 0.5$), which shortens the step enough for optically thick cells to reach $\phi = 1$, floored at the unreduced ($\hat{c} = c$) CFL step so that radiation-dominated runs like `radhydro_pulse_dyn` never subcycle finer than $\hat{c} = c$.
- `RadiationPhysics(wavespeed_correction = true)` throws an error unless the limiter is `ConstantChat`. Because the face factor $\epsilon$ depends on both adjacent cells, a thick cell next to a thin cell had its damping reduced by the shared face while still running at $\phi = 1$, growing by $\sim 10^{43}$ in 200 steps; in uniform media, reduced HLL dissipation also destabilises intermediate-wavelength modes ($\sim \lambda = 4\Delta x$) that the checkerboard bound does not capture.

On `radhydro_shock_super` (the Mach 5 Lowrie & Edwards supercritical shock matching Quokka's `RadhydroShockSuper`), the current ARSA achieves an $L_1$ error of 0.17% versus 3.39% for the standard RSLA. In `examples/linear_waves_rhd.jl`, complex eigenfrequencies of linearised radiation-hydrodynamic waves match exact linear theory to within ~1% across $\tau = 0.1$ to $10^3$ for $\hat{c} = c$, standard RSLA, and ARSA alike.

### A2. Removing the $\hat{c}$ dependence on the radiative shock

Lowrie & Edwards Mach 3 radiative shock, $n = 256$, evolved for three shock-crossing times (IRSLA column measured with the pre-`b287f03` limiter):

| $\phi_0$ | RSLA | IRSLA | gain | steps |
|---:|---:|---:|---:|---:|
| 0.0231 | 0.1129% | 0.1321% | 0.85$\times$ | 2726 / 2726 |
| 0.0069 | 0.6066% | 0.1307% | 4.64$\times$ | 2724 / 2726 |
| 0.0023 | 2.5124% | 0.1309% | 19.19$\times$ | 2721 / 2726 |

The IRSLA error stays constant at 0.131% across all three values of $\phi_0$, while the standard RSLA error grows by a factor of 22 over the same range. The "gain" ratio simply reflects how far the standard RSLA degrades as $\phi_0$ decreases. At $\phi_0 = 0.0231$, the standard RSLA error (0.113%) is already at the spatial discretisation floor of the scheme, so raising $\phi$ locally cannot reduce it further.

![The Mach 3 radiative shock at $\phi_0 = 0.0069$: the improved RSLA sits on the $\hat{c} = c$ reference through the precursor and the Zel'dovich spike, where the standard RSLA does not. The lower panel is the reduction factor the limiter chose. Drawn with the earlier limiter.](figures/rsla_limiter.png)

On the multigroup Marshak wave of Vaytet et al. ($n = 64$, four groups), the error improves by roughly a factor of two while retaining a $\phi_0$ dependence:

| $\phi_0$ | RSLA | IRSLA | gain |
|---:|---:|---:|---:|
| 0.03 | 5.276% | 3.172% | 1.66$\times$ |
| 0.01 | 13.303% | 4.903% | 2.71$\times$ |
| 0.003 | 28.620% | 22.894% | 1.25$\times$ |

Here the groups carrying the front's energy sit at $\tau_\mathrm{cell} \lesssim 1$, where the stability bound genuinely restricts $\phi$, so a stability-limited local $\hat{c}$ cannot eliminate the $\phi_0$ dependence.

### A3. Using the M1 signal speed in the stability bound

In the initial paper draft, the transport-versus-damping bound balanced the amplification $2\nu_\phi/\sqrt{3} - 1$ of an upwind P1 step against the implicit damping $1/(1 + \nu_\phi \tau_\mathrm{cell})$, yielding

$$\nu_\phi\left(\frac{2}{\sqrt{3}} - \tau_\mathrm{cell}\right) \leq 2 .$$

The factor $2/\sqrt{3}$ uses the isotropic P1 wave speed $c/\sqrt{3}$. Under the M1 closure in free-streaming regions, the Eddington factor approaches 1 and the HLL wave speeds reach $\pm\hat{c}$, making the P1 bound too permissive by a factor of $\sqrt{3}$ in optically thin cells where source damping is absent. With the $2/\sqrt{3}$ bound, the Vaytet Marshak wave has an $L_1$ error of 130% at every $\phi_0$, and the slab heated by a 1000 K boundary overshoots to 2158 K (consistent with the $\nu = \sqrt{3}$ overflow in S3).

Replacing $1/\sqrt{3}$ with the closure's signal speed $\sigma$ gives

$$\nu_\phi\left(2\sigma - \tau_\mathrm{cell}\right) \leq 2 ,
\qquad
\sigma = \sqrt{\chi\!\left(|\boldsymbol{F}|/cE\right)} ,$$

evaluated per cell from `eddington_factor`. This recovers $1/\sqrt{3}$ in isotropic regions and tightens by $\sqrt{3}$ in free-streaming beams, reducing the Vaytet Marshak wave error from 130% to 3.2% (the value reported in A2). Consequently, the optical depth threshold above which $\phi = 1$ is unconditionally stable shifts from $2/\sqrt{3} \approx 1.15$ to $2\sigma$ (up to $2$ in a beam), and the optically thin limit becomes $\phi \leq \phi_0/(\sigma\, C_r)$.

The current bound in A1 refines this further by evaluating $\sigma_d = \max(0.1, \sqrt{T_{dd}})$ along each active axis $d$ and summing $\sigma_d/\Delta x_d$ (`_signal_sum`), so a beam aligned with one axis in 2D or 3D only tightens the bound along that axis. In 1D with $K \ll 2$, A1's energy block reduces to the formula above.

### A4. Non-monotonicity in $\beta$ on the Mach 3 shock (not seen in Quokka)

**Cross-check (2026-08-19):** Running the same $\beta$ scan on Quokka's Mach 5 supercritical shock gives a monotonically decreasing $L_1$ error right up to the stability limit: 1.29%, 0.50%, 0.23%, 0.12%, and 0.068% at $\beta = 0.3, 0.5, 0.7, 0.9, 1.0$, followed by instability (172%) at $\beta = 1.2$. The non-monotonicity below is specific to the Mach 3 setup or the earlier limiter in Honghu and does not carry over to Quokka.

Scanning the safety factor $\beta$ in Honghu with the earlier limiter gives:

| $\beta$ | shock $L_1$ | Vaytet $L_1$ | Vaytet max $T$ |
|---:|---:|---:|---:|
| 0.3 | 0.1233% | 13.301% | 932 K |
| 0.6 | **0.0590%** | 12.346% | 934 K |
| 0.9 | 0.1307% | 4.903% | 937 K |
| 1.0 | 0.2165% | **3.663%** | 936 K |
| 1.5 | 290.5% | 412.1% | 2947 K |
| 2.0 | 301.5% | 460.3% | 3272 K |

On the Marshak wave, error decreases monotonically up to $\beta = 1$. On the Mach 3 shock with the earlier limiter, error reaches a minimum at $\beta = 0.6$ before rising to 0.217% at $\beta = 1.0$ (possibly because higher $\phi$ in optically thin cells increases HLL numerical diffusion, which scales with the wave speed). Above $\beta = 1$, both problems go unstable between $\beta = 1.0$ and $1.5$, matching the analytical stability boundary; on the Marshak wave, $\beta = 1.5$ completes in the same number of steps while producing a 412% temperature error.

### A5. Conservation and accuracy of per-group versus per-cell $\phi$

Although frequency groups are independent during explicit transport, they couple through the shared gas temperature during the implicit source step. When $\phi_g$ varies across groups, the energy identity of S2 no longer collapses. In a single-cell two-group relaxation test over 400 steps without transport, setting $\phi = (1, 0.1)$ produces a $-44.5\%$ drift in $e + \sum_g E_g$, and $\phi = (1, 0.01)$ produces $-56.7\%$.

Initially, this led us to default to a single $\phi$ per cell (taking the minimum over groups). **Commit `b287f03` reversed that choice.** Because the S2 identity also requires $\phi$ to be uniform across space, any spatially varying limiter already loses conservation of $e + \sum_g E_g$. Instead, the time-derivative formulation of A1 exactly conserves the weighted energy $e + \sum_g E_g/\phi_g$, both in the local coupling step and across cell interfaces, whether $\phi$ is chosen per cell or per group. Since both choices are stable under the coupled multigroup von Neumann analysis, the choice is decided by accuracy on multigroup problems such as the Vaytet Marshak wave:

| $\phi_0$ | earlier limiter, per cell | earlier limiter, per group | current limiter, per cell | current limiter, per group |
|---:|---:|---:|---:|---:|
| 0.03 | 3.172% | 3.975% | 4.3% | 2.7% |
| 0.01 | 4.903% | 2.830% | 11.5% | 7.8% |
| 0.003 | 22.894% | 18.579% | 26.0% | 21.2% |

With the current limiter (`src/rsla.jl`), per-group $\phi$ is more accurate at all three $\phi_0$ values at identical cost. `StabilityBound` therefore always uses one $\phi$ per group (matching Quokka; see A7), while `OpticalDepthFit` retains a `per_group` keyword to reproduce older runs.

### A6. Behaviour under grid refinement

Because $\tau_\mathrm{cell} \propto \Delta x$ decreases as the grid is refined, cells move toward the optically thin regime where the limiter has less headroom to raise $\phi$. Even so, refining the Mach 3 shock at $\phi_0 = 0.0069$ (measured with the earlier limiter) increases the relative advantage over the standard RSLA:

| $n$ | RSLA | IRSLA | gain |
|---:|---:|---:|---:|
| 64 | 0.6491% | 0.2164% | 3.00$\times$ |
| 128 | 0.5379% | 0.1755% | 3.07$\times$ |
| 256 | 0.6066% | 0.1307% | 4.64$\times$ |
| 512 | 0.6742% | 0.0624% | 10.81$\times$ |

From $n = 64$ to $n = 512$, the IRSLA error drops by a factor of 3.5 as the grid resolves the shock, whereas the standard RSLA error stalls around 0.6% because its equilibrium mismatch is a resolution-independent modelling error (S1).

### A7. Cross-checks and fixes in Quokka

Checking A3 and A5 against Quokka's `ComputeReducedSpeedOfLightFactor` and multigroup source term (`0bf3741b` on `chong/iRSLA-framework-202607-improved`, with `radiation.variable_chat_legacy` preserving the prior behaviour) confirmed four issues:

| Issue | Effect on Quokka's tests |
|:---|:---|
| P1 signal speed in the bound (A3) | supercritical shock 0.159% $\to$ 0.116%; Vaytet at $\phi_0 = 0.01$, 8.30% $\to$ 3.65% |
| bound evaluated at nominal rather than actual CFL (A8) | Vaytet 3.65% $\to$ 1.97% with $\sigma$ also fixed, versus **57%** if fixed without $\sigma$ |
| $\beta$ multiplying an unreduced cap, limiting thick cells to $\phi = 0.9$ | negligible $L_1$ change; restores exact $\phi = 1$ in thick cells |
| face interpolation indexing a non-adjacent cell (Quokka-specific bug) | $\leq 0.2\%$ on Vaytet wave, zero on shock |

In the second row, Quokka's Vaytet setup ran at $C_r = 0.4$ while evaluating the stability bound with $C_r = 0.3$, allowing $\phi$ to exceed the bound by a third. Fixing only the CFL mismatch pushed $\phi$ into the unstable region left open by the P1 signal speed, raising the error from 8.3% to 57% (and the slab temperature to 1118 K under a 1000 K boundary); fixing both the CFL input and the M1 signal speed reduced the error to 1.97%.

Quokka also uses per-group $\phi_g$ (`cscale[g] = 1/phi_g`), which both codes now retain for accuracy on multigroup problems (A5).

### A8. Additional details

**Interface value in the flux-divergence form.** Section 5 of the initial paper draft defined $\phi_{i+1/2} = \min(\phi_i, \phi_{i+1})$ in the prose (so a thick cell cannot impose a CFL-violating wave speed on an adjacent thin cell) while writing $\hat{c}_{i+1/2} = \max(\phi_i, \phi_{i+1})c$ in the end-of-section summary table. With $\phi$ on the time derivative (`flux_form = false`, the default since `b287f03`), no face value is needed; when `flux_form = true`, Honghu uses $\min(\phi_i, \phi_{i+1})$, verified in `test/test_rsla.jl`.

**Evaluating the bound with the actual substep $\Delta t$.** Writing the bound in terms of the nominal radiation CFL number $C_r$ assumes $\Delta t = C_r \Delta x/(\phi_0 c)$. Whenever $\Delta t$ is smaller (because hydrodynamics restricted the step or subcycling divided it unevenly), using the actual $\Delta t$ passed to `update_chat!` allows a larger stable $\phi$. Because `run!` shortens the final step to hit `tmax` exactly, $\phi$ on that final step is larger than during the rest of the run; `examples/irsla_paper.jl` therefore records $\phi$ from the last full step.

### A9. Open questions

If the Mach 3 non-monotonicity in A4 comes from HLL numerical diffusion growing with $\phi\,c\,\Delta x$ in thin cells while the RSLA source error scales with $(1-\phi)$, choosing $\phi$ to minimise their sum (subject to the stability bound) could improve accuracy below the stability limit. In addition, the directional signal speed $\sigma_d$ is currently evaluated at the start of the IMEX substep and held fixed across both stages; in a cell that transitions from isotropic to free-streaming within a single step, the end-of-step wave speed exceeds the start-of-step estimate.

