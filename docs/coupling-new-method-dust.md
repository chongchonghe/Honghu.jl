# Dust-gas thermal coupling with the one-variable method: implementation plan

Status: **implemented, 2026-09-28.** All five recommendations in §8 were accepted as written. §9 says what was built and what it measured. The math in §2–§5 was first checked by a prototype, `docs/coupling-experiments/dust_prototype.jl`, and §6 gives its numbers.

Scope: the thermal coupling of gas, dust and multigroup radiation, with the Bate & Keto (2015) dust model as used in the draft `external/quokka-dust-gas-tex/main.tex` §2.1 and in Quokka's `SolveGasDustRadiationEnergyExchange`. Photoelectric heating, cosmic-ray heating and line cooling are out of scope. The numerical method is **not** the draft's: the draft (and Quokka) runs a $(1+N_G)$ Newton iteration in $(E_{\rm gas}, R_g)$ with a regime switch between "well-coupled" and "decoupled" dust. This plan reduces the step to one scalar equation in the dust temperature, solved inside a bracket, in the same way `BracketedCoupling` reduces the dust-free step to one equation in $E_{\rm gas}$ (`docs/coupling-new-method.md`).

## 1. The physics

Radiation interacts with the dust only. The gas has no opacity. The dust has no heat capacity, so at every instant it is in energy balance: what it gains from the gas by collisions it loses to the radiation. The three equations are (draft eqs. 2.1, 2.2, dust1 and Td, without transport)

$$\frac{\partial E_{\rm gas}}{\partial t} = -\Lambda_{\rm gd},$$

$$\frac{\partial E_g}{\partial t} = \hat{c}\left[\rho\kappa_{P,g}(T_d)\,\frac{4\pi B_g(T_d)}{c} - \rho\kappa_{E,g}(T_d)\,E_g\right],$$

$$\Lambda_{\rm gd} = c\sum_g \left[\rho\kappa_{P,g}(T_d)\,\frac{4\pi B_g(T_d)}{c} - \rho\kappa_{E,g}(T_d)\,E_g\right],$$

with the collisional rate of Keto & Field (2005)

$$\Lambda_{\rm gd} = \Lambda^0_{\rm gd}\, n_{\rm H}^2\, T^{1/2}\,(T - T_d) \equiv K\, T^{1/2}(T - T_d), \qquad K = \Lambda^0_{\rm gd}\, n_{\rm H}^2 .$$

Here $T$ is the gas temperature, $T_d$ the dust temperature, and $\kappa$ the dust opacity per unit mass of gas (it includes the dust-to-gas ratio), evaluated at $T_d$. The third equation is algebraic, so $T_d$ is not a state variable: it is fixed at each instant by $T$ and the $E_g$. The sign in the draft's eq. 2.2 is a typo; eq. 3.2 and the equations above have the physical sign.

Quokka writes the same model. With `enable_dust_gas_thermal_coupling_model`, it evaluates the emission and every opacity at $T_d$, and uses $n_{\rm H} = \rho / m$, with $m$ = `mean_molecular_mass`. Its `coeff_n` is $(\hat{c}/c)\,\Delta t\, K$, the draft's $N_{\rm dust}$.

Two properties follow directly and are used below:

- **Total energy is conserved.** $E_{\rm gas} + \sum_g (c/\hat{c}_g) E_g$ does not change, because the dust holds no energy. Both terms are shifted by $\Lambda_{\rm gd}$ with opposite signs.
- **The two limits are known.** As $K \to \infty$ the dust is locked to the gas, $T_d \to T$, and the system becomes the dust-free coupling with $\kappa$ evaluated at $T$. As $K \to 0$ the gas is isolated, and the dust is in radiative equilibrium with the field.

## 2. The backward-Euler step, reduced to one unknown

Take one backward-Euler step of length $\Delta t$. Use the same notation as `CouplingState`: $S_g$ is the external source and $W_g$ the lagged work term, and

$$\mathrm{rad0}_g = E_g^0 + S_g + W_g, \qquad \mathrm{gas0} = E_{\rm gas}^0 - \sum_g \frac{c}{\hat{c}_g} W_g .$$

The unknowns are $E_{\rm gas}$, $E_g$ and $T_d$. Write $\varepsilon_g(T_d) = \rho\kappa_{P,g}(T_d)\,4\pi B_g(T_d)/c$ and $\alpha_g(T_d) = \rho\kappa_{E,g}(T_d)$, which are exactly what `coupling_coefficients(s, T_d)` returns. The discrete system is

$$E_g - \mathrm{rad0}_g = \Delta t\,\hat{c}_g\,(\varepsilon_g - \alpha_g E_g), \tag{a}$$

$$E_{\rm gas} - \mathrm{gas0} = -\Delta t\, K\, T^{1/2}(T - T_d), \tag{b}$$

$$K\, T^{1/2}(T - T_d) = c \sum_g (\varepsilon_g - \alpha_g E_g). \tag{c}$$

**Choose $T_d$ as the unknown.** Everything else then follows from it in closed form.

1. **Group energies.** Equation (a) is linear in $E_g$ at fixed $T_d$, and no group appears in another's equation:
   $$E_g(T_d) = \frac{\mathrm{rad0}_g + \Delta t\,\hat{c}_g\,\varepsilon_g(T_d)}{1 + \Delta t\,\hat{c}_g\,\alpha_g(T_d)} .$$
   This is `_eliminate_groups` evaluated at $T_d$ rather than at $T$, unchanged.
2. **Gas energy, from conservation.** Add (b) to $\sum_g (c/\hat{c}_g)\times$(a), and substitute (c). The exchange terms cancel:
   $$E_{\rm gas}(T_d) = \mathrm{gas0} - \sum_g \frac{c}{\hat{c}_g}\left[E_g(T_d) - \mathrm{rad0}_g\right] .$$
   This is `energy_balance(s, E_gas, E_rad) = 0` solved for $E_{\rm gas}$, so it is $-$`energy_balance(s, 0, E_rad)`. Then $T = E_{\rm gas}/c_V$.
3. **The one equation left** is (b), the gas equation:
   $$H(T_d) \equiv E_{\rm gas}(T_d) - \mathrm{gas0} + \Delta t\, K\, T^{1/2}\,(T - T_d) = 0, \qquad T = T\!\left(E_{\rm gas}(T_d)\right).$$

Steps 1 and 2 make (a) and the sum of (b) and (c) hold exactly, for every trial $T_d$. So **the total energy is conserved to round-off at every iterate, not only at convergence.** $H$ is the part of the gas equation that is left, in energy units.

### Why $T_d$ and not $E_{\rm gas}$

With $E_{\rm gas}$ as the unknown (the dust-free method's choice), each evaluation of the outer equation needs $T_d$ at the trial $T$, which is a root of (c) with (a) substituted: a nested root-find, costing about ten evaluations per outer evaluation. With $T_d$ as the unknown, one evaluation of $H$ costs one evaluation of `coupling_coefficients`, the same as one evaluation of the dust-free equation. The cost per cell should therefore be about the same as `BracketedCoupling` today.

### Why no regime switch is needed

Quokka and the draft recover the dust temperature from the transfer, $T_d = T - \sum_g R_g / (N_{\rm dust}\sqrt{T})$. This divides by $N_{\rm dust} \propto K$, so it breaks down when the dust decouples. That is why they need a second, "decoupled" model with its own Jacobian, selected by `gas_dust_coupling_threshold`, and adaptive damping when that branch oscillates. In the formulation above, $K$ only multiplies. At $K = 0$, $H$ becomes $E_{\rm gas}(T_d) - \mathrm{gas0}$, whose root is the dust in radiative equilibrium with the gas untouched, which is the correct decoupled answer. One equation covers every $K$ from $0$ to $\infty$, and §6 checks both ends.

## 3. The bracket

**Lower end, $T_d = T_{\rm floor}$.** There $\varepsilon_g \approx 0$, so step 1 gives $E_g \le \mathrm{rad0}_g$. Step 2 then gives $E_{\rm gas} \ge \mathrm{gas0}$, so $T \ge T_{\rm floor}$ and the collision term is $\ge 0$. Therefore $H(T_{\rm floor}) \ge 0$, up to the thermal emission at the floor temperature.

**Upper end, $T_d \to \infty$.** Here $\varepsilon_g \to \infty$ in any group with $\kappa_P > 0$. The group energies grow without bound, $E_{\rm gas}(T_d) \to -\infty$, and $H \to -\infty$. For $E_{\rm gas} < c_V T_{\rm floor}$, $T$ is held at the floor. The collision term is then negative too, so $H < 0$ throughout that region. Therefore every root has $E_{\rm gas} \ge c_V T_{\rm floor}$, and the floor on $T$ never selects the answer.

**The march.** As in `_energy_bracket`, start from a guess and step by factors of two in the direction the sign of $H$ indicates, until the sign changes. Stop at $T_{\rm floor}$ going down; going up, the sign change is guaranteed. The guess is $T_d^{\rm init} = T^0$, the gas temperature at the start of the step. That is also Quokka's choice: it calls `ComputeDustTemperatureBateKeto(T_gas0, T_gas0, …)`, whose second argument is the initial guess. The walk needs `_energy_bracket` with the lower bound and the starting point as arguments instead of `c_V T_floor` and `E_gas^0`. That generalisation leaves the dust-free call identical.

**Uniqueness.** Two separate facts:

- The collision law by itself is harmless. At fixed $T_d$, the backward-Euler gas equation $T - T^0 = k\sqrt{T}(T_d - T)$ has exactly one root in $(0, T_d)$, because $\sqrt{T}(T_d - T)$ is concave.
- The dust balance can have several roots when the dust opacity rises steeply with $T_d$. With $\kappa \propto T_d^2$, the absorption term $\rho\kappa(T_d)(B(T_d) - E)$ is not monotone in $T_d$. The multiple roots are then equilibria of the model itself, not artefacts of the discretisation: an ODE integration faces the same choice at $t = 0$. The prototype sweep found this in 2 of 1600 cells (one case, for both $N_G = 1$ and $N_G = 4$): $\kappa \propto T_d^2$, gas at $0.01$, radiation at $1$, with roots at $T_d = 0.0114$, $0.134$ and $0.738$. The march from $T^0 = 0.01$ returns $0.0114$, the root continuously connected to cold dust in cold gas. This is the same selection rule the dust-free solver uses, and the same starting point Quokka uses.

## 4. The convergence test: on the state, not the residual

**The residual test fails.** $H$ has the same round-off amplification that Quokka's replaced variables were designed to avoid. The amplification now comes from the collision term instead of the radiation term: $\Delta t K \sqrt{T}(T - T_d)$ multiplies round-off in $T - T_d$ by $\Delta t K \sqrt{T}$. In the prototype sweep with tolerance $10^{-9}$, the test $|H| \le {\rm tol}\cdot E_{\rm tot}$ fails in 255 of the 320 cells at $\Delta t K / c_V = 10^8$. It fails even when $T_d$ is converged to $10^{-13}$, and it fails nowhere else. The answer is accurate; only the test is wrong.

**The test to use.** Conservation holds exactly by construction (§2), so the only thing left to control is how the energy is shared between gas and radiation. A bracketing method returns a final bracket $[T_d^-, T_d^+]$ that contains the root. So test the state across it:

$$\left|E_{\rm gas}(T_d^-) - E_{\rm gas}(T_d^+)\right| \le {\rm tol}\cdot E_{\rm tot} .$$

This bounds the error of the returned $E_{\rm gas}$, and through conservation the error of $\sum_g (c/\hat{c}_g)E_g$, by ${\rm tol}\cdot E_{\rm tot}$. It is not amplified by anything. It costs two extra evaluations of steps 1 and 2 (no root-finder call).

**The tolerance conversion.** The root-finder stops on bracket width in $T_d$. Start with `abstol` $= \tfrac12\,{\rm tol}\, T_d^+$. If the state test fails, re-solve on the returned bracket with `abstol` $= {\rm tol}\,E_{\rm tot} / (4\,|\Delta E_{\rm gas} / \Delta T_d|)$, using the slope measured across the current bracket. Repeat at most three times. The factor 4 is there because `ZBrent` stops at a half-width of `abstol`: a factor of 2 turns that into full width, and another factor of 2 is a margin for the slope changing across a wide first bracket. Without that margin, 21 cells failed. This is the same pattern as the dust-free solver's conversion, but it converts through $dE_{\rm gas}/dT_d$ instead of $dG/dE_{\rm gas}$.

## 5. Algorithm, per cell

1. Guess: $T_d \leftarrow \max(E_{\rm gas}^0 / c_V,\ T_{\rm floor})$.
2. March by factors of two on $H(T_d)$ until the sign changes. If no sign change is found, report the cell unconverged and keep the old state, as `BracketedCoupling` does.
3. `solve(IntervalNonlinearProblem(H, (lo, hi), s), root_finder; abstol, maxiters)`, with `ZBrent()` as the default.
4. Test the state across the final bracket and convert the tolerance (§4).
5. Set $E_g = E_g(T_d)$ and $E_{\rm gas} = E_{\rm gas}(T_d)$. Apply `apply_energy_floors`. Return $T_d$ too, because the flux update needs it (§7).

## 6. What the prototype shows

`julia --project=. docs/coupling-experiments/dust_prototype.jl` reproduces every number here.

**Sweep.** 1600 one-step cells: $N_G \in \{1, 4\}$; constant and $\kappa \propto T_d^2$ dust opacity; gas temperature $\{0.01, 0.3, 1, 10\}$; radiation temperature $\{0, 0.1, 1, 5\}$; $\tau = \Delta t\,\hat{c}\rho\kappa \in \{10^{-4}, 1, 10^4, 10^8, 10^{12}\}$; $\Delta t K / c_V \in \{0, 10^{-6}, 1, 10^4, 10^8\}$; tolerance $10^{-9}$.

| quantity | result |
|:---|:---|
| cells that failed to converge | 0 of 1600 |
| worst state error across the final bracket | $9.8\times10^{-10}\,E_{\rm tot}$ |
| march steps plus root-finder calls, mean per cell | 5.7 |
| cells with more than one root (scan of $H$ on 4000 points) | 2 (see §3) |
| residual-test failures with a $10^{-13}$-converged $T_d$ | 255, all at $\Delta t K / c_V = 10^8$ |

**Time accuracy.** Quokka's `RadDust` problem: $c = 10^8$, $\hat{c} = 10^7$, $\rho\kappa = 10^4$, $\Lambda^0_{\rm gd} n_{\rm H}^2 = 10^6$, $c_V = 1.5$, $T^0 = 1$, $E^0 = 0$, run to $t = 3\times10^{-5}$. It uses $B = aT^4$, not Quokka's linearised $B = aT$; see decision 4 in §8. The reference is the reduced ODE ($T_d$ solved inside the right-hand side) integrated with `Rodas5P` at `reltol = 1e-11`.

| $\Delta t$ | error in $T$ | error in $E$ | energy error |
|:---|:---|:---|:---|
| $10^{-7}$ | $5.2\times10^{-9}$ | $-5.6\times10^{-9}$ | $4\times10^{-16}$ |
| $10^{-8}$ | $5.0\times10^{-10}$ | $-5.7\times10^{-10}$ | $2\times10^{-15}$ |
| $10^{-9}$ | $1.1\times10^{-9}$ | $-1.2\times10^{-9}$ | $9\times10^{-16}$ |

The first two rows are first order in $\Delta t$. At $10^{-9}$ the $3\times10^4$ steps are limited by the per-step tolerance instead. The end time is near equilibrium, so these errors are small in absolute terms; the implemented test will compare the whole history, as Quokka's does.

**Limits.** At $\Delta t K / c_V = 10^{12}$, the dust solve reproduces the dust-free `BracketedCoupling` result to $6\times10^{-13}$ in $E_{\rm gas}$, with $|T_d/T - 1| \le 6\times10^{-13}$. At $K = 0$ the gas energy is unchanged to $6\times10^{-13}$. Both hold for $N_G = 1$ and $4$ and for constant and $\kappa \propto T^{-3/2}$ opacity.

## 7. Implementation plan

About 250 lines in `src/` and 200 in `test/`. About half a day, including the test problem.

**Physics: the dust model** (new file `src/dust.jl`, included after `opacity.jl`):

- `AbstractDustModel`; `NoDust()`, the default; `BateKetoDust(; coefficient, particle_mass = 1)`.
- `dust_gas_coefficient(d, rho) = d.coefficient * (rho / d.particle_mass)^2`, which is $K$. `particle_mass` is Quokka's `mean_molecular_mass`, so `coefficient` is Quokka's `dust_gas_interaction_coeff`, and $K$ is the same number in both codes.

**The state** (`src/coupling.jl`):

- `CouplingState` gets a `dust` field and a type parameter for it. The dust law travels on the state, like the opacity law.
- A new method `dust_energy_balance(s, Td) -> (H, Egas, Erad)`: steps 1–3 of §2, built only from `coupling_coefficients`, `_eliminate_groups` and `energy_balance`. There is no new physics formula except the collision rate.
- `CouplingResult` gets `Temit`: the temperature the emitting matter is at. It is $T$ without dust and $T_d$ with dust.

**The solver** (`src/coupling_solvers.jl`):

- A second `solve_coupling(m::BracketedCoupling, s)` method, dispatching on `s.dust isa BateKetoDust`. It uses the same `tolerance`, `max_iterations` and `root_finder`, so users choose the solver exactly as today.
- `_energy_bracket` gets the lower bound and the starting point as arguments. The dust-free call passes today's values.

**The driver** (`src/source_terms.jl`, `src/radiation_system.jl`):

- `RadiationPhysics` gets `dust = NoDust()`. The constructor throws an `ArgumentError` if a dust model is combined with `QuokkaCoupling` or `AdaptiveCoupling` (see decision 2).
- `multigroup_source_step!` evaluates the flux-mean opacity, the Planck term in `_beta_force`, and the work term at `res.Temit` instead of at the gas temperature. Quokka does the same (`ComputeModelDependentKappaFAndDeltaTerms(T_d, …)`). The first work estimate uses $T^0$, as it does now.
- The grey driver (`NG = 1`, the inline 2×2 Newton) has no dust. Its dispatch is narrowed to `NoDust`, so a grey run with dust goes to `multigroup_source_step!`, which works unchanged for one group. Grey runs without dust are not affected.

**Tests** (`test/test_dust.jl`, new):

- Both limits of §6, as invariants: at large $K$ the result equals dust-free `BracketedCoupling`; at $K = 0$ the gas is unchanged.
- Exact energy conservation on every cell of a small sweep, and 0 failures on it.
- The multi-root cell: the march returns the root near cold dust.
- 0 bytes allocated per cell, added to `test_performance.jl`.
- `RadDust` end to end against the `Rodas5P` reference of §6, over the whole history.

**Problems and docs:**

- `Problems.rad_dust` (grey), and `rad_dust_mg` (Quokka's 4-group `RadDustMG`), if you want it.
- Results appended to this file.
- A CLAUDE.md paragraph under "The multigroup coupling solvers".

**Not planned:** photoelectric heating, cosmic-ray heating, line cooling; gas and dust opacity at the same time (§8, decision 3); storing $T_d$ between steps (§8, decision 5).

## 8. Decisions (all five recommendations accepted)

1. **The API shape.** Recommended: `RadiationPhysics(…; dust = BateKetoDust(coefficient = 1e6))`. Dust is part of the physics, not the solver, and `coupling = BracketedCoupling()` stays the solver choice. The alternative, a separate `DustCoupling <: AbstractCoupling`, would mix physics and solver in one type.
2. **The other two solvers.** Recommended: dust is supported by `BracketedCoupling` only, and the constructor refuses the other combinations. `QuokkaCoupling` with dust would mean porting Quokka's two-regime Newton, which this plan exists to replace. `AdaptiveCoupling` with dust would need an inner $T_d$ solve inside every right-hand-side evaluation, plus its implicit-function Jacobian. That is a useful reference, but the test can build its own reference, as §6 does.
3. **Gas opacity together with dust.** Recommended: not now. The draft and Quokka both assume the gas has no opacity when the dust model is on. Supporting both at once would bring back two temperatures in the emission, and then no single-variable closed form exists. It would need a nested bracket.
4. **The `RadDust` reference.** Quokka's test replaces $B = aT^4$ by $B = aT$, so that its exact solution can be written down. Honghu has no hook to override the emission law. Recommended: keep $aT^4$ and use the `Rodas5P` reference of §6, which the test can compute in a few seconds. The alternative is an emission-law hook, which is a new API for one test.
5. **The initial guess for $T_d$.** Recommended: $T^0$, as Quokka does. Storing $T_d$ as a diagnostic field would give a better starting point, and root continuity across steps in the multi-root case, but it adds a field to the state arrays.

## 9. Implementation and results

The code follows §7. The only difference from the plan is in method dispatch. The dust-free `solve_coupling` methods of all three solvers are now written for `CouplingState{…,NoDust}`, and one fallback `solve_coupling(::AbstractCoupling, ::CouplingState{…,<:BateKetoDust})` throws `ArgumentError`. With the methods left general, Julia reported the `BracketedCoupling` dust method as ambiguous. A hand-built dust state handed to `QuokkaCoupling` or `AdaptiveCoupling` now fails loudly; it cannot be solved as if there were no dust.

| file | what |
|:---|:---|
| `src/dust.jl` | `AbstractDustModel`, `NoDust`, `BateKetoDust(; coefficient, particle_mass = 1)`, `dust_gas_coefficient` |
| `src/coupling.jl` | `CouplingState` gains `dust` (the old 15-argument constructor still works and means `NoDust`); `CouplingResult` gains `Tdust` (`NaN` without dust); `dust_energy_balance(s, Td) -> (H, Egas, Erad)` |
| `src/coupling_solvers.jl` | the `BateKetoDust` method of `solve_coupling(::BracketedCoupling, …)`; `_energy_bracket(G, s, x0, xmin)` |
| `src/radiation_system.jl` | `RadiationPhysics(; dust = NoDust())`, and the refusal of any other solver |
| `src/source_terms.jl` | the grey driver is `NoDust` only; the multigroup driver passes the dust model and evaluates the flux opacity and $B$ at `res.Tdust` |
| `src/problems/radiation.jl` | `Problems.rad_dust(; groups)` and `Problems.rad_dust_reference(ts; groups)` |
| `test/test_dust.jl` | the tests below |

`H` is defined as the negative of §2's, so that it increases with $T_d$. The march and the root-finder are then the same ones the dust-free equation uses.

**Tests** (58 new, and the suite passes 1401 of 1401):

- A sweep of 432 cells ($N_G \in \{1,4\}$; two opacity laws; gas temperature, radiation temperature, $\rho$ and $K$ spread over the regimes of §6): 0 failures, worst relative energy error below $10^{-13}$, and the state within $1.01\times10^{-9}\,E_{\rm tot}$ of a solve at tolerance $10^{-12}$. A mutation check confirms the width test matters: with the residual test instead, this sweep reports 127 cells unconverged.
- $K \to \infty$ equals dust-free `BracketedCoupling` to $10^{-11}$, and $K = 0$ leaves the gas unchanged with the dust balance satisfied.
- The multi-root cell of §3 returns $T_d \in (0.0113, 0.0115)$.
- The `ArgumentError`s, and 0 bytes per cell (`test_dust.jl`, and `test_performance.jl` through the full grey driver with dust and `beta_order = 1`).
- `RadDust` over the full history against `rad_dust_reference`, grey and with `RadDustMG`'s four groups: relative L1 error $2.2\times10^{-4}$ for both (Quokka's tolerance is $3\times10^{-3}$), 0 failures, 5.1 iterations per cell on average, 8 at most, about 1 s for 3000 steps.
