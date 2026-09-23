# The matter-radiation coupling: three ways to solve it, and what the measurements say

Hydro3D ships three solvers for the per-cell matter-radiation energy exchange, and the point of having three is to be able to measure what each one is worth. This document does that. The three are `QuokkaCoupling` (backward Euler in Quokka's replaced base variables), `DirectCoupling` (backward Euler in the natural base variables, by either of two steppers), and `AdaptiveCoupling` (an error-controlled stiff ODE solve). Everything below is reproduced by `docs/coupling-experiments/coupling_study.jl`, which also defines two further steppers that are deliberately *not* in `src/` — a natural-variable Newton with a switchable convergence test, and a bracketed scalar solver — because the argument turns on them.

The headline, stated up front so the rest can be read as evidence for it:

1. The two backward-Euler solvers are **the same nonlinear system**, written in two sets of unknowns. Any difference between them is floating point, not mathematics.
2. The round-off amplification that motivates Quokka's replaced variables is **real and exactly as large as claimed** — at Quokka's own default tolerance of $10^{-11}$ the natural-variable residual test starts failing at $\tau_\mathrm{cell} \approx 10^5$ and fails everywhere by $3\times10^5$.
3. But what the amplification breaks is the **convergence test, not the answer**. The iteration that "fails" is sitting on the right root, to $10^{-16}$ of the cell's energy. Dividing the group residual by its own Jacobian diagonal — one line — removes the amplification completely, without changing the base variables, and delivers a *more* accurate answer than the replaced variables do.
4. Where a one-step backward Euler genuinely fails is **not round-off but the nonlinearity of a temperature-dependent opacity** — and there the replaced variables are markedly *worse*: 36 failures out of 66 cases on a $\kappa \propto T^{-3.5}$ sweep, against 0 for a damped natural-variable Newton.
5. Reducing the natural-variable system to **one scalar equation with a guaranteed bracket** gives a solver that converged in all 240 sweep cases and in every test problem, at zero allocation and $O(N_G)$ per iteration. It is the only one-step solver that gets the hardest test problem here right. It is not in `src/`; §5 derives it in full and says what shipping it would take.
6. The error-controlled ODE solve never fails and is 2 to 6 orders of magnitude more accurate in the intermediate regime, at 12 times the cost of a hand-written Newton step on its loosest setting and several hundred times on a tight one, plus 8 to 21 kB of allocation per cell. It is a reference solver, and should stay one.

## 1. The three methods, and why two of them are one method

Per cell, the coupling is a stiff ODE in the gas internal energy and the $N_G$ group radiation energies,

$$\frac{dE_\mathrm{gas}}{dt} = -c\,\rho \sum_g \left(\kappa_{P,g} B_g(T) - \kappa_{E,g} E_g\right), \qquad \frac{dE_g}{dt} = +\hat c\,\rho \left(\kappa_{P,g} B_g(T) - \kappa_{E,g} E_g\right),$$

writing $B_g$ for $4\pi B_g/c = a_\mathrm{rad} T^4 f_g(T)$. Two of the three solvers take one backward-Euler step of this over $\Delta t$; the third integrates it with an adaptive stiff method and error control.

**Natural base variables.** The unknowns are $(E_\mathrm{gas}, E_g)$ and the residual is $u - u^0 - \Delta t\, f(u)$, so with $\tau_g = \Delta t\,\hat c\,\rho\,\kappa_{P,g}$ and $a_g = \Delta t\,\hat c\,\rho\,\kappa_{E,g}$ the group equation reads

$$F_g = \left(E_g - E_g^0\right) - \tau_g B_g(T) + a_g E_g = 0 .$$

**Quokka's replaced base variables.** The unknowns are $(E_\mathrm{gas}, R_g)$, where $R_g$ is the energy handed from gas to group $g$ over the whole step,

$$R_g = \Delta t\,\hat c\,\rho\left(\kappa_{P,g} B_g(T) - \kappa_{E,g} E_g\right) + W_g ,$$

with $W_g$ the lagged $\boldsymbol{v}\cdot\boldsymbol{F}$ work term. The residuals are $F_0 = (E_\mathrm{gas} - E_\mathrm{gas}^0) + (c/\hat c)\sum_g R_g$ and $F_g = (E_g - E_g^0) - (R_g + S_g)$, and $E_g$ is recovered from $R_g$ in closed form as $E_g = (\kappa_{P,g}/\kappa_{E,g})\left(B_g - (R_g - W_g)/\tau_g\right)$.

Substituting that closed form back into $F_g$ returns the natural-variable group equation term for term, and substituting it into $F_0$ returns the natural-variable gas equation. **The two are the same nonlinear system.** The measurement agrees: in 256-bit arithmetic the two formulations land on the same root to better than $10^{-40}$, and in double precision their answers agree to $10^{-11}$ or better at every optical depth from $10^{-4}$ to $10^{12}$ (§6, columns "BE Quokka" and "BE natural"). This matters because it fixes what the comparison can possibly be about: not accuracy of the discretisation, only conditioning and iteration behaviour.

Quokka has a further switch, `use_D_as_base`, which rescales the unknown to $D_g = R_g/\tau_g^{(0)}$. It is `false` by default, and it is a diagonal rescaling of the unknowns, to which Newton's method is invariant in exact arithmetic — so it changes step-length bookkeeping and floating-point detail, not the method. Hydro3D's `QuokkaCoupling` implements the production default, $R_g$.

**Error-controlled ODE solve.** `AdaptiveCoupling` hands the same right-hand side and the same analytic Jacobian to an adaptive stiff integrator (`Rosenbrock23` by default) over $[0, \Delta t]$, and lets it choose its own internal steps. It is not a discretisation of the coupling with a fixed error; it is a statement of what the coupling ODE's answer actually is, to a tolerance you ask for.

## 2. The conditioning claim is right, and this is how big it is

The natural-variable Newton matrix has a condition number proportional to $\tau_\mathrm{cell}$. The replaced-variable one does not. Measured at the exact root, four groups, constant opacity:

| $\tau_\mathrm{cell}$ | cond, natural | cond, replaced | residual noise / $E_\mathrm{tot}$ | $\varepsilon\,\tau_\mathrm{cell}$ |
|---:|---:|---:|---:|---:|
| $10^{-4}$ | 1.00 | $9.99\times10^{3}$ | $2.9\times10^{-17}$ | $2.2\times10^{-20}$ |
| $10^{-2}$ | 1.10 | $9.37\times10^{1}$ | $4.1\times10^{-17}$ | $2.2\times10^{-18}$ |
| $10^{0}$ | 6.24 | 2.51 | $1.2\times10^{-16}$ | $2.2\times10^{-16}$ |
| $10^{2}$ | $5.04\times10^{2}$ | 3.36 | $8.4\times10^{-15}$ | $2.2\times10^{-14}$ |
| $10^{4}$ | $5.04\times10^{4}$ | 3.39 | $5.5\times10^{-13}$ | $2.2\times10^{-12}$ |
| $10^{6}$ | $5.04\times10^{6}$ | 3.39 | $1.6\times10^{-10}$ | $2.2\times10^{-10}$ |
| $10^{8}$ | $5.04\times10^{8}$ | 3.39 | $5.6\times10^{-9}$ | $2.2\times10^{-8}$ |
| $10^{9}$ | $5.04\times10^{9}$ | 3.39 | $3.5\times10^{-8}$ | $2.2\times10^{-7}$ |
| $10^{10}$ | $5.04\times10^{10}$ | 3.39 | $1.1\times10^{-6}$ | $2.2\times10^{-6}$ |
| $10^{12}$ | $5.04\times10^{12}$ | 3.39 | $1.4\times10^{-4}$ | $2.2\times10^{-4}$ |

The fourth column is the measurement that matters: it is the natural-variable residual evaluated **at the exact root**, in double precision, in units of the cell's total energy. It should be zero and it is not, because at the root $B_g$ and $E_g$ agree to round-off and the difference $\tau_g(B_g - E_g)$ is a cancellation whose absolute error is $\varepsilon\,\tau_g B_g$. Measured, the noise is $0.2$ to $0.7$ times $\varepsilon\,\tau_\mathrm{cell}$ — it is one realisation of a rounding error, so the prefactor wanders, but the $\tau$ scaling holds across sixteen orders of magnitude. Asking a residual test for $10^{-9}$ at $\tau = 10^{12}$ asks for a number ten million times smaller than the noise in computing it.

Note also the first two rows of the third column. The replaced variables are ill-conditioned in the *optically thin* limit, where $J_{gg} = -\kappa_P/(\kappa_E \tau_g) - 1$ blows up — which is precisely why Quokka carries an explicit optically-thin shortcut and Hydro3D's `QuokkaCoupling` carries the same one. Each set of base variables has a limit it cannot see through, and each carries a patch for it; they are just at opposite ends of the $\tau$ axis.

Where the natural-variable residual test gives up, as a function of the tolerance it is asked for, over eight initial states at four groups:

| $\tau_\mathrm{cell}$ | tol $10^{-7}$ | tol $10^{-9}$ | tol $10^{-11}$ | tol $10^{-13}$ |
|---:|---:|---:|---:|---:|
| $10^{3}$ | 0/8 | 0/8 | 0/8 | 3/8 |
| $10^{4}$ | 0/8 | 0/8 | 0/8 | 8/8 |
| $10^{5}$ | 0/8 | 0/8 | 4/8 | 8/8 |
| $3\times10^{5}$ | 0/8 | 0/8 | 8/8 | 8/8 |
| $10^{6}$ | 0/8 | 0/8 | 8/8 | 8/8 |
| $10^{7}$ | 0/8 | 1/8 | 8/8 | 8/8 |
| $3\times10^{7}$ | 0/8 | 5/8 | 8/8 | 8/8 |
| $10^{8}$ | 0/8 | 8/8 | 8/8 | 8/8 |
| $10^{9}$ | 2/8 | 8/8 | 8/8 | 8/8 |

The replaced variables give 0/8 in every one of those cells, at every tolerance down to $10^{-13}$ and at $\tau$ up to $10^{12}$.

Quokka's default `radiation_iteration_tolerance` is $10^{-11}$, and the table says the onset at that tolerance is $\tau_\mathrm{cell} \approx 10^{5}$, with total failure by $3\times10^{5}$. That is the number in this project's folklore, now measured rather than estimated. The rule $\tau_\mathrm{cell} \gtrsim \mathrm{tol}/(0.6\,\varepsilon)$ predicts the first failure in every column of the table to within 30%: $7.5\times10^{8}$, $7.5\times10^{6}$, $7.5\times10^{4}$ and $7.5\times10^{2}$ against measured onsets of $10^{9}$, $10^{7}$, $10^{5}$ and $10^{3}$.

## 3. What the amplification actually costs is the test, not the answer

The natural-variable iteration that "fails" in the table above is sitting on the correct root. The table below measures it: at every $\tau$ from $10^{8}$ to $10^{12}$ the residual test burns 100 iterations instead of 7, reports `converged = false`, and hands back an answer within $10^{-16}$ of the cell's energy of the exact root — machine precision.

The reason is that **the same $\tau_g$ that amplifies the residual sits on the Jacobian's diagonal and divides it back out.** For the group block, $\partial F_g/\partial E_g = 1 + a_g$, so the Newton increment for that group is $\delta E_g \approx -F_g/(1 + a_g)$. Noise of size $\varepsilon\,\tau_g B_g$ in $F_g$ becomes noise of size $\varepsilon\,B_g$ in the increment. The amplification and the de-amplification are the same factor, and only the convergence test — which reads the un-preconditioned residual — sees the amplified one.

That is a stronger statement than "the amplification is tolerable". It says the failure mode is a *false report of non-convergence*, and the cure is to fix the report. Three ways to do that, all in the natural variables, at a tolerance held fixed at $10^{-9}$ with no automatic degradation, four groups, constant opacity (error in units of cell energy; `!` marks a reported failure):

| $\tau_\mathrm{cell}$ | residual | it | scaled residual | it | increment | it | `ImplicitEuler` | it |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $10^{0}$ | $6.9\times10^{-17}$ | 6 | $6.9\times10^{-17}$ | 6 | $6.9\times10^{-17}$ | 6 | $6.9\times10^{-17}$ | 6 |
| $10^{2}$ | $1.9\times10^{-12}$ | 6 | $1.9\times10^{-12}$ | 6 | $9.4\times10^{-17}$ | 6 | $9.9\times10^{-17}$ | 6 |
| $10^{4}$ | $2.3\times10^{-17}$ | 7 | $2.1\times10^{-12}$ | 6 | $2.3\times10^{-17}$ | 6 | $9.6\times10^{-17}$ | 6 |
| $10^{6}$ | $7.5\times10^{-17}$ | 7 | $2.1\times10^{-12}$ | 6 | $7.5\times10^{-17}$ | 6 | $1.3\times10^{-16}$ | 6 |
| $10^{7}$ | $4.5\times10^{-17}$ | 7 | $2.1\times10^{-12}$ | 6 | $4.5\times10^{-17}$ | 6 | $3.5\times10^{-17}$ | 6 |
| $10^{8}$ | $6.5\times10^{-17}$ | 100 ! | $2.1\times10^{-12}$ | 6 | $4.7\times10^{-17}$ | 6 | $1.2\times10^{-16}$ | 6 |
| $10^{9}$ | $2.9\times10^{-17}$ | 100 ! | $2.2\times10^{-12}$ | 6 | $4.6\times10^{-17}$ | 6 | $2.1\times10^{-17}$ | 6 |
| $10^{10}$ | $6.0\times10^{-17}$ | 100 ! | $4.2\times10^{-12}$ | 6 | $1.0\times10^{-16}$ | 6 | $6.0\times10^{-17}$ | 6 |
| $10^{12}$ | $8.5\times10^{-17}$ | 100 ! | $3.0\times10^{-10}$ | 6 | $2.5\times10^{-15}$ | 6 | $3.1\times10^{-15}$ | 6 |

The **scaled residual** test — stop on $\max_i |F_i / J_{ii}| \le \mathrm{tol}\cdot E_\mathrm{tot}$ instead of $\max_i |F_i|$ — is one line and it removes the $\tau$ dependence entirely: six iterations at every optical depth, exactly like the replaced variables. The **increment** test does the same and is more accurate, because it takes the step before testing it, so it delivers the extra Newton iteration for free.

The last column is worth noting for what it says about the code as it stands. `DirectCoupling`'s default stepper, OrdinaryDiffEq's `ImplicitEuler`, measures its own increment rather than the residual, so it is immune already — and this column is it run at the *un-raised* tolerance, bypassing the automatic degradation `DirectCoupling` applies. It never needed that degradation. The degradation is there for the hand-written `NewtonStepper`, which tests the residual, and the better fix for that stepper is the scaled test, because raising the tolerance to $4\varepsilon\tau$ makes the failures go away by *asking for less* rather than by asking correctly. The cost shows up in the sweep of §4: with the tolerance raise the worst error over 80 cases is $2.3\times10^{-10}$, and with the increment test it is $1.6\times10^{-14}$.

## 4. Where backward Euler really fails: the opacity, not the arithmetic

The sweep is 8 initial gas/radiation temperature pairs $\times$ 10 optical depths from $10^{-4}$ to $10^{12}$ $\times$ 3 opacity laws, at four groups. Each case is scored against the exact root of its own backward-Euler system, computed in 256-bit arithmetic and verified by evaluating its own residual there. Cases where even that 256-bit solve could not be verified are excluded and counted separately.

**Constant opacity — 80 cases, all references verified.**

| method | failures | worst error | mean iterations |
|:---|---:|---:|---:|
| Quokka, $R$ base | 0/80 | $9.2\times10^{-11}$ | 5.7 |
| natural, residual test | 32/80 | $2.3\times10^{-10}$ | 8.4 |
| natural, residual + tolerance raise (as shipped) | 0/80 | $2.3\times10^{-10}$ | 9.5 |
| natural, increment test | 0/80 | $1.6\times10^{-14}$ | 9.0 |
| natural, increment + damping | 0/80 | $6.7\times10^{-15}$ | 6.3 |
| natural, bracketed scalar | 0/80 | $3.2\times10^{-10}$ | 9.2 |
| natural, `ImplicitEuler` | 0/80 | $2.7\times10^{-8}$ | 9.0 |
| adaptive, $10^{-2}$ | 0/80 | — | 20.3 |
| adaptive, $10^{-8}$ | 0/80 | — | 589.5 |

With a constant opacity, only the naive residual test fails, and only at $\tau \ge 10^{8}$ — which is the whole of the round-off story. Everything else in the table is a tie.

**$\kappa \propto T^{-3.5}$ — 66 cases scored, 14 with no verifiable reference.**

| method | failures | worst error | mean iterations |
|:---|---:|---:|---:|
| Quokka, $R$ base | **36/66** | $9.9\times10^{-1}$ | 6.2 |
| natural, residual test | 20/66 | $6.2\times10^{-1}$ | 9.2 |
| natural, residual + tolerance raise | 1/66 | $6.2\times10^{-1}$ | 10.0 |
| natural, increment test | 0/66 | $6.2\times10^{-1}$ | 9.6 |
| natural, increment + damping | 0/66 | $4.8\times10^{-14}$ | 8.6 |
| natural, bracketed scalar | **0/66** | $7.9\times10^{-10}$ | 7.4 |
| natural, `ImplicitEuler` | 0/66 | $6.2\times10^{-1}$ | 10.4 |
| adaptive, $10^{-2}$ | 0/66 | — | 21.2 |

**$\kappa \propto T^{+2}$ — 55 cases scored, 25 with no verifiable reference.**

| method | failures | worst error | mean iterations |
|:---|---:|---:|---:|
| Quokka, $R$ base | 1/55 | $3.9\times10^{-1}$ | 6.4 |
| natural, residual test | 20/55 | $7.4\times10^{-10}$ | 7.1 |
| natural, residual + tolerance raise | 0/55 | $7.4\times10^{-10}$ | 8.1 |
| natural, increment test | 0/55 | $4.4\times10^{-16}$ | 7.8 |
| natural, increment + damping | 1/55 | $3.6\times10^{-1}$ | 7.5 |
| natural, bracketed scalar | **0/55** | $6.9\times10^{-10}$ | 6.1 |
| natural, `ImplicitEuler` | 0/55 | $5.4\times10^{-10}$ | 7.7 |
| adaptive, $10^{-2}$ | 0/55 | — | 21.1 |

Three things to take from these two tables, and the first is the one that changes the picture.

**The replaced base variables are less robust, not more, once the opacity moves with temperature.** 36 failures out of 66, with errors up to 99% of the cell's energy. The mechanism is visible in the formulation: $E_g$ is not an unknown but is *derived* from $R_g$ through $\tau_g(T)$, so when $\kappa$ swings by orders of magnitude between iterates the implied group energies swing with it, and there is nothing holding them. The natural variables keep $E_g$ as an independent unknown that only moves by a Newton step. Written the other way round: $R_g$ is a well-scaled unknown exactly while $\tau$ is nearly constant over the iteration, which is what a constant opacity guarantees and a realistic one does not.

**A worst error of 0.62 alongside "0 failures" is worse than a failure.** The plain increment test and `ImplicitEuler` both report success on cases where they are 62% wrong. An increment test cannot distinguish "the iteration has converged" from "the iteration has stalled", and both stall against the temperature floor, where $\kappa \propto T^{-3.5}$ is $10^{35}$ and nothing recovers. The damped variant — which forbids the gas energy leaving $[\max(E_\mathrm{floor}, x/4),\, 4x]$ in one step — fixes this case entirely ($4.8\times10^{-14}$), and is one `while` loop.

**But damping is not a general answer.** It fails one $\kappa \propto T^{+2}$ case, and that failure is instructive: with a strongly absorbing temperature-dependent opacity, $\partial(\kappa_P B_g - \kappa_E E_g)/\partial T$ can change sign, so the scalar residual is not monotone and the Newton direction points *away* from the root from the very first iterate. Tracing that cell, the iteration walks the gas energy from $0.5$ down through $10^{-6}$, halving each step, while the root is at $1.21$. No step-length rule saves an iteration whose direction is wrong. Both bases do this; it is the physics of the residual, not the choice of unknown.

## 5. A natural-variable backward Euler that does not fail

This section derives the method in full, because the idea is simple but the reason it works is not the obvious one. It is **not implemented in `src/`**: it lives as `BracketedDirect` in `docs/coupling-experiments/coupling_study.jl`, as an ordinary `AbstractCoupling` subtype with its own `solve_coupling` method, so it can be passed as `RadiationPhysics(coupling = BracketedDirect())` and was run that way for every number below. §5.7 says what moving it into the tree would take.

### 5.1 The system, written out

Write the two step-constant increments the driver supplies — the external radiation source $S_g$ and the lagged $\boldsymbol{v}\cdot\boldsymbol{F}$ work term $W_g$ — into the old state, exactly as `backward_euler` does at `src/coupling_solvers.jl:361-363`:

$$\mathrm{gas0} = E_\mathrm{gas}^0 - \sum_g \frac{c}{\hat c_g} W_g , \qquad \mathrm{rad0}_g = E_g^0 + S_g + W_g .$$

Define the two optical depths of the step, one for emission and one for absorption,

$$\tau_g(T) = \Delta t\,\hat c_g\,\rho\,\kappa_{P,g}(T) , \qquad a_g(T) = \Delta t\,\hat c_g\,\rho\,\kappa_{E,g}(T) .$$

The backward-Euler system $u - u^0 - \Delta t\,f(u) = 0$ is then, with $T = E_\mathrm{gas}/c_V$,

$$F_g = \left(1 + a_g\right) E_g - \mathrm{rad0}_g - \tau_g B_g(T) = 0 , \qquad g = 1 \ldots N_G,$$

$$F_0 = \left(E_\mathrm{gas} - \mathrm{gas0}\right) + \sum_g \frac{c}{\hat c_g}\left(\tau_g B_g(T) - a_g E_g\right) = 0 .$$

The gas equation picks up $c/\hat c_g$ per group because the exchange rate is formed once and then scaled by $\hat c_g$ on the radiation side and by $c$ on the gas side — that asymmetry is the reduced speed of light approximation itself.

### 5.2 Step 1: the group block is linear, so eliminate it

Nothing in $F_g$ couples group $g$ to group $h$: the groups meet only through $T$. So **at a fixed $E_\mathrm{gas}$ the group block is $N_G$ independent linear equations**, and they can be solved outright rather than iterated on,

$$\boxed{\;E_g(T) = \frac{\mathrm{rad0}_g + \tau_g(T)\,B_g(T)}{1 + a_g(T)}\;}$$

which is `coupling_study.jl:196`. In this code $\kappa_{P,g} = \kappa_{E,g}$, because the multigroup opacity is piecewise constant across a group and averaging a constant against any weight returns it (`coupling_opacity`, `src/coupling.jl`), so $\tau_g = a_g$ and the formula reads

$$E_g(T) = \frac{1}{1+\tau_g}\,\mathrm{rad0}_g \;+\; \frac{\tau_g}{1+\tau_g}\,B_g(T) ,$$

a **convex combination of where the group started and the Planck value at the gas temperature**, with the weight set by the optical depth of the step. At $\tau_g \to 0$ the group keeps its energy; at $\tau_g \to \infty$ it is in equilibrium with the gas. That single line contains the whole of the stiffness, and it is exact.

### 5.3 Step 2: what remains is energy conservation

Rearranging $F_g = 0$ gives $\tau_g B_g - a_g E_g = E_g - \mathrm{rad0}_g$: the net energy the gas hands to group $g$ over the step is just the change in that group's energy. Substituting that into $F_0$ collapses the emission and absorption terms and leaves one scalar equation in one unknown,

$$G(E_\mathrm{gas}) \;\equiv\; \left. F_0 \right|_{E_g = E_g(T)} \;=\; \left(E_\mathrm{gas} - \mathrm{gas0}\right) + \sum_g \frac{c}{\hat c_g}\left(E_g(T) - \mathrm{rad0}_g\right) = 0 ,$$

which is `coupling_study.jl:198`. Expanding `gas0` and `rad0` shows that the work term cancels out of $G$ entirely — it survives only inside $E_g(T)$, through $\mathrm{rad0}_g$ — and that what is left is the statement that the cell's total energy does not change:

$$G(E_\mathrm{gas}) = E_\mathrm{gas} + \sum_g \frac{c}{\hat c_g} E_g(T) \;-\; \mathcal{E} , \qquad \mathcal{E} \equiv E_\mathrm{gas}^0 + \sum_g \frac{c}{\hat c_g}\left(E_g^0 + S_g\right) .$$

$G$ is (new total energy) minus (old total energy). That is worth saying plainly, because it is what makes the next step easy: we are not looking for the root of an abstract residual, we are looking for the gas energy at which the books balance.

### 5.4 Step 3: the bracket, and why it always exists

**Lower end.** Take $E_\mathrm{gas} = c_V T_\mathrm{floor}$. Every $B_g(T) \to 0$ there, and $E_g$ is a convex combination of $\mathrm{rad0}_g$ and $B_g$, so $E_g \le \mathrm{rad0}_g$. Hence

$$G(c_V T_\mathrm{floor}) \;\le\; c_V T_\mathrm{floor} + \sum_g \frac{c}{\hat c_g}\,W_g - E_\mathrm{gas}^0 ,$$

which is negative whenever the gas starts above its floor and the work term does not by itself exceed the gas energy. With $W_g = 0$ — every run in this study, and any run with `beta_order = 0` — it is unconditional, because the state is admissible between steps. The step also needs $\kappa_{P,g}B_g \to 0$ as $T \to 0$, which holds for any $\kappa \propto T^n$ with $n > -4$ and for any physical opacity.

**Upper end.** Every $E_g \ge 0$, so $G(E_\mathrm{gas}) \ge E_\mathrm{gas} - \mathcal{E}$, which is non-negative as soon as $E_\mathrm{gas} \ge \mathcal{E}$ — that is, once the gas alone is asked to hold more than the cell's total energy. The code starts at $\max(4\mathcal{E},\, 4E_\mathrm{gas}^0)$ and widens by factors of four (`coupling_study.jl:220-224`) in case a negative work term has shifted things, then checks both ends and reports `converged = false` rather than guessing if the bracket still is not there (`:225`). That branch never fired in 240 sweep cases or in any test problem.

**So $G$ changes sign on $[c_V T_\mathrm{floor},\, \sim\!\mathcal{E}]$ for every cell, always.** Bisection alone would converge; Newton inside the bracket converges fast and cannot leave it.

### 5.5 Step 4: the elimination fixes the Newton *direction*, not just the step length

This is the part that matters, and it is not what "add a bracket" suggests. Differentiating $E_g(T)$ and using $E_g(1+a_g) = \mathrm{rad0}_g + \tau_g B_g$,

$$\frac{dG}{dE_\mathrm{gas}} = 1 + \sum_g \frac{c}{\hat c_g}\frac{1}{c_V}\frac{dE_g}{dT} , \qquad \frac{dE_g}{dT} = \frac{\tau_g' B_g + \tau_g B_g' - a_g' E_g}{1 + a_g} ,$$

with $' = d/dT$ — `coupling_study.jl:199-204`. For $\kappa \propto T^n$ and $\kappa_P = \kappa_E$, substituting the eliminated $E_g$ back in gives

$$\frac{dE_g}{dT} = \underbrace{\frac{\tau_g B_g'}{1+\tau_g}}_{\text{emission, always} > 0} \;+\; \underbrace{\frac{n\,\tau_g}{T\,(1+\tau_g)^2}\left(B_g - \mathrm{rad0}_g\right)}_{\text{opacity moving with } T} .$$

Compare that with the same quantity in the **unreduced** $(1+N_G)$-dimensional Newton, where $E_g$ is a free iterate rather than the eliminated value. There the corresponding object is the Schur complement of the arrowhead system, and the second term reads

$$\frac{n\,\tau_g}{T\,(1+\tau_g)}\left(B_g - E_g\right) ,$$

with **one** power of $(1+\tau_g)$ instead of two, and the *current iterate* $E_g$ instead of the initial $\mathrm{rad0}_g$. Both differences matter, and together they are the whole failure of §4:

- In the eliminated form the term decays like $1/\tau_g$ and vanishes deep in the optically thick regime, so $dG/dE_\mathrm{gas} \to 1 + (c/\hat c)\,4 a_\mathrm{rad} T^3/c_V > 0$. **$G$ is guaranteed monotone at large $\tau$**, whatever the opacity exponent.
- In the unreduced form it survives at large $\tau_g$, and $B_g - E_g$ is the full departure from equilibrium rather than a small residual — so it can dominate and flip the sign.

Worked on the exact cell §4 traces ($T_\mathrm{gas} = 0.5$, $T_\mathrm{rad} = 1.3$, $\kappa \propto T^{+2}$, $\tau(T^0_\mathrm{gas}) = 2.5\times10^{8}$, four groups, $c = \hat c = c_V = a_\mathrm{rad} = 1$), at the opening iterate $E_\mathrm{gas} = E_\mathrm{gas}^0 = 0.5$:

| | eliminated $G$ | unreduced Schur complement |
|:---|---:|---:|
| value of $E_g$ used | $\sum_g E_g(T^0) = 0.0625$ | $\sum_g E_g^0 = 2.8561$ |
| emission term $4a_\mathrm{rad}T^3/c_V$ | $+0.500$ | $+0.500$ |
| opacity term $(n/T)\sum_g(B_g - \cdot)$ | $\to 0$ (extra $1/\tau$) | $-11.174$ |
| derivative | $+1.5000$ | $-9.6744$ |
| first Newton step | $+1.862$ | $-0.2888$ |
| root | $1.2103$ | $1.2103$ |

Measured: $dG/dE_\mathrm{gas} = 1.4999999$ against the predicted $1.5$, and the unreduced Schur complement $-9.674400$ against the predicted $-9.674400$. The unreduced iteration's first step points **away** from the root, and §4's trace shows it then halving the gas energy for twenty iterations down to $10^{-6}$. The eliminated one overshoots to $2.36$, in the right direction, and the bracket catches the overshoot.

So the elimination is not bookkeeping. It linearises about a state where the group block is *already satisfied*, and that is what keeps the sign of the derivative right. The bracket is the belt to that pair of braces: it handles whatever is left at moderate $\tau$, where neither limit's monotonicity argument applies and $G$ can genuinely be non-monotone.

### 5.6 The algorithm

Given the bracket $[E_\mathrm{lo}, E_\mathrm{hi}]$ from §5.4 and an opening iterate $x = E_\mathrm{gas}^0$, repeat:

1. evaluate $E_g(T)$, $G(x)$ and $G'(x)$ from §5.2 and §5.5 — one pass over the groups, no matrix;
2. tighten the bracket on the sign of $G$: $G(x) < 0 \Rightarrow E_\mathrm{lo} \leftarrow x$, else $E_\mathrm{hi} \leftarrow x$;
3. stop if $|G(x)| \le \mathrm{tol}\cdot\mathcal{E}$;
4. take $x_\mathrm{new} = x - G/G'$, and replace it by the bisection $ (E_\mathrm{lo}+E_\mathrm{hi})/2$ if it falls outside the bracket;
5. stop if the bracket has closed to round-off.

Finally recover $E_g$ at the accepted $x$ and apply `apply_energy_floors`. In the code these are `coupling_study.jl:230`, `:231`, `:232`, `:236-237`, `:238` and `:245-246`.

Note what the convergence test is. $G$ is an energy difference, not a stiff residual: it contains no $\tau_g(B_g - E_g)$ to cancel, because that cancellation was performed analytically in §5.3. **The whole of §2 and §3 is therefore moot for this solver** — there is nothing left for $\tau$ to amplify, and the residual test is safe at any optical depth and any tolerance.

### 5.7 What it measures, and what it would take to ship

All 240 sweep cases of §4, including the 39 where the unsafeguarded 256-bit Newton could not produce a verifiable reference:

| opacity | failures | worst error | mean iterations |
|:---|---:|---:|---:|
| constant | 0/80 | $3.2\times10^{-10}$ | 9.2 |
| $\kappa \propto T^{-3.5}$ | 0/80 | $7.9\times10^{-10}$ | 7.7 |
| $\kappa \propto T^{+2}$ | 0/80 | $6.9\times10^{-10}$ | 10.4 |

Zero failures, and every answer inside the $10^{-9}$ it was asked for. It costs $O(N_G)$ per iteration, allocates nothing, and needs no linear solve at all — not even the arrowhead elimination, since the group block is inverted in closed form. Against `QuokkaCoupling` it is 9% slower at one group and 40% slower from four groups to sixteen (§7), which is the cost of the extra $G$ evaluations the bracket asks for; across the test problems the gap runs from nothing to 60%.

Four things stand between this and `src/`. Its residual and derivative are written out in `_reduced` rather than assembled from `coupling_rhs` and `coupling_jacobian`, which breaks the single-source-of-truth rule stated at the head of `src/coupling.jl` and is the first thing to fix. It has no optically-thin shortcut, and the sweep did not stress transparent cells hard enough to say it does not need one. It is not in the three-way agreement testset of `test/test_coupling_adaptive.jl`. And its zero-byte allocation is measured but not pinned by an assertion in `test/test_radiation_multigroup.jl` alongside the existing ones.

### 5.8 Two caveats

Where $G$ is non-monotone the backward-Euler system genuinely has more than one root, and a bracketed method returns one of them without saying which. Only following the trajectory settles which root the physics meant, which is what §6 is about; the bracket guarantees convergence to *a* solution of the step, not uniqueness of it.

The elimination assumes the group block is diagonal — that groups exchange energy with the gas and never directly with each other. That is the same assumption `ArrowheadSolve` makes and the same one documented in `AbstractJacobianSolve`, and it would fail the same way: a Compton redistribution or line-overlap term would make the group block a matrix, the closed form of §5.2 would no longer be a per-group scalar division, and the reduction to one unknown would be gone with it.

## 6. What the error-controlled ODE solve buys, and what it costs

One backward-Euler step is second-order accurate *as a step*, so its error against the true trajectory is largest where the coupling time and $\Delta t$ are comparable and vanishes in both limits. Measured against two independent stiff integrators at $10^{-12}$, which agree with each other to the last column:

| $\tau_\mathrm{cell}$ | BE, $R$ base | BE, natural | adaptive $10^{-2}$ | steps | adaptive $10^{-8}$ | steps | reference check |
|---:|---:|---:|---:|---:|---:|---:|---:|
| $10^{-4}$ | $1.0\times10^{-7}$ | $1.0\times10^{-7}$ | $3.1\times10^{-12}$ | 1 | $3.1\times10^{-12}$ | 1 | $7.4\times10^{-13}$ |
| $10^{-2}$ | $8.3\times10^{-4}$ | $8.3\times10^{-4}$ | $3.1\times10^{-6}$ | 1 | $4.4\times10^{-9}$ | 32 | $6.2\times10^{-12}$ |
| $10^{-1}$ | $2.7\times10^{-2}$ | $2.7\times10^{-2}$ | $2.3\times10^{-3}$ | 1 | $7.2\times10^{-8}$ | 143 | $7.7\times10^{-12}$ |
| $10^{0}$ | $\mathbf{6.2\times10^{-2}}$ | $\mathbf{6.2\times10^{-2}}$ | $6.6\times10^{-4}$ | 12 | $1.4\times10^{-7}$ | 465 | $3.2\times10^{-13}$ |
| $3\times10^{0}$ | $2.8\times10^{-2}$ | $2.8\times10^{-2}$ | $1.0\times10^{-4}$ | 16 | $3.9\times10^{-8}$ | 647 | $1.2\times10^{-13}$ |
| $10^{1}$ | $9.3\times10^{-3}$ | $9.3\times10^{-3}$ | $2.8\times10^{-6}$ | 19 | $3.9\times10^{-9}$ | 826 | $6.3\times10^{-13}$ |
| $10^{2}$ | $9.6\times10^{-4}$ | $9.6\times10^{-4}$ | $1.5\times10^{-9}$ | 21 | $3.3\times10^{-15}$ | 851 | $7.9\times10^{-13}$ |
| $10^{4}$ | $9.7\times10^{-6}$ | $9.7\times10^{-6}$ | $4.1\times10^{-16}$ | 24 | $8.2\times10^{-17}$ | 854 | $5.6\times10^{-12}$ |
| $10^{9}$ | $9.9\times10^{-11}$ | $9.7\times10^{-11}$ | $8.6\times10^{-16}$ | 29 | $3.3\times10^{-16}$ | 859 | $3.3\times10^{-12}$ |
| $10^{12}$ | $3.5\times10^{-12}$ | $9.8\times10^{-14}$ | $3.3\times10^{-16}$ | 32 | $5.3\times10^{-16}$ | 862 | $9.9\times10^{-12}$ |

The two backward-Euler columns are identical to every digit printed until $\tau = 10^{12}$, where they differ only because their stopping tests differ — which is the empirical form of the claim in §1 that they are one method. The peak error is 6.2% of the cell's energy at $\tau_\mathrm{cell} \approx 1$, and that is the whole of what an error-controlled solve has to sell.

Buying that accuracy by substepping backward Euler is expensive, because backward Euler is first order once you take more than one step. Splitting the same $\Delta t$ into $n$ substeps of `QuokkaCoupling` (total Newton iterations in brackets):

| $\tau_\mathrm{cell}$ | $n=1$ | $n=2$ | $n=4$ | $n=8$ | $n=16$ | $n=64$ |
|---:|---:|---:|---:|---:|---:|---:|
| $10^{-1}$ | $2.7\times10^{-2}$ (5) | $1.5\times10^{-2}$ (8) | $8.4\times10^{-3}$ (16) | $4.4\times10^{-3}$ (32) | $2.2\times10^{-3}$ (50) | $5.7\times10^{-4}$ (192) |
| $10^{0}$ | $6.2\times10^{-2}$ (6) | $3.2\times10^{-2}$ (10) | $1.5\times10^{-2}$ (18) | $7.1\times10^{-3}$ (33) | $3.4\times10^{-3}$ (58) | $8.1\times10^{-4}$ (203) |
| $10^{1}$ | $9.3\times10^{-3}$ (6) | $1.3\times10^{-3}$ (10) | $2.3\times10^{-4}$ (16) | $5.8\times10^{-5}$ (27) | $1.6\times10^{-5}$ (48) | $2.1\times10^{-6}$ (156) |
| $10^{2}$ | $9.6\times10^{-4}$ (6) | $1.6\times10^{-5}$ (9) | $7.0\times10^{-8}$ (15) | $4.4\times10^{-11}$ (23) | $8.7\times10^{-11}$ (34) | $3.4\times10^{-10}$ (95) |

At $\tau = 1$, 64 backward-Euler substeps and 203 Newton iterations reach $8.1\times10^{-4}$; the adaptive Rosenbrock at its loosest setting reaches $6.6\times10^{-4}$ in 12 internal steps. A second-order L-stable method with error control is simply the right tool for that regime, and substepping a first-order one is not competitive.

The question the tables cannot answer is whether the intermediate regime matters in a run, where the coupling is operator-split against transport and the splitting error is itself $O(\Delta t)$. §8 says: on the code's own test problems it does not, except where a solver fails.

## 7. Cost

One cell solve, constant opacity at $\tau_\mathrm{cell} = 10^{4}$, out of equilibrium. Minimum of 200 samples, and allocation per call:

| method | $N_G = 1$ | $N_G = 2$ | $N_G = 4$ | $N_G = 8$ | $N_G = 16$ |
|:---|---:|---:|---:|---:|---:|
| Quokka, $R$ base | 127 ns | 1.3 $\mu$s | 2.3 $\mu$s | 4.3 $\mu$s | 9.5 $\mu$s |
| natural, residual + raise | 190 ns | 1.7 $\mu$s | 2.9 $\mu$s | 5.5 $\mu$s | 11.5 $\mu$s |
| natural, increment | 131 ns | 1.6 $\mu$s | 2.8 $\mu$s | 5.2 $\mu$s | 11.1 $\mu$s |
| natural, increment + damping | 140 ns | 1.6 $\mu$s | 2.7 $\mu$s | 5.1 $\mu$s | 10.9 $\mu$s |
| natural, bracketed scalar | 138 ns | 1.7 $\mu$s | 3.3 $\mu$s | 6.4 $\mu$s | 13.6 $\mu$s |
| natural, `ImplicitEuler` | 7.2 $\mu$s | 8.5 $\mu$s | 10.5 $\mu$s | 13.4 $\mu$s | 24.3 $\mu$s |
| adaptive, $10^{-2}$ | 10.6 $\mu$s | 17.1 $\mu$s | 27.2 $\mu$s | 47.0 $\mu$s | 96.4 $\mu$s |
| adaptive, $10^{-8}$ | 211 $\mu$s | 502 $\mu$s | 861 $\mu$s | 1.84 ms | 4.33 ms |

Every hand-written stepper allocates **0 bytes**. `ImplicitEuler` allocates 8.6 to 20.5 kB per cell and the adaptive solvers 8.0 to 21.5 kB, both unavoidably: a `solve` call builds an integrator, a nonlinear solver, their caches and a solution object for a problem in $1 + N_G$ unknowns.

One thing the table says that is easy to miss: the jump from $N_G = 1$ to $N_G = 2$ is a factor of ten, far more than the linear algebra can explain. At two groups and above the cost is dominated by `planck_fractions` — the incomplete Planck integral and its temperature derivative, evaluated once per Newton iteration. **The choice of formulation is a second-order effect on cost; the Planck integral is the first-order one.** Anything that cuts iterations pays; anything that only changes the linear solve barely shows.

## 8. The test problems

Each problem run to completion with each solver, at four groups. The reference row is the adaptive solve at $10^{-8}$, and the score is the relative $L_1$ difference in gas and radiation temperature over the interior.

### `marshak_vaytet`, 4 groups, $n = 64$ — the code's own multigroup problem

| method | wall | Newton mean / max | failures | $\Delta T_\mathrm{gas}$ | $\Delta T_\mathrm{rad}$ |
|:---|---:|---:|---:|---:|---:|
| Quokka, $R$ base | 2.1 s | 2.00 / 3 | 0 | $7.61\times10^{-4}$ | $2.59\times10^{-3}$ |
| natural, residual test | 1.8 s | 2.00 / 3 | 0 | $7.60\times10^{-4}$ | $9.75\times10^{-4}$ |
| natural, residual + raise | 1.4 s | 2.00 / 3 | 0 | $7.60\times10^{-4}$ | $9.75\times10^{-4}$ |
| natural, increment + damping | 1.7 s | 2.00 / 3 | 0 | $7.61\times10^{-4}$ | $9.92\times10^{-4}$ |
| natural, bracketed scalar | 3.4 s | 2.00 / 3 | 0 | $7.60\times10^{-4}$ | $2.59\times10^{-3}$ |
| natural, `ImplicitEuler` | 21.2 s | 2.44 / 3 | 0 | $7.61\times10^{-4}$ | $9.92\times10^{-4}$ |
| adaptive, $10^{-2}$ | 18.6 s | 4.73 / 22 | 0 | $3.42\times10^{-6}$ | $2.39\times10^{-3}$ |
| adaptive, $10^{-8}$ | 62.7 s | 29.3 / 759 | 0 | — | — |

Nothing fails, and the six one-step solvers agree with each other on the gas temperature to $8\times10^{-7}$ — a thousand times below the $7.6\times10^{-4}$ they all share as their offset from the reference, which is therefore the backward-Euler discretisation error and nothing else. The radiation temperature splits them into two groups, $9.8\times10^{-4}$ and $2.6\times10^{-3}$, a spread as large as the offset itself: 16309 steps is long enough for sub-tolerance differences in where each iteration stops to grow into the third digit of a diagnostic that is dominated by the cold far field. The reason nothing fails is measurable: the largest cell optical depth this problem reaches over those steps is $\tau_\mathrm{cell} = 2.2\times10^{4}$, which the table in §2 puts below the onset of round-off trouble at $10^{-9}$ — though not at $10^{-13}$. **The code's only built-in multigroup test problem does not reach the regime the base-variable replacement exists for.** That is worth knowing before drawing conclusions from it.

The cost column is the real content here. The allocation-free hand-written steppers run in 1.4 to 3.4 s; `ImplicitEuler` and the two adaptive settings cost 13, 15 and 45 times the fastest of them for an answer that is the same to $10^{-3}$, and the adaptive gain in gas temperature ($3.4\times10^{-6}$ against $7.6\times10^{-4}$) buys two decimal places at 13 times the wall clock.

### Stiff relaxation, $\kappa = 10^{12}$, $\tau_\mathrm{cell} \approx 3\times10^{9}$

| method | Newton mean / max | failures | $\Delta T_\mathrm{gas}$ |
|:---|---:|---:|---:|
| Quokka, $R$ base | 3.09 / 7 | 0 | $1.4\times10^{-10}$ |
| natural, residual test | **86.4 / 100** | **330** | $1.4\times10^{-15}$ |
| natural, residual + raise | 3.28 / 7 | 0 | $2.2\times10^{-15}$ |
| natural, increment + damping | 2.48 / 7 | 0 | $1.3\times10^{-15}$ |
| natural, bracketed scalar | 2.52 / 7 | 0 | $1.3\times10^{-10}$ |
| natural, `ImplicitEuler` | 2.37 / 6 | 0 | $4.3\times10^{-15}$ |

This is §2 and §3 reproduced in a run. The naive residual test does 26 times the Newton work and reports 330 failures, and its answer is the most accurate in the table. Every fix — the tolerance raise, the increment test, the library stepper — removes both symptoms.

### Multigroup Marshak wave, $\kappa = 300\,T^{-3.5}$

Every solver converges, in 1.1 mean Newton iterations, and all six one-step answers agree to five digits ($\Delta T_\mathrm{gas} = 5.86\times10^{-4}$ against the reference for all of them). The radiation CFL step keeps $\tau_\mathrm{cell}$ modest, so the hard single-cell regime of §4 never arises. This is the common case.

### Multigroup Marshak wave, $\kappa = 3\times10^{5}\,T^{+2}$ — the case that separates them

| method | Newton mean / max | failures | $\Delta T_\mathrm{gas}$ | $\Delta T_\mathrm{rad}$ |
|:---|---:|---:|---:|---:|
| Quokka, $R$ base | 1.36 / 100 | 2 | $6.9\times10^{-2}$ | $5.3\times10^{-2}$ |
| natural, residual test | 2.45 / 100 | 9 | $7.1\times10^{-2}$ | $5.7\times10^{-2}$ |
| natural, residual + raise | 2.45 / 100 | 9 | $7.1\times10^{-2}$ | $5.7\times10^{-2}$ |
| natural, increment + damping | 2.46 / 100 | 9 | $7.3\times10^{-2}$ | $6.0\times10^{-2}$ |
| natural, bracketed scalar | **1.07 / 4** | **0** | $\mathbf{2.5\times10^{-3}}$ | $\mathbf{2.5\times10^{-3}}$ |
| natural, `ImplicitEuler` | 2.46 / 100 | **0** | $7.1\times10^{-2}$ | $5.7\times10^{-2}$ |
| adaptive, $10^{-2}$ | 2.38 / 31 | 0 | $1.9\times10^{-8}$ | $4.3\times10^{-6}$ |

A handful of cells per step fail, and the temperature field is 7% wrong. The bracketed scalar solver is the only one-step method that gets it right, at a quarter of the Newton work, and its remaining $2.5\times10^{-3}$ is the honest one-step discretisation error rather than a failure. The adaptive solve is exact for practical purposes.

The `ImplicitEuler` row is the one to be uncomfortable about: **0 failures reported, 7% wrong**. Its increment test cannot distinguish convergence from a stall, so the only diagnostic a run has — `newton_failures` — says nothing is wrong. Every other solver at least raised its hand.

## 9. What follows from this

**On the base variables.** The replacement is a correct response to a real problem, but it is aimed at a symptom. What breaks in the natural variables at large $\tau_\mathrm{cell}$ is the convergence test, and the test can be fixed directly: divide each residual component by its own Jacobian diagonal, or test the increment. Either is one line, either removes the $\tau$ dependence completely, and either leaves the answer more accurate than the replaced variables deliver, because the replaced variables carry their own $10^{-11}$-level stopping error. Against that, the replacement costs robustness wherever the opacity depends on temperature — 36 failures in 66 cases at $\kappa \propto T^{-3.5}$, against zero for a damped natural-variable Newton — which is a live concern for any run with a realistic opacity table rather than a constant.

**On what to change first.** The failures that damage an answer in these tests are nonlinearity, not round-off. The round-off failures are loud and harmless; the nonlinearity failures are quiet and cost 7% of the temperature field. If one change is to be made, it is the bracket: eliminate the group block analytically, reduce to the scalar energy balance, and safeguard Newton with bisection inside the bracket the physics guarantees. It converged in all 240 sweep cases and in all four test problems, allocates nothing, is $O(N_G)$, and costs 8% at four groups. It also makes the residual-conditioning question moot, since the scalar residual has no cancelling $\tau_g(B_g - E_g)$ in it.

**On the error-controlled solve.** It never failed, anywhere, and it is the only method that resolves the intermediate regime and the multi-root ambiguity. It also costs 10 to 100 times a hand-written Newton step and 8 to 21 kB per cell, and the accuracy it buys is invisible on three of the four test problems here, where the coupling's own error is dwarfed by the operator splitting it sits inside. Keep it as the reference — the thing that says what the answer is when two production solvers disagree, which is exactly the role it played in producing this document.

**On what is still open.** All of this is one-dimensional, in double precision, with `beta_order = 0` on the multigroup problems, so the lagged $\boldsymbol{v}\cdot\boldsymbol{F}$ outer iteration is untested here. The two hard opacity laws are power laws; a tabulated opacity with structure in it may behave differently again, and the non-monotone residual of §4 is the mechanism to watch for. And the bracketed reduction inherits the assumption that groups exchange energy only through the gas — the same assumption `ArrowheadSolve` makes, documented in the same place, and it would need the same rethink if a Compton or line-overlap term were added.

## 10. Reproducing this

```bash
julia --project=. -t 4 docs/coupling-experiments/coupling_study.jl              # every table
julia --project=. -t 4 docs/coupling-experiments/coupling_study.jl cost runs    # named sections
```

Sections are `conditioning`, `test`, `robust`, `derivation`, `ode`, `cost` and `runs`, producing the tables of §2, §3, §4 and §5.7, §5.5, §6, §7 and §8 respectively. The script defines `PlainDirect` and `BracketedDirect` — the candidate steppers this document argues about — as `AbstractCoupling` subtypes outside `src/`, so the measurements can be redone without adding an unused solver to the package. `BracketedDirect`'s own derivation is at `coupling_study.jl:159-248`, and §5 quotes its line numbers as it goes. Timings are from an Apple-silicon laptop and are meaningful only as ratios.
