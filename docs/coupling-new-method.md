# The production matter-radiation coupling: `BracketedCoupling`

This document describes the multigroup matter-radiation coupling solver Honghu uses in production, `BracketedCoupling`, and is complete on its own: the physics, the derivation, the algorithm, the choice of root-finder, and the measurements. It is the default for every run with $N_G \ge 2$ frequency groups; `RadiationPhysics(coupling = BracketedCoupling())` states it explicitly. The code is `solve_coupling(::BracketedCoupling, …)` in `src/coupling_solvers.jl` and the functions it calls in `src/coupling.jl`. Every number below is reproduced by `docs/coupling-experiments/coupling_study.jl` (§8).

In one paragraph: one backward-Euler step of the coupling is a system of $1 + N_G$ nonlinear equations, but its $N_G$ group equations are linear in the group energies, so they are solved in closed form and the whole step reduces to **one scalar equation in the gas energy** — the statement that the cell's total energy is unchanged. That equation always has a sign change on a known interval, so it is handed as a lambda to a **bracketing root-finder** — by default the `Brent` of SciML's BracketingNonlinearSolve, Brent's method as Brent wrote it. Nothing in the solve forms a matrix or allocates memory, and over a 240-cell sweep spanning three opacity laws and sixteen decades of optical depth it fails on none.

## 1. The problem

### 1.1 The equations

In each cell, the gas internal energy $E_\mathrm{gas}$ and the group radiation energies $E_g$, $g = 1 \ldots N_G$, exchange energy by emission and absorption:

$$\frac{dE_g}{dt} = \hat c_g \left(\varepsilon_g(T) - \alpha_g(T)\,E_g\right) , \qquad \frac{dE_\mathrm{gas}}{dt} = -c \sum_g \left(\varepsilon_g(T) - \alpha_g(T)\,E_g\right) ,$$

with gas temperature $T = E_\mathrm{gas}/c_V$ and two coefficients per group,

$$\varepsilon_g(T) = \rho\,\kappa_{P,g}(T)\,\frac{4\pi B_g(T)}{c} , \qquad \alpha_g(T) = \rho\,\kappa_{E,g}(T) .$$

$\varepsilon_g$ is the rate at which the gas emits into group $g$, $\alpha_g$ the coefficient of the rate at which the group is absorbed, and $4\pi B_g/c = a_\mathrm{rad} T^4 f_g(T)$ is the Planck energy density in the group, with $f_g$ the fraction of a Planck spectrum between the group's edges. $\kappa_{P,g}$ and $\kappa_{E,g}$ are the Planck- and energy-mean opacities; in this code they are equal, because the multigroup opacity is piecewise constant across a group.

The two equations differ by the factor $c/\hat c_g$. That asymmetry is the reduced speed of light approximation: the radiation side runs at $\hat c_g \le c$ and the gas side at the full $c$. It means the conserved quantity is not $E_\mathrm{gas} + \sum_g E_g$ but

$$E_\mathrm{gas} + \sum_g \frac{c}{\hat c_g}\,E_g ,$$

which every formula below respects.

The system is stiff. In an optically thick cell the coupling time $1/(\hat c\,\rho\,\kappa)$ is shorter than the transport step by factors up to $10^{12}$ in the cells tested here, so an explicit step is out of the question.

### 1.2 One backward-Euler step

Over a step $\Delta t$, two increments are held constant and supplied by the driver: an external radiation source $S_g$, and the lagged $\boldsymbol{v}\cdot\boldsymbol{F}$ work term $W_g$. Folding them into the old state,

$$\mathrm{gas0} = E_\mathrm{gas}^0 - \sum_g \frac{c}{\hat c_g} W_g , \qquad \mathrm{rad0}_g = E_g^0 + S_g + W_g ,$$

and writing the step's two optical depths as

$$\tau_g(T) = \Delta t\,\hat c_g\,\rho\,\kappa_{P,g}(T) , \qquad a_g(T) = \Delta t\,\hat c_g\,\alpha_g(T) = \Delta t\,\hat c_g\,\rho\,\kappa_{E,g}(T) ,$$

the backward-Euler step $u - u^0 - \Delta t\,f(u) = 0$ is

$$F_g \equiv (1 + a_g)\,E_g - \mathrm{rad0}_g - \Delta t\,\hat c_g\,\varepsilon_g(T) = 0 , \qquad g = 1 \ldots N_G ,$$

$$F_0 \equiv (E_\mathrm{gas} - \mathrm{gas0}) + \sum_g \frac{c}{\hat c_g}\left(\Delta t\,\hat c_g\,\varepsilon_g(T) - a_g E_g\right) = 0 .$$

That is $1 + N_G$ nonlinear equations in $1 + N_G$ unknowns. The obvious approach is Newton's method on all of them. `BracketedCoupling` does not do that.

## 2. The reduction to one equation

### 2.1 The group block is linear: eliminate it

No group appears in another group's equation; the groups meet only through the gas temperature. So at a fixed $E_\mathrm{gas}$ — hence a fixed $T$ — the $N_G$ equations $F_g = 0$ are independent and **linear** in $E_g$, and are solved outright:

$$\boxed{\;E_g(T) = \frac{\mathrm{rad0}_g + \Delta t\,\hat c_g\,\varepsilon_g(T)}{1 + \Delta t\,\hat c_g\,\alpha_g(T)}\;}$$

With $\kappa_{P,g} = \kappa_{E,g}$ this reads

$$E_g(T) = \frac{1}{1+\tau_g}\,\mathrm{rad0}_g + \frac{\tau_g}{1+\tau_g}\,\frac{4\pi B_g(T)}{c} ,$$

a **convex combination of where the group started and the Planck value at the gas temperature**, weighted by the optical depth of the step. As $\tau_g \to 0$ the group keeps its energy; as $\tau_g \to \infty$ it is in equilibrium with the gas. All of the stiffness is in that one weight, and it is handled exactly, with no iteration.

### 2.2 What is left is energy conservation

Rearranging $F_g = 0$ gives $\Delta t\,\hat c_g\,\varepsilon_g - a_g E_g = E_g - \mathrm{rad0}_g$: the net energy the gas hands to group $g$ over the step is the change in that group's energy. Substituting it into $F_0$ removes the emission and absorption terms and leaves one scalar equation in one unknown,

$$G(E_\mathrm{gas}) \equiv (E_\mathrm{gas} - \mathrm{gas0}) + \sum_g \frac{c}{\hat c_g}\left(E_g(T) - \mathrm{rad0}_g\right) = 0 .$$

Expanding `gas0` and `rad0`, the work term cancels out of $G$ completely — it survives only inside $E_g(T)$ — and

$$G(E_\mathrm{gas}) = E_\mathrm{gas} + \sum_g \frac{c}{\hat c_g}\,E_g(T) - \mathcal{E} , \qquad \mathcal{E} \equiv E_\mathrm{gas}^0 + \sum_g \frac{c}{\hat c_g}\left(E_g^0 + S_g\right) .$$

**$G$ is the new total energy minus the old.** The step is: find the gas energy at which the cell's books balance. It also means that $G$ at the returned answer is, exactly, the solver's energy-conservation error, which matters in §4.3.

One contract follows from the derivation and is worth stating because it is easy to break. Exactly, $F_0 = G - \sum_g (c/\hat c_g)\,F_g$, so $G$ is the residual of the step **only where the group block is solved**. Given arbitrary group energies it vanishes in the wrong place; given the $E_g(T)$ of §2.1 it is the whole problem.

### 2.3 In the code

The physics is written once and everything else is built from it:

| function (`src/coupling.jl`) | returns | role |
|:---|:---|:---|
| `coupling_coefficients(s, T)` | $(\varepsilon_g, \alpha_g)$ | the single source of truth for the coupling physics |
| `coupling_coefficients_dT(s, T)` | $(d\varepsilon_g/dT,\, d\alpha_g/dT)$ | their temperature derivatives |
| `group_energy_values(s, Egas)` | $E_g(T)$ | §2.1; what the root-finder evaluates |
| `group_energies(s, Egas)` | $E_g(T)$, $dE_g/dE_\mathrm{gas}$ | §2.1 plus derivative; used once per cell, in §4.3 |
| `energy_balance(s, Egas, Erad)` | $G$ | §2.2; contains no physics, only the $c/\hat c_g$ weights |

The same two coefficients also build `coupling_rhs` and `coupling_jacobian`, which the error-controlled reference solver `AdaptiveCoupling` integrates, so the production solver and its reference cannot drift apart.

## 3. The bracket, and which root

### 3.1 A sign change always exists

$G$ is a function of a *trial* gas energy, and bracketing means evaluating it at trial values deliberately below and above the answer — including values the cell could never physically have — and checking that the sign differs.

**Below: $E_\mathrm{gas} = c_V T_\mathrm{floor}$.** Every $4\pi B_g/c \to 0$ there, and $E_g$ is a convex combination of $\mathrm{rad0}_g$ and $4\pi B_g/c$, so $E_g \le \mathrm{rad0}_g$ and

$$G(c_V T_\mathrm{floor}) \le c_V T_\mathrm{floor} + \sum_g \frac{c}{\hat c_g}W_g - E_\mathrm{gas}^0 \le 0 .$$

The last inequality holds whenever the gas starts above its floor and the work term does not by itself exceed the gas energy; with $W_g = 0$ it is unconditional. It also needs $\kappa_{P,g}B_g \to 0$ as $T \to 0$, true for $\kappa \propto T^n$ with $n > -4$ and for any physical opacity.

**Above: $E_\mathrm{gas} = \mathcal{E}$**, the whole cell's energy in the gas. The first and last terms of $G$ cancel outright:

$$G(\mathcal{E}) = \sum_g \frac{c}{\hat c_g}\,E_g(T = \mathcal{E}/c_V) \ge 0 ,$$

because every group energy is non-negative whenever $\mathrm{rad0}_g \ge 0$. Asking the gas to hold the whole cell's energy leaves the radiation with some over, so the books over-balance.

$G$ is continuous, so a root lies in $[c_V T_\mathrm{floor}, \mathcal{E}]$ — which also shows that the solved gas energy never exceeds the cell's total. Checked over the 240-cell sweep of §5.2: the identity $G(\mathcal{E}) = \sum_g (c/\hat c_g)E_g$ holds to one ulp, and the interval brackets the root in every cell.

### 3.2 How $G$ behaves, and why a bracket is still needed

Differentiating §2.1,

$$\frac{dG}{dE_\mathrm{gas}} = 1 + \sum_g \frac{c}{\hat c_g}\,\frac{dE_g}{dE_\mathrm{gas}} , \qquad \frac{dE_g}{dE_\mathrm{gas}} = \frac{\Delta t\,\hat c_g\left(\varepsilon_g' - \alpha_g' E_g\right)}{c_V\,(1 + \Delta t\,\hat c_g\,\alpha_g)} ,$$

with $' = d/dT$. For $\kappa \propto T^n$ and $\kappa_P = \kappa_E$, substituting the eliminated $E_g$:

$$\frac{dE_g}{dT} = \underbrace{\frac{\tau_g}{1+\tau_g}\,\frac{d}{dT}\!\left(\frac{4\pi B_g}{c}\right)}_{\text{emission, always} > 0} + \underbrace{\frac{n\,\tau_g}{T\,(1+\tau_g)^2}\left(\frac{4\pi B_g}{c} - \mathrm{rad0}_g\right)}_{\text{opacity moving with } T} .$$

The second term vanishes when the opacity is constant, and decays like $1/\tau_g$ in the optically thick limit, where

$$\frac{dG}{dE_\mathrm{gas}} \to 1 + \frac{c}{\hat c}\,\frac{4 a_\mathrm{rad} T^3}{c_V} > 0 .$$

So **$G$ is monotone for a constant opacity, and monotone at large $\tau$ for any opacity law**. At moderate $\tau$, with a steep opacity law and a cell far from equilibrium, the second term can win and $G$ can have more than one root. That is why the solver needs a bracket at all, and why the bracket needs care (§3.3).

A worked example of $G$ at a hard cell — gas at $T = 0.5$, radiation at $1.3$, $\kappa \propto T^2$, $\tau = 2.5\times10^{8}$, four groups, code units — shows the reduction behaving well where the unreduced $(1+N_G)$-dimensional Newton does not. At the opening iterate $E_\mathrm{gas} = 0.5$ the eliminated group energies are already near equilibrium with the gas ($\sum_g E_g = 0.0625$), $G = -2.79$ and $dG/dE_\mathrm{gas} = 1.500$ against the predicted $1 + 4a_\mathrm{rad}T^3/c_V = 1.5$, so the first step points at the root, $E_\mathrm{gas} = 1.2103$. An iteration that keeps $E_g$ as independent unknowns linearises instead about the initial group energies ($\sum_g E_g = 2.856$), where the opacity term above is $-11.2$; its derivative comes out $-9.67$ and its first step points the wrong way.

### 3.3 Which root: march from the old state

A bracketing method returns *a* root in its bracket and does not say which. On the sweep of §5.2 one cell has three: $\kappa \propto T^2$, gas at $0.1$, radiation at $3$, $\tau = 10^{-2}$, where $G$ changes sign at $E_\mathrm{gas} = 0.110$, $1.17$ and $2.52$. An error-controlled integration of the ODE of §1.1 across the step ends at $0.109$, so $0.110$ is the physical root. Solved on the wide bracket $[c_V T_\mathrm{floor}, 4\mathcal{E}]$, ModAB and Bisection both returned $2.52$ — the gas handed 23 times the energy it physically reaches — while the library's Brent and ITP happened to return $0.110$. Which root came back depended on the method.

So the bracket is not taken as the whole interval of §3.1. It is **built by marching outward from the old gas energy** by factors of two, in the direction the sign of $G$ says the root lies, until the sign changes (`_energy_bracket` in `src/coupling_solvers.jl`). This isolates the root continuously connected to where the cell started — the one a small $\Delta t$ would reach — unless two roots lie within a factor of two of each other. With it, every bracketing method tried (`ZBrent`, the library's Brent, ModAB, ITP, Bisection) returns $0.110$ on that cell and agrees with the others to $10^{-9}$ on all 240. The march stops at the temperature floor going down and after 200 doublings going up, and if it finds no sign change the cell is reported unconverged rather than guessed at; this never happened on the sweep or the test problems. Typical cost is one or two evaluations of $G$.

## 4. The algorithm

### 4.1 The solve

```julia
G = (Egas, s) -> energy_balance(s, Egas, group_energy_values(s, Egas))   # the equation
lo, hi, nbracket, found = _energy_bracket(G, s)                          # §3.3
sol = solve(IntervalNonlinearProblem{false}(G, (lo, hi), s), m.root_finder;
            abstol = atol, maxiters = m.max_iterations)                  # the algorithm
```

The root-finder is a field of the solver, `BracketedCoupling(root_finder = Brent())` by default, the `Brent` of SciML's BracketingNonlinearSolve. It is called through SciMLBase's `IntervalNonlinearProblem` interface, so any bracketing method written to that interface is interchangeable with it, in particular the package's other methods; §5 compares them. Both packages were already in OrdinaryDiffEq's dependency tree and already loaded, so using them added nothing to the install or to the load time.

The cell state travels as the problem parameter rather than being captured by the lambda, so the lambda closes over nothing and costs nothing. `_energy_bracket` takes the equation as `G::F … where {F}`, because Julia does not specialise on a function argument it only passes through, and without that the same code allocated 2.8 kB per cell.

### 4.2 Why derivative-free

$dG/dE_\mathrm{gas}$ is known in closed form (§3.2), so Newton's method is an obvious alternative. It is not the better one, because of what an evaluation costs: $G$ needs one incomplete Planck integral per call, and $dG/dE_\mathrm{gas}$ needs its temperature derivative as well. At $N_G \ge 2$ that integral dominates the cost of the whole solve. A derivative-free method that takes about as many, cheaper evaluations wins: Brent's and `ModAB`'s 12 value-only evaluations per cell take 0.57 of the time of the 9 value-and-derivative evaluations a Newton-in-bracket needed on the same sweep.

### 4.3 Converting the tolerance

Every bracketing method in BracketingNonlinearSolve, `Brent` included, stops when the bracket is narrower than `abstol` **in $E_\mathrm{gas}$**. The solver promises something else: $|G| \le \mathrm{tol}\cdot\mathcal{E}$, and by §2.2 $G$ is the cell's energy-conservation error. The two differ by $dG/dE_\mathrm{gas}$, which by §3.2 reaches $1 + (c/\hat c)\,4a_\mathrm{rad}T^3/c_V$ — thousands in a radiation-dominated cell, and more under a strongly reduced $\hat c$. Passing $\mathrm{tol}\cdot\mathcal{E}$ as `abstol` alone silently loosens conservation by that factor: measured with `ZBrent` over the sweep, 72 of 240 cells then conserve energy only to $1.9\times10^{-6}$ against a promised $10^{-9}$.

So after the first solve the residual is checked. If it is too large, the solve is repeated on the library's own final bracket, `(sol.left, sol.right)`, with `abstol` divided by $dG/dE_\mathrm{gas}$ evaluated at the answer — effectively constant across a bracket that narrow, so one repeat always suffices, and it starts from an interval a few `abstol` wide. This is the one place the derivative is used, through `group_energies`. **Convergence is judged on $|G|$, not on the library's return code.**

### 4.4 A convergence test that $\tau$ cannot break

In a Newton iteration on the $1 + N_G$ equations of §1.2, the group residual contains $\tau_g\,(4\pi B_g/c - E_g)$. At the solution $4\pi B_g/c$ and $E_g$ agree to round-off, so the difference carries an absolute error of $\varepsilon_\mathrm{mach}\cdot 4\pi B_g/c$ and the residual an error of $\varepsilon_\mathrm{mach}\,\tau_g$ times the cell's energy. A residual test at tolerance $10^{-11}$ cannot then be satisfied once $\tau_g \gtrsim 10^{5}$, however exact the answer. $G$ contains no such term: the cancellation was carried out analytically in §2.2, and $G$ is a difference of energies of ordinary size. The tolerance means what it says at every optical depth.

### 4.5 What `iterations` reports

`CouplingResult.iterations`, which feeds the run's `newton_mean` / `newton_max` diagnostics, counts the evaluations of $G$ made by the bracket march plus one per call to the root-finder. The root-finder interface returns no evaluation count without allocating (SciMLBase's statistics type is mutable), and counting through the lambda allocates 16 bytes per cell. Read a `newton_mean` of about 3 for this solver as "one march step each way plus one solve", not as three Newton iterations.

## 5. The root-finder: Brent's method, and how it compares

Three bracketing methods are compared here. All interpolate where interpolation helps and never leave the bracket. Until 2026-10-17 the default was `ZBrent`, an in-tree copy of Brent's method, written because the library's `Brent` was not that method (§5.1). The fix went upstream and `ZBrent` is gone (§5.6). The measurements below were taken with `ZBrent`; the library's `Brent` from 1.12.8 on reproduces its rows exactly, so in the tables `ZBrent()` stands for today's default and "`Brent()` (library)" for the library before 1.12.8.

- **`ZBrent`**, the default until 2026-10-17, formerly in `src/rootfind.jl`: Brent's method as Brent (1973) wrote it and as Numerical Recipes transcribes it in `zbrent` — inverse quadratic interpolation or a secant step when either shrinks the bracket fast enough, bisection otherwise, and a minimum step (§5.1).
- **`Brent`** from SciML's BracketingNonlinearSolve before 1.12.8, which was a transcription of textbook pseudocode rather than of Brent's algorithm.
- **`ModAB`** from the same package, a modified Anderson–Björck method: regula falsi with a correction that stops one end of the bracket from sticking.

All three plug into the same interface, and the test suite checks the default against `ModAB`, `ITP` and `Bisection`.

### 5.1 What the library's `Brent` got wrong before 1.12.8

Two defects, both visible in its source then and both measured:

- **No minimum step.** When the interpolated step would move the estimate $b$ by less than $\mathrm{tol}_1 = 2\varepsilon_\mathrm{mach}|b| + \mathtt{abstol}$, Brent's algorithm moves it by $\mathrm{tol}_1$ towards the contrapoint instead. On a smooth convex function the interpolated iterates converge to the root from one side, so without that step the far end of the bracket is never replaced except by bisection — and the method stops only when the *bracket* is narrow, which then takes many halvings. The minimum step deliberately probes just past the root, and the far end collapses in one evaluation.
- **Two evaluations per iteration.** It recomputes $f(c)$ at the top of every iteration, where $c$ is always an earlier point whose value is already known.

On two analytic functions, counting every evaluation of $f$:

| $f(x)$, bracket, `abstol` | `ZBrent` | library `Brent` | `ModAB` | `ZBrent` without the minimum step |
|:---|---:|---:|---:|---:|
| $x^2 - 2$, $[1, 2]$, $10^{-15}$ | 9 | 20 (9 distinct points) | 10 | 9 |
| $x^4 - 2$, $[0, 4]$, $5\times10^{-11}$ | **12** | 62 (30 distinct points) | 15 | 50 |

$x^2 - 2$ shows the second defect alone: the library's `Brent` visits the same 9 points as `ZBrent` but evaluates most of them twice. $x^4 - 2$ — the same $T^4$ shape as the energy balance $G$ — shows the first: with the minimum step removed, `ZBrent` itself needs 50 evaluations instead of 12. (A report that prompted this comparison found $x^2 - 2$ taking 37 evaluations instead of 8 under FMA contraction in compiled C and CUDA. Julia does not contract $x^2 - 2$, and there the minimum step makes no difference; on $x^4 - 2$ and on $G$ it does.)

On $G$ itself, over the sweep of §5.2 and counting only the root-finder's own evaluations: the library's `Brent` takes 20.1 on average and 70 in the worst cell; `ZBrent` without the minimum step 13.0 and 61; `ZBrent` 7.7 and 12; `ModAB` 9.5 and 12.

### 5.2 Robustness and cost over the sweep

The sweep is 240 single cells: 8 initial gas and radiation temperature pairs (gas hotter, radiation hotter, and both near equilibrium, over two decades), 10 optical depths $\tau$ from $10^{-4}$ to $10^{12}$, and three opacity laws — constant, $\kappa \propto T^{-3.5}$ and $\kappa \propto T^{2}$ — at four groups and tolerance $10^{-9}$. "$G$ evaluations" counts everything the solve does: the bracket march, the root-finder, and the §4.3 repeat. "vs ref" is the largest difference in $E_\mathrm{gas}$ from a Bisection solve at tolerance $10^{-13}$, and "energy error" the largest $|E_\mathrm{gas} + \sum_g E_g - \mathcal{E}|$; both are in units of the cell's total energy.

| `root_finder` | failures | $G$ evaluations | cells re-solved (§4.3) | vs ref | energy error | time per cell |
|:---|---:|---:|---:|---:|---:|---:|
| `ZBrent()` (default) | 0 / 240 | **12.0** | 72 | $3.5\times10^{-10}$ | $4.9\times10^{-10}$ | **1.88 μs** |
| `Brent()` (library) | 0 / 240 | 25.0 | 24 | $9.3\times10^{-10}$ | $9.3\times10^{-10}$ | 3.33 μs |
| `ModAB()` | 0 / 240 | 12.7 | 0 | $6.8\times10^{-14}$ | $8.8\times10^{-16}$ | **1.88 μs** |

All three are perfectly robust on the sweep, and `ZBrent` and `ModAB` cost the same. They differ in how they finish. Both Brent variants stop essentially when the bracket reaches the requested width, so their answers sit just inside the tolerance and many cells need the §4.3 repeat — 72 for `ZBrent`, whose narrow final brackets put its answers closer to the edge. `ModAB`'s final steps converge superlinearly past the requested width, so it lands at machine precision and never needs the repeat. `ZBrent` makes up the repeats with fewer evaluations to get there.

### 5.3 Cost against group count

One cell, constant opacity, $\tau = 10^{4}$, out of equilibrium; minimum over 200 samples.

| `root_finder` | $N_G = 2$ | $N_G = 4$ | $N_G = 8$ | $N_G = 16$ |
|:---|---:|---:|---:|---:|
| `ZBrent()` | **0.88 μs** | **1.50 μs** | **2.69 μs** | **5.51 μs** |
| `Brent()` (library) | 1.29 μs | 2.26 μs | 4.13 μs | 8.43 μs |
| `ModAB()` | 0.94 μs | 1.66 μs | 2.84 μs | 6.63 μs |

All three allocate **0 bytes** at every group count, and all scale linearly in $N_G$, because each evaluation of $G$ is one pass over the groups. `ZBrent` is 1.5 times faster than the library's `Brent` and 5 to 17 per cent faster than `ModAB`.

### 5.4 The test problems

Each problem run to completion at four groups; error is the relative $L_1$ difference in gas temperature from `AdaptiveCoupling` at relative tolerance $10^{-8}$, which integrates the ODE of §1.1 rather than taking one backward-Euler step.

| problem | `ZBrent()` | `Brent()` (library) | `ModAB()` | error in $T_\mathrm{gas}$ |
|:---|---:|---:|---:|---:|
| `marshak_vaytet`, $n = 64$ | **2.47 s** | 4.91 s | 2.82 s | $7.61\times10^{-4}$ (all three) |
| stiff relaxation, $\kappa = 10^{12}$ | 0.15 s | 0.15 s | 0.15 s | $1.2\times10^{-10}$, $1.9\times10^{-10}$, $1.4\times10^{-15}$ |
| Marshak wave, $\kappa = 300\,T^{-3.5}$ | 0.18 s | 0.21 s | 0.17 s | $5.859\times10^{-4}$ (all three) |
| Marshak wave, $\kappa = 3\times10^{5}\,T^{2}$ | < 0.01 s | < 0.01 s | < 0.01 s | $2.497\times10^{-3}$ (all three) |

No solve failed on any problem. `marshak_vaytet` is the code's built-in multigroup test problem (Vaytet et al., frequency-dependent opacity, 16309 steps). The stiff relaxation is a periodic box with $\tau_\mathrm{cell} \approx 3\times10^{9}$. The last two are Marshak waves with temperature-dependent opacity; the $T^{2}$ one is where solvers that iterate on all $1 + N_G$ unknowns fail on some cells and leave the temperature field 7% wrong.

The three root-finders give the same temperature field on every problem, to the digits shown, except where all are already at the $10^{-10}$ level. The errors in the table are not the root-finders' errors but the discretisation error of taking one backward-Euler step, which all three share (§6.2). The difference is cost: on `marshak_vaytet`, where the coupling runs in every cell of 16309 steps, the library's `Brent` takes twice as long as `ZBrent`.

### 5.5 Which to use

**Brent's method is the default**, now as the library's `Brent()` from 1.12.8, which is the `ZBrent` of the tables: the fastest of the three per cell and on the one test problem long enough to time, and it keeps the solver's promise exactly — conservation to the requested tolerance. **`ModAB` is equivalent** in robustness and cost on everything measured, and conserves energy to machine precision rather than to the tolerance; pass `BracketedCoupling(root_finder = ModAB())` if that margin is wanted, with `using BracketingNonlinearSolve: ModAB`. **The library's `Brent` before 1.12.8** was as robust but twice as expensive, for the reasons of §5.1; `Project.toml` now requires 1.12.8. Keep `Bisection()` for checking: it is slow — 37 evaluations on average — and cannot return a wrong answer from a valid bracket, which is what makes it a reference.

### 5.6 The upstream fix, and the switch to it

The two defects of §5.1 are reported upstream with a fix, as [SciML/NonlinearSolve.jl#1340](https://github.com/SciML/NonlinearSolve.jl/pull/1340) (opened 2026-09-28, from the branch `brent-zbrent-minimum-step` of the fork `chongchonghe/NonlinearSolve.jl`). It replaces the library's loop with Brent's `zero` / `zbrent`, so the library's `Brent` evaluates $f$ once per iteration and takes the minimum step. On the functions of §5.1 it brings the library's counts down to `ZBrent`'s: 20 to 9 on $x^2 - 2$ and 62 to 12 on $x^4 - 2$. It was merged the same day and released in BracketingNonlinearSolve 1.12.8 on 2026-09-29.

**Honghu now uses it.** On 2026-10-17 `BracketingNonlinearSolve.Brent()` became the default `root_finder`, BracketingNonlinearSolve moved from `[extras]` to `[deps]` with compat 1.12.8, and `src/rootfind.jl` was deleted. Three checks came first, on 1.12.8:

1. The `rootfind` section of `coupling_study.jl`, run with both: over the sweep the same 12.0 evaluations of $G$ a cell, the same 72 re-solves and the same deviations to every printed digit, 1.76 against 1.76 μs a cell; 0 bytes at every group count; and the same results on all four test problems.
2. On $x^2 - 2$ and $x^4 - 2$, at the positive tolerances the solver uses, the two visit the same points and return the same root to the last bit. At `abstol = 0` the library takes one evaluation more and stops with `FloatingPointLimit`, which the solver never asks for.
3. `test/test_rootfind.jl` now tests the library's `Brent` as the default: each point evaluated once, the minimum step on $x^4 - 2$ in at most 14 evaluations, zero allocation.

## 6. Accuracy and cost

### 6.1 The solve is exact to its tolerance

Over the sweep of §5.2 the solver never fails and every answer is within $10^{-9}$ of the cell's energy of a Bisection solve at $10^{-13}$, for each of the three root-finders. It allocates nothing and costs $O(N_G)$ per evaluation, with no matrix formed and no linear system solved.

### 6.2 One backward-Euler step is not exact

`BracketedCoupling` solves the backward-Euler equations exactly; those equations are a first-order discretisation of the ODE of §1.1, and one step of them has an error. It is largest where the coupling time and $\Delta t$ are comparable, and vanishes in both limits: nothing happens in a thin cell, and a thick cell goes to the same equilibrium either way. Measured on one cell (four groups, gas at 1.3, radiation at 0.5, constant opacity) against two independent stiff integrators at $10^{-12}$, in units of the cell's total energy:

| $\tau$ | $10^{-4}$ | $10^{-2}$ | $10^{-1}$ | $1$ | $3$ | $10$ | $10^{2}$ | $10^{4}$ | $10^{9}$ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| error of one step | $1.0\times10^{-7}$ | $8.3\times10^{-4}$ | $2.7\times10^{-2}$ | $6.2\times10^{-2}$ | $2.8\times10^{-2}$ | $9.3\times10^{-3}$ | $9.6\times10^{-4}$ | $9.7\times10^{-6}$ | $9.7\times10^{-11}$ |

The peak, 6% at $\tau \sim 1$, is a property of the time discretisation, not of the solver, and in a run it sits inside the operator-splitting error of coupling to transport. `AdaptiveCoupling` integrates the ODE with error control and removes it, at about 12 times the cost on its loosest setting and hundreds of times on a tight one, plus 8 to 21 kB of allocation per cell; it is the reference for checking, not a production option.

## 7. Limitations

- **Groups exchange energy only through the gas.** The closed form of §2.1 is a per-group division because no group appears in another's equation. A term that couples groups directly — Compton redistribution, overlapping lines — makes the group block a matrix, and the reduction to one equation is gone. `BracketedCoupling` would have to be replaced, not adjusted.
- **Roots closer than a factor of two.** The march of §3.3 isolates the root connected to the old state unless another root lies within a factor of two of it. None did on the sweep.
- **Negative starting group energies.** The bracket of §3.1 needs $\mathrm{rad0}_g = E_g^0 + S_g + W_g \ge 0$; a lagged work term more negative than a group's own energy can break it. The march then finds no sign change and the cell is reported unconverged, not solved wrongly.
- **What has been tested.** One-dimensional problems, double precision, `beta_order = 0` on the multigroup problems (so the lagged $\boldsymbol{v}\cdot\boldsymbol{F}$ outer iteration is not exercised), and power-law opacities. A tabulated opacity with structure in it may produce the non-monotone $G$ of §3.2 more often than the sweep did.

## 8. Code map and reproducing

| where | what |
|:---|:---|
| `src/coupling.jl` | `coupling_coefficients`, `group_energy_values`, `group_energies`, `energy_balance` (§2.3) |
| `src/coupling_solvers.jl` | `BracketedCoupling`, its `solve_coupling`, `_energy_bracket` (§3.3, §4) |
| `src/Honghu.jl` | the selective imports of `IntervalNonlinearProblem` from SciMLBase and `Brent` from BracketingNonlinearSolve |
| `test/test_rootfind.jl` | the library's `Brent`, as the default: each point evaluated once, the minimum step on $x^4 - 2$ (fails without it: 50 evaluations against a limit of 14), zero allocation |
| `test/test_radiation_multigroup.jl` | zero-byte allocation of the coupling; the testset "BracketedCoupling: the root-finder chooses how, not what", which pins §3.3, §4.3 and agreement between Brent, ModAB, ITP and Bisection |

```bash
julia --project=. -t 4 docs/coupling-experiments/coupling_study.jl rootfind     # §5, every table
julia --project=. -t 4 docs/coupling-experiments/coupling_study.jl derivation   # §3.1, §3.2
julia --project=. -t 4 docs/coupling-experiments/coupling_study.jl ode          # §6.2
julia --project=. -t 4 test/runtests.jl                                          # the full suite
```

The `rootfind` section counts evaluations with a re-assembly of the solve that carries a counter, and asserts that it returns the shipped solver's gas energy to the last bit before reporting its counts. Timings are from an Apple-silicon laptop and are meaningful only as ratios.
