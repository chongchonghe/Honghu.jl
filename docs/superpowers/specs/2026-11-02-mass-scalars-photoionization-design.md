# Mass scalars and hydrogen photoionization for Honghu: design

Status: revised after a second review and a cost measurement, 2026-11-02, waiting for the author's approval. Items marked "my call" are decisions I made and the author has not explicitly approved.

## 1. Intent

Honghu is a bench for algorithms and test problems that are destined for Quokka. Quokka has a photoionization module (beta), described in `quokka-local/docs/markdown/photoionization.md`, and Honghu has none. The goal is to evolve species abundances (H$^+$ now, He$^+$ and He$^{++}$ later) as passive mass scalars in conserved form, and to evolve the hydrogen ionization state and the gas temperature from the ionizing radiation field, following Aubert & Teyssier (2008) with the on-the-spot approximation (OTSA).

The first physical target is a Stromgren sphere around a point source. The benchmark is Iliev et al. (2006), Test 1 (isothermal) and Test 2 (heating and cooling).

The work has two layers, and each is useful alone:

- Layer 1: mass scalars. Extra conserved variables $\rho X_s$ that are advected with the gas, with no effect on the hydrodynamics. This is also the substrate for any later species.
- Layer 2: photoionization chemistry. A species network that is independent of the integrator, a hydrogen OTSA network on top of it, two integrators, and the coupling to the existing M1 radiation.

## 2. Goals, non-goals, anti-goals

Goals:

- Mass scalars work in every reconstruction, both Riemann solvers, all boundary conditions, initial conditions, output, checkpoints and `conserved_totals`.
- A chemistry network is a plain physics object (species, rates, Jacobian). An integrator never contains hydrogen-specific code, and a network never contains integrator-specific code.
- The hydrogen network follows the Quokka note: photoionization, case B recombination, collisional ionization, photoheating, and cooling. Temperature evolves in Test 2 and is held fixed in Test 1 (an explicit isothermal mode).
- Two integrators on the same network. The main one is an error-controlled Rosenbrock method. The second is a single backward-Euler step without sub-cycling, kept for comparison.
- Iliev Tests 1 and 2 run as an example and as tests, and the two integrators are compared on them in a short findings note.
- Extensibility is tested, not asserted: a small test-only network with an exact solution runs through both integrators unchanged.

Non-goals for this version:

- Case A diffuse recombination photons, helium, metals, molecules, Compton cooling, dust coupling.
- Momentum deposition by absorbed photons. Chemistry requires `beta_order = 0` (my call).
- Chemistry with any limiter other than `ConstantChat` (confirmed by the author). The reason is that the `StabilityBound` derivation has no photoabsorption term, so its stability bound does not cover the chemistry's absorption. Chemistry throws for any other limiter, as `wavespeed_correction` does.
- Validated dynamic HII-region problems. Running with hydro on must work and the scalars must advect, but no hydro-on photoionization benchmark is part of this work.
- ATON-style sub-cycling of the chemistry step, and an in-tree integrator (only a last fallback, section 5).
- More than one ionizing band in an actual run. The network interface allows it; only one band is exercised.

Anti-goals, which would make this a failure even if the benchmark passes:

- Any change to existing results when there are no scalars and no chemistry. Those runs must stay bit for bit identical, as `ConstantChat` is.
- Scalars that feed back on the hydrodynamics (they are passive: pressure uses a fixed $\gamma$).
- Allocation in the hydro kernels because of scalars, or flooring inside a conversion kernel (CLAUDE.md invariants).

## 3. What the code looks like today

These are facts I read from the repository, from Quokka and from the Iliev paper, not decisions.

- The hydro state is fixed at five variables. `NHYDRO = 5` is used in `simulation.jl` (workspace allocation, sweeps, `primitives`, and the size of `RadiationWorkspace.Utmp`, which `imex_substep!` fills with `copyto!`) and in `integrators.jl` (`combine_gas!` reads `Val(NHYDRO)`). In `boundary.jl` it appears only in docstrings. `reconstruct!` and `flux_divergence!` already take `nvar`.
- `riemann_flux` returns an `SVec{5}`, and `HydroState` and the solver contract are tested as an extension point (`state.jl`, `test_extensions.jl`). They do not have to change (section 4.1).
- `Reflecting` flips the normal momentum by striding `NRAD = 4` from index `dir + 1`. That stride stops at the end of a five-variable vector, so it is correct today. With scalars appended it would also flip scalar rows (index 6 for `dir = 1`). It has to be changed.
- `Fixed{N}` and `FunctionBC` check that the boundary state length equals the state length, so they need the scalar entries too.
- `hydro_system.jl` is part of the change: `cons2prim!` must carry the scalars, the first-order fallback inside `hydro_fluxes!` must cover them, and the density-floor reset leaves $\rho X_s$ untouched, so $X_s$ would jump. With hydro off, `enforce_floors!` never runs (`simulation.jl`, `enforce_state!`).
- `set_initial_conditions!` has no keys for species, and its `T` key uses the fixed-$\mu$ equation of state. The checkpoint header records no variable count, so a checkpoint with a different number of scalars would be misread silently.
- The matter-radiation coupling already has the pattern wanted here: `AbstractCoupling`, a per-cell `CouplingState`, `solve_coupling`, and `coupling_rhs` / `coupling_jacobian` as the single source of truth. `AdaptiveCoupling` in `coupling_ode.jl` runs OrdinaryDiffEq (default `Rosenbrock23`) with an analytic Jacobian and an exact `tgrad`. Its own docstring reports that it allocates 8 to 21 kB per cell and is 12 times to several hundred times slower than the one-step solvers.
- In Honghu's reduced-speed-of-light convention, $E$ is the physical energy density in a steady state, $\phi = \hat c / c$ multiplies the time derivative of the radiation, and the gas side of an exchange is never scaled by $\hat c$. The existing `source` hook injects $\phi\,S\,\Delta t$ into $E$ (`source_terms.jl`: `src = stot * cdt` with `stot = S/c`). It splits the total over the groups by a Planck spectrum, so it cannot put photons into one band only.
- The radiation substep is $\Delta t = {\rm cfl}/(\hat c \sum_d 1/\Delta x_d)$ (`radiation_system.jl`, `radiation_timestep`).
- The Quokka checkout is `/Users/u1149259/softwares/quokka/quokka-local`. CLAUDE.md now says so; it used to name `../quokka`, which does not exist.
- Quokka's order per radiation substep is: transport and emission of the chemical bands, then the photochemistry burn over the full substep, then the IMEX step of the thermal groups. Its hydrogen network is in `src/networks/photoionization_H_caseB_handwritten/actual_rhs.H`, the burn wrapper in `src/radiation/photochemistry.hpp`, and its Stromgren test in `src/problems/StromgrenSphere/`.
- Quokka reconstructs the conserved scalar $\rho X_s$ (it copies it into the primitive array), and each Riemann solver treats the scalars as extra components of the conserved vector (`HLLC.hpp`, `LLF.hpp`). Honghu's HLLC has no HLL blend, so for it this is the same as mass flux times upwind $X_s$. For LLF it is not the same.
- Quokka's burn has no emission term and no separate flux variable: with `SKIP_PHOTOCHEMFLUX` it scales the flux by $N_{\rm end}/N_{\rm start}$.
- Iliev et al. (2006), checked in the paper: all tests are on $128^3$ grids; the box is $L = 6.6$ kpc with the source in a corner; $n_{\rm H} = 10^{-3}$ cm$^{-3}$ and $\dot N_\gamma = 5\times10^{48}$ photons s$^{-1}$. Test 1 is monochromatic at $h\nu = 13.6$ eV, with $T$ fixed at $10^4$ K, $\alpha_B = 2.59\times10^{-13}$ cm$^3$ s$^{-1}$ and an initial ionization fraction $1.2\times10^{-3}$. Test 2 uses a $10^5$ K blackbody, starts fully neutral at 100 K, and lets $T$ vary. The paper does not prescribe rates beyond the Test 1 $\alpha_B$: each code used its own (its Table 2).

## 4. Design

### 4.1 Layer 1: mass scalars

Storage: extra rows after the five hydro rows of `sim.U`, so the array is `(5 + ns, i, j, k)`, as in Quokka. The conserved variable is $\rho X_s$ (species mass density). Reconstruction acts on $\rho X_s$, as in Quokka (the primitive array carries $\rho X_s$ unchanged, so `cons2prim!` only copies it).

Flux: the scalar flux is the Riemann mass flux times the upwind mass fraction $X_s = \rho X_s / \rho$ of the reconstructed interface state, upwind by the sign of the mass flux. It is computed in `hydro_fluxes!` from the mass component of the Riemann flux, so `riemann_flux`, `HydroState` and the extension contract stay as they are (my call). For HLLC this is what Quokka does. For LLF it is a deliberate departure from Quokka's component-wise flux; it keeps the solver contract and gives a monotone $X_s$.

Interface: `Simulation` gains `nscalars` (default 0) and a `chemistry` argument. With a network, `nscalars` is the number of species. A plain scalar run without a network is supported. `set_initial_conditions!` accepts species (mass fractions or an ionization fraction), and its `T` key must use the network's temperature when there is one.

Repair: a negative $X_s$ is projected back by `enforce_floors!` and counted in `Diagnostics`, never inside a conversion. The density-floor reset must rescale $\rho X_s$ with $\rho$, as Quokka does (`hydro_system.hpp`). With hydro off, the scalar repair must still run (from `enforce_state!` or from the chemistry step).

Other places that must change: the `NHYDRO` uses listed in section 3, the `Reflecting` stride, `Fixed` / `FunctionBC` lengths, `RadiationWorkspace.Utmp`, the checkpoint header (record the variable count and check it on load), VTK output, `field` / `available_fields`, and `conserved_totals` (it should report scalar masses). The IMEX stages do not change scalars, so `combine_gas!` can leave the scalar rows alone; the builder must confirm this.

### 4.2 Layer 2: the network and the two integrators

A network describes what is evolved in one cell over one step and nothing about how:

- Species: masses and charges. Electrons are not stored; they follow from charge neutrality (my call; enough for hydrogen and helium, not for negative ions).
- Bands: which radiation groups it reads, and the mean photon energy $\varepsilon_b$ of each.
- The per-cell ODE state: species number densities, gas internal energy, and the photon number density $N_b = E_b / \varepsilon_b$ of each band. There is no separate flux variable: absorption acts on $N$ and on the flux with the same linear rate, so the flux of a band is scaled by $N_{\rm end} / N_{\rm start}$ afterwards, as Quokka does with `SKIP_PHOTOCHEMFLUX`. Emission is injected outside the burn.
- `rhs`, `jacobian` (exact, analytic), an exact `tgrad` of zero, the gas temperature from internal energy and composition, and the tolerance scales of each unknown.
- An isothermal mode (needed for Test 1): the temperature function returns a fixed $T$, the energy equation is dropped, and the gas internal energy in `sim.U` is set from $n_{\rm tot}$ and $T$ after the step. Without it, $T$ would halve as $n_{\rm tot}$ doubles at fixed $e$.

Following `coupling.jl`: the rate terms are written once, and every integrator's residual and linearisation come from them.

An integrator is a subtype of an abstract chemistry-solver type with one method, `solve_chemistry(solver, network, cell) -> result`, returning the new state, a step or iteration count, and a converged flag. The shape is the same as `solve_coupling`.

- `RosenbrockChem` (main, default): an adaptive stiff Rosenbrock method from OrdinaryDiffEq, handed the analytic Jacobian and `tgrad`, with one reused integrator per loop chunk (section 5). The defaults are `Rosenbrock23` and a relative tolerance of $10^{-2}$, set by the author from Quokka experience (they are also the defaults of `AdaptiveCoupling`). The algorithm, the relative tolerance and the absolute tolerances are all constructor arguments, passed through to OrdinaryDiffEq, because the author will test other combinations later; any stiff OrdinaryDiffEq algorithm works. The absolute tolerances are dimensionless fractions of cell scales (species relative to $n_{\rm H}$, energy relative to the cell's internal energy), as `AdaptiveCoupling` does with its energy scale, so they mean the same in any unit system.
- `BackwardEulerChem` (comparison): one backward-Euler step with a Newton iteration on the same `rhs` and `jacobian`. No adaptivity and no sub-cycling. If Newton does not converge it says so in the result; it does not hide it.

A failed or unconverged cell is counted in `Diagnostics` and reported in the `run!` summary, like `newton_failures` today.

Chemistry needs CGS units, because the rates are in CGS. The rest of Honghu is unit-agnostic; a chemistry run uses the `CGS` constants.

### 4.3 Hydrogen OTSA network and its coupling to the radiation

Species: H$^0$ and H$^+$, both stored as scalars. Their sum is fixed by the hydrogen mass fraction, so $n_{\rm H^0} + n_{\rm H^+} = n_{\rm H}$ is a built-in invariant check. $n_e = n_{\rm H^+}$. Temperature comes from $e = \frac{3}{2} n_{\rm tot} k_B T$ with $n_{\rm tot} = n_{\rm H^0} + n_{\rm H^+} + n_e$, $\gamma = 5/3$.

Rates, as in the Quokka note and its `actual_rhs.H`, with the changes below:

- Photoionization on the radiation side $\hat c \sigma_\gamma n_{\rm H^0} N_\gamma$ and on the gas side $c \sigma_\gamma n_{\rm H^0} N_\gamma$, with $\sigma_\gamma$ and the photon energy as constants of the band (see the reduced-speed-of-light item below).
- Band constants are per test. Test 1: $\varepsilon_b = 13.6$ eV (monochromatic) and $\sigma_\gamma = 6.3\times10^{-18}$ cm$^2$, with no heating because $T$ is fixed. Test 2: the $10^5$ K blackbody, $\varepsilon_b = 29.65$ eV as the number-weighted mean energy of the ionizing photons, and the photon-number-weighted cross-section $\sigma_N \approx 1.63\times10^{-18}$ cm$^2$. I checked the energy by integrating a $10^5$ K Planck spectrum above 13.6 eV (29.61 eV). The Quokka note's section 1.2 calls 29.65 eV the "mean excess" energy; that wording is wrong. The builder checks $\sigma$ against the Planck integral with a hydrogen cross-section. The constants are plain fields.
- Heating per ionization $\varepsilon_{\rm heat}$ defaults to $\varepsilon_b - \chi_{\rm H}$ (16 eV in Test 2), as in Quokka, with $\chi_{\rm H} = 13.6$ eV. It is a field. Expect the ionized interior of Test 2 to be too hot: in the optically thin limit the physical heating per ionization is about $\bar\varepsilon\,\sigma_E/\sigma_N - \chi_{\rm H} \approx 6.3$ eV (with $\sigma_E \approx 1.10\times10^{-18}$ cm$^2$ from the same Planck integral), so 16 eV overheats by about a factor 2.5. This is why the Test 2 check is qualitative for now. If $\varepsilon_{\rm heat} \neq \varepsilon_b - \chi_{\rm H}$, the energy invariant below has an explicit leftover term (my call).
- Case B recombination $\alpha_B(T)$ and collisional ionization $\beta(T)$. Quokka uses $\alpha_B = 2.63\times10^{-13}\,(T/10^4\,{\rm K})^{-0.7}$ cm$^3$ s$^{-1}$ and the Krumholz et al. (2007) $\beta$. For Test 1 the tests override $\alpha_B$ to the paper's constant $2.59\times10^{-13}$.
- Cooling, for pure hydrogen: recombination cooling (Quokka, Krumholz et al. 2007), collisional-ionization cooling ($\chi_{\rm H}$ per ionization), free-free, and collisional excitation (Ly$\alpha$). For free-free, use only the $1.3\times1.427\times10^{-27}\sqrt T\,n_e n_{\rm H^+}$ term of Quokka's `get_ion_ff_cooling_coefficient`. Its other part, `get_cle_term` (Frazer & Heitsch 2019), is line cooling by metal ions and does not belong in a hydrogen-only test. Quokka has no separate Ly$\alpha$ term; its "KI" terms are neutral-ISM heating and cooling and are not used, because the constant KI heating would warm the cold neutral gas ahead of the front in Test 2. Iliev prescribes none of these rates, so I choose the Cen (1992) collisional-excitation fit for Ly$\alpha$ (my call; the builder checks the fit against the paper).
- Every coefficient is a field of the network, so tests can override it without editing code.

Coupling to the existing radiation:

- The ionizing photons are one group of the existing multigroup M1 (my guess, to be confirmed by the author: the last group). That group must have zero thermal opacity, so the standard coupling leaves it alone. Quokka asserts the same for its chemical bands. A grey run (`NG = 1`) is allowed if its opacity is zero.
- Emission goes into the band only, through a function giving photons per second per volume. It follows Honghu's existing convention: $\phi\,\dot N\,\Delta t$ is added to $N$, as the `source` hook does. A one-cell point source gives artifacts along the grid axes, so the source is a narrow Gaussian that is normalised to the exact photon rate, as Quokka's test does. For the corner-source octant the rate is $\dot N / 8$, as in Quokka's `testStromgrenSphere.cpp`.
- Order per radiation substep (my call): the existing IMEX substep as now (the band's transport is effectively explicit RK2 when its opacity is zero), then the chemistry solve over the substep length for every cell. It updates the species rows, the gas internal energy and total energy, the band energy $E = N \varepsilon$, and the band flux. For the chemical band Quokka does the same order, transport and then chemistry; only the order relative to the thermal-group coupling differs, and the benchmark has none.
- Reduced speed of light. Read from the code, this is settled and the tests confirm it. Quokka's chemical bands store $N_Q = N_{\rm phys}/\phi$, do not scale the source by $\hat c / c$, and use $\hat c$ in both the gas and the photon rate, so the budget is $n_{\rm H^+} + N_Q$. Honghu's convention is the same scheme with $N = \phi N_Q$. It requires: the photon loss $\hat c \sigma n_{\rm H^0} N$ in the radiation equation, the ionization rate $c \sigma n_{\rm H^0} N$ on the gas side, and the emission multiplied by $\phi$. The conserved budget is $n_{\rm H^+} + N/\phi$, and "photons emitted equals $\dot N t$" holds for $\int N/\phi$, not for $\int N$. Using $c$ on the gas side with unscaled emission gives a Stromgren volume proportional to $1/\phi$.
- The test that confirms it: the ionization equilibrium behind the front must not depend on $\hat c$.
- Energy bookkeeping. With cooling, recombination and collisional ionization switched off, and with the default $\varepsilon_{\rm heat}$, $e + \chi_{\rm H} n_{\rm H^+} + E_{\rm band}/\phi$ is conserved. It is the exact invariant for the photoionization step, in the style of the existing tests. The $\chi_{\rm H}$ term is chemical energy and is a diagnostic only.
- `gas_temperature(sim)` and the `:T` plot field must use the network's temperature when a network is present, because the constant-$\mu$ equation of state is wrong in partially ionized gas.

### 4.4 Benchmark and tests

Setup from the paper (section 3). The benchmark and the test suite use Iliev's own geometry: source in the corner cell, `Reflecting` walls on the three low faces, `Outflow` on the high faces, run to 500 Myr (about $4\,t_{\rm rec}$), with $\dot N / 8$ injected (my call about the octant).

- Test 1 has an analytic reference, $r_I(t) = r_S\,[1 - e^{-t/t_{\rm rec}}]^{1/3}$ with $r_S = (3 \dot N / 4\pi \alpha_B n^2)^{1/3} = 5.39$ kpc and $t_{\rm rec} = 1/(\alpha_B n) = 122.3$ Myr. For a finite $\hat c$ there is an exact reference as well, the front equation of Shapiro et al. (2006), $dR/dt = (\dot N - \tfrac{4\pi}{3} R^3 \alpha_B n^2) / (\dot N/\hat c + 4\pi R^2 n)$, which Quokka's test integrates (`integrate_radius`). The reduced-light-speed run must match it at every $\hat c$, and approach the analytic curve as $\hat c \rightarrow c$.
- Test 2 has no analytic solution. The author has no reference data at hand and accepts a qualitative check for now: a front that grows monotonically, hot ionized gas behind it, and cold gas ahead of it. (The author's words were "a front slower than $r_S$"; the plan checks monotonic growth instead, because hot gas recombines more slowly and the Test 2 front can pass the Test 1 $r_S$.) The interior temperature will be too high for the reason in section 4.3. If the published $r_I(t)$ and $T(r)$ of the comparison project become available, they are added as a quantitative comparison. A single band with constant cross-section cannot reproduce a multi-frequency result exactly, so differences are documented and attributed, not tuned away.
- The example puts the source at the box centre, as requested. That needs a 13.2 kpc box, which is 8 times the cells at the same cell size. The example's default resolution is chosen after the cost gate in section 5, and the corner-source octant is available in the example as an option.
- Reduced speed of light. The author expects the result to be independent of $\hat c$ when $\hat c > 3 c_s$. With hydro off the relevant speed is the front speed, $v_I = (r_S/3t_{\rm rec})\,e^{-t/t_{\rm rec}}\,(1 - e^{-t/t_{\rm rec}})^{-2/3}$ with $r_S/3t_{\rm rec} = 14.4$ km s$^{-1}$. That is about 70 km s$^{-1}$ at 10 Myr, 30 km s$^{-1}$ at 30 Myr and 9 km s$^{-1}$ at 100 Myr, and $3 c_s$ is about 45 km s$^{-1}$, so $3 c_s$ is below the front speed before about 25 Myr. The plan includes a sweep of $\hat c$ against the finite-$\hat c$ front equation to find where the result starts to change, and development runs use a lower resolution to save time.
- Pass criteria are shape checks, in the style of the existing suite, and not fixed numbers chosen in advance. The numerical error thresholds are set after the first converged runs.

Other tests:

- Layer 1: a uniform $X_s$ stays uniform through a density wave to round-off, in both Riemann solvers; $\sum_s X_s = 1$ is preserved; a contact discontinuity in $X_s$ is monotone; the density-floor reset keeps $X_s$ continuous; a run with no scalars is bit for bit unchanged; a hydro sweep with scalars allocates under 8 bytes per cell, added to `test_performance.jl`.
- Layer 2: the toy network (a linear chain $A \rightarrow B \rightarrow C$ with an exact solution) through both integrators, with order-of-accuracy checks (backward Euler first order, the Rosenbrock method higher). The $n_{\rm H^0} + n_{\rm H^+}$ and photon-ionization budget invariants. A cell-level agreement test between the two integrators at small $\Delta t$. An isothermal-mode test.
- The `BackwardEulerChem` path should allocate nothing per cell.

### 4.5 Housekeeping required by CLAUDE.md

A docstring for every exported name; a line in `docs/make.jl` `PAGES` for each new source file; a `CHANGELOG.md` entry in the newest section as each change is made; tests as above; the example follows `examples/common.jl` (flags, `outpath`, `plotting_note`, figures built from `plotstate` / `plotpanels`). The solver comparison goes in a findings note in `docs/`, in the manner of `docs/coupling-findings.md`.

## 5. Cost of the main integrator

The main integrator is the Rosenbrock method of OrdinaryDiffEq, as the author asked. No integrator is written in tree. The cost has to be managed and measured, not avoided.

A measurement on a toy hydrogen-like cell (three unknowns, one step of $3.5\times10^{12}$ s, one thread, automatic-differentiation Jacobian, 2000 random cell states). The numbers are the same at the default relative tolerance $10^{-2}$ and at $10^{-3}$ (within a few percent), so the cost is dominated by the per-call overhead and not by the number of steps:

- A one-shot `solve` per cell costs about 19 $\mu$s and 8 kB with `Rosenbrock23`, and 21 $\mu$s and 11 kB with `Rodas5P`.
- A reused integrator (`init` once, then `reinit!` and `solve!` for each cell) costs about 11 $\mu$s and 1.2 kB with `Rosenbrock23`, and 19 $\mu$s with `Rodas5P`.
- The 12 to 300 $\mu$s and 8 to 21 kB quoted in `coupling_ode.jl` are for the one-shot `solve` on extremely stiff cells (optical depths up to $10^{12}$). A Stromgren cell is milder.
- The toy is not the real network. The real one has more rate terms, and cells in the front take more steps. These numbers are lower bounds to be measured again at the gate.

Scaling: the radiation substep is about $0.1\,\Delta x / \hat c$ in 3D, so the cell solves scale as $N^4$ and as $\phi$. A $64^3$ octant over 500 Myr is about $3.9\times10^{12}\,\phi$ cell solves. At $\phi = 3\times10^{-4}$ (about 100 km s$^{-1}$, enough for the front after the first few Myr) that is $1.2\times10^9$ solves, about 0.4 to 0.8 hours on 8 threads at 10 to 20 $\mu$s per solve if the threads scale. At $\phi = 10^{-3}$ it is about 1.4 to 2.7 hours. A $32^3$ octant is 16 times cheaper, which is fine for development.

Plan, in order:

- Reuse one integrator per loop chunk, for example one per $(j, k)$ row, created inside the row. It is never keyed on `threadid()`, as CLAUDE.md requires.
- Give the integrator the analytic Jacobian and `tgrad`.
- Measure $\mu$s per cell on the real network, on Test 1 and Test 2 cell states, before any 3D benchmark.
- If the gate still fails: a fast path for cells whose state changes by less than the tolerance (dark neutral cells), a lower $\hat c$ where the front speed allows it, and a lower resolution.
- Only as a last fallback: an in-tree allocation-free Rosenbrock method for small fixed-size networks, validated against OrdinaryDiffEq. That is a new numerical component and needs the author's decision at that point.

The gate numbers (about 30 minutes for the example on 8 threads, about 30 s added to the test suite) are my call. If they are missed, the plan stops and reports the measurements.

## 6. Calls I made

- Scalars live as extra rows of `sim.U`; reconstruction on $\rho X_s$, as Quokka; scalar flux is mass flux times upwind $X_s$ computed in `hydro_fluxes!` (a departure from Quokka for LLF only).
- Electrons are derived from charge neutrality and not stored. Both H$^0$ and H$^+$ are stored, so their sum is a checkable invariant.
- The ionizing band is the last radiation group, with zero thermal opacity; $\sigma_\gamma$, $\varepsilon_b$ and $\varepsilon_{\rm heat}$ are constants of the band; no flux variable in the ODE.
- Order per substep: IMEX, then chemistry. Emission is a normalised Gaussian carrying the factor $\phi$.
- OrdinaryDiffEq's `Rosenbrock23` with relative tolerance $10^{-2}$ is the default main integrator (set by the author); algorithm and tolerances are constructor arguments. It uses a reused integrator per loop chunk; an in-tree integrator is a last fallback only. Absolute tolerances are dimensionless fractions of cell scales.
- Chemistry requires `beta_order = 0`, `ConstantChat` and CGS units. An explicit isothermal mode is added for Test 1.
- Cooling is hydrogen-only: Quokka's recombination and collisional-ionization cooling, pure free-free (not Quokka's metal term), and the Cen (1992) Ly$\alpha$ fit. KI terms and Compton cooling are excluded.
- Heating per ionization defaults to $\varepsilon_b - \chi_{\rm H}$, and Test 2 is expected to be too hot inside.
- The benchmark and the test suite use Iliev's corner-source octant with $\dot N/8$; the centred example needs a 13.2 kpc box.
- Pass criteria are shape checks and the finite-$\hat c$ front equation; thresholds come after the first runs. Test 2 is qualitative.

## 7. Left to the builder

File and function names; how a band is declared; how the emission is weighted inside the IMEX stages; variable scaling of the ODE state; thread safety of the per-cell ODE solves; the `Reflecting` fix; the exact form of the density-floor reset; the choice of cross-section for the Planck averages; the Gaussian width of the source; the Cen (1992) fit and every other rate fit, checked against its paper.
