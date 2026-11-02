# Photochemistry

This page explains the photochemistry module of Honghu: the physics it solves, the numerical methods it uses, and how the code is put together. It is written for a reader who wants to understand the module well enough to use it with confidence, to change it, or to carry an idea across to Quokka. The function-by-function documentation is generated from the docstrings, on the reference page [Photoionization chemistry](reference/chemistry.qmd); this page links to it wherever a name appears. What the module costs and how accurate it is, measured, is in a separate note, [Photoionization: the chemistry integrators and the reduced speed of light, measured](photoionization-findings.md).

In one paragraph: the module evolves the ionization state and the temperature of hydrogen gas that is lit by ionizing photons. The species are carried by the gas as passive mass scalars, the ionizing photons are one frequency group of the two-moment (M1) radiation, and in every cell, after every radiation substep, a small stiff system of ordinary differential equations (ODEs) couples the two: photoionization, recombination, collisional ionization, photoheating and cooling. The ODE is written once, by a *network* (here [`HydrogenOTSA`](reference/chemistry.qmd#HydrogenOTSA)), and solved by an *integrator* (the adaptive [`RosenbrockChem`](reference/chemistry.qmd#RosenbrockChem), or the single-step [`BackwardEulerChem`](reference/chemistry.qmd#BackwardEulerChem) for comparison). The physics follows Aubert & Teyssier (2008) in the on-the-spot approximation, as Quokka's photoionization module does, and the benchmark is the Stromgren-sphere comparison of Iliev et al. (2006).

## 1. The physics

### 1.1 The problem: an H II region

A star that emits $\dot N$ ionizing photons per second into uniform hydrogen of number density $n_{\rm H}$ ionizes a sphere around itself. Inside the sphere almost every atom is ionized, and the photons are used up by keeping it so: each recombination makes a neutral atom that the next photon ionizes again. The sphere stops growing when the recombinations inside it use up all the photons. Its radius is then the Stromgren radius,

$$r_S = \left(\frac{3\dot N}{4\pi\,\alpha_B\,n_{\rm H}^2}\right)^{1/3} ,$$

with $\alpha_B$ the case B recombination coefficient (section 1.2). The sphere approaches $r_S$ on the recombination time $t_{\rm rec} = 1/(\alpha_B n_{\rm H})$. If the front between ionized and neutral gas is sharp and photons travel infinitely fast, the photon budget gives the radius of the front at every time,

$$r_I(t) = r_S\left(1 - e^{-t/t_{\rm rec}}\right)^{1/3} ,$$

and its speed, $v_I = dr_I/dt$, falls from very large values at early times to zero. For the parameters of Iliev et al. (2006) Test 1 ($\dot N = 5\times10^{48}$ s$^{-1}$, $n_{\rm H} = 10^{-3}$ cm$^{-3}$, $\alpha_B = 2.59\times10^{-13}$ cm$^3$ s$^{-1}$) the radius is $r_S = 5.39$ kpc and the recombination time is $t_{\rm rec} = 122.3$ Myr. These two formulas are [`stromgren_radius_analytic`](reference/problems.qmd#stromgren_radius_analytic).

When the photons travel at a finite speed $\hat c$, the photons in flight are a third reservoir besides the ions and the recombinations. Shapiro et al. (2006) write the budget as

$$\frac{dR}{dt} = \frac{\dot N - \tfrac{4\pi}{3}R^3\alpha_B n_{\rm H}^2}{\dot N/\hat c + 4\pi R^2 n_{\rm H}} ,$$

which keeps $dR/dt \leq \hat c$ and tends to the formula above as $\hat c \to \infty$. Honghu integrates this equation in [`stromgren_radius_finite_c`](reference/problems.qmd#stromgren_radius_finite_c), which is a transliteration of Quokka's `integrate_radius`. It is approximate: it counts the photons in flight as $\dot N R/\hat c$, as if none of them were used up by recombinations inside $R$.

### 1.2 Hydrogen: species, rates, and the on-the-spot approximation

The gas is pure hydrogen with two species, neutral hydrogen H$^0$ and protons H$^+$, of number densities $n_{\rm H^0}$ and $n_{\rm H^+}$. The electrons come only from hydrogen, so $n_e = n_{\rm H^+}$, and the ionization fraction is $x = n_{\rm H^+}/n_{\rm H}$ with $n_{\rm H} = n_{\rm H^0} + n_{\rm H^+}$. Three processes change the species:

- **Photoionization** at the rate $\Gamma\,n_{\rm H^0}$ per unit volume, with $\Gamma = c\,\sigma\,N$ the number of ionizations per neutral atom per second, $\sigma$ the photoionization cross-section and $N$ the number density of ionizing photons.
- **Recombination** at the rate $\alpha_B(T)\,n_e\,n_{\rm H^+}$.
- **Collisional ionization** at the rate $\beta(T)\,n_e\,n_{\rm H^0}$, important only above a few $10^4$ K.

So

$$\frac{dn_{\rm H^+}}{dt} = \Gamma\,n_{\rm H^0} + \beta\,n_e n_{\rm H^0} - \alpha_B\,n_e n_{\rm H^+} = -\frac{dn_{\rm H^0}}{dt} .$$

A recombination straight to the ground state emits a photon that can itself ionize. In the **on-the-spot approximation** (OTSA) that photon is absorbed again close to where it was made, so the two events cancel. Recombinations to the ground state are then left out of the rate, which is why the coefficient is the case B one, $\alpha_B$, and the diffuse recombination radiation is left out of the radiation field. The approximation is good deep inside a large H II region and poor near the front and in very diffuse, highly ionized gas. Honghu, like Quokka today, uses it throughout.

The rate coefficients of [`HydrogenOTSA`](reference/chemistry.qmd#HydrogenOTSA), in CGS units, are:

| Process | Coefficient | Source |
|:---|:---|:---|
| Case B recombination | $\alpha_B = \alpha_0\,(T/10^4\,{\rm K})^{p}$ cm$^3$ s$^{-1}$, by default $\alpha_0 = 2.63\times10^{-13}$, $p = -0.7$ | Quokka's `get_recombination_coefficient` |
| Collisional ionization | $\beta = 5.84\times10^{-11}\,\sqrt{T}\,\exp(-2.18\times10^{-11}\,{\rm erg}/k_B T)$ cm$^3$ s$^{-1}$ | Krumholz et al. (2007), as in Quokka |

Iliev's Test 1 uses a constant $\alpha_B = 2.59\times10^{-13}$, which is `alpha0 = 2.59e-13, alpha_exp = 0`.

### 1.3 The ionizing photons: one band of the M1 radiation

The ionizing radiation is one frequency group of Honghu's two-moment radiation transport: an energy density $E$ and a flux $\mathbf F$ per group, transported with the M1 closure and an HLL Riemann solver at the reduced speed of light $\hat c$. The chemistry calls that group a *band* and works with the photon number density

$$N = E/\varepsilon ,$$

with $\varepsilon$ the mean energy of a photon in the band. A single band stands for the whole ionizing spectrum through two numbers, the cross-section $\sigma$ and the photon energy $\varepsilon$. For Iliev's Test 1 the spectrum is monochromatic at 13.6 eV, with $\sigma = 6.3\times10^{-18}$ cm$^2$. For Test 2 it is a $10^5$ K blackbody, which Honghu represents by $\varepsilon = 29.65$ eV, the number-weighted mean energy of its ionizing photons, and $\sigma = 1.63\times10^{-18}$ cm$^2$, the photon-number-averaged cross-section.

Absorption removes photons at the rate $\sigma\,n_{\rm H^0}$ per photon, so the band is opaque wherever the gas is neutral, and it affects the flux and the energy density alike: with no source in the cell, $\mathbf F$ and $N$ decay at the same rate. The band must have no thermal opacity in the sense of Honghu's matter-radiation coupling (emission and absorption by a Planck spectrum at the gas temperature), so that only the chemistry changes it. Section 3.5 explains why every other group must have none as well.

### 1.4 Heating, cooling, and the temperature

Each photoionization gives the gas the excess energy of the photon above the ionization threshold, $\chi_{\rm H} = 13.6$ eV. With one band, that is one number, the heating energy $\varepsilon_{\rm heat}$, and the heating rate is $\varepsilon_{\rm heat}\,\Gamma\,n_{\rm H^0}$. The default is $\varepsilon_{\rm heat} = \varepsilon - \chi_{\rm H}$, which is 16.05 eV for Test 2. This is Quokka's choice, without Quokka's floor at zero.

That default is too large, and it is worth understanding why. The cross-section falls steeply with frequency, roughly as $\nu^{-3}$, so in optically thin gas the low-energy photons do most of the ionizing. The mean excess energy per ionization is then $\langle\varepsilon\rangle\,\sigma_E/\sigma_N - \chi_{\rm H} \approx 6.3$ eV for a $10^5$ K blackbody, with $\sigma_E$ the energy-weighted and $\sigma_N$ the number-weighted cross-section. One band cannot be right for both the photon count and the energy. The measured effect on Test 2 is modest: 2.5 times the heating raises the temperature of the ionized gas by about a tenth ([findings, section 4](photoionization-findings.md#the-overheating-of-test-2)).

The gas loses energy by four processes, each per unit volume:

| Process | Rate | Source |
|:---|:---|:---|
| Recombination cooling | $6.1\times10^{-10}\,k_B\,T^{0.11}\;n_e n_{\rm H^+}$, zero below 100 K | Krumholz et al. (2007), as Quokka's `get_recombination_cooling_coefficient` |
| Free-free | $1.3\times1.427\times10^{-27}\sqrt{T}\;n_e n_{\rm H^+}$ | the first term of Quokka's `get_ion_ff_cooling_coefficient` |
| Collisional excitation (Ly$\alpha$) | $7.5\times10^{-19}\,\dfrac{e^{-118348\,{\rm K}/T}}{1 + \sqrt{T/10^5\,{\rm K}}}\;n_e n_{\rm H^0}$ | Cen (1992), in the form of Katz, Weinberg & Hernquist (1996), Table 1 |
| Collisional ionization | $\chi_{\rm H}\,\beta\,n_e n_{\rm H^0}$ | the energy each collisional ionization costs |

Two of Quokka's terms are left out on purpose. The second term of Quokka's free-free coefficient, `get_cle_term` (Frazer & Heitsch 2019), is line cooling by metal ions, and the neutral-ISM ("KI") heating and cooling are molecular and dust physics. A pure-hydrogen test must not see either. The collisional-ionization term belongs to collisional ionization, not to cooling: it is switched with `collisional_ionization`, so that switching cooling off never breaks energy conservation.

The temperature follows from the internal energy density $e$ of a monatomic gas,

$$e = \frac32\,k_B\,(n_{\rm H^0} + n_{\rm H^+} + n_e)\,T = \frac32\,k_B\,(n_{\rm H^0} + 2n_{\rm H^+})\,T ,$$

so the mean molecular weight changes from 1 to 1/2 as the gas ionizes. At fixed $e$, ionizing a cell halves its temperature. This is why the network, and not the equation of state, says what the temperature is when chemistry is on ([`network_temperature`](reference/chemistry.qmd#network_temperature)), and why the gas must have $\gamma = 5/3$: the hydrodynamic pressure $(\gamma - 1)\,e$ must be the pressure of the same monatomic gas.

### 1.5 The reduced speed of light

Light crosses an H II region in a few hundred years, while the region evolves over hundreds of millions of years. An explicit transport step limited by $c$ would therefore be absurdly short. Honghu, like Quokka, uses the reduced speed of light approximation (RSLA): the radiation is transported at $\hat c = \phi\,c$, with $\phi < 1$, which lengthens the radiation step by $1/\phi$.

In Honghu's convention $\phi$ multiplies the time derivative of the radiation and nothing on the gas side: the photon density obeys

$$\frac{\partial N}{\partial t} = \phi\left(-\nabla\cdot\frac{\mathbf F}{\varepsilon} + q - c\,\sigma\,n_{\rm H^0} N\right) ,$$

with $q$ the photons emitted per unit volume and time, while the gas still sees the full photoionization rate $\Gamma = c\,\sigma\,N$. Three consequences follow, and the code is built on them:

- **Steady states are those of the real problem.** Where the radiation is in a steady state, $\phi$ drops out, so $N$ is the physical photon density and the ionization equilibrium does not depend on $\hat c$. The photon loss in the band is $\hat c\,\sigma\,n_{\rm H^0} N$, while the ionization rate of the gas is $c\,\sigma\,n_{\rm H^0} N$.
- **The emission is scaled.** A source adds $\phi\,q\,\Delta t$ to $N$ in a step of length $\Delta t$.
- **The conserved budget is $n_{\rm H^+} + N/\phi$.** Each photon removed from the band, counted as $N/\phi$, makes one ion.

Quokka writes the same scheme with the variable $N_Q = N/\phi$: there the source is not scaled, and $\hat c$ appears in both the photon loss and the ionization rate, $\hat c\,\sigma\,N_Q = c\,\sigma\,N$. The two forms give the same ionization history.

The RSLA is good when $\hat c$ is faster than everything the radiation must keep up with. With hydrodynamics off that is the ionization front, whose speed is

$$v_I = \frac{r_S}{3t_{\rm rec}}\,e^{-t/t_{\rm rec}}\left(1 - e^{-t/t_{\rm rec}}\right)^{-2/3} ,$$

with $r_S/3t_{\rm rec} = 14.4$ km s$^{-1}$ for Test 1: about 70, 30 and 9 km s$^{-1}$ at 10, 30 and 100 Myr. At early times a front faster than $\hat c$ is held back by the light-travel time. The measured accuracy as a function of $\hat c$ is in [findings, section 3](photoionization-findings.md#the-reduced-speed-of-light-on-test-1).

### 1.6 What is conserved

With recombination, collisional ionization and cooling switched off (`alpha0 = 0`, `collisional_ionization = false`, `cooling = false`) and the default heating energy, the equations of one cell conserve three quantities exactly:

$$n_{\rm H^0} + n_{\rm H^+} , \qquad n_{\rm H^+} + \frac{N}{\phi} , \qquad e + \chi_{\rm H}\,n_{\rm H^+} + \varepsilon\,\frac{N}{\phi} .$$

The first is the number of hydrogen nuclei, the second the photon budget, and the third the energy: the photon's energy $\varepsilon$ is split into $\chi_{\rm H}$ stored as chemical energy and $\varepsilon - \chi_{\rm H}$ given to the gas. All three are linear in the unknowns, so both integrators preserve them to round-off, and the tests check them in a single cell and over a whole simulation (section 5). The third is why the default heating energy has no floor at zero, unlike Quokka's: a floor would break it for a band below 13.6 eV.

### 1.7 The full system of one cell

Put together, the network evolves four unknowns per cell, $u = (n_{\rm H^0}, n_{\rm H^+}, e, N)$:

$$\begin{aligned}
\frac{dn_{\rm H^0}}{dt} &= \alpha_B\,n_e n_{\rm H^+} - \Gamma\,n_{\rm H^0} - \beta\,n_e n_{\rm H^0} , \\
\frac{dn_{\rm H^+}}{dt} &= -\frac{dn_{\rm H^0}}{dt} , \\
\frac{de}{dt} &= \varepsilon_{\rm heat}\,\Gamma\,n_{\rm H^0} - \chi_{\rm H}\,\beta\,n_e n_{\rm H^0} - (\Lambda_{\rm rec} + \Lambda_{\rm ff})\,n_e n_{\rm H^+} - \Lambda_{\rm ce}\,n_e n_{\rm H^0} , \\
\frac{dN}{dt} &= -\hat c\,\sigma\,n_{\rm H^0} N ,
\end{aligned}$$

with $n_e = n_{\rm H^+}$, $\Gamma = c\,\sigma\,N$ and $T$ from section 1.4. The system is stiff: in a cell at the front, $\Gamma$ can exceed $10^{-10}$ s$^{-1}$ while the radiation substep is $10^{12}$ s or more, so $\Gamma\,\Delta t$ reaches $10^2$ to $10^3$. Recombination and cooling are slower, but still much faster than the step in dense gas. An explicit integrator would drive $N$ or $n_{\rm H^0}$ negative within one step, because the photon loss and the ionization share the product $n_{\rm H^0}N$.

The network also has an **isothermal mode**, `isothermal = T0`, for Iliev's Test 1: the energy equation is dropped, $T$ is held at $T_0$, and the energy written back to the grid is the one that $T_0$ and the new composition imply ([`chem_internal_energy`](reference/chemistry.qmd#chem_internal_energy)). Without it, ionizing a cell at fixed $e$ would halve its temperature.

## 2. The numerical method

### 2.1 Where the chemistry sits in a step

One call of `step!` advances the simulation by $\Delta t$ through a fixed sequence of operators. With chemistry on it is:

```text
step!(sim, Δt)
├─ Strang source terms, first half (Δt/2)
├─ advance_hydro!            SSP Runge-Kutta: gas and species (mass scalars) advected together
└─ advance_radiation!
   ├─ renormalise the species so that Σ ρX_s = ρ           (only when hydro is on)
   └─ for each radiation substep Δt_r = Δt / n_sub:
      ├─ imex_substep!       M1 transport of every group, matter-radiation coupling
      ├─ chemistry_step!     in every cell: emission, the network over Δt_r, write-back
      └─ amend_rad_state!    project the radiation back onto E ≥ floor, |F| ≤ cE
├─ Strang source terms, second half (Δt/2)
└─ enforce_state!            floors; species clipped at zero and renormalised
```

The radiation substep is set by the CFL condition of the transport at $\hat c$, $\Delta t_r = {\rm cfl}/(\hat c\sum_d 1/\Delta x_d)$, with ${\rm cfl} = 0.3$ by default. With hydrodynamics on, the hydro step is capped at `max_radiation_substeps` radiation substeps. With hydrodynamics off, the step is one radiation substep.

The band has no thermal opacity, so the implicit part of the IMEX step leaves it alone, and its transport reduces to the explicit part of the IMEX PD-ARS scheme, a second-order Runge-Kutta step. The chemistry then acts on what the transport delivered, over the whole substep. This is the order Quokka uses for its chemical bands: transport, then the chemistry burn. It is also the order of ATON (Aubert & Teyssier 2008), up to where the source is injected.

The splitting has a cost. The emission is added to $N$ all at once at the start of the burn, so in an opaque cell the burn ionizes with those photons early in the substep and recombines for the rest of it. The steady state is then biased by about $\alpha_B n_{\rm H} x^2\,\Delta t_r/2$, which is first order in the substep: in a uniform test box with an exact equilibrium of $x = 0.5$, it came out as 0.4939 at $\Delta t_r = 0.05\,t_{\rm rec}$ and 0.4450 at $0.5\,t_{\rm rec}$ ([findings, section 5](photoionization-findings.md#two-choices-in-the-scheme-that-the-results-depend-on)). The photons that arrive by transport are split the same way. Strang-split injection, or the emission inside the ODE, would remove the bias, and neither is implemented.

### 2.2 The species as mass scalars

The species are stored as **mass scalars**: conserved densities $\rho X_s$, with $X_s$ the mass fraction of species $s$, in extra rows of the gas state `sim.U` after the five hydrodynamic variables ([`scalar_index`](reference/simulation.qmd#scalar_index)). A simulation with a network carries one per species, in the network's order; for hydrogen, `:X_1` is the H$^0$ fraction and `:X_2` the H$^+$ fraction. The network works in number densities, $n_s = \rho X_s/m_s$ with $m_s$ = [`species_mass`](reference/chemistry.qmd#species_mass), here $m_{\rm H}$ for both species. The electron mass is neglected.

The scalars are **passive**: the hydrodynamics never reads them. They are advected with the gas in three steps, inside [`hydro_fluxes!`](reference/hydro.qmd#hydro_fluxes-bang):

1. Each $\rho X_s$ is reconstructed to the cell faces like the other variables. Quokka also reconstructs $\rho X_s$, not $X_s$.
2. The Riemann solver computes the mass flux $F_\rho$ through the face, and knows nothing of the scalars.
3. The scalar flux is the mass flux times the mass fraction on the upwind side, $F_{\rho X_s} = F_\rho\,X_s^{\rm up}$, with $X_s^{\rm up} = (\rho X_s)_{\rm face}/\rho_{\rm face}$ on the side the mass comes from.

A uniform $X_s$ is therefore advected exactly, to round-off, in any flow. For HLLC this flux is the same as Quokka's, which carries the scalars as extra conserved components inside the solver. For LLF it is a deliberate departure: it is less diffusive for the scalar, and the Riemann solvers keep their five-component interface.

Because each $\rho X_s$ is reconstructed and limited separately, the face values of the species do not add up to $\rho_{\rm face}$, and the species fluxes do not add up to the mass flux. With PLM or PPM the sum $\sum_s X_s$ of a cell then drifts away from 1, by $10^{-2}$ in ten steps of a coarse test. With a network, [`enforce_scalar_floors!`](reference/hydro.qmd#enforce_scalar_floors-bang) therefore rescales the species of each cell so that $\sum_s \rho X_s = \rho$. It does so at the start of every hydrodynamic right-hand side, before the chemistry reads the species, and at the end of every step, after clipping any negative $\rho X_s$ to zero. This is Quokka's cure (its `EnforceLimits` renormalises its mass scalars). It conserves the total mass but not each species separately. The conservative alternative, consistent multi-fluid advection (CMA; Plewa & Müller 1999), scales the face fractions instead so that the species fluxes add up to the mass flux, and it is not implemented. A simulation of plain scalars, without a network, is never renormalised.

### 2.3 One cell as an ODE

For every interior cell, the driver [`chemistry_step!`](reference/chemistry.qmd#chemistry_step-bang) builds the unknown vector

$$u = (n_1, \ldots, n_{N_S},\; e,\; N_1, \ldots, N_{N_B})$$

from the grid: the species from the mass scalars, $e$ from the total gas energy minus the kinetic energy, and the photon density of each band from its group's energy. The constants a rate may need travel in a [`ChemCell`](reference/chemistry.qmd#ChemCell): the density $\rho$, $c$, the $\hat c$ of the band's group, and the step $\Delta t$.

Three things the network does to keep the integration well behaved:

- **Clamping.** A negative $n_{\rm H^0}$, $n_{\rm H^+}$ or $N$ enters every rate as zero, as Quokka clamps $N$, and its column of the Jacobian is zero. An integrator's trial state can be a rounding error below zero, and a negative photon density must not create neutral atoms.
- **A temperature floor.** The rates see $\max(T, T_{\min})$, with $T_{\min} = 1$ K by default, so that no rate is evaluated at a negative or zero temperature. A cell without hydrogen gets $T_{\min}$ and no rates at all, so it is left unchanged.
- **An exact Jacobian.** [`chem_jacobian!`](reference/chemistry.qmd#chem_jacobian-bang) returns the analytic Jacobian, including the temperature dependence of every rate through $\partial T/\partial n_s$ and $\partial T/\partial e$. Each rate coefficient is coded once, as a function that returns the value and its temperature derivative together, so the right-hand side and the Jacobian cannot disagree about a formula. A test compares the Jacobian with a finite difference in 256-bit arithmetic.

The tolerances of the integrators are measured against one absolute **scale** per unknown, from [`chem_scales!`](reference/chemistry.qmd#chem_scales-bang): $n_{\rm H}$ for both species; the cell's energy for $e$ (but at least the energy of the cell at $T_{\min}$, so that a cold cell does not ask for a vanishing absolute error); and $\phi\,n_{\rm H}$ for $N$, the photon density whose budget $N/\phi$ is one photon per hydrogen nucleus. A tolerance is then a fraction of a physically meaningful amount, and it means the same thing in any unit system and in any cell.

### 2.4 The integrators

Both integrators solve the same ODE: they read only the network's right-hand side, its Jacobian and its scales. Any difference between them is a difference in how the equations are integrated, not in what equations are integrated.

**[`RosenbrockChem`](reference/chemistry.qmd#RosenbrockChem)**, the default, integrates each cell over the substep with an adaptive, error-controlled stiff integrator from OrdinaryDiffEq, `Rosenbrock23` by default: the second-order Rosenbrock-W method with a third-order error estimate of MATLAB's `ode23s` (Shampine & Reichelt 1997). It is linearly implicit, so each internal step solves linear systems with the matrix $I - \gamma h J$ and never iterates to convergence. It takes as many internal steps as the cell needs: the default relative tolerance is $10^{-2}$, and the absolute tolerance is $10^{-8}$ of each unknown's scale. On the Iliev tests it takes about 15 internal steps per cell per substep and never fails ([findings, sections 1 and 2](photoionization-findings.md#cost-of-the-chemistry-step)). Four implementation details matter for cost and correctness:

- **One integrator per row of cells, re-aimed at each cell.** Building an integrator costs about 8 kB and 19 µs; re-aiming an existing one with `reinit!` costs much less. The cell's constants are passed by overwriting a mutable parameter object, and the per-unknown absolute tolerances are written in place into the integrator's own tolerance vector.
- **No step cap.** The integrator is built on a placeholder time interval of length 1, and OrdinaryDiffEq caps the step at the length of that interval unless told otherwise; `dtmax` is set to infinity, or no internal step could exceed one second.
- **A linear solver written in Julia.** The linear systems are solved with LinearSolve's `SimpleLUFactorization`, not with OpenBLAS: OpenBLAS takes a global lock on every call, which made a threaded chemistry step slower on four threads (184 ms) than on one (85 ms), against 74 ms and 20 ms with the Julia solver.
- **No warnings.** OrdinaryDiffEq's per-cell warnings are switched off. A failed cell is reported through its [`ChemStats`](reference/chemistry.qmd#ChemStats) and counted by the driver, which a threaded run needs: otherwise every failing cell would print a long warning.

**[`BackwardEulerChem`](reference/chemistry.qmd#BackwardEulerChem)** takes one backward-Euler step over the whole substep: it solves

$$G(u) = u - u_0 - \Delta t\,f(u) = 0$$

by Newton's method, $(I - \Delta t\,J)\,\delta = -G(u)$. The linear system is solved by Gaussian elimination with partial pivoting, written out so that it allocates nothing. After each update the species and photon densities are clamped at zero. The update is damped by a backtracking line search on the scaled residual $\max_i |G_i|/s_i$, with $s_i$ the scales: it tries the whole update, then a half, a quarter and so on, and keeps the first trial that lowers the residual, or the whole update if none does. The iteration stops when the scaled update or the scaled residual is below $10^{-10}$. The line search changes how the equation of the step is solved, never the equation, so it is still one first-order step. It costs about 0.1 µs per cell, 45 to 100 times less than `RosenbrockChem`.

Newton can still fail, and in three known situations the one-step equation, not the damping, is at fault:

- **Stiff growth.** When collisional ionization grows fast ($\beta\,n_{\rm H^0}\,\Delta t > 1$), the backward-Euler root is positive, but the Newton direction from the old state points into the clamp at zero. A line search on the residual cannot cross it.
- **Very stiff front cells.** Cells with $\Gamma\,\Delta t$ of order $10^2$ to $10^3$, where the temperature and the clamps interact within the step.
- **Discontinuous rates.** The recombination cooling switches on at 100 K, and a cell near that temperature can make the iteration cycle across the switch.

On a 2592-cell stress grid it left 10% of the cells unconverged; on the Iliev tests, none. Every failure is reported, never hidden ([findings, section 2](photoionization-findings.md#the-two-integrators-on-test-2)).

**Which to use.** `RosenbrockChem` is the safe choice and the default. `BackwardEulerChem` is the comparison, and it is the cheap choice on problems where it converges: on Test 2 it agrees with the Rosenbrock reference to 0.1% in the radius of the ionized region, and to within 29% in the temperature of the few cells the front crosses.

### 2.5 Writing the answer back

After the solve, the driver writes the new state into the grid:

- the species, $\rho X_s = \max(n_s, 0)\,m_s$;
- the gas energy, changed by the change of the internal energy, $E_{\rm gas} \mathrel{+}= e_{\rm new} - e_{\rm old}$; the density and the momentum are not touched, so the kinetic energy is unchanged (and the momentum of the absorbed photons is not modelled);
- the band's energy, $E = \max(N, 0)\,\varepsilon$;
- the band's flux, multiplied by $N_{\rm end}/N_{\rm start}$.

The last item replaces three more unknowns. During the burn, absorption removes photons from $N$ and from $\mathbf F$ at the same rate $\hat c\,\sigma\,n_{\rm H^0}$, and there is no source in the burn, so the flux decays by exactly the factor the photon density does. Quokka does the same with its `SKIP_PHOTOCHEMFLUX` option. $N_{\rm start}$ includes the emission added before the burn, which adds photons but no flux.

The emission and the flux scaling are the reason the driver, and not the network or the integrator, owns these steps. They are properties of how the cell's ODE is embedded in the radiation-hydrodynamics scheme, not terms of the ODE. Putting them into the right-hand side would make the two integrators integrate different problems.

### 2.6 A point source on a grid

A star is a point, and a grid cannot hold one. The Stromgren problems of [`stromgren_iliev`](reference/problems.qmd#stromgren_iliev) spread the source over a Gaussian of width $1.5\,\Delta x$, cut at 2.5 widths, as Quokka's `testStromgrenSphere.cpp` does. Its amplitude is normalised over the cells of the grid, not analytically, so the cells receive exactly $\dot N$ photons per second whatever the resolution. A point source in one cell would print the grid axes onto the front. In the default octant geometry (the source in the corner, with reflecting walls on the three low faces) the box receives $\dot N/8$, so that the full sphere would receive $\dot N$.

The emission function is evaluated once per cell and per radiation substep, at the start of the substep, from several threads at once. It must therefore be thread-safe: it may read captured constants, but must not cache results in a global or update a counter.

## 3. The design of the code

### 3.1 Three layers

The module is three layers, each in its own file, and each knows nothing of the layers above it:

| Layer | File | Knows | Does not know |
|:---|:---|:---|:---|
| Network | `chemistry_hydrogen.jl` | the species, the rates, the Jacobian, the scales, the temperature | how the ODE is integrated, the grid |
| Integrator | `chemistry_solvers.jl` | how to integrate any network's ODE over one step | any rate, any species, the grid |
| Driver | `chemistry_step.jl` | the grid: packing cells, the emission, the write-back, threads | any rate, how the ODE is integrated |

`chemistry.jl` holds the interface between them, as `coupling.jl` does for Honghu's matter-radiation coupling. A new network is a new subtype of [`AbstractNetwork`](reference/chemistry.qmd#AbstractNetwork) with its methods, and a new integrator is a new subtype of [`AbstractChemSolver`](reference/chemistry.qmd#AbstractChemSolver) with two methods. Neither requires an edit to the other or to the driver. The test suite holds a three-species toy network with an exact solution, which runs through both integrators and through the driver unchanged; it is what shows that the interface is really independent of hydrogen.

The rule for where a piece of physics or numerics belongs:

- A term that depends on the cell's own state is part of the ODE, and belongs to the **network**: every integrator must see it.
- A choice of how to integrate belongs to the **integrator**.
- Anything that comes from outside the cell during the step, or that is a property of the grid, belongs to the **driver**: the emission, the flux scaling, the clamp of the answer at zero, the conversion between mass scalars and number densities.

### 3.2 The network interface

A network implements these methods (each is documented on the [reference page](reference/chemistry.qmd#AbstractNetwork)):

| Method | Returns |
|:---|:---|
| `nspecies(net)`, `nbands(net)` | the number of species and of photon bands |
| `species_mass(net, s)` | the mass of a particle of species `s`, in g |
| `band_group(net, b)`, `band_photon_energy(net, b)` | the radiation group band `b` reads, and its mean photon energy in erg |
| `chem_rhs!(du, u, net, cell)` | the time derivative of the unknowns |
| `chem_jacobian!(J, u, net, cell)` | its exact, dense Jacobian |
| `chem_scales!(scale, u, net, cell)` | one positive absolute scale per unknown |

Four more have defaults that a network may override: `network_temperature` and `network_eint` (the temperature from the energy and back; the default is a monatomic gas of the stored species, without electrons), `chem_internal_energy` (the energy written back; overridden for an isothermal mode), and `initial_mass_fractions` (the species of an initial condition given without the key `X`; [`HydrogenOTSA`](reference/chemistry.qmd#HydrogenOTSA) reads `x_HII`). The layout of `u` is fixed: species first, then the energy, then one photon density per band ([`nunknowns`](reference/chemistry.qmd#nunknowns), [`ienergy`](reference/chemistry.qmd#ienergy), [`iband`](reference/chemistry.qmd#iband)).

The skeleton of a new network, for instance hydrogen and helium with three bands:

```julia
struct HHeNetwork <: Honghu.AbstractNetwork
    # cross-sections, photon energies, groups, switches ...
end
Honghu.nspecies(::HHeNetwork) = 5            # H0, H+, He0, He+, He++
Honghu.nbands(::HHeNetwork) = 3              # H-, He0- and He+-ionizing photons
Honghu.species_mass(::HHeNetwork, s) = s <= 2 ? Honghu.CGS.m_H : 4 * Honghu.CGS.m_u
Honghu.band_group(net::HHeNetwork, b) = net.groups[b]
Honghu.band_photon_energy(net::HHeNetwork, b) = net.eps[b]
function Honghu.chem_rhs!(du, u, net::HHeNetwork, cell)
    # n_e = n_H+ + n_He+ + 2 n_He++ from charge neutrality; the gas side uses cell.c,
    # the photon loss of each band uses cell.chat
end
function Honghu.chem_jacobian!(J, u, net::HHeNetwork, cell)
    # every entry, exactly; a test against a finite difference is the safety net
end
function Honghu.chem_scales!(scale, u, net::HHeNetwork, cell)
    # the nuclei of each element for its species, the energy, phi times the nuclei for photons
end
function Honghu.network_temperature(net::HHeNetwork, n, e)
    # e / (3/2 k_B (sum of the species + n_e)), with n_e from charge neutrality
end
```

Two assumptions of the current driver limit what a network may do. The electrons are not stored, so a network takes them from charge neutrality, which is enough for hydrogen and helium but not for negative ions. And only the first band receives the emission of `Photochemistry`; a multi-band source would need a new keyword.

### 3.3 The integrator interface

An integrator implements two methods:

```julia
chem_workspace(solver, net, ::Type{T}) -> work            # scratch space, allocated once
solve_chemistry!(work, solver, net, cell, u) -> ChemStats  # u: initial state in, answer out
```

[`ChemStats`](reference/chemistry.qmd#ChemStats) reports the number of internal steps or Newton iterations and whether the solve converged. An unconverged cell keeps the integrator's last state, and the driver counts it. The integrator reads the network only through `chem_rhs!`, `chem_jacobian!` and `chem_scales!`.

### 3.4 Threads and memory

The driver loops over the rows of cells, $(j, k)$, with Honghu's `threaded_jk`, as the matter-radiation coupling does. Each row owns one workspace, built once when the simulation is built, and a row is handled by one task at a time. So no workspace is shared, nothing is keyed on `Threads.threadid()` (which is not stable under dynamic scheduling), and no locking is needed. The work of a row is in an ordinary function, not in the closure that `threaded_jk` calls, so that no variable the closure captures is reassigned; a reassigned captured variable is boxed by Julia and allocates on every use, the trap described in `CLAUDE.md`.

With `BackwardEulerChem` the chemistry step allocates nothing per cell, and the test suite checks it. `RosenbrockChem` allocates about 350 bytes per cell and call inside OrdinaryDiffEq's `reinit!` and `solve!`. That is an accepted exception to Honghu's rule that hot kernels do not allocate, and a test bounds it at 2 kB.

### 3.5 What a simulation with chemistry requires

[`Photochemistry`](reference/chemistry.qmd#Photochemistry) is passed to [`Simulation`](reference/simulation.qmd#Simulation) as `chemistry`. The simulation refuses to be built (an `ArgumentError`) unless:

| Requirement | Why |
|:---|:---|
| radiation is enabled | it carries the photons |
| the reduced speed of light is constant (`ConstantChat`) | the conserved budget $n_{\rm H^+} + N/\phi$ needs one $\phi$, and the stability bound of the adaptive limiter has no photoabsorption term |
| `beta_order = 0` | the momentum of the absorbed photons is not modelled |
| every radiation group has zero thermal opacity | the matter-radiation coupling takes the gas temperature from the constant-$\mu$ equation of state, which ignores the species and is off by a factor of 2 in ionized gas |
| `nscalars` is 0 or the number of species | the species are the mass scalars |
| $\gamma = 5/3$ | the network's temperature is that of a monatomic gas |
| each band's group exists | the band is a radiation group |

The rates are in CGS, so the run must be in CGS units; that is not checked. Each requirement is an error rather than a quiet wrong answer, because a run that breaks one is not slightly off, it is a different problem.

### 3.6 Diagnostics

[`Diagnostics`](reference/simulation.qmd#Diagnostics) counts the cells the chemistry solved (`chem_cells`), the integrator's steps summed over them (`chem_steps`), the most any one cell took (`chem_max`), and the cells that did not converge (`chem_failures`). The summary that `run!` prints gives the mean and worst steps per cell, and a warning when any cell failed. Species clipped at zero count in `floors_scalar` and in `nrepaired`. A radiation cell sitting exactly at the energy floor, as the dark cells ahead of the front do, is not counted as a repair, so `nrepaired(sim) == 0` remains a useful check on a photoionization run.

## 4. Using it

A minimal run: a uniform box of neutral hydrogen with a uniform source of ionizing photons.

```julia
using Honghu
const CGS = Honghu.CGS

net = HydrogenOTSA(sigma = 1.63e-18, photon_energy = 29.65 * CGS.eV)
chem = Photochemistry(net; emission = (x, y, z, t) -> 1e-17)    # photons cm^-3 s^-1

rad = RadiationPhysics(c = CGS.c_light, chat = 1e-3 * CGS.c_light, a_rad = CGS.a_rad,
                       k_B = CGS.k_B, opacity = ConstantOpacity(0.0), beta_order = 0)
sim = Simulation(n = (8, 8, 8), xhi = (1e21, 1e21, 1e21), hydro = false, radiation = rad,
                 eos = IdealGas(gamma = 5/3, cv = cv_from_mu(1.0, 5/3)),
                 boundaries = Boundaries(Periodic()), chemistry = chem)
set_initial_conditions!(sim) do x, y, z
    (rho = 1e-3 * CGS.m_H, T = 100.0, x_HII = 0.0, Erad = 0.0)
end
run!(sim, tmax = 1e14)

xHII, _ = field(sim, :X_2)         # the ionized fraction
T, _ = field(sim, :temperature)    # the network's temperature
```

Three things in the initial condition matter:

- `x_HII` sets the ionization fraction. The key `X` sets the mass fractions directly instead.
- `T` is turned into an energy by the network's equation of state.
- `Erad = 0.0` is needed: without it the radiation starts at $a_r T^4$, which for the ionizing band would be about $10^{12}$ photons cm$^{-3}$.

The Iliev tests are ready-made in [`stromgren_iliev`](reference/problems.qmd#stromgren_iliev), with [`ionized_radius`](reference/problems.qmd#ionized_radius) to measure the front:

```julia
P = Honghu.Problems
sim, tmax = P.stromgren_iliev(test = 1, n = 32, chat_factor = 3e-4)
run!(sim; tmax)
r = P.ionized_radius(sim; x0 = 1.2e-3)
```

The example [`examples/stromgren_sphere.jl`](https://github.com/chongchonghe/Honghu.jl/blob/main/examples/stromgren_sphere.jl) runs either test in the whole sphere or in the octant, prints the radius at Iliev's output times against both references, and draws the neutral fraction, the temperature and the radius history. The centred $64^3$ run of Test 1 to 500 Myr takes about 9 minutes on 8 threads.

## 5. How it is tested

The tests pin properties, not numbers that a better scheme would change:

| Test | What it pins | File |
|:---|:---|:---|
| a toy network $A \to B \to C$ with an exact solution | the integrators' accuracy and order (backward Euler first order), workspace reuse, zero allocation | `test_chemistry.jl` |
| the Jacobian against a 256-bit finite difference, in six configurations | every entry of `chem_jacobian!`, including clamps and the isothermal mode | `test_chemistry.jl` |
| the rate coefficients at fixed temperatures | the constants and exponents, which the finite difference cannot see | `test_chemistry.jl` |
| one cell with recombination, collisional ionization and cooling off | the three invariants of section 1.6, to $10^{-10}$, for both integrators | `test_chemistry.jl` |
| one cell at fixed photon density | the ionization equilibrium $x^2/(1-x) = \Gamma/(\alpha_B n_{\rm H})$ | `test_chemistry.jl` |
| a periodic box with uniform emission | the photon budget and the energy invariant over a whole run, to $10^{-8}$ | `test_chemistry.jl` |
| the same box at two values of $\hat c$ and equal $\Delta t$ | that the equilibrium does not depend on $\hat c$ | `test_chemistry.jl` |
| hydrodynamics on, with a density wave and a step in $X$ | hydrogen mass conserved, $\sum_s X_s = 1$ in every cell and where the chemistry reads it | `test_chemistry.jl` |
| advection of a uniform scalar, a contact, reflecting walls | $X_s$ advected exactly and monotonely, the hydrodynamics unaffected by scalars | `test_scalars.jl` |
| Iliev Test 1 at $16^3$ | the radius within 4.5% of the finite-$\hat c$ reference at 0.5, 1 and 2 $t_{\rm rec}$, and the neutral fraction behind the front within 2% between two $\hat c$ | `test_stromgren.jl` |
| Iliev Test 2 at $16^3$ | a front that grows, hot ionized gas behind it and cold gas ahead | `test_stromgren.jl` |

The measurements behind the choices on this page, the cost of the chemistry step, the comparison of the two integrators and the sweep in $\hat c$, are in [the findings note](photoionization-findings.md), with the script that reproduces its tables.

## 6. Limitations

What the module does not do, and what would be needed:

- **One band, hydrogen only.** The network interface allows more species and bands, and electrons from charge neutrality. A helium network would also need emission into several bands, which `Photochemistry` cannot express yet.
- **No diffuse recombination radiation.** The on-the-spot approximation is used throughout. Case A recombination with the diffuse photons as a source in the band is not implemented.
- **No radiation pressure.** The momentum of the absorbed photons is not given to the gas, so `beta_order = 0` is required.
- **No thermal groups alongside the chemistry.** Until the equation of state reads the species, a group with opacity would couple at the wrong temperature, so it is refused.
- **Only the constant reduced speed of light.** The adaptive limiter, `StabilityBound`, has no photoabsorption term in its stability bound.
- **A first-order splitting of the emission** (section 2.1), and a renormalisation of the species that conserves the total mass but not each species (section 2.2).
- **One band cannot get the heating right** for a broad spectrum (section 1.4).
- **The front is not quite round** at low resolution, and part of the difference from the finite-$\hat c$ reference is not explained ([findings, section 3](photoionization-findings.md#the-reduced-speed-of-light-on-test-1)).

## References

- Aubert, D. & Teyssier, R. 2008, MNRAS, 387, 295: a radiative transfer scheme for cosmological reionization based on a local Eddington tensor (ATON). The model of section 1 and the operator splitting of section 2.1.
- Cen, R. 1992, ApJS, 78, 341: the collisional excitation cooling of hydrogen.
- Iliev, I. T., et al. 2006, MNRAS, 371, 1057: the Cosmological Radiative Transfer Codes Comparison Project, Tests 1 and 2.
- Katz, N., Weinberg, D. H. & Hernquist, L. 1996, ApJS, 105, 19: the form of the cooling rates used here (Table 1).
- Krumholz, M. R., Stone, J. M. & Gardiner, T. A. 2007, ApJ, 671, 518: the collisional ionization and recombination cooling rates.
- Plewa, T. & Müller, E. 1999, A&A, 342, 179: consistent multi-fluid advection (CMA).
- Shampine, L. F. & Reichelt, M. W. 1997, SIAM J. Sci. Comput., 18, 1: the MATLAB ODE suite, and the `ode23s` method that `Rosenbrock23` implements.
- Shapiro, P. R., Iliev, I. T., Alvarez, M. A. & Scannapieco, E. 2006, ApJ, 648, 922: the expansion of an ionization front at a finite speed of light.
- Quokka's photoionization module: `docs/markdown/photoionization.md`, `src/radiation/photochemistry.hpp` and `src/networks/photoionization_H_caseB_handwritten/actual_rhs.H` in the Quokka repository.
