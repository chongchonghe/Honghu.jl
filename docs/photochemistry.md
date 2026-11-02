# Photochemistry

This page describes the photochemistry module of Honghu: the equations it solves and the numerical methods that solve them. The functions and types, with their arguments and defaults, are documented on the reference page [Photoionization chemistry](reference/chemistry.qmd), which is generated from the docstrings. Measurements of the module's cost and accuracy are in the note [Photoionization: the chemistry integrators and the reduced speed of light, measured](photoionization-findings.md).

The model is that of Aubert & Teyssier (2008), the ATON scheme, in the on-the-spot approximation, as in Quokka's photoionization module. Hydrogen gas carries its ionization state as passive mass scalars, the ionizing photons are one frequency band of the two-moment (M1) radiation, and photoionization, recombination, collisional ionization, photoheating and cooling couple the two. The benchmark is the Stromgren-sphere comparison of Iliev et al. (2006).

## 1. Governing equations

### 1.1 Gas and species

The gas obeys the Euler equations, with the thermal energy changed by the heating $\mathcal H$ and the cooling $\Lambda$ of section 1.4, and carries the mass density $\rho X_s$ of each species $s$:

$$\begin{aligned}
\frac{\partial\rho}{\partial t} + \nabla\cdot(\rho\mathbf v) &= 0 , \\
\frac{\partial(\rho\mathbf v)}{\partial t} + \nabla\cdot(\rho\mathbf v\mathbf v + p\,\mathsf I) &= 0 , \\
\frac{\partial E}{\partial t} + \nabla\cdot\left[(E + p)\,\mathbf v\right] &= \mathcal H - \Lambda , \\
\frac{\partial(\rho X_s)}{\partial t} + \nabla\cdot(\rho X_s\mathbf v) &= m_s\,\mathcal R_s .
\end{aligned}$$

Here $E = e + \rho v^2/2$ is the total energy density, $e$ the internal energy density, $X_s$ the mass fraction of species $s$, $m_s$ its particle mass and $\mathcal R_s$ its net rate of creation per unit volume (section 1.3). The gas is pure hydrogen with two species, H$^0$ and H$^+$, both of mass $m_{\rm H}$, so $n_s = \rho X_s/m_{\rm H}$ and $n_{\rm H^0} + n_{\rm H^+} = n_{\rm H} = \rho/m_{\rm H}$. The electrons come only from hydrogen, so $n_e = n_{\rm H^+}$, and their mass is neglected. The radiation exerts no force on the gas: the momentum of the absorbed photons is not modelled.

The gas is a monatomic ideal gas, $p = (\gamma - 1)\,e$ with $\gamma = 5/3$, and its temperature counts the electrons:

$$e = \frac32\,k_B\,(n_{\rm H^0} + n_{\rm H^+} + n_e)\,T = \frac32\,k_B\,(n_{\rm H^0} + 2n_{\rm H^+})\,T .$$

The mean mass per particle therefore falls from $m_{\rm H}$ to $m_{\rm H}/2$ as the gas ionizes, and at fixed $e$, ionizing the gas halves its temperature.

### 1.2 Ionizing photons

The first two moments of the radiative transfer equation, integrated over the ionizing band, give conservation laws for the photon number density $N_\gamma$ and the photon flux $\mathbf F_\gamma$:

$$\begin{aligned}
\frac{\partial N_\gamma}{\partial t} + \phi\,\nabla\cdot\mathbf F_\gamma &= \phi\,\dot N_\gamma - \hat c\,\sigma\,n_{\rm H^0} N_\gamma , \\
\frac{\partial \mathbf F_\gamma}{\partial t} + \hat c\,c\,\nabla\cdot\mathsf P_\gamma &= -\hat c\,\sigma\,n_{\rm H^0}\,\mathbf F_\gamma .
\end{aligned}$$

| Symbol | Definition |
|:---|:---|
| $N_\gamma$ | ionizing photon number density (cm$^{-3}$) |
| $\mathbf F_\gamma$ | ionizing photon number flux (cm$^{-2}$ s$^{-1}$) |
| $\mathsf P_\gamma = \mathsf D\,N_\gamma$ | photon pressure tensor (cm$^{-3}$), with $\mathsf D$ the Eddington tensor of the M1 closure |
| $\sigma$ | photoionization cross-section, averaged over the band (cm$^2$) |
| $\dot N_\gamma$ | photons emitted by sources, per unit volume and time (cm$^{-3}$ s$^{-1}$) |
| $c$, $\hat c = \phi\,c$ | the speed of light and the reduced speed of light, $\phi \leq 1$ (section 1.6) |

With $\phi = 1$ these are the physical equations, as in Quokka's note. The closure is M1: the Eddington tensor $\mathsf D$ is a function of the reduced flux $\mathbf f = \mathbf F_\gamma/(c N_\gamma)$, with the Eddington factor of Levermore (1984), and it is exact for radiation that is isotropic ($|\mathbf f| = 0$) or a single beam ($|\mathbf f| = 1$). The admissible states have $|\mathbf F_\gamma| \leq c N_\gamma$.

A single band stands for the whole ionizing spectrum through two numbers: the cross-section $\sigma$ and the mean energy $\varepsilon$ of a photon in the band, which converts between photons and energy, $E_\gamma = \varepsilon N_\gamma$. In the code the band is one frequency group of the radiation, whose energy density is $E_\gamma$. For Iliev's Test 1 the spectrum is monochromatic at 13.6 eV, with $\sigma = 6.3\times10^{-18}$ cm$^2$. For Test 2 it is a $10^5$ K blackbody, represented by $\varepsilon = 29.65$ eV, the number-weighted mean energy of its ionizing photons, and $\sigma = 1.63\times10^{-18}$ cm$^2$, the photon-number-weighted cross-section.

The band has no thermal emission or absorption: the gas, at a few $10^4$ K at most, emits a negligible number of ionizing photons, and only the chemistry changes the band.

### 1.3 Hydrogen chemistry

Three processes change the species: photoionization, recombination and collisional ionization. The net rate of creation of protons is

$$\mathcal R_{\rm H^+} = \Gamma\,n_{\rm H^0} + \beta(T)\,n_e n_{\rm H^0} - \alpha_B(T)\,n_e n_{\rm H^+} , \qquad \mathcal R_{\rm H^0} = -\mathcal R_{\rm H^+} ,$$

with $\Gamma = c\,\sigma\,N_\gamma$ the photoionization rate per neutral atom. The coefficients, in CGS units, are:

| Process | Coefficient | Source |
|:---|:---|:---|
| Case B recombination | $\alpha_B = \alpha_0\,(T/10^4\,{\rm K})^{p}$ cm$^3$ s$^{-1}$, by default $\alpha_0 = 2.63\times10^{-13}$ and $p = -0.7$ | Quokka |
| Collisional ionization | $\beta = 5.84\times10^{-11}\,\sqrt{T}\,\exp(-\chi_{\rm H}/k_B T)$ cm$^3$ s$^{-1}$ | Krumholz et al. (2007) |

with $\chi_{\rm H} = 13.6$ eV. Iliev's Test 1 uses a constant $\alpha_B = 2.59\times10^{-13}$ cm$^3$ s$^{-1}$.

### 1.4 Heating and cooling

Each photoionization gives the gas the energy of the photon above the threshold. With one band that is one number, the heating energy $\varepsilon_{\rm heat}$, and the heating rate is

$$\mathcal H = \varepsilon_{\rm heat}\,\Gamma\,n_{\rm H^0} .$$

The cooling has four terms:

$$\Lambda = \chi_{\rm H}\,\beta\,n_e n_{\rm H^0} + (\Lambda_{\rm rec} + \Lambda_{\rm ff})\,n_e n_{\rm H^+} + \Lambda_{\rm ce}\,n_e n_{\rm H^0} ,$$

| Process | Coefficient (erg cm$^3$ s$^{-1}$) | Source |
|:---|:---|:---|
| Collisional ionization | $\chi_{\rm H}\,\beta$, the energy each collisional ionization costs | |
| Recombination | $\Lambda_{\rm rec} = 6.1\times10^{-10}\,k_B\,T^{0.11}$ for $T \geq 100$ K, and zero below | Krumholz et al. (2007) |
| Free-free | $\Lambda_{\rm ff} = 1.3\times1.427\times10^{-27}\,\sqrt T$ | Quokka |
| Collisional excitation (Ly$\alpha$) | $\Lambda_{\rm ce} = 7.5\times10^{-19}\,e^{-118348\,{\rm K}/T}/(1 + \sqrt{T/10^5\,{\rm K}})$ | Cen (1992), in the form of Katz, Weinberg & Hernquist (1996) |

This is Quokka's set of processes for pure hydrogen, with two of its terms left out: metal-line cooling, which Quokka adds to the free-free term, and the heating and cooling of the neutral interstellar medium. Both belong to gas with metals and dust, and a pure-hydrogen test must not see them. Quokka has no separate collisional-excitation term.

The default heating energy is $\varepsilon_{\rm heat} = \varepsilon - \chi_{\rm H}$, which is 16.05 eV for Test 2, as in Quokka. It is too large for a broad spectrum. The cross-section falls steeply with frequency, roughly as $\nu^{-3}$, so in optically thin gas the low-energy photons do most of the ionizing, and the mean energy left over per ionization is $\langle\varepsilon\rangle\,\sigma_E/\sigma_N - \chi_{\rm H} \approx 6.3$ eV for a $10^5$ K blackbody, with $\sigma_E$ the energy-weighted and $\sigma_N$ the number-weighted cross-section. One band cannot be right for both the number of ionizations and the heating. On Test 2, 2.5 times the heating raises the temperature of the ionized gas by only about a tenth, because the cooling rises steeply with temperature ([findings, section 4](photoionization-findings.md#the-overheating-of-test-2)).

Iliev's Test 1 holds the gas at $10^4$ K. For it, the energy equation is replaced by $T = T_0$, and the internal energy follows from $T_0$ and the composition through the relation of section 1.1.

### 1.5 The on-the-spot approximation

A recombination straight to the ground state emits a photon that can itself ionize. In the on-the-spot approximation that photon is absorbed again close to where it was emitted, so the recombination and the re-ionization cancel. Recombinations to the ground state are then left out: the recombination coefficient is the case B one, $\alpha_B$, which omits them, and the diffuse recombination radiation, the source $n_e n_{\rm H^+}(\alpha_A - \alpha_B)$ in the photon equation of Quokka's note, is dropped. The approximation is good deep inside a large H II region, where the mean free path of a recombination photon is short, and poor near the ionization front and in diffuse, highly ionized gas. Honghu, like Quokka at present, uses it everywhere.

### 1.6 The reduced speed of light

Light crosses an H II region in a few hundred years, and the region evolves over hundreds of millions of years. An explicit transport step limited by $c$ would be impractically short. The reduced speed of light approximation (RSLA; Gnedin & Abel 2001) transports the photons at $\hat c = \phi\,c$ with $\phi < 1$, which lengthens the transport step by $1/\phi$ (section 2.1).

The equations of section 1.2 are the RSLA in Honghu's form: $\phi$ multiplies the time derivative of the radiation, and nothing on the gas side. Three consequences follow:

- **The steady states are those of the real problem.** Where $\partial/\partial t = 0$, $\phi$ drops out of the photon equations, so $N_\gamma$ is the physical photon density, and the ionization equilibrium, set by $\Gamma = c\,\sigma N_\gamma$, does not depend on $\hat c$.
- **The radiation and the gas see different rates of the same process.** The band loses photons at $\hat c\,\sigma\,n_{\rm H^0}N_\gamma$, while the gas is ionized at $c\,\sigma\,n_{\rm H^0}N_\gamma$, and sources add photons at $\phi\,\dot N_\gamma$.
- **The conserved photon budget is $n_{\rm H^+} + N_\gamma/\phi$.** Each photon removed from the band, counted as $N_\gamma/\phi$, makes one ion.

Quokka writes the same scheme with $N_Q = N_\gamma/\phi$ as its variable. Its sources are then not scaled, and $\hat c$ appears both in the photon loss and in the ionization rate, $\hat c\,\sigma N_Q = c\,\sigma N_\gamma$. The two forms give the same ionization history.

The RSLA is accurate when $\hat c$ is faster than anything the radiation must keep up with. With the gas at rest, that is the ionization front (section 1.8), which at early times can be faster than any affordable $\hat c$. Before the front speed falls below $\hat c$, the front is held back by the travel time of the photons. The accuracy as a function of $\hat c$ is measured in [findings, section 3](photoionization-findings.md#the-reduced-speed-of-light-on-test-1).

### 1.7 Conservation laws

Summed over the species, the chemistry conserves hydrogen nuclei. With recombination, collisional ionization and cooling switched off, and $\varepsilon_{\rm heat} = \varepsilon - \chi_{\rm H}$, the local equations conserve two more quantities exactly:

$$n_{\rm H^+} + \frac{N_\gamma}{\phi} , \qquad e + \chi_{\rm H}\,n_{\rm H^+} + \varepsilon\,\frac{N_\gamma}{\phi} .$$

The first is the photon budget. The second is the energy: a photon's energy $\varepsilon$ is split into $\chi_{\rm H}$, stored as chemical energy, and $\varepsilon - \chi_{\rm H}$, given to the gas. Both are linear in the unknowns, so any Runge-Kutta or Rosenbrock method and Newton's method preserve them to round-off, which makes them exact checks of an implementation. This is also why the default heating energy has no floor at zero, unlike Quokka's: a floor would break the energy balance of a band below 13.6 eV.

### 1.8 A reference solution: the H II region

A source of $\dot N$ ionizing photons per second in uniform hydrogen of density $n_{\rm H}$ ionizes a sphere around itself. Inside, almost every atom is ionized, and the photons are used up by keeping it so: each recombination makes a neutral atom that the next photon ionizes again. The sphere stops growing when the recombinations inside it use up all the photons, at the Stromgren radius

$$r_S = \left(\frac{3\dot N}{4\pi\,\alpha_B\,n_{\rm H}^2}\right)^{1/3} .$$

With a sharp front, a constant temperature and an infinite speed of light, the photon budget gives the radius at every time,

$$r_I(t) = r_S\left(1 - e^{-t/t_{\rm rec}}\right)^{1/3} , \qquad t_{\rm rec} = \frac{1}{\alpha_B\,n_{\rm H}} ,$$

and the speed of the front,

$$v_I = \frac{r_S}{3t_{\rm rec}}\,e^{-t/t_{\rm rec}}\left(1 - e^{-t/t_{\rm rec}}\right)^{-2/3} .$$

For Iliev's Test 1 ($\dot N = 5\times10^{48}$ s$^{-1}$, $n_{\rm H} = 10^{-3}$ cm$^{-3}$, $\alpha_B = 2.59\times10^{-13}$ cm$^3$ s$^{-1}$), $r_S = 5.39$ kpc and $t_{\rm rec} = 122.3$ Myr, and the front moves at about 70, 30 and 9 km s$^{-1}$ at 10, 30 and 100 Myr.

At a finite speed $\hat c$ the photons in flight are a third reservoir, besides the ions and the recombinations. Shapiro et al. (2006) write the budget as

$$\frac{dR}{dt} = \frac{\dot N - \tfrac{4\pi}{3}R^3\alpha_B n_{\rm H}^2}{\dot N/\hat c + 4\pi R^2 n_{\rm H}} ,$$

which keeps $dR/dt \leq \hat c$ and tends to $r_I(t)$ as $\hat c \to \infty$. It is approximate: it counts the photons in flight as $\dot N R/\hat c$, as if recombinations inside $R$ used none of them. Both formulas are in `Honghu.Problems`, as [`stromgren_radius_analytic`](reference/problems.qmd#stromgren_radius_analytic) and [`stromgren_radius_finite_c`](reference/problems.qmd#stromgren_radius_finite_c), and the two Iliev tests as [`stromgren_iliev`](reference/problems.qmd#stromgren_iliev).

## 2. Numerical method

### 2.1 Operator splitting

The equations of section 1 have three parts of very different character:

- **Hydrodynamics,** with the advection of the species: hyperbolic, with signal speeds $|\mathbf v| + c_s$.
- **Radiation transport:** hyperbolic, with signal speed $\hat c$, usually far larger than $|\mathbf v| + c_s$.
- **The local source terms:** chemistry, heating and cooling, and the absorption and emission of photons. They couple the variables within a cell but not between cells, and they are stiff (section 2.3).

Each part is advanced by the method that suits it, and the parts are combined by operator splitting. Over one step $\Delta t$, with $\mathcal H$, $\mathcal T$ and $\mathcal C$ the solution operators of the three parts,

$$U^{n+1} = \Big[\mathcal C_{\Delta t_r}\,\mathcal T_{\Delta t_r}\Big]^{n_{\rm sub}}\,\mathcal H_{\Delta t}\,U^n , \qquad \Delta t_r = \Delta t/n_{\rm sub} .$$

The hydrodynamics takes one step of $\Delta t$, and the radiation subcycles within it, because its stable step, set by the Courant condition at $\hat c$, is far shorter: $\Delta t_r \propto \Delta x/\hat c$. The reduced speed of light lengthens it by $1/\phi$, and that is what makes the problem affordable. The two transport operators are those of Honghu's hydrodynamics and radiation transport, and are not described here. In each radiation substep the photons are transported first, and the chemistry then acts on what the transport delivered, over the whole substep. This is the order of ATON and of Quokka's chemical bands. It is Lie splitting, so the splitting error is first order in the step.

The chemistry step is one radiation substep. A smaller $\hat c$ therefore makes the gas-side rates (ionization, recombination and cooling) stiffer relative to the step, and the splitting error below larger.

The emission is part of the chemistry operator: the photons a source emits during a substep are added to $N_\gamma$ at the start of it, and the chemistry then absorbs them. In an opaque cell that makes the gas ionize early in the substep and recombine for the rest of it, so the ionization fraction at the end of the substep lies below the equilibrium by about $\alpha_B n_{\rm H} x^2\,\Delta t_r/2$, with $x = n_{\rm H^+}/n_{\rm H}$. This is a first-order splitting error. In a uniform test with an exact equilibrium of $x = 0.5$, it gave $x = 0.4939$ for $\Delta t_r = 0.05\,t_{\rm rec}$ and $0.4450$ for $\Delta t_r = 0.5\,t_{\rm rec}$, and the error fell linearly as the substep was cut ([findings, section 5](photoionization-findings.md#two-choices-in-the-scheme-that-the-results-depend-on)). The photons that arrive by transport enter each substep the same way. A Strang splitting of the emission, half before and half after the chemistry, or the emission as a term of the local ODE, would raise the order; neither is implemented.

### 2.2 Advection of the species

The species are advected with the gas consistently. The mass flux $F_\rho$ through a cell face comes from the hydrodynamic Riemann solver, and the flux of each species is that mass flux times the mass fraction on the side the mass comes from,

$$F_{\rho X_s} = F_\rho\,X_s^{\rm upwind} , \qquad X_s^{\rm upwind} = \frac{(\rho X_s)_{\rm face}}{\rho_{\rm face}} ,$$

with each $\rho X_s$ reconstructed like the density. A uniform $X_s$ is therefore advected exactly, to round-off, in any flow. For the HLLC solver this flux is the same as treating $\rho X_s$ as extra conserved components inside the solver, as Quokka does; for the local Lax-Friedrichs solver it is less diffusive. The face values of the species are limited separately, so they do not add up to the face density, and the species fluxes do not add up to the mass flux: with second- and third-order reconstruction the sum $\sum_s X_s$ of a cell drifts from one, by $10^{-2}$ after ten steps of a coarse test. The species of each cell are therefore rescaled to add up to $\rho$ at the start of every Runge-Kutta stage, before the chemistry reads them, and at the end of every step, as Quokka does with its mass scalars. This conserves the total mass but not each species separately. Consistent multi-fluid advection (CMA; Plewa & Müller 1999), which normalises the face mass fractions so that the species fluxes add up to the mass flux, would conserve each species, and is not implemented.

### 2.3 The local problem: an error-controlled stiff integrator

In every cell, for every substep, the chemistry operator integrates the local part of the equations,

$$\frac{\mathrm du}{\mathrm dt} = S(u) , \qquad u = (n_{\rm H^0},\, n_{\rm H^+},\, e,\, N_\gamma) ,$$

with $S$ the source terms of sections 1.1 to 1.4: the rates $\mathcal R_s$, the heating minus the cooling, and the photon loss $-\hat c\,\sigma\,n_{\rm H^0}N_\gamma$. The emission enters through the initial value of $N_\gamma$ (section 2.1).

**Why the local problem is stiff.** The local rates are fast compared with the substep wherever the neutral cells are optically thick, the gas is dense, or a source is near. Over one substep, the photon loss of a neutral cell is $\hat c\,\sigma\,n_{\rm H^0}\Delta t_r \approx (C_r/D)\,\tau_{\rm cell}$, with $\tau_{\rm cell} = \sigma\,n_{\rm H^0}\Delta x$ the optical depth of the cell and $D$ the number of dimensions. That is about 0.8 for Iliev's Test 1 at $16^3$ cells, but it grows in proportion to the density and the cell size. For gas of $100$ cm$^{-3}$ in cells of 1 pc, with the 13.6 eV cross-section, it is about 200. Near a source $\Gamma\,\Delta t_r \gg 1$, and recombination and cooling grow with the density. The terms are also tightly coupled. The photon loss and the ionization share the product $n_{\rm H^0}N_\gamma$, so an explicit step that overshoots drives $N_\gamma$ or $n_{\rm H^0}$ negative. And the heating and cooling change the temperature, and with it every rate. An explicit method would need many steps per substep in such cells and could not keep the abundances positive, so the local problem is solved implicitly.

**The default: a Rosenbrock method with error control.** A Rosenbrock method is a linearly implicit Runge-Kutta method. Each of its stages solves a linear system with the matrix $I - \gamma h J$, with $J = \partial S/\partial u$ and $h$ the internal step, instead of solving the nonlinear implicit equations by Newton iteration. It is stable on stiff problems and costs one Jacobian and one factorisation per step. The default is `Rosenbrock23` of OrdinaryDiffEq: the method of MATLAB's `ode23s` (Shampine & Reichelt 1997), a second-order Rosenbrock-W method with an embedded third-order error estimate. A W-method keeps its order even if the Jacobian is not exact. The integrator compares the error estimate of each internal step with the tolerance $\epsilon_i = a_i + r\,|u_i|$ of each unknown, and shrinks or grows the next step to match it, so a cell takes as many internal steps as its own evolution needs, and the result has a controlled error instead of an assumed one.

The tolerances are a relative $r = 10^{-2}$ and absolute tolerances $a_i$ of $10^{-8}$ times a physical scale of each unknown: the hydrogen density $n_{\rm H}$ for the species, the cell's own internal energy for $e$, and $\phi\,n_{\rm H}$ for the photons, which is the photon density whose budget $N_\gamma/\phi$ is one photon per hydrogen nucleus. A tolerance is then a fraction of a physically meaningful amount in every cell and in any unit system, and a species that holds a negligible share of a cell does not drive the step size to zero on its own relative error.

Two more choices make the integration robust:

- **An exact Jacobian.** The Jacobian is computed analytically, including the temperature dependence of every rate through $\partial T/\partial n_s$ and $\partial T/\partial e$, and not by finite differences. A W-method does not need it exact to keep its order, but an exact one is cheaper than differencing and gives the step-size controller an accurate linearisation.
- **Positivity.** A negative $n_{\rm H^0}$, $n_{\rm H^+}$ or $N_\gamma$ enters every rate as zero, and the rates use the temperature $\max(T, T_{\min})$. A trial state of the integrator can fall a rounding error below zero, and a negative photon density must not create neutral atoms.

On the Iliev tests the integrator takes about 15 internal steps per cell per substep, costs about 5 to 6.5 µs per cell on one core, and never fails ([findings, section 1](photoionization-findings.md#cost-of-the-chemistry-step)).

**The flux.** The absorption attenuates the photon flux at the same rate as the photon density, and the chemistry adds no flux: emission is isotropic. So the flux is not integrated as further unknowns. It is multiplied by $N_\gamma^{\rm end}/N_\gamma^{\rm start}$, the factor by which the photon density fell over the substep, which is the exact attenuation factor of the flux. Quokka does the same with its `SKIP_PHOTOCHEMFLUX` option.

### 2.4 The alternative: one backward-Euler step

The local problem can also be solved with a single implicit step over the whole substep. The backward-Euler step is the root of

$$G(u) = u - u^n - \Delta t_r\,S(u) = 0 ,$$

which Newton's method solves with the linear systems $(I - \Delta t_r J)\,\delta = -G(u)$. The Newton update is damped by a backtracking line search on the size of $G$, measured with the scales of section 2.3, and the species and photon densities are kept non-negative. It is first order in $\Delta t_r$, needs no error control, and costs about 0.1 µs per cell, 45 to 100 times less than the Rosenbrock integrator.

Its weakness is the nonlinear solve. A single backward-Euler step is an equation that Newton's method can fail to solve, in three situations that are known:

- **A stiff growth term,** such as collisional ionization, whose rate per proton is $\beta\,n_{\rm H^0}$, with $\beta\,n_{\rm H^0}\,\Delta t_r > 1$. The root is positive, but the Newton direction from the old state points into the bound at zero, and no line search crosses it.
- **Very stiff photoionization,** with $\Gamma\,\Delta t_r$ of $10^2$ to $10^3$, where the temperature and the bounds interact within the step.
- **A discontinuous rate,** such as the recombination cooling that switches on at 100 K, across which the iteration can cycle.

On a stress grid of 2592 cells it failed to converge in 10% of them, and on the Iliev tests in none. On Test 2 it agrees with the Rosenbrock reference to 0.1% in the radius of the ionized region, and to within 29% in the temperature of the few cells the front crosses at the time ([findings, section 2](photoionization-findings.md#the-two-integrators-on-test-2)). It is the comparison for the Rosenbrock integrator, and the cheap choice on a problem where it converges. A cell that fails is reported, never hidden.

## 3. Design

Two principles shape the module.

**The physics is written once, and the numerics never restate it.** The local problem is defined by a *network* ([`AbstractNetwork`](reference/chemistry.qmd#AbstractNetwork)): its species, its bands, the source term $S(u)$ and its exact Jacobian, and the scales of its unknowns. It is solved by an *integrator* ([`AbstractChemSolver`](reference/chemistry.qmd#AbstractChemSolver)) that reads the network only through those. The two integrators of sections 2.3 and 2.4 therefore solve exactly the same problem, so a difference between their answers is a difference of method and nothing else. A new network, for instance hydrogen and helium with three bands, or a new integrator, can be added without changing the other. This is the pattern of Honghu's matter-radiation coupling, and the test suite runs a toy network with an exact solution through both integrators to keep the interface honest.

**What is not a term of the local ODE stays out of it.** The emission is a source from outside the cell, the flux attenuation follows from the solution rather than being part of it, and keeping the stored state admissible is the grid's business. All three are done around the local solve, by the operator of section 2.1, and none is in $S(u)$. Putting any of them into $S$ would make the integrators solve different problems.

## 4. Verification

The tests pin properties that a correct implementation must have, not numbers that a better scheme would change. The analytic Jacobian is compared with a finite difference in 256-bit arithmetic, and the rate coefficients with values computed by hand. With the sinks switched off, the photon budget and the energy of section 1.7 are conserved to $10^{-10}$ in a single cell, for both integrators, and to $10^{-8}$ over a whole simulation with a uniform source. The ionization equilibrium $x^2/(1-x) = \Gamma/(\alpha_B n_{\rm H})$ is reached, and it does not depend on $\hat c$. With the gas moving, hydrogen mass is conserved and the species add up to $\rho$ in every cell. On Iliev's Test 1 the radius of the ionized region follows the finite-$\hat c$ reference to within 4.5% from 0.5 to 2 recombination times at $16^3$ cells, and on Test 2 the front grows, with hot ionized gas behind it and cold gas ahead. The cost of the chemistry, the comparison of the two integrators and the dependence on $\hat c$ are measured in [the findings note](photoionization-findings.md).

## 5. Limitations

The model and the method have these limits, each of which a run must respect:

- **Hydrogen only, one band.** More species and bands fit the model of section 1, with the electrons from charge neutrality, but only one band is implemented, and sources emit into one band.
- **On the spot.** The diffuse recombination radiation is not followed (section 1.5).
- **No radiation pressure.** The momentum of the absorbed photons is not given to the gas, so the velocity-dependent radiation terms must be off (`beta_order = 0`).
- **No thermal radiation alongside.** The thermal matter-radiation coupling takes the gas temperature from an equation of state of constant mean molecular weight, which ignores the ionization state and is wrong by up to a factor of 2 in ionized gas. A simulation with chemistry therefore refuses any radiation group with thermal opacity.
- **A constant reduced speed of light.** The photon budget of section 1.6 needs one $\phi$, and the stability bound of Honghu's adaptive reduced speed of light has no photoabsorption term.
- **A monatomic gas in CGS units.** The temperature of section 1.1 needs $\gamma = 5/3$, and the rates are in CGS units.
- **First-order splitting** of the emission and of the transport (section 2.1), and a renormalisation of the species that conserves the total mass but not each species (section 2.2).
- **One band cannot get the heating right** for a broad spectrum (section 1.4).
- **The front is not quite round** at low resolution, and part of the difference from the finite-$\hat c$ reference is not explained ([findings, section 3](photoionization-findings.md#the-reduced-speed-of-light-on-test-1)).

## References

- Aubert, D. & Teyssier, R. 2008, MNRAS, 387, 295: a radiative transfer scheme for cosmological reionization based on a local Eddington tensor (ATON).
- Cen, R. 1992, ApJS, 78, 341: the collisional-excitation cooling of hydrogen.
- Gnedin, N. Y. & Abel, T. 2001, New Astron., 6, 437: the reduced speed of light approximation.
- Iliev, I. T., et al. 2006, MNRAS, 371, 1057: the Cosmological Radiative Transfer Codes Comparison Project, Tests 1 and 2.
- Katz, N., Weinberg, D. H. & Hernquist, L. 1996, ApJS, 105, 19: the form of the cooling rates (their Table 1).
- Krumholz, M. R., Stone, J. M. & Gardiner, T. A. 2007, ApJ, 671, 518: the collisional ionization and recombination cooling rates.
- Levermore, C. D. 1984, J. Quant. Spectrosc. Radiat. Transfer, 31, 149: the Eddington factor of the M1 closure.
- Plewa, T. & Müller, E. 1999, A&A, 342, 179: consistent multi-fluid advection.
- Shampine, L. F. & Reichelt, M. W. 1997, SIAM J. Sci. Comput., 18, 1: the MATLAB ODE suite and its `ode23s` method.
- Shapiro, P. R., Iliev, I. T., Alvarez, M. A. & Scannapieco, E. 2006, ApJ, 648, 922: ionization fronts at a finite speed of light.
- Quokka's photoionization module: `docs/markdown/photoionization.md` and `src/networks/photoionization_H_caseB_handwritten/actual_rhs.H` in the Quokka repository.
