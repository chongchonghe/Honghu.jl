# Radiation test problems.
#
# All of these run in code units with c = a_rad = 1, which keeps the numbers
# readable; nothing in the code depends on that choice.

"""
    radiation_streaming(; n = 128, chat_over_c = 1.0, closure = LevermoreClosure(),
                        beta_order = 0, kwargs...) -> (sim, tmax)

Free-streaming limit: radiation injected at the left boundary of a transparent
(`kappa = 0`), static medium. The front must advance at exactly `chat` while
keeping the reduced flux at the causal limit, which is where the M1 closure
differs from flux-limited diffusion.

At `tmax = 0.5` the front sits at `x = 0.5 chat`.
"""
function radiation_streaming(; n::Integer=128, nx::Integer=n, chat_over_c=1.0,
                             closure::AbstractClosure=LevermoreClosure(),
                             beta_order::Integer=0, kwargs...)
    c = 1.0
    E0 = 1.0
    rad = RadiationPhysics(; c=c, chat=c * chat_over_c, a_rad=1.0, closure, beta_order,
                           opacity=ConstantOpacity(0.0))
    # ghost state: a beam at (very nearly) the causal limit
    beam = Fixed((E0, 0.999 * c * E0, 0.0, 0.0))
    sim = Simulation(; n=(nx, 1, 1), xhi=(1.0, 1.0, 1.0),
                     radiation=rad, hydro=false,
                     boundaries=Boundaries(Outflow()),
                     radiation_boundaries=Boundaries((beam, Outflow()),
                                                     (Outflow(), Outflow()),
                                                     (Outflow(), Outflow())),
                     kwargs...)
    set_initial_conditions!(sim) do x, y, z
        (rho=1.0, p=1.0, Erad=1e-10, Fx=0.0)
    end
    return sim, 0.5
end

"""
    radiation_diffusion(; n = 128, kappa = 300.0, sigma0 = 0.05,
                        closure = LevermoreClosure(), beta_order = 0, kwargs...)
        -> (sim, tmax)

Static-diffusion limit: a Gaussian pulse of radiation energy in a periodic,
purely scattering medium (`kappa_P = kappa_E = 0`, `kappa_F = kappa`), with the
matter held fixed.

The M1 system reduces to `∂E/∂t = D ∂²E/∂x²` with `D = c/(3 rho kappa)`, so the
pulse stays Gaussian with `sigma² = sigma0² + 2 D t` and its peak falls as
`sigma0/sigma`. This is the test of the scheme's asymptotic-preserving property:
the answer must not depend on how many mean free paths a cell spans.

Note that a reduced speed of light rescales `D` by `chat/c`, so this problem is
set up with `chat = c`.
"""
function radiation_diffusion(; n::Integer=128, nx::Integer=n, kappa=300.0, sigma0=0.05, E0=1.0,
                             closure::AbstractClosure=LevermoreClosure(),
                             beta_order::Integer=0, kwargs...)
    c = 1.0
    rad = RadiationPhysics(; c=c, chat=c, a_rad=1.0, closure, beta_order,
                           opacity=ConstantOpacity(planck=0.0, energy=0.0, flux=kappa))
    sim = Simulation(; n=(nx, 1, 1), xhi=(1.0, 1.0, 1.0),
                     radiation=rad, hydro=false,
                     boundaries=Boundaries(Periodic()), kwargs...)
    set_initial_conditions!(sim) do x, y, z
        (rho=1.0, p=1.0, Erad=E0 * exp(-(x - 0.5)^2 / (2 * sigma0^2)), Fx=0.0)
    end
    return sim, 1.0
end

"""
    diffusion_exact(x, t; sigma0 = 0.05, E0 = 1.0, D = 1/900, x0 = 0.5)

Analytic solution of [`radiation_diffusion`](@ref): a Gaussian of width
`sigma² = sigma0² + 2 D t` whose integral is conserved.
"""
function diffusion_exact(x, t; sigma0=0.05, E0=1.0, D=1 / 900, x0=0.5)
    s = sqrt(sigma0^2 + 2 * D * t)
    E0 * (sigma0 / s) * exp(-(x - x0)^2 / (2 * s^2))
end

"""
    matter_coupling(; n = 4, kappa = 1e3, T_gas = 1.0, T_rad = 0.1, cv = 1.0, kwargs...)
        -> (sim, tmax)

Stiff matter-radiation coupling with no spatial structure: a uniform box whose
gas and radiation start far out of equilibrium and relax towards `a_rad T⁴ =
E_rad` on a timescale `1/(c rho kappa)` that is orders of magnitude shorter than
the transport step.

Passing this test means the implicit solve is doing its job: total energy
`E_int + (c/chat) E_rad` must be conserved to round-off, and the final state
must sit on the equilibrium curve regardless of how large the step is.

The state is uniform, so `n` changes nothing but the number of cells solving the
same problem; it exists so that every problem takes resolution under one name.
"""
function matter_coupling(; n::Integer=4, kappa=1e3, T_gas=1.0, T_rad=0.1, cv=1.0,
                         gamma=5 / 3, kwargs...)
    rad = RadiationPhysics(; c=1.0, chat=1.0, a_rad=1.0, beta_order=0,
                           opacity=ConstantOpacity(kappa))
    sim = Simulation(; n=(n, 1, 1), xhi=(1.0, 1.0, 1.0),
                     eos=IdealGas(; gamma, cv), radiation=rad, hydro=false,
                     boundaries=Boundaries(Periodic()), kwargs...)
    set_initial_conditions!(sim) do x, y, z
        (rho=1.0, T=T_gas, Erad=T_rad^4)
    end
    return sim, 1e-2
end

"""
    rad_dust(; n = 8, groups = GREY, chat_over_c = 0.1, coefficient = 1e6,
             kappa = 1e4, kwargs...) -> (sim, tmax)

Gas, dust and radiation in a uniform static medium, from Quokka's `RadDust` and
`RadDustMG` tests: the gas starts at `T = 1` with no radiation, and cools by
colliding with dust that radiates. The radiation couples to the dust only
([`BateKetoDust`](@ref)), so the gas cools at the collisional rate
`coefficient n^2 sqrt(T) (T - T_d)` rather than at the radiative one -- here a
hundred times slower -- and the dust temperature sits between the two.

Units and numbers are Quokka's: `c = 1e8`, `chat = 0.1 c`, `rho = 1`,
`a_rad = 1`, `cv = 1.5` (`mu = 1`, `k_B = 1`, `gamma = 5/3`), `rho kappa = 1e4`,
`coefficient = 1e6`, and the radiation time step `0.8 dx / chat = 1e-8` at eight
cells. One difference: Quokka replaces `B = a T^4` by `B = a T` so that its
reference solution is a linear ODE; this code has no hook for the emission law,
so it keeps `a T^4`, and [`rad_dust_reference`](@ref) integrates the model's own
ODE instead.

`groups` other than `GREY` gives the multigroup form (`RadDustMG` uses edges
`1e-3, 0.1, 1, 10, 1e3`); with a constant opacity the answer does not depend on
the grouping, which is the point of that test.
"""
function rad_dust(; n::Integer=8, groups::Honghu.RadiationGroups=Honghu.GREY, chat_over_c=0.1,
                  coefficient=1e6, kappa=1e4, kwargs...)
    c = 1e8
    rad = RadiationPhysics(; c, chat=chat_over_c * c, a_rad=1.0, groups, beta_order=0,
                           opacity=ConstantOpacity(kappa),
                           dust=BateKetoDust(; coefficient))
    sim = Simulation(; n=(n, 1, 1), xhi=(1.0, 1.0, 1.0),
                     eos=IdealGas(; gamma=5 / 3, cv=1.5), radiation=rad, hydro=false,
                     boundaries=Boundaries(Periodic()), floors=Floors(erad=1e-20),
                     cfl_radiation=0.8, kwargs...)
    set_initial_conditions!(sim) do x, y, z
        (rho=1.0, T=1.0, Erad=1e-20, Fx=0.0)
    end
    return sim, 3e-5
end

"""
    rad_dust_reference(ts; groups = GREY, chat_over_c = 0.1, coefficient = 1e6,
                       kappa = 1e4) -> (T_gas, E_rad)

The error-controlled solution of [`rad_dust`](@ref) at the times `ts`: the gas
temperature and the total radiation energy density. The model's ODE -- `E_gas`
and the `E_g` evolved, `T_d` solved at every evaluation from the dust balance --
integrated with `Rodas5P` at `reltol = 1e-10`. With a constant opacity the dust
balance is monotone in `T_d`, so that inner solve has one root and a bracket.
"""
function rad_dust_reference(ts; groups::Honghu.RadiationGroups{NG}=Honghu.GREY, chat_over_c=0.1,
                            coefficient=1e6, kappa=1e4) where {NG}
    c, chat, cv, rho, arad, kB = 1e8, chat_over_c * 1e8, 1.5, 1.0, 1.0, 1.0
    K = coefficient * rho^2
    chi = rho * kappa
    function dust_T(T, E)
        h(x, _) = c * chi * (arad * x^4 * sum(Tuple(planck_fractions(groups, x, kB))) -
                             sum(E)) - K * sqrt(T) * (T - x)
        # at fixed T and E the balance increases with T_d: bracket from zero to
        # above both the gas and the radiation temperature
        hi = 2 * max(T, (max(sum(E), 0.0) / arad)^0.25) + 1
        Honghu.solve(Honghu.IntervalNonlinearProblem{false}(h, (1e-12, hi), nothing),
                     Honghu.Brent(); abstol=0.0).u
    end
    function rhs(u, _, _)
        T = max(u[1] / cv, 1e-12)
        E = u[2:end]
        Td = dust_T(T, E)
        f = planck_fractions(groups, Td, kB)
        dE = [chat * chi * (arad * Td^4 * f[g] - E[g]) for g in 1:NG]
        return vcat(-K * sqrt(T) * (T - Td), dE)
    end
    u0 = vcat(cv * 1.0, fill(1e-20 / NG, NG))
    prob = ODEProblem(rhs, u0, (0.0, maximum(ts)))
    sol = Honghu.solve(prob, Rodas5P(autodiff=AutoFiniteDiff()); reltol=1e-10,
                       abstol=1e-14, saveat=ts)
    return ([u[1] / cv for u in sol.u], [sum(u[2:end]) for u in sol.u])
end

"""
    marshak_wave(; n = 256, kappa = 300.0, T_wall = 1.0, cv = 1.0,
                 closure = LevermoreClosure(), beta_order = 0, kwargs...)
        -> (sim, tmax)

A non-equilibrium Marshak wave: a cold, static, optically thick slab heated from
`x = 0` by a wall held in radiative equilibrium at `T_wall`. Radiation diffuses
inward and drags the matter temperature behind it, so the gas and radiation
temperatures separate at the front and merge behind it.

The wall is imposed as a Dirichlet condition on the radiation moments,
`E = a_rad T_wall⁴` with zero flux; the HLL solver then supplies the inward flux
set by the jump across the boundary.
"""
function marshak_wave(; n::Integer=256, nx::Integer=n, kappa=300.0, T_wall=1.0, T_init=1e-4,
                      cv=1.0, gamma=5 / 3,
                      closure::AbstractClosure=LevermoreClosure(),
                      beta_order::Integer=0, kwargs...)
    a_rad = 1.0
    rad = RadiationPhysics(; c=1.0, chat=1.0, a_rad=a_rad, closure, beta_order,
                           opacity=ConstantOpacity(kappa))
    wall = Fixed((a_rad * T_wall^4, 0.0, 0.0, 0.0))
    sim = Simulation(; n=(nx, 1, 1), xhi=(1.0, 1.0, 1.0),
                     eos=IdealGas(; gamma, cv), radiation=rad, hydro=false,
                     boundaries=Boundaries(Outflow()),
                     radiation_boundaries=Boundaries((wall, Outflow()),
                                                     (Outflow(), Outflow()),
                                                     (Outflow(), Outflow())),
                     kwargs...)
    set_initial_conditions!(sim) do x, y, z
        (rho=1.0, T=T_init, Erad=a_rad * T_init^4, Fx=0.0)
    end
    return sim, 1.0
end

"""
    radhydro_shock(; n = 256, closure = EddingtonClosure(), beta_order = 1,
                   chat_factor = 10.0, chat_limiter = ConstantChat(),
                   kwargs...) -> (sim, tmax)

The Mach 3 radiative shock of Lowrie & Edwards (2008), in the form used by
Quokka's `RadhydroShock` test. This is the most demanding one-dimensional
radiation-hydrodynamics problem here: every part of the scheme has to be right
at once.

A Mach 3 shock sits in a medium that is optically thick (about nine mean free
paths across the domain) and strongly coupled, so the radiation both preheats
the gas ahead of the shock -- the radiative precursor -- and, immediately behind
the front, leaves the matter temperature briefly *above* its downstream value.
That overshoot is the Zel'dovich spike, `T_matter` reaching 4.26 against a
downstream 3.66, and it is what the test is really for: it only appears if the
velocity-dependent coupling terms, the implicit solve and the shock capturing
are all correct together.

Everything is in the dimensionless units of the reference solution: `c_s0 = 1`
upstream, `T = 1` upstream, `rho = 1` upstream, `a_rad = 1e-4`, absorption cross
section `1e6`, so that `c = sqrt(3 sigma_a) = 1732`. The reduced speed of light
is `10 (v_0 + c_s0) = 40`.

Both ends are held at the upstream and downstream Rankine-Hugoniot states, and
the shock is initialised slightly left of where it settles, because it drifts a
little during the run.

Runs with [`EddingtonClosure`](@ref) rather than M1, to match the assumption the
semi-analytic solution is derived under -- exactly as Quokka overrides
`ComputeEddingtonFactor` for this problem. Pass `closure = LevermoreClosure()`
to measure what that assumption costs; the difference is the closure error, not
a scheme error. `chat_factor` scales the reduced speed of light, whose default
of `10 (v_0 + c_s0)` is Quokka's -- and which is already a fairly aggressive
reduction, `phi0 = chat/c = 0.023`. `chat_factor = 433` recovers `chat = c` and
is the reference a `chat_limiter` should be scored against.

Compare against [`radhydro_shock_exact`](@ref); `tmax` is three shock-crossing
times.
"""
function radhydro_shock(; n::Integer=256, nx::Integer=n,
                        closure::AbstractClosure=EddingtonClosure(),
                        beta_order::Integer=1, chat_factor=10.0,
                        chat_limiter::Honghu.AbstractChatLimiter=ConstantChat(),
                        kwargs...)
    # upstream (left) and downstream (right) Rankine-Hugoniot states at Mach 3
    Lx = 0.01578396467532876                      # 9.11 mean free paths
    sim = _lowrie_edwards_shock(; nx, closure, beta_order, chat_limiter,
                                mach=3.0, T_hi=3.661912665809719,
                                rho1=3.0021676971081166, xlo=0.0, xhi=Lx,
                                shock_position=0.0130,
                                chat=chat_factor * (3.0 + 1.0), kwargs...)
    return sim, 3 * Lx / 3.0                      # three shock-crossing times
end

"""
    radhydro_shock_super(; n = 128, chat_over_c = 0.025, closure = EddingtonClosure(),
                         beta_order = 1, chat_limiter = ConstantChat(), kwargs...)
        -> (sim, tmax)

The Mach 5 *supercritical* radiative shock of Lowrie & Edwards (2008), in the
form of Quokka's `RadhydroShockSuper` test: the same dimensionless units and
opacity as [`radhydro_shock`](@ref), downstream state `rho = 3.5979`,
`T = 8.5572`, on `x in [-0.04, 0.01]` with the shock initially at `x = 0`.

Supercritical means the radiative precursor heats the upstream gas to nearly the
downstream temperature, and that makes it the problem most sensitive to the RSLA:
the strong exchange in the precursor amplifies the RSLA's energy error. Quokka's
default reduction for it is `chat/c = 0.025`. `chat_over_c = 1` is the reference
a limiter should be scored against; `tmax` is three shock-crossing times.
"""
function radhydro_shock_super(; n::Integer=128, nx::Integer=n, chat_over_c=0.025,
                              closure::AbstractClosure=EddingtonClosure(),
                              beta_order::Integer=1,
                              chat_limiter::Honghu.AbstractChatLimiter=ConstantChat(),
                              kwargs...)
    c = sqrt(3 * 1.0e6)
    sim = _lowrie_edwards_shock(; nx, closure, beta_order, chat_limiter,
                                mach=5.0, T_hi=8.557199218476, rho1=3.597910653061,
                                xlo=-0.04, xhi=0.01, shock_position=0.0,
                                chat=chat_over_c * c, kwargs...)
    return sim, 3 * 0.05 / 5.0                    # three shock-crossing times
end

# The Lowrie & Edwards (2008) radiative shock at a given Mach number, shared by
# `radhydro_shock` (Mach 3) and `radhydro_shock_super` (Mach 5).
function _lowrie_edwards_shock(; nx, closure, beta_order, chat_limiter, mach, T_hi,
                               rho1, xlo, xhi, shock_position, chat, kwargs...)
    a_rad = 1.0e-4
    sigma_a = 1.0e6
    c_s0 = 1.0
    c = sqrt(3 * sigma_a) * c_s0                  # 1732.0508...
    gamma = 5 / 3
    mu = gamma                                    # chosen so that c_s0 == 1
    k_B = c_s0^2
    cv = k_B / (mu * (gamma - 1))                 # 0.9
    kappa_abs = sigma_a * (c_s0 / c)              # absorption coefficient, 1/length

    T_lo, rho0 = 1.0, 1.0
    v0 = mach * c_s0
    v1 = v0 * rho0 / rho1

    eos = IdealGas(; gamma, cv)
    rad = RadiationPhysics(; c, chat, a_rad, closure, beta_order, chat_limiter,
                           opacity=UserOpacity((rho, T) -> kappa_abs / rho))

    egas(rho, T) = rho * cv * T
    lo_gas = Fixed((rho0, rho0 * v0, 0.0, 0.0, egas(rho0, T_lo) + 0.5 * rho0 * v0^2))
    hi_gas = Fixed((rho1, rho1 * v1, 0.0, 0.0, egas(rho1, T_hi) + 0.5 * rho1 * v1^2))
    lo_rad = Fixed((a_rad * T_lo^4, 0.0, 0.0, 0.0))
    hi_rad = Fixed((a_rad * T_hi^4, 0.0, 0.0, 0.0))
    outflow = (Outflow(), Outflow())

    sim = Simulation(; n=(nx, 1, 1), xlo=(xlo, 0.0, 0.0), xhi=(xhi, 1.0, 1.0), eos=eos,
                     radiation=rad,
                     boundaries=Boundaries((lo_gas, hi_gas), outflow, outflow),
                     radiation_boundaries=Boundaries((lo_rad, hi_rad), outflow, outflow),
                     cfl=0.4, cfl_radiation=0.4, kwargs...)
    set_initial_conditions!(sim) do x, y, z
        if x < shock_position
            (rho=rho0, vx=v0, T=T_lo, Erad=a_rad * T_lo^4, Fx=0.0)
        else
            (rho=rho1, vx=v1, T=T_hi, Erad=a_rad * T_hi^4, Fx=0.0)
        end
    end
    return sim
end

"""
    radiation_shadow(; n = (280, 112), kappa = 1.0, rho_blob = 1e3,
                     closure = LevermoreClosure(), beta_order = 0, kwargs...)
        -> (sim, tmax)

The shadow test of Hayes & Norman (2003) in 2D: a collimated beam enters from
the left of a transparent box and strikes an opaque, absorbing elliptical clump
of semi-axes `(0.10, 0.06)` centred at `x = 0.25`.

`n` may be the pair of cell counts or a single number, which is taken as the `x`
resolution and gives `ny = 2 nx / 5` -- the ratio that makes cells square on
this `1 x 0.4` domain, and the one the default `(280, 112)` already has.

A two-moment scheme with the M1 closure preserves the direction of the
radiation and leaves a sharp shadow behind the clump; a diffusion-based scheme
lets radiation leak around the obstacle and fills the shadow in. Far downstream
M1 does fill the shadow in slowly, because it carries a single flux vector per
cell and so cannot represent the two crossing beams that meet there.

The clump is given a very large heat capacity so that it absorbs without
re-emitting, which isolates the transport.
"""
function radiation_shadow(; n=(280, 112), kappa=1.0, rho_blob=1e3, rho_bg=1e-2,
                          closure::AbstractClosure=LevermoreClosure(),
                          beta_order::Integer=0, kwargs...)
    c = 1.0
    E0 = 1.0
    # The domain is 1 x 0.4, so the two cell counts are not independent: square
    # cells need `ny = 2 nx / 5`. A scalar `n` is the x resolution and supplies
    # the other, which is what makes this problem refinable by `compare`; the
    # default (280, 112) is exactly that ratio.
    nxy = n isa Integer ? (n, round(Int, 2n / 5)) : n
    rad = RadiationPhysics(; c=c, chat=c, a_rad=1.0, closure, beta_order,
                           opacity=ConstantOpacity(kappa))
    beam = Fixed((E0, 0.999 * c * E0, 0.0, 0.0))
    sim = Simulation(; n=(nxy[1], nxy[2], 1), xlo=(0.0, -0.2, 0.0), xhi=(1.0, 0.2, 1.0),
                     eos=IdealGas(gamma=5 / 3, cv=1e10),
                     radiation=rad, hydro=false,
                     boundaries=Boundaries(Outflow()),
                     radiation_boundaries=Boundaries((beam, Outflow()),
                                                     (Outflow(), Outflow()),
                                                     (Outflow(), Outflow())),
                     kwargs...)
    ax, ay = 0.10, 0.06
    set_initial_conditions!(sim) do x, y, z
        inside = ((x - 0.25) / ax)^2 + (y / ay)^2 < 1
        (rho=inside ? rho_blob : rho_bg, p=1.0, Erad=1e-10, Fx=0.0)
    end
    return sim, 1.5
end

"""
    marshak_vaytet(; n = 64, ngroups = 4, Lx = 20.0, chat_over_c = 1.0,
                   chat_limiter = ConstantChat(), coupling = QuokkaCoupling(),
                   closure = LevermoreClosure(), kwargs...) -> (sim, tmax)

The frequency-dependent Marshak wave of Vaytet et al. (2011), in the form of
Quokka's `RadMarshakVaytet` test: a cold slab of `rho = 10⁻³ g cm⁻³` at 300 K,
heated from `x = 0` by a wall in radiative equilibrium at `T_L = 1000 K`, with
an opacity that falls as `nu⁻²`.

This is the multigroup counterpart of [`marshak_wave`](@ref), and it is here
because the two disagree: with `kappa = 2000 (nu / 4×10¹³ Hz)⁻² cm² g⁻¹` the
groups span four decades of optical depth at once — a mean free path of 10⁻⁵ cm
in the lowest group and 10 cm in the highest — so the high-frequency groups run
ahead of the wave while the low-frequency ones are trapped in it. A grey
calculation, which has one opacity to describe all of that, cannot reproduce the
front. The number of groups is the convergence parameter of the test, hence
`ngroups`: its edges are always logarithmically spaced from `6×10¹⁰` to
`6×10¹⁴ Hz`, which reproduces Quokka's tabulated edges exactly at every group
count it lists (4 gives `6e10, 6e11, 6e12, 6e13, 6e14`).

Everything is CGS, the medium is static (`hydro = false`, `beta_order = 0`) and
`chat = c` unless `chat_over_c` says otherwise. This problem is genuinely
time-dependent, so the answer *does* depend on the reduced speed of light: it is
the one here that measures what a `chat_limiter` buys. The opacity is piecewise constant across a group, evaluated at the
group's geometric centre, which is Quokka's `piecewise_constant_opacity` model.

The domain length is a keyword because the choice matters and Quokka keeps it in
an inputs file rather than in the problem source. `Lx = 20 cm` is the value that
file uses, and it is the right one: at `tmax = 1.36×10⁻⁷ s` and 64 cells the
thermal front (`T_gas = 650 K`, half way from 300 K to the wall) sits at about
3.8 cm, so the wave itself is comfortably inside. What does reach the far
boundary is the top group, whose mean free path is 11 cm: it is optically thin,
it streams across the slab, and it has warmed the far end to about 370 K by
`tmax`. That is the physics the test exists to show, not an error, and it is why
the far boundary is outflow rather than a cold wall — a shorter `Lx` would
simply move the outflow closer, while a longer one costs proportionally more
steps for no extra physics.

Has a [`default_metrics`](@ref) method: the front position and the near-wall
radiation temperature.
"""
function marshak_vaytet(; n::Integer=64, nx::Integer=n, ngroups::Integer=4, Lx=20.0,
                        chat_over_c=1.0,
                        chat_limiter::Honghu.AbstractChatLimiter=ConstantChat(),
                        coupling::Honghu.AbstractCoupling=QuokkaCoupling(),
                        closure::AbstractClosure=LevermoreClosure(), kwargs...)
    rho0 = 1.0e-3               # g cm^-3
    T_initial = 300.0           # K
    T_L = 1000.0                # K, the wall
    rho_C_V = 1.0e-3            # erg cm^-3 K^-1, so cv is 1 erg g^-1 K^-1
    cv = rho_C_V / rho0
    gamma = 5 / 3
    kappa0 = 2000.0             # cm^2 g^-1 at the pivot frequency
    nu_pivot = 4.0e13           # Hz

    c = Honghu.CGS.c_light
    a_rad = Honghu.CGS.a_rad
    k_B = Honghu.CGS.k_B

    groups = RadiationGroups(6.0e10, 6.0e14, ngroups;
                             energy_unit=Honghu.CGS.h_planck, spacing=:log)
    # No grey value: kappa really does depend on frequency here, and letting a
    # single-group run silently pick one would be the bug FrequencyOpacity exists
    # to prevent.
    opacity = FrequencyOpacity((nu, rho, T) -> kappa0 * (nu_pivot / nu)^2)

    rad = RadiationPhysics(; c=c, chat=c * chat_over_c, a_rad=a_rad, k_B=k_B,
                           groups=groups, opacity=opacity, closure=closure,
                           coupling=coupling, chat_limiter=chat_limiter, beta_order=0)

    # The wall: a Planck spectrum at T_L with no net flux. The HLL solver then
    # supplies the inward flux set by the jump across the boundary, exactly as in
    # the grey Marshak wave.
    fracL = planck_fractions(groups, T_L, k_B)
    EL = a_rad * T_L^4
    wall = Fixed(ntuple(Honghu.NRAD * ngroups) do m
        gr, comp = divrem(m - 1, Honghu.NRAD)
        comp == 0 ? EL * fracL[gr+1] : 0.0
    end)
    outflow = (Outflow(), Outflow())

    sim = Simulation(; n=(nx, 1, 1), xhi=(Lx, 1.0, 1.0),
                     eos=IdealGas(; gamma, cv), radiation=rad, hydro=false,
                     boundaries=Boundaries(Outflow()),
                     radiation_boundaries=Boundaries((wall, Outflow()), outflow, outflow),
                     floors=Floors(erad=a_rad * T_initial^4 * 1e-20),
                     cfl_radiation=0.8, kwargs...)
    set_initial_conditions!(sim) do x, y, z
        (rho=rho0, T=T_initial, Erad=a_rad * T_initial^4, Fx=0.0)
    end
    # A scalar `Erad` is a total, and `set_initial_conditions!` splits it over
    # the groups by the Planck spectrum at the temperature it implies -- here
    # `T_initial`, exactly. So the interior starts as a 300 K blackbody without
    # this problem writing a single group by hand; only the wall, which is a
    # *boundary*, still has to.
    return sim, 1.36e-7
end

"""
    total_erad(sim) -> Array

Radiation energy density summed over every frequency group, over the interior.

Redundant, and kept only because `default_metrics(marshak_vaytet)` and
`test/test_radiation_multigroup.jl` call it: `first(field(sim, :Erad))` sums the
groups too and returns the identical array, as does
[`radiation_temperature`](@ref) for the temperature it implies. Prefer the
accessors in new code — this dates from before they learned about groups.
"""
function total_erad(sim)
    g = sim.grid
    NG = ngroups(sim.radiation)
    ir, jr, kr = g.lo[1]:g.hi[1], g.lo[2]:g.hi[2], g.lo[3]:g.hi[3]
    out = zeros(eltype(sim.R), length(ir), length(jr), length(kr))
    for gr in 1:NG
        out .+= @view sim.R[Honghu.gbase(gr)+Honghu.IERAD, ir, jr, kr]
    end
    return out
end

"""
    radhydro_pulse_dyn(; n = 128, chat_over_c = 1.0, v0 = 3e7, kappa0 = 500.0,
                       chat_limiter = ConstantChat(), kwargs...) -> (sim, tmax)

The advecting radiation pulse in the dynamic diffusion limit of He, Wibking &
Krumholz (2024a), in the form of Quokka's `RadhydroPulseDyn` test, in CGS: a
Gaussian temperature pulse, `T` from `1e7` to `2e7` K over a width of 24 cm, in
pressure balance between gas and radiation, carried by a uniform flow of
`v0 = 3e7` cm/s through a periodic box 1024 cm wide. The opacity is `500 cm² g⁻¹`
at `rho ≈ 1.2 g cm⁻³`, so a cell is thousands of mean free paths thick and the
optical depth of the pulse times `v0/c` is about ten: the radiation is carried
with the gas rather than diffusing through it.

This is the regime in which Wibking & Krumholz (2022) show the RSLA makes
order-unity errors whatever `chat` is, and it is radiation-rich -- radiation
holds about a third of the heat capacity -- which is the regime in which the
stability bound caps `phi` at the RSLA step (see [`StabilityBound`](@ref)).
`chat_over_c = 1` is the reference; `tmax = 4.8e-6` s moves the pulse by 144 cm.
"""
function radhydro_pulse_dyn(; n::Integer=128, nx::Integer=n, chat_over_c=1.0, v0=3e7,
                            kappa0=500.0,
                            chat_limiter::Honghu.AbstractChatLimiter=ConstantChat(),
                            beta_order::Integer=1, wavespeed_correction::Bool=false,
                            kwargs...)
    c = Honghu.CGS.c_light
    a_rad = Honghu.CGS.a_rad
    k_B = Honghu.CGS.k_B
    mu = 2.33 * Honghu.CGS.m_u
    gamma = 5 / 3
    cv = k_B / (mu * (gamma - 1))
    T_lo, T_hi, rho0, width = 1.0e7, 2.0e7, 1.2, 24.0
    Tprof(x) = T_lo + (T_hi - T_lo) * exp(-x^2 / (2 * width^2))
    # gas + radiation pressure balance: rho k T/mu + a T^4/3 = const
    rhoprof(x) = (T = Tprof(x); rho0 * T_lo / T + (a_rad * mu / (3 * k_B)) * (T_lo^4 / T - T^3))
    rad = RadiationPhysics(; c, chat=chat_over_c * c, a_rad, k_B, beta_order, chat_limiter,
                           opacity=ConstantOpacity(kappa0), wavespeed_correction)
    sim = Simulation(; n=(nx, 1, 1), xlo=(-512.0, 0.0, 0.0), xhi=(512.0, 1.0, 1.0),
                     eos=IdealGas(; gamma, cv), radiation=rad,
                     boundaries=Boundaries(Periodic()), cfl=0.8, cfl_radiation=0.4,
                     kwargs...)
    set_initial_conditions!(sim) do x, y, z
        T = Tprof(x)
        (rho=rhoprof(x), vx=v0, T=T, Erad=a_rad * T^4, Fx=0.0)
    end
    return sim, 4.8e-6
end
