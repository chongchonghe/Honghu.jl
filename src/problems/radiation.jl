# Radiation test problems.
#
# All of these run in code units with c = a_rad = 1, which keeps the numbers
# readable; nothing in the code depends on that choice.

"""
    radiation_streaming(; nx = 128, chat_over_c = 1.0, kwargs...) -> (sim, tmax)

Free-streaming limit: radiation injected at the left boundary of a transparent
(`kappa = 0`), static medium. The front must advance at exactly `chat` while
keeping the reduced flux at the causal limit, which is where the M1 closure
differs from flux-limited diffusion.

At `tmax = 0.5` the front sits at `x = 0.5 chat`.
"""
function radiation_streaming(; nx::Integer=128, chat_over_c=1.0, kwargs...)
    c = 1.0
    E0 = 1.0
    rad = RadiationPhysics(; c=c, chat=c * chat_over_c, a_rad=1.0,
                           opacity=ConstantOpacity(0.0), beta_order=0)
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
    radiation_diffusion(; nx = 128, kappa = 300.0, sigma0 = 0.05, kwargs...)
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
function radiation_diffusion(; nx::Integer=128, kappa=300.0, sigma0=0.05, E0=1.0, kwargs...)
    c = 1.0
    rad = RadiationPhysics(; c=c, chat=c, a_rad=1.0, beta_order=0,
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
    matter_coupling(; kappa = 1e3, T_gas = 1.0, T_rad = 0.1, cv = 1.0, kwargs...)
        -> (sim, tmax)

Stiff matter-radiation coupling with no spatial structure: a uniform box whose
gas and radiation start far out of equilibrium and relax towards `a_rad T⁴ =
E_rad` on a timescale `1/(c rho kappa)` that is orders of magnitude shorter than
the transport step.

Passing this test means the implicit solve is doing its job: total energy
`E_int + (c/chat) E_rad` must be conserved to round-off, and the final state
must sit on the equilibrium curve regardless of how large the step is.
"""
function matter_coupling(; kappa=1e3, T_gas=1.0, T_rad=0.1, cv=1.0, gamma=5 / 3, kwargs...)
    rad = RadiationPhysics(; c=1.0, chat=1.0, a_rad=1.0, beta_order=0,
                           opacity=ConstantOpacity(kappa))
    sim = Simulation(; n=(4, 1, 1), xhi=(1.0, 1.0, 1.0),
                     eos=IdealGas(; gamma, cv), radiation=rad, hydro=false,
                     boundaries=Boundaries(Periodic()), kwargs...)
    set_initial_conditions!(sim) do x, y, z
        (rho=1.0, T=T_gas, Erad=T_rad^4)
    end
    return sim, 1e-2
end

"""
    marshak_wave(; nx = 256, kappa = 300.0, T_wall = 1.0, cv = 1.0, kwargs...)
        -> (sim, tmax)

A non-equilibrium Marshak wave: a cold, static, optically thick slab heated from
`x = 0` by a wall held in radiative equilibrium at `T_wall`. Radiation diffuses
inward and drags the matter temperature behind it, so the gas and radiation
temperatures separate at the front and merge behind it.

The wall is imposed as a Dirichlet condition on the radiation moments,
`E = a_rad T_wall⁴` with zero flux; the HLL solver then supplies the inward flux
set by the jump across the boundary.
"""
function marshak_wave(; nx::Integer=256, kappa=300.0, T_wall=1.0, T_init=1e-4,
                      cv=1.0, gamma=5 / 3, kwargs...)
    a_rad = 1.0
    rad = RadiationPhysics(; c=1.0, chat=1.0, a_rad=a_rad, beta_order=0,
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
    radhydro_shock(; nx = 256, kwargs...) -> (sim, tmax)

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
semi-analytic solution is derived under. Compare against
[`radhydro_shock_exact`](@ref); `tmax` is three shock-crossing times.
"""
function radhydro_shock(; nx::Integer=256, kwargs...)
    a_rad = 1.0e-4
    sigma_a = 1.0e6
    c_s0 = 1.0
    c = sqrt(3 * sigma_a) * c_s0                  # 1732.0508...
    gamma = 5 / 3
    mu = gamma                                    # chosen so that c_s0 == 1
    k_B = c_s0^2
    cv = k_B / (mu * (gamma - 1))                 # 0.9
    kappa_abs = sigma_a * (c_s0 / c)              # absorption coefficient, 1/length

    # upstream (left) and downstream (right) Rankine-Hugoniot states at Mach 3
    T_lo, rho0 = 1.0, 1.0
    v0 = 3.0 * c_s0
    T_hi = 3.661912665809719
    rho1 = 3.0021676971081166
    v1 = v0 * rho0 / rho1

    chat = 10 * (v0 + c_s0)
    Lx = 0.01578396467532876                      # 9.11 mean free paths
    shock_position = 0.0130

    eos = IdealGas(; gamma, cv)
    rad = RadiationPhysics(; c, chat, a_rad, beta_order=1,
                           closure=EddingtonClosure(),
                           opacity=UserOpacity((rho, T) -> kappa_abs / rho))

    egas(rho, T) = rho * cv * T
    lo_gas = Fixed((rho0, rho0 * v0, 0.0, 0.0, egas(rho0, T_lo) + 0.5 * rho0 * v0^2))
    hi_gas = Fixed((rho1, rho1 * v1, 0.0, 0.0, egas(rho1, T_hi) + 0.5 * rho1 * v1^2))
    lo_rad = Fixed((a_rad * T_lo^4, 0.0, 0.0, 0.0))
    hi_rad = Fixed((a_rad * T_hi^4, 0.0, 0.0, 0.0))
    outflow = (Outflow(), Outflow())

    sim = Simulation(; n=(nx, 1, 1), xhi=(Lx, 1.0, 1.0), eos=eos, radiation=rad,
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
    return sim, 3 * Lx / v0                       # three shock-crossing times
end

"""
    radiation_shadow(; n = (280, 112), kappa = 1.0, rho_blob = 1e3, kwargs...)
        -> (sim, tmax)

The shadow test of Hayes & Norman (2003) in 2D: a collimated beam enters from
the left of a transparent box and strikes an opaque, absorbing elliptical clump
of semi-axes `(0.10, 0.06)` centred at `x = 0.25`.

A two-moment scheme with the M1 closure preserves the direction of the
radiation and leaves a sharp shadow behind the clump; a diffusion-based scheme
lets radiation leak around the obstacle and fills the shadow in. Far downstream
M1 does fill the shadow in slowly, because it carries a single flux vector per
cell and so cannot represent the two crossing beams that meet there.

The clump is given a very large heat capacity so that it absorbs without
re-emitting, which isolates the transport.
"""
function radiation_shadow(; n=(280, 112), kappa=1.0, rho_blob=1e3, rho_bg=1e-2, kwargs...)
    c = 1.0
    E0 = 1.0
    rad = RadiationPhysics(; c=c, chat=c, a_rad=1.0, beta_order=0,
                           opacity=ConstantOpacity(kappa))
    beam = Fixed((E0, 0.999 * c * E0, 0.0, 0.0))
    sim = Simulation(; n=(n[1], n[2], 1), xlo=(0.0, -0.2, 0.0), xhi=(1.0, 0.2, 1.0),
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
