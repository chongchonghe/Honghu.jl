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
    radiation_shadow(; n = (280, 80), kappa = 1.0, rho_blob = 1e3, kwargs...)
        -> (sim, tmax)

The shadow test of Hayes & Norman (2003) in 2D: a collimated beam enters from
the left of a transparent box and strikes an opaque elliptical clump of
semi-axes `(0.10, 0.06)` centred at `x = 0.3`.

A two-moment scheme with the M1 closure preserves the direction of the
radiation and leaves a shadow behind the clump; a diffusion-based scheme lets
radiation leak around the obstacle and fills the shadow in. (M1 does still fill
the far shadow in slowly, because it carries only one flux vector per cell and
so cannot represent the two crossing beams that meet there.)

The opacity is purely scattering-like (`kappa_P = kappa_E = 0`), so the result
depends only on the transport.
"""
function radiation_shadow(; n=(280, 80), kappa=1.0, rho_blob=1e3, rho_bg=1e-2, kwargs...)
    c = 1.0
    E0 = 1.0
    rad = RadiationPhysics(; c=c, chat=c, a_rad=1.0, beta_order=0,
                           opacity=ConstantOpacity(planck=0.0, energy=0.0, flux=kappa))
    beam = Fixed((E0, 0.999 * c * E0, 0.0, 0.0))
    sim = Simulation(; n=(n[1], n[2], 1), xlo=(0.0, -0.12, 0.0), xhi=(1.0, 0.12, 1.0),
                     radiation=rad, hydro=false,
                     boundaries=Boundaries(Outflow()),
                     radiation_boundaries=Boundaries((beam, Outflow()),
                                                     (Outflow(), Outflow()),
                                                     (Outflow(), Outflow())),
                     kwargs...)
    ax, ay = 0.10, 0.06
    set_initial_conditions!(sim) do x, y, z
        inside = ((x - 0.3) / ax)^2 + (y / ay)^2 < 1
        (rho=inside ? rho_blob : rho_bg, p=1.0, Erad=1e-10, Fx=0.0)
    end
    return sim, 1.5
end
