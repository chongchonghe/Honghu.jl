# Standard hydrodynamic verification problems.

"""
    sod_shock_tube(; nx = 256, gamma = 1.4, kwargs...) -> (sim, tmax)

The Sod (1978) shock tube in 1D: `(rho, p) = (1, 1)` for `x < 0.5` and
`(0.125, 0.1)` for `x > 0.5`, at rest, with outflow boundaries. Compare the
result at `tmax = 0.2` against [`sod_exact`](@ref).
"""
function sod_shock_tube(; nx::Integer=256, gamma=1.4, kwargs...)
    sim = Simulation(; n=(nx, 1, 1), xlo=(0.0, 0.0, 0.0), xhi=(1.0, 1.0, 1.0),
                     eos=IdealGas(; gamma), boundaries=Boundaries(Outflow()), kwargs...)
    set_initial_conditions!(sim) do x, y, z
        x < 0.5 ? (rho=1.0, p=1.0) : (rho=0.125, p=0.1)
    end
    return sim, 0.2
end

"""
    smooth_advection(; nx = 64, amplitude = 0.5, velocity = 1.0, gamma = 5/3, kwargs...)
        -> (sim, tmax)

A smooth density perturbation `rho = 1 + A sin(2πx)` advected at constant
velocity and pressure through a periodic box. After `tmax = 1/velocity` the
solution equals the initial condition exactly, which makes this the convergence
test of the code: the L1 error should fall like `dx^p` with `p` set by the
reconstruction.
"""
function smooth_advection(; nx::Integer=64, amplitude=0.5, velocity=1.0, gamma=5 / 3, kwargs...)
    sim = Simulation(; n=(nx, 1, 1), xlo=(0.0, 0.0, 0.0), xhi=(1.0, 1.0, 1.0),
                     eos=IdealGas(; gamma), boundaries=Boundaries(Periodic()), kwargs...)
    set_initial_conditions!(sim) do x, y, z
        (rho=1 + amplitude * sinpi(2x), vx=velocity, p=1.0)
    end
    return sim, 1 / velocity
end

"""
    kelvin_helmholtz(; n = 256, gamma = 1.4, kwargs...) -> (sim, tmax)

The Kelvin-Helmholtz instability in a doubly periodic 2D box, in the smoothly
seeded form of Springel (2010): a dense slab `rho = 2` moving at `vx = +1/2`
between two `rho = 1` layers moving at `vx = -1/2`, perturbed by a single
transverse mode localised on the two shear layers.
"""
function kelvin_helmholtz(; n::Integer=256, gamma=1.4, kwargs...)
    sim = Simulation(; n=(n, n, 1), xlo=(0.0, 0.0, 0.0), xhi=(1.0, 1.0, 1.0),
                     eos=IdealGas(; gamma), boundaries=Boundaries(Periodic()), kwargs...)
    sigma = 0.05
    set_initial_conditions!(sim) do x, y, z
        inner = abs(y - 0.5) < 0.25
        rho = inner ? 2.0 : 1.0
        vx = inner ? 0.5 : -0.5
        vy = 0.1 * sinpi(4x) * (exp(-(y - 0.25)^2 / (2 * sigma^2)) +
                                exp(-(y - 0.75)^2 / (2 * sigma^2)))
        (rho=rho, vx=vx, vy=vy, p=2.5)
    end
    return sim, 2.0
end

"""
    sedov_blast(; n = 64, energy = 1.0, gamma = 5/3, kwargs...) -> (sim, tmax)

A Sedov-Taylor point explosion in 3D: `energy` deposited as thermal energy in a
small sphere at the centre of a unit-density, cold, unit box with outflow
boundaries. The blast radius should follow `R(t) = ξ (E t² / rho)^(1/5)` with
`ξ ≈ 1.15` for `gamma = 5/3`.
"""
function sedov_blast(; n::Integer=64, energy=1.0, gamma=5 / 3, kwargs...)
    sim = Simulation(; n=(n, n, n), xlo=(-0.5, -0.5, -0.5), xhi=(0.5, 0.5, 0.5),
                     eos=IdealGas(; gamma), boundaries=Boundaries(Outflow()), kwargs...)
    dx = 1 / n
    r0 = 3.5 * dx                       # spread over a few cells to limit start-up error
    pin = (gamma - 1) * energy / (4 / 3 * pi * r0^3)
    set_initial_conditions!(sim) do x, y, z
        r2 = x * x + y * y + z * z
        (rho=1.0, p=r2 < r0^2 ? pin : 1e-5)
    end
    return sim, 0.05
end
