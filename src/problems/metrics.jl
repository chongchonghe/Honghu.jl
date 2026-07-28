# Default metric sets for the problems that have a reference solution.
#
# One method per problem, each returning a NamedTuple of `metric(sim) -> Real`.
# The test suite, the verification figures and `compare` all read the same
# definitions from here, so "the error" means one thing in this package.
#
# Problems with no reference solution -- Kelvin-Helmholtz, the Marshak wave, the
# shadow -- deliberately have no method. `compare` then asks for explicit
# metrics rather than inventing a number that looks authoritative.

using ..Hydro3D: relative_l1, line, coordinates, gas_temperature, radiation_temperature

"""
    default_metrics(sod_shock_tube)

`l1_density` against the exact Riemann solution at the final time.
"""
Hydro3D.default_metrics(::typeof(sod_shock_tube)) = (;
    l1_density = function (sim)
        x = coordinates(sim.grid, 1)
        gam = sim.eos.gamma
        ref = [sod_exact(xi, sim.t; gamma=gam)[1] for xi in x]
        relative_l1(line(sim, :density), ref)
    end,
)

"""
    default_metrics(smooth_advection)

`l1_density` against the initial profile, which the exact solution returns to
after a whole number of periods. This is the problem the order-of-accuracy
measurements use, so the metric is the one whose slope is the convergence rate.
"""
Hydro3D.default_metrics(::typeof(smooth_advection)) = (;
    l1_density = function (sim)
        x = coordinates(sim.grid, 1)
        relative_l1(line(sim, :density), 1 .+ 0.5 .* sinpi.(2 .* x))
    end,
)

"""
    default_metrics(radiation_diffusion)

`l1_Erad` against the analytic Gaussian `sigma² = sigma0² + 2Dt`.
"""
Hydro3D.default_metrics(::typeof(radiation_diffusion)) = (;
    l1_Erad = function (sim)
        x = coordinates(sim.grid, 1)
        # D = c/(3 rho kappa_F); rho = 1 and kappa_F is the flux mean
        kap = Hydro3D.flux_mean_opacity(sim.radiation.opacity, 1.0, 1.0)
        D = sim.radiation.c / (3 * kap)
        relative_l1(line(sim, :Erad), [diffusion_exact(xi, sim.t; D=D) for xi in x])
    end,
)

"""
    default_metrics(radiation_streaming)

`front_error`, the relative error in the position of the radiation front, which
must travel at exactly `chat`.
"""
Hydro3D.default_metrics(::typeof(radiation_streaming)) = (;
    front_error = function (sim)
        x = coordinates(sim.grid, 1)
        er = line(sim, :Erad)
        i = findlast(>(0.5), er)
        i === nothing && return 1.0
        abs(x[i] - sim.radiation.chat * sim.t) / (sim.radiation.chat * sim.t)
    end,
)

"""
    default_metrics(radhydro_shock)

Quokka's own pair for this test:

  - `l1_Trad`, the relative `L1` error in the radiation temperature, with the
    numerical profile interpolated onto the reference abscissae. Quokka's pass
    threshold is 1% at 256 cells.
  - `spike`, the peak matter temperature. The Zel'dovich spike is the feature
    the test exists for and an error norm alone would pass a solution with no
    spike at all; the reference value is 4.2618.
"""
Hydro3D.default_metrics(::typeof(radhydro_shock)) = (;
    l1_Trad = function (sim)
        xe, _, _, _, Tre = radhydro_shock_exact()
        Tr = radiation_temperature(sim)[:, 1, 1]
        relative_l1(interpolate_to(xe, coordinates(sim.grid, 1), Tr), Tre)
    end,
    spike = sim -> maximum(gas_temperature(sim)[:, 1, 1]),
)

"""
    default_metrics(sedov_blast)

`radius_error`, the relative error in the shock radius against the
Sedov-Taylor similarity radius `xi (E t²/rho)^(1/5)` with `xi = 1.15` for
`gamma = 5/3`.
"""
Hydro3D.default_metrics(::typeof(sedov_blast)) = (;
    radius_error = function (sim)
        g = sim.grid
        rho, _ = field(sim, :density)
        ic = cld(g.n[1], 2)
        i = argmax(@view rho[ic:end, ic, ic])
        abs((i - 0.5) * g.dx[1] - 1.15 * (sim.t^2)^(1 / 5)) / (1.15 * (sim.t^2)^(1 / 5))
    end,
)
