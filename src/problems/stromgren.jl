# The Stromgren-sphere tests of Iliev et al. (2006), on the hydrogen networks JAFF generates
# from `networks/hydrogen_iliev1` and `networks/hydrogen_iliev2`.
#
# Everything here is in CGS, which the chemistry requires. Hydrodynamics is off: the tests
# are about the ionization front, and the gas does not move.

# Iliev et al. (2006) section 3: the box, the density and the source.
const _ILIEV_L = 6.6e3 * Honghu.CGS.parsec        # 6.6 kpc, in cm
const _ILIEV_NH = 1e-3                              # hydrogen number density, cm^-3
const _ILIEV_NDOT = 5e48                            # ionizing photons per second
const _ILIEV_ALPHA_B = 2.59e-13                     # Test 1 case B recombination coefficient, cm^3 s^-1

# The Gaussian source: width `1.5 dx` and cut at `2.5` widths, as Quokka's
# `testStromgrenSphere.cpp` (there the width is 1.5 / 16 of the box). Its rate is
# normalised over the *grid*, so the photons per second that the cells receive are exactly
# `Ndot`, whatever the resolution and wherever the truncation falls among the cells. A
# point source in one cell would print the grid axes onto the front instead.
function _gaussian_source(xlo, dx, n, Ndot)
    s = 1.5 * dx
    rmax2 = (2.5 * s)^2
    w(x, y, z) = (r2 = x^2 + y^2 + z^2; r2 < rmax2 ? exp(-r2 / (2 * s^2)) : 0.0)
    total = 0.0
    for k in 1:n, j in 1:n, i in 1:n
        total += w(xlo + (i - 0.5) * dx, xlo + (j - 0.5) * dx, xlo + (k - 0.5) * dx)
    end
    A = Ndot / (total * dx^3)                        # photons cm^-3 s^-1 at the source centre
    # Called from the threads of the chemistry step: it reads only captured constants.
    return (x, y, z, t) -> A * w(x, y, z)
end

"""
    stromgren_iliev(; test = 1, n = 32, centered = false, chat_factor = 1e-3,
                    solver = RosenbrockChem(), kwargs...) -> (sim, tmax)

Test 1 or 2 of the Stromgren-sphere comparison of Iliev et al. (2006): an ionizing point
source switched on in a uniform, static, hydrogen-only medium, run with the network
[`HydrogenIliev1`](@ref) or [`HydrogenIliev2`](@ref) and the photon band of
[`Photochemistry`](@ref). `n` is the number of
cells along each side of the box; `chat_factor` is `chat / c`; `solver` is the chemistry
integrator; the other keywords go to [`Simulation`](@ref). `tmax` is 500 Myr, the end of the
paper's runs.

From the paper: the box side `L = 6.6` kpc, the hydrogen density `n_H = 1e-3` cm^-3, the
photon rate `Ndot = 5e48` s^-1, the source in the corner of the box with reflecting walls on
the three low faces (the octant `[0, L]^3` of a sphere about the source), and the two tests
below. Test 1 is monochromatic at 13.6 eV, with the gas held at `1e4` K, the recombination
coefficient `alpha_B = 2.59e-13` cm^3 s^-1 and an initial ionization fraction `1.2e-3`.
Test 2 starts neutral and cold (`100` K), and lets the gas heat and cool. The paper's source
in Test 2 is a `1e5` K blackbody.

Choices that the paper does not make, and that are ours:

  - **One band for Test 2.** The blackbody is replaced by a single band with
    `sigma = 1.63e-18` cm^2 and `photon_energy = 29.65 eV`. These are the project's values
    for the `1e5` K blackbody, the number-weighted mean energy of the ionizing photons and
    their photon-number-averaged cross-section, and not numbers the paper prescribes (a
    cross-section `~ nu^-3` gives 29.6 eV and 1.5e-18 cm^2). Each ionization heats the gas
    by `29.65 - 13.6` eV, Quokka's default. That overheats the gas compared with the real
    spectrum, whose energy-weighted value is about 6.3 eV, so the Test 2 temperature behind
    the front is too high; and the paper prescribes no rates (each code used its own), so
    the result is checked for its shape and not against data. The network has the case B
    `alpha_B(T)`, collisional ionization, and recombination, free-free and Lyman alpha
    cooling (`networks/hydrogen_iliev2`).
  - **Test 1 in the network.** `isothermal = 1e4`, no heating, a constant `alpha_B`, and no
    collisional ionization, which keeps the background ionization at `1.2e-3` instead of
    letting `1e4` K gas ionize itself (`networks/hydrogen_iliev1`).
  - **The photon rate of the octant.** With `centered = false` (the default) the box is the
    paper's octant `[0, L]^3`, with [`Reflecting`](@ref) walls on the low faces and
    [`Outflow`](@ref) on the high ones. Emitting `Ndot / 8` into it, so that the full sphere
    would receive `Ndot`, is our choice (it is also Quokka's). `centered = true` is the whole
    sphere in `[-L, L]^3`, `Outflow` everywhere, and `Ndot` emitted, at eight times the cells
    for the same cell size.
  - **The source.** A Gaussian at the origin of width `1.5 dx`, cut at `2.5` widths, whose
    rate is normalised so that the cells receive exactly `Ndot / 8` (or `Ndot`) photons per
    second. The width and the cut are Quokka's (`testStromgrenSphere.cpp`), not the paper's.
  - **The radiation.** One group, zero opacity, `beta_order = 0` and a constant reduced
    speed of light, as a chemistry run requires. The initial radiation energy is zero: the
    default `a_r T^4` would put about `1e12` ionizing photons per cm^3 in the box.

The front is measured by [`ionized_radius`](@ref) and compared with
[`stromgren_radius_analytic`](@ref) (`chat = c`) or [`stromgren_radius_finite_c`](@ref) (a
reduced speed of light, an approximate reference). At `n = 16` and `chat_factor = 3e-4` the
Test 1 radius is within 3% of the finite-`chat` reference from 0.5 to 2 recombination times,
and the difference grows as `chat_factor` falls (`test/test_stromgren.jl` has the numbers and
the causes not yet examined).

```julia
sim, tmax = stromgren_iliev(test = 1, n = 32, chat_factor = 3e-4)
run!(sim; tmax = 0.5 / (2.59e-13 * 1e-3))
r = ionized_radius(sim; x0 = 1.2e-3)
```
"""
function stromgren_iliev(; test::Integer=1, n::Integer=32, centered::Bool=false,
                         chat_factor=1e-3, solver=RosenbrockChem(), kwargs...)
    test in (1, 2) || throw(ArgumentError("test must be 1 or 2, got $test"))
    CGS = Honghu.CGS
    eV = CGS.eV
    net = test == 1 ? HydrogenIliev1(photon_energy=13.6 * eV, isothermal=1e4) :
          HydrogenIliev2(photon_energy=29.65 * eV)
    T0, x0 = test == 1 ? (1e4, 1.2e-3) : (100.0, 0.0)

    L = _ILIEV_L
    xlo, side = centered ? (-L, 2L) : (0.0, L)
    dx = side / n
    emission = _gaussian_source(xlo, dx, n, centered ? _ILIEV_NDOT : _ILIEV_NDOT / 8)

    rad = RadiationPhysics(c=CGS.c_light, chat=chat_factor * CGS.c_light, a_rad=CGS.a_rad,
                           k_B=CGS.k_B, opacity=ConstantOpacity(0.0), beta_order=0)
    bc = centered ? Boundaries(Outflow()) :
         Boundaries((Reflecting(), Outflow()), (Reflecting(), Outflow()),
                    (Reflecting(), Outflow()))
    sim = Simulation(; n=(n, n, n), xlo=(xlo, xlo, xlo), xhi=(xlo + side, xlo + side, xlo + side),
                     hydro=false, radiation=rad, eos=IdealGas(gamma=5 / 3, cv=cv_from_mu(1.0, 5 / 3)),
                     boundaries=bc, radiation_boundaries=bc,
                     chemistry=Photochemistry(net; solver, emission), kwargs...)
    # The species are H, H+ and e-, and the mass of an H+ ion and an electron is that of an
    # H atom, so a hydrogen density `n_H` with ionization fraction `x` has these fractions.
    m = Honghu.species_masses(net)
    rho = _ILIEV_NH * m[1]
    X = (1 - x0, x0 * m[2] / m[1], x0 * m[3] / m[1])
    set_initial_conditions!(sim) do x, y, z
        (rho=rho, T=T0, X=X, Erad=0.0)
    end
    return sim, 500e6 * CGS.year
end

"""
    ionized_radius(sim; centered = false, x0 = 0.0)

The radius of the ionized region of a [`stromgren_iliev`](@ref) run, `r = (3 V / 4 pi)^(1/3)`,
with `V` the volume of ionized gas, `sum max(x - x0, 0) / (1 - x0) dV` over the cells: the
ionization fraction `x = n_H+ / (n_H0 + n_H+)` counted above the background `x0`, which is the
initial `1.2e-3` of Test 1 and zero for Test 2. It is a volume-weighted radius, not the
position of a front, so a sharp front and a smeared one of the same total give the same
`r`. With `centered = false` the grid is the octant and `V` is multiplied by 8.
"""
function ionized_radius(sim; centered::Bool=false, x0=0.0)
    g = sim.grid
    U = sim.U
    i1, i2 = Honghu.scalar_index(1), Honghu.scalar_index(2)
    m1, m2 = Honghu.species_masses(sim.chemistry.network)
    V = 0.0
    for k in g.lo[3]:g.hi[3], j in g.lo[2]:g.hi[2], i in g.lo[1]:g.hi[1]
        n0, n1 = U[i1, i, j, k] / m1, U[i2, i, j, k] / m2
        x = n1 / (n0 + n1)
        V += max(x - x0, 0.0) / (1 - x0)
    end
    V *= prod(g.dx)
    centered || (V *= 8)
    return cbrt(3V / (4pi))
end

"""
    stromgren_radius_analytic(t; Ndot = 5e48, n = 1e-3, alpha_B = 2.59e-13)

The radius of the ionized sphere of a point source in a uniform neutral medium, at the speed
of light, with the recombination of Test 1 of Iliev et al. (2006): `r_I = r_S (1 - exp(-t /
t_rec))^(1/3)`, with the Stromgren radius `r_S = (3 Ndot / 4 pi alpha_B n^2)^(1/3)` (5.39 kpc
for the defaults) and the recombination time `t_rec = 1 / (alpha_B n)` (122.3 Myr). `t` is in
seconds and the radius in cm. This is the paper's equation for the front of Test 1, which
holds when the front is much slower than light.
"""
function stromgren_radius_analytic(t; Ndot=_ILIEV_NDOT, n=_ILIEV_NH, alpha_B=_ILIEV_ALPHA_B)
    rS = cbrt(3Ndot / (4pi * alpha_B * n^2))
    trec = 1 / (alpha_B * n)
    return rS * cbrt(-expm1(-t / trec))
end

"""
    stromgren_radius_finite_c(t; chat, Ndot = 5e48, n = 1e-3, alpha_B = 2.59e-13)

The radius of the ionized sphere at time `t` when the radiation travels at the speed `chat`,
from the front equation of Shapiro et al. (2006),

    dR/dt = (Ndot - 4 pi R^3 alpha_B n^2 / 3) / (Ndot / chat + 4 pi R^2 n),

which is approximate: its term `Ndot / chat` counts the photons in flight as `Ndot R / chat`
and ignores how many of them recombinations inside `R` have used up. It is integrated from `R = 0` with fourth-order Runge-Kutta. It starts with 256 steps and doubles
the number of steps until two successive radii differ by less than `1e-6` of the result, as
the `integrate_radius` of Quokka's `testStromgrenSphere.cpp` does, and throws an error if
that has not happened after 12 doublings. With `chat` the speed of light it approaches
[`stromgren_radius_analytic`](@ref), and it is the reference of Shapiro et al. (2006) for a
run with a reduced speed of light, which [`stromgren_iliev`](@ref) reduces in the photon
equation while it keeps the chemistry unchanged. `t` is in seconds and the radius in cm.
"""
function stromgren_radius_finite_c(t; chat, Ndot=_ILIEV_NDOT, n=_ILIEV_NH, alpha_B=_ILIEV_ALPHA_B)
    t <= 0 && return 0.0
    rhs(R) = (Ndot - 4pi * R^3 * alpha_B * n^2 / 3) / (Ndot / chat + 4pi * R^2 * n)
    N = 256
    Rprev = NaN
    for _ in 1:12
        h = t / N
        R = 0.0
        for _ in 1:N
            k1 = rhs(R)
            k2 = rhs(R + h * k1 / 2)
            k3 = rhs(R + h * k2 / 2)
            k4 = rhs(R + h * k3)
            R = max(R + h * (k1 + 2k2 + 2k3 + k4) / 6, 0.0)   # the radius cannot go negative
        end
        abs(R - Rprev) <= 1e-6 * R && return R
        Rprev = R
        N *= 2
    end
    error("stromgren_radius_finite_c: the Runge-Kutta integration did not converge for t = $t")
end
