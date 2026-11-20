# Standard MHD verification problems.

# The 1D shock tube of a left and a right state along x, with uniform `bx` and `bz`
# and a jump in `by`, the field set on the faces (exactly divergence-free, since no
# component varies along its own direction). The states are those of Quokka's
# MHDBrioWuShockTube and MHDRyuJones2aShockTube.
function _mhd_shock_tube(nx, gamma, L, R, bx, bz; kwargs...)
    sim = Simulation(; n=(nx, 1, 1), xlo=(0.0, 0.0, 0.0), xhi=(1.0, 1.0, 1.0), mhd=true,
                     eos=IdealGas(; gamma), boundaries=Boundaries(Outflow()), kwargs...)
    set_initial_conditions!(sim) do x, y, z
        s = x < 0.5 ? L : R
        (rho=s.rho, p=s.p, vx=s.vx, vy=s.vy, vz=s.vz, bx=bx, by=s.by, bz=bz)
    end
    return sim
end

"""
    brio_wu(; n = 256, gamma = 2, kwargs...) -> (sim, tmax)

The MHD shock tube of Brio & Wu (1988) in 1D: `(rho, p, b_y) = (1, 1, 1)` for
`x < 0.5` and `(0.125, 0.1, -1)` for `x > 0.5`, at rest, with `b_x = 0.75` and
`gamma = 2`, to `tmax = 0.1`. The solution has a fast rarefaction, a slow compound
wave, a contact, a slow shock and a fast rarefaction. Quokka's
`MHDBrioWuShockTube`, with outflow boundaries (Quokka holds the initial states),
which make no difference until a fast wave reaches one.
"""
function brio_wu(; n::Integer=256, nx::Integer=n, gamma=2.0, kwargs...)
    L = (rho=1.0, p=1.0, vx=0.0, vy=0.0, vz=0.0, by=1.0)
    R = (rho=0.125, p=0.1, vx=0.0, vy=0.0, vz=0.0, by=-1.0)
    return _mhd_shock_tube(nx, gamma, L, R, 0.75, 0.0; kwargs...), 0.1
end

"""
    ryu_jones_2a(; n = 256, gamma = 5/3, kwargs...) -> (sim, tmax)

The shock tube of Ryu & Jones (1995), their Fig. 2a, in 1D, to `tmax = 0.2`: all
seven waves, with
`(rho, p, v, b_y) = (1.08, 0.95, (1.2, 0.01, 0.5), 3.6/sqrt(4 pi))` on the left,
`(1, 1, 0, 4/sqrt(4 pi))` on the right, and `b_x = b_z = 2/sqrt(4 pi)`. Quokka's
`MHDRyuJones2aShockTube`, with outflow boundaries.
"""
function ryu_jones_2a(; n::Integer=256, nx::Integer=n, gamma=5 / 3, kwargs...)
    s4pi = sqrt(4pi)
    L = (rho=1.08, p=0.95, vx=1.2, vy=0.01, vz=0.5, by=3.6 / s4pi)
    R = (rho=1.0, p=1.0, vx=0.0, vy=0.0, vz=0.0, by=4 / s4pi)
    return _mhd_shock_tube(nx, gamma, L, R, 2 / s4pi, 2 / s4pi; kwargs...), 0.2
end

"""
    orszag_tang(; n = 128, gamma = 5/3, kwargs...) -> (sim, tmax)

The Orszag-Tang vortex in a doubly periodic unit square: `rho = 25/(36 pi)`,
`p = 5/(12 pi)`, `v = (-sin 2 pi y, sin 2 pi x)`, `b = (-sin 2 pi y, sin 4 pi x)/sqrt(4 pi)`,
to `tmax = 0.5`, where the shocks have interacted into the usual pattern. Quokka's
`OrszagTang` is the same vortex shifted by half the box. Each field component
varies only across its own direction, so the faces set from it are exactly
divergence-free.
"""
function orszag_tang(; n::Integer=128, gamma=5 / 3, kwargs...)
    sim = Simulation(; n=(n, n, 1), mhd=true, eos=IdealGas(; gamma),
                     boundaries=Boundaries(Periodic()), kwargs...)
    b0 = 1 / sqrt(4pi)
    set_initial_conditions!(sim) do x, y, z
        (rho=25 / (36pi), p=5 / (12pi), vx=-sinpi(2y), vy=sinpi(2x),
         bx=-b0 * sinpi(2y), by=b0 * sinpi(4x))
    end
    return sim, 0.5
end

"""
    alfven_wave(; n = 32, amplitude = 0.1, kwargs...) -> (sim, tmax)

The circularly polarised Alfven wave of Toth (2000), an exact nonlinear solution,
travelling obliquely across a 2D periodic box `[0, 2] x [0, 1]` of `(2n, n)`
cells, to `tmax` = one period. The wave vector `k = 2 pi (1/2, 1)` puts one
wavelength `lambda = 2/sqrt(5)` across each side, at `atan 2 = 63.4` degrees to
`x`. Along `k`: `rho = 1`, `p = 0.1`, `b_par = 1`, `v_par = 0`, and across it
`b_perp = A (sin phi, cos phi)` and `v_perp = -b_perp`, with `phi = k . x` and
`A = amplitude`, so the wave moves at `v_A = 1` along `k` and the period is
`lambda`. The uniform `b_par` is set from the keys and the rotating part from a
vector potential, so `div b` starts at round-off. After one period the solution
equals the initial condition, which makes this the convergence test of the MHD
scheme (Gardiner & Stone 2005, Sec. 5.1).
"""
function alfven_wave(; n::Integer=32, amplitude=0.1, gamma=5 / 3, kwargs...)
    Lx, Ly = 2.0, 1.0
    kx, ky = 1 / Lx, 1 / Ly                    # wave numbers over 2 pi
    kk = sqrt(kx^2 + ky^2)
    ca, sa = kx / kk, ky / kk                  # direction of k
    lambda = 1 / kk
    A = amplitude
    sim = Simulation(; n=(2n, n, 1), xhi=(Lx, Ly, 1.0), mhd=true, eos=IdealGas(; gamma),
                     boundaries=Boundaries(Periodic()), kwargs...)
    phase(x, y) = 2pi * (kx * x + ky * y)
    # b_perp = A (sin phi) e2 + A (cos phi) z, with e2 = (-sin a, cos a, 0), is the curl
    # of A2 e2 + A3 z with A2 = A lambda/(2 pi) sin phi and A3 = A lambda/(2 pi) cos phi
    potential(x, y, z) = begin
        a = A * lambda / 2pi
        (-sa * a * sin(phase(x, y)), ca * a * sin(phase(x, y)), a * cos(phase(x, y)))
    end
    set_initial_conditions!(sim; vector_potential=potential) do x, y, z
        s, c = sin(phase(x, y)), cos(phase(x, y))
        (rho=1.0, p=0.1, vx=A * sa * s, vy=-A * ca * s, vz=-A * c, bx=ca, by=sa)
    end
    return sim, lambda
end
