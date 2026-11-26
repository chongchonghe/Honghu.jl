#!/usr/bin/env julia
#
# Test 5 of Iliev et al. (2009): the expansion of an HII region, with hydrodynamics on, in the
# centred box (the whole sphere in [-15, 15]^3 kpc), n^3 cells, chat = 3e-4 c. How round are
# the ionization front and the shock through the R-type and the D-type phase, for a few source
# kernels?
#
#   julia --project=. -t auto docs/photoionization-experiments/dtype.jl [kernel ...] [--n=64] [--solver=be|ros] [--outdir=DIR]
#
# The kernels are those of sphericity.jl; the default set is gauss, wendland2 and gauss3. At
# each output time it writes, per kernel, to dtype_<kernel>_<solver>.txt:
#   - r_I, the volume-equivalent radius of the ionized gas (`ionized_radius`);
#   - the front: the radius where x_H+ falls through 0.5, along 4000 directions (mean, spread
#     std / mean, min, max);
#   - the shock: the radius of the densest point along each of the 4000 directions (mean,
#     spread, min, max), and the largest density and speed in the box;
# and the shell-averaged radial profiles of x_H+, n / n_0, T and v_r to
# dtype_<kernel>_<solver>_profiles.txt, and x_H+, n / n_0 and T as raw Float32 (n^3, x
# fastest) to x_/rho_/T_<kernel>_<solver>_<t>Myr.f32, for dtype_plots.jl.

include(joinpath(@__DIR__, "sphericity.jl"))      # the kernels and the front measurement

const DT_TIMES_MYR = (10, 30, 100, 200, 300, 400, 500)
const DT_OUT = opt("outdir", joinpath(@__DIR__, "dtype"))
mkpath(DT_OUT)

# Density relative to the initial one, temperature and radial velocity of every interior cell.
function hydro_fields(sim, rho0)
    g = sim.grid
    rho, _ = field(sim, :density)
    T, _ = field(sim, :temperature)
    vx, _ = field(sim, :vx)
    vy, _ = field(sim, :vy)
    vz, _ = field(sim, :vz)
    vr = similar(vx)
    for k in 1:g.n[3], j in 1:g.n[2], i in 1:g.n[1]
        x, y, z = (g.xlo[d] + ((i, j, k)[d] - 0.5) * g.dx[d] for d in 1:3)
        r = sqrt(x^2 + y^2 + z^2)
        vr[i, j, k] = (vx[i, j, k] * x + vy[i, j, k] * y + vz[i, j, k] * z) / r
    end
    return rho ./ rho0, T, vr
end

# The radius of the largest value of f along the unit vector d, sampled every dx / 20 out to
# the edge of the box.
function peak_radius(f, d, xlo, dx)
    h = dx / 20
    best, rbest = -Inf, 0.0
    r = h
    while r < -xlo * 0.98
        v = trilinear(f, (r * d[1], r * d[2], r * d[3]), xlo, dx)
        v > best && ((best, rbest) = (v, r))
        r += h
    end
    return rbest
end

# NaN where no direction found the surface, as when the ionized region is smaller than a cell.
function stats(v)
    g = filter(!isnan, v)
    length(g) < 2 && return (NaN, NaN, NaN, NaN, length(v) - length(g))
    m = mean(g)
    return (m, std(g) / m, minimum(g), maximum(g), length(v) - length(g))
end

# Shell averages in bins of width dx out to the edge of the box.
function profiles(x, rho, T, vr, xlo, dx)
    n = size(x, 1)
    nb = n ÷ 2
    acc = zeros(nb, 5)
    for k in 1:n, j in 1:n, i in 1:n
        r = sqrt(sum((xlo + (c - 0.5) * dx)^2 for c in (i, j, k)))
        b = floor(Int, r / dx) + 1
        b > nb && continue
        acc[b, 1] += 1
        acc[b, 2] += x[i, j, k]
        acc[b, 3] += rho[i, j, k]
        acc[b, 4] += T[i, j, k]
        acc[b, 5] += vr[i, j, k]
    end
    return [((b - 0.5) * dx, acc[b, 2] / acc[b, 1], acc[b, 3] / acc[b, 1], acc[b, 4] / acc[b, 1],
             acc[b, 5] / acc[b, 1]) for b in 1:nb if acc[b, 1] > 0]
end

function run_dtype(name)
    solver = SOLVER == "be" ? BackwardEulerChem() : RosenbrockChem()
    sim, _ = P.stromgren_iliev(; test=5, n=N, centered=true, chat_factor=CHAT_FACTOR,
                               solver, kernel=KERNELS[name])
    g = sim.grid
    xlo, dx = g.xlo[1], g.dx[1]
    rho0 = first(field(sim, :density))[1, 1, 1]
    file = joinpath(DT_OUT, "dtype_$(name)_$(SOLVER).txt")
    pfile = joinpath(DT_OUT, "dtype_$(name)_$(SOLVER)_profiles.txt")
    io, pio = open(file, "w"), open(pfile, "w")
    println(io, "# Iliev et al. (2009) Test 5, centred, n = $N, chat = $CHAT_FACTOR c, kernel $name, solver $SOLVER, dx = $(dx / KPC) kpc")
    println(io, "# radii in kpc; front: x_H+ = 0.5; shock: densest point along the ray; spread = std / mean")
    println(io, "# t_Myr  r_I  front_mean  front_spread  front_min  front_max  shock_mean  shock_spread  shock_min  shock_max  rho_max/rho0  vmax_kms  wall_s")
    println(pio, "# t_Myr  r_kpc  x_H+  n/n0  T_K  v_r_kms   (shell averages, bins of width dx)")
    t0 = time()
    for tm in DT_TIMES_MYR
        run!(sim; tmax=tm * MYR, verbose=false)
        x = ionization(sim)
        rho, T, vr = hydro_fields(sim, rho0)
        fr = stats([front_radius(x, d, xlo, dx) for d in DIRS])
        sh = stats([peak_radius(rho, d, xlo, dx) for d in DIRS])
        v, _ = field(sim, :speed)
        wall = time() - t0
        rI = P.ionized_radius(sim; centered=true)
        @printf(io, "%4d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.3f %.0f\n", tm, rI / KPC,
                fr[1] / KPC, fr[2], fr[3] / KPC, fr[4] / KPC, sh[1] / KPC, sh[2], sh[3] / KPC, sh[4] / KPC,
                maximum(rho), maximum(v) / 1e5, wall)
        @printf("%-10s %4d Myr  r_I %.3f  front %.3f (spread %.2f%%)  shock %.3f (spread %.2f%%)  n_max/n0 %.2f  v_max %.1f km/s  (%.0f s)\n",
                name, tm, rI / KPC, fr[1] / KPC, 100fr[2], sh[1] / KPC, 100sh[2], maximum(rho), maximum(v) / 1e5, wall)
        fr[5] > 0 && println("  warning: $(fr[5]) directions found no front")
        for row in profiles(x, rho, T, vr, xlo, dx)
            @printf(pio, "%4d %.4f %.5f %.4f %.1f %.3f\n", tm, row[1] / KPC, row[2], row[3], row[4], row[5] / 1e5)
        end
        for (tag, f) in (("x", x), ("rho", rho), ("T", T))
            write(joinpath(DT_OUT, "$(tag)_$(name)_$(SOLVER)_$(tm)Myr.f32"), Float32.(f))
        end
        flush(io); flush(pio)
    end
    d = sim.diag
    @printf(io, "# chemistry: %d solves, %.2f steps per cell, %d failed; %d steps; floors: %d gas, %d radiation\n",
            d.chem_cells, d.chem_steps / d.chem_cells, d.chem_failures, sim.step, d.floors_hydro, d.floors_rad)
    close(io); close(pio)
    println("  wrote $file")
end

if abspath(PROGRAM_FILE) == @__FILE__
    names = filter(a -> !startswith(a, "--"), ARGS)
    for name in (isempty(names) ? ("gauss", "wendland2", "gauss3") : names)
        run_dtype(name)
    end
end
