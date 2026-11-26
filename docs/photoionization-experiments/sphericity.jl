#!/usr/bin/env julia
#
# How round is the ionized region of the Stromgren sphere, and does the kernel that deposits
# the source photons change it? Test 1 of Iliev et al. (2006), centred box, n^3 cells,
# chat = 3e-4 c, one run per kernel.
#
#   julia --project=. -t auto docs/photoionization-experiments/sphericity.jl [kernel ...] [--n=64] [--solver=be|ros] [--outdir=DIR]
#
# Kernels: gauss (Quokka's default: width 1.5 dx, cut at 2.5 widths), gauss_nocut (the same
# Gaussian cut at 5 widths), gauss2 and gauss3 (width 2 dx and 3 dx, cut at 4 widths), cells8 (the 8 cells that
# touch the source), tophat3 (a uniform sphere of radius 3 dx, weighted by the volume of each
# cell inside it). All of them by default.
#
# At each output time it finds the radius where the ionization fraction falls through 0.5
# along 4000 directions (trilinear interpolation of the cell values) and writes, per kernel,
# sphericity_<kernel>_<solver>.txt with the mean, the spread and the radii along the axes,
# the face diagonals and the body diagonals, and the ionization fraction as raw Float32
# (n^3, x fastest) at each output time, for the figures of sphericity_plots.jl.

using Honghu
using Printf
const P = Honghu.Problems

mean(v) = sum(v) / length(v)
std(v) = (m = mean(v); sqrt(sum((a - m)^2 for a in v) / (length(v) - 1)))

const MYR = 1e6 * Honghu.CGS.year
const KPC = 1e3 * Honghu.CGS.parsec
const TIMES_MYR = (30, 100, 200, 500)
const CHAT_FACTOR = 3e-4

opt(name, default) = (i = findfirst(a -> startswith(a, "--$name="), ARGS);
                      i === nothing ? default : split(ARGS[i], "=")[2])
const N = parse(Int, opt("n", "64"))
const SOLVER = opt("solver", "be")
const OUTDIR = opt("outdir", joinpath(@__DIR__, "sphericity"))
mkpath(OUTDIR)

# The volume of the cell centred at (x, y, z) inside the sphere |r| < R, by sampling the cell
# at m^3 points.
function _sphere_fraction(x, y, z, dx, R; m=8)
    c = 0
    for a in 1:m, b in 1:m, d in 1:m
        px = x + ((a - 0.5) / m - 0.5) * dx
        py = y + ((b - 0.5) / m - 0.5) * dx
        pz = z + ((d - 0.5) / m - 0.5) * dx
        c += px^2 + py^2 + pz^2 < R^2
    end
    return c / m^3
end

const KERNELS = Dict(
    "gauss" => P.gaussian_kernel(),
    "gauss_nocut" => P.gaussian_kernel(cut=5.0),
    "gauss2" => P.gaussian_kernel(width=2.0, cut=4.0),
    "gauss3" => P.gaussian_kernel(width=3.0, cut=4.0),
    "cells8" => (x, y, z, dx) -> max(abs(x), abs(y), abs(z)) < dx ? 1.0 : 0.0,
    "tophat3" => (x, y, z, dx) -> _sphere_fraction(x, y, z, dx, 3dx),
)
const ORDER = ("gauss", "gauss_nocut", "gauss2", "gauss3", "cells8", "tophat3")

# The ionization fraction of every interior cell.
function ionization(sim)
    g = sim.grid
    m1, m2 = Honghu.species_masses(sim.chemistry.network)
    i1, i2 = Honghu.scalar_index(1), Honghu.scalar_index(2)
    x = zeros(g.n...)
    for k in 1:g.n[3], j in 1:g.n[2], i in 1:g.n[1]
        I, J, K = g.lo[1] + i - 1, g.lo[2] + j - 1, g.lo[3] + k - 1
        n0, n1 = sim.U[i1, I, J, K] / m1, sim.U[i2, I, J, K] / m2
        x[i, j, k] = n1 / (n0 + n1)
    end
    return x
end

# Trilinear interpolation of the cell-centred field `f` at the point p, in a box [xlo, xlo +
# n dx]^3. Clamped to the outermost cell centres.
function trilinear(f, p, xlo, dx)
    n = size(f, 1)
    s = ntuple(d -> clamp((p[d] - xlo) / dx + 0.5, 1.0, n - 1e-9), 3)
    i = floor.(Int, s)
    t = s .- i
    v = 0.0
    for c in 0:1, b in 0:1, a in 0:1
        w = (a == 0 ? 1 - t[1] : t[1]) * (b == 0 ? 1 - t[2] : t[2]) * (c == 0 ? 1 - t[3] : t[3])
        v += w * f[i[1]+a, i[2]+b, i[3]+c]
    end
    return v
end

# The first radius along the unit vector d where f falls through `level`, marching out from
# the source in steps of dx / 20. NaN if it never does inside the box.
function front_radius(f, d, xlo, dx; level=0.5)
    h = dx / 20
    rmax = -xlo * 0.98
    rprev, fprev = 0.0, trilinear(f, (0.0, 0.0, 0.0), xlo, dx)
    r = h
    while r < rmax
        fr = trilinear(f, (r * d[1], r * d[2], r * d[3]), xlo, dx)
        if fprev >= level > fr
            return rprev + (fprev - level) / (fprev - fr) * h
        end
        rprev, fprev = r, fr
        r += h
    end
    return NaN
end

# Directions spread evenly over the sphere (a Fibonacci lattice).
function fibonacci_directions(m)
    ga = pi * (3 - sqrt(5))
    [(z = 1 - 2 * (q - 0.5) / m; s = sqrt(1 - z^2); (s * cos(ga * q), s * sin(ga * q), z))
     for q in 1:m]
end

const DIRS = fibonacci_directions(4000)
const AXES = [(s, 0.0, 0.0) for s in (-1, 1)] ∪ [(0.0, s, 0.0) for s in (-1, 1)] ∪
             [(0.0, 0.0, s) for s in (-1, 1)]
const FACE = vcat([(a, b, 0.0) ./ sqrt(2) for a in (-1, 1), b in (-1, 1)][:],
                  [(a, 0.0, b) ./ sqrt(2) for a in (-1, 1), b in (-1, 1)][:],
                  [(0.0, a, b) ./ sqrt(2) for a in (-1, 1), b in (-1, 1)][:])
const BODY = [(a, b, c) ./ sqrt(3) for a in (-1, 1), b in (-1, 1), c in (-1, 1)][:]

function measure(sim, xlo, dx)
    x = ionization(sim)
    R = [front_radius(x, d, xlo, dx) for d in DIRS]
    good = filter(!isnan, R)
    mean_dir(ds) = mean(front_radius(x, d, xlo, dx) for d in ds)
    Rm = mean(good)
    return (; x, Rmean=Rm, Rstd=std(good), Rmin=minimum(good), Rmax=maximum(good),
            Raxis=mean_dir(AXES), Rface=mean_dir(FACE), Rbody=mean_dir(BODY),
            nan=length(R) - length(good),
            rI=P.ionized_radius(sim; centered=true, x0=1.2e-3))
end

function run_kernel(name)
    solver = SOLVER == "be" ? BackwardEulerChem() : RosenbrockChem()
    sim, _ = P.stromgren_iliev(; test=1, n=N, centered=true, chat_factor=CHAT_FACTOR,
                               solver, kernel=KERNELS[name])
    xlo, dx = sim.grid.xlo[1], sim.grid.dx[1]
    file = joinpath(OUTDIR, "sphericity_$(name)_$(SOLVER).txt")
    open(file, "w") do io
        println(io, "# Stromgren Test 1, centred, n = $N, chat = $CHAT_FACTOR c, kernel $name, solver $SOLVER, dx = $(dx / KPC) kpc")
        println(io, "# radii in kpc, of the surface x_H+ = 0.5; spread = std / mean; aniso = (R_body - R_axis) / R_mean")
        println(io, "# t_Myr  r_I  R_mean  spread  R_min  R_max  R_axis  R_face  R_body  aniso  wall_s")
        t0 = time()
        for tm in TIMES_MYR
            run!(sim; tmax=tm * MYR, verbose=false)
            m = measure(sim, xlo, dx)
            wall = time() - t0
            row = (tm, m.rI / KPC, m.Rmean / KPC, m.Rstd / m.Rmean, m.Rmin / KPC, m.Rmax / KPC,
                   m.Raxis / KPC, m.Rface / KPC, m.Rbody / KPC, (m.Rbody - m.Raxis) / m.Rmean, wall)
            @printf(io, "%4d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %+.4f %.0f\n", row...)
            @printf("%-12s %4d Myr  r_I %.3f  R %.3f  spread %.2f%%  axis %.3f  face %.3f  body %.3f  aniso %+.2f%%  (%.0f s)\n",
                    name, tm, row[2], row[3], 100row[4], row[7], row[8], row[9], 100row[10], wall)
            m.nan > 0 && println("  warning: $(m.nan) directions found no front")
            write(joinpath(OUTDIR, "x_$(name)_$(SOLVER)_$(tm)Myr.f32"), Float32.(m.x))
        end
        d = sim.diag
        @printf(io, "# chemistry: %d solves, %.2f steps per cell, %d failed\n",
                d.chem_cells, d.chem_steps / d.chem_cells, d.chem_failures)
    end
    println("  wrote $file")
end

# Run only when called as a script, so that sphericity_plots.jl can include the functions.
if abspath(PROGRAM_FILE) == @__FILE__
    names = filter(a -> !startswith(a, "--"), ARGS)
    for name in (isempty(names) ? ORDER : names)
        run_kernel(name)
    end
end
