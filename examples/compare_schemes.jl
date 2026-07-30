#!/usr/bin/env julia
#
# Comparing numerical schemes: HLLC against Lax-Friedrichs, and the three
# reconstructions, scored against exact solutions and refined to see whether
# the answer survives.
#
#   julia --project=. examples/compare_schemes.jl [--no-plot] [--outdir=DIR]
#
# Each sweep is one call. `compare` runs the problem once per (scheme,
# resolution), scores it with the problem's own metrics, prints a table and
# writes it to disk; `plotcompare` draws the figure from the table. The table
# is the durable artefact -- months later,
#
#   plotcompare("sod_shock_tube_riemann_compare.txt")
#
# redraws the figure, with the git hash that produced it stamped on, without
# re-running a single simulation.

using Hydro3D
using Printf
include(joinpath(@__DIR__, "common.jl"))
const P = Hydro3D.Problems

const SIZES = (64, 128, 256, 512)
const SOLVERS = (HLLC(), LLF())

# --- 1. the solver, on a problem that is nothing but discontinuities ---------
#
# The Sod tube has a shock, a contact and a rarefaction. This is where an
# approximate Riemann solver earns its keep: LLF applies the largest wave speed
# to every characteristic, so it smears the contact, which no amount of
# refinement recovers from at the same rate.

println("\n=== Riemann solver, Sod shock tube ===")
sod = compare(P.sod_shock_tube, :riemann, SOLVERS; sizes=SIZES, outdir=OUTDIR)
display(sod)

# --- 2. the same solvers, on smooth flow ------------------------------------
#
# Advecting a sine wave through a periodic box has no discontinuity for the
# solver to resolve, so the reconstruction sets the accuracy and the two
# solvers should very nearly agree. Running both problems is what turns "HLLC
# is better" into "HLLC is better *at discontinuities*".

println("\n=== Riemann solver, smooth advection ===")
smooth = compare(P.smooth_advection, :riemann, SOLVERS; sizes=SIZES, outdir=OUTDIR)
display(smooth)

# --- 3. a different knob, same one-line call --------------------------------
#
# The textbook ladder: piecewise constant, linear and parabolic reconstruction
# on the smooth problem, whose exact solution is the initial condition.

println("\n=== Reconstruction, smooth advection ===")
recon = compare(P.smooth_advection, :reconstruction, (PCM(), PLM(), PPM());
                sizes=SIZES, outdir=OUTDIR)
display(recon)

# --- 4. a difference that is not a keyword at all ---------------------------
#
# The three sweeps above all vary something `Simulation` takes as a keyword. A
# variant is really a *builder* -- a function of resolution returning
# `(sim, tmax)` -- so it can differ in anything, including the initial condition,
# which lives inside the factory's closure and no keyword can reach.
#
# The question here is whether starting a shock tube from a grid-aligned jump
# costs anything: does smoothing the discontinuity over a couple of cells change
# where the scheme converges? Both variants are scored against the same exact
# Riemann solution, so the comparison is fair.

function smoothed_sod(nx; width=2.0)
    sim = Simulation(; n=(nx, 1, 1), eos=IdealGas(gamma=1.4),
                     boundaries=Boundaries(Outflow()))
    dx = 1 / nx
    set_initial_conditions!(sim) do x, y, z
        s = 0.5 * (1 + tanh((x - 0.5) / (width * dx)))       # 0 on the left, 1 on the right
        (rho=1.0 + s * (0.125 - 1.0), p=1.0 + s * (0.1 - 1.0))
    end
    return sim, 0.2
end

println("\n=== Initial condition, Sod shock tube ===")
startup = compare(["sharp" => nx -> P.sod_shock_tube(; n=nx),
                   "smoothed" => nx -> smoothed_sod(nx)];
                  sizes=SIZES, metrics=default_metrics(P.sod_shock_tube),
                  name="sod_startup", outdir=OUTDIR)
display(startup)

# --- what the tables say -----------------------------------------------------

ratio(t) = t[:l1_density, "HLLC"] ./ t[:l1_density, "LLF"]
trend(t) = join((@sprintf("%.0f%%", 100r) for r in ratio(t)), " -> ")

println("""

Summary
  HLLC's L1 density error as a percentage of LLF's, from $(SIZES[1]) to $(SIZES[end]) cells:

    shock tube        $(trend(sod))
    smooth advection  $(trend(smooth))

  The gap holds on the shock tube and closes on smooth flow. That is the
  useful statement: LLF's extra dissipation is spent on the contact
  discontinuity, which is a fixed cost the grid cannot refine away, whereas in
  smooth flow the reconstruction error dominates and the choice of solver
  stops mattering. Comparing on one problem alone would have given a single
  ratio and no way to tell those two cases apart.

  Convergence rates: about 1 for both solvers on the shock tube, since a
  discontinuity caps the order whatever the scheme, and 1 / 2 / 2.3 for
  PCM / PLM / PPM on the smooth problem (see README.md for why PPM stops
  short of 3).

  Smoothing the shock tube's initial jump over two cells costs a factor of
  $(round(startup[:l1_density, "smoothed"][end] / startup[:l1_density, "sharp"][end],
          digits = 1)) in error at $(SIZES[end]) cells without changing the rate, so it is a
  constant start-up penalty rather than a different answer. That variant is
  not expressible as a keyword -- the initial condition lives inside the
  problem's closure -- which is what the builder form is for.""")

if PLOTTING
    # Two tables in one figure: the same two solvers, on both problems, so the
    # persistent gap and the closing one sit next to each other.
    saveplot(plotcompare(sod, smooth), "schemes_riemann.png")
    saveplot(plotcompare(recon), "schemes_reconstruction.png")
    saveplot(plotcompare(startup), "schemes_startup.png")
end
plotting_note()
