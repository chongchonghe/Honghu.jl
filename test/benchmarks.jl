#!/usr/bin/env julia
#
# The cost of a step, per cell and per CPU, and where it goes. Not part of the
# test suite: run directly, on one thread, and paste the table into README.md.
#
#   julia --project=. -t 1 test/benchmarks.jl [n]
#
# One problem, the blast wave of `Problems.sedov_blast` on n^3 cells (128 by
# default), in three forms: hydrodynamics; with grey M1 radiation coupled, one
# radiation substep per step; and with a uniform magnetic field (ideal MHD, beta = 1
# in the ambient gas). The same setup in every column makes the differences the cost
# of the physics. PLM and the default SSPRK2 throughout.
#
# The total comes from timing whole steps. The rows below it share that total out by
# a profile of whole steps: each sample goes to the kernel it was in, so the rows add
# up to the total.

using Honghu
using Printf
using Profile

const N = isempty(ARGS) ? 128 : parse(Int, ARGS[1])
const NSTEPS = 10

# The Sedov blast of `Problems.sedov_blast`, written out so that radiation and a
# field can be added to it.
function blast(n; physics::Symbol)
    gamma = 5 / 3
    rad = physics === :rhd ?
          RadiationPhysics(c=1e3, chat=1e2, a_rad=1e-12, opacity=ConstantOpacity(100.0)) :
          nothing
    sim = Simulation(; n=(n, n, n), xlo=(-0.5, -0.5, -0.5), xhi=(0.5, 0.5, 0.5),
                     eos=IdealGas(; gamma, cv=1.0), boundaries=Boundaries(Outflow()),
                     radiation=rad, max_radiation_substeps=1, mhd=physics === :mhd,
                     reconstruction=PLM())
    r0 = 3.5 / n
    pin = (gamma - 1) / (4 / 3 * pi * r0^3)
    pamb = 1e-5
    b0 = sqrt(2pamb) / sqrt(2)                  # beta = 1 in the ambient gas
    set_initial_conditions!(sim) do x, y, z
        r2 = x * x + y * y + z * z
        s = (rho=1.0, p=r2 < r0^2 ? pin : pamb)
        physics === :mhd ? (; s..., bx=b0, by=b0, bz=0.0) : s
    end
    return sim
end

# Which row a profile sample belongs to. Two rows are whole call trees: anything
# under `rad_rhs!` is radiation transport and anything under
# `radiation_source_step!` is the coupling, whatever kernel it was in. Every other
# sample goes to the innermost frame that names a row.
const SUBTREE = ("radiation_source_step!" => "matter-radiation coupling",
                 "rad_rhs!" => "radiation transport (HLL)")
const KERNEL = Dict(
    "_reconstruct_cell!" => "reconstruction (PLM)",
    "reconstruct!" => "reconstruction (PLM)",
    "_hydro_flux_cell!" => "Riemann solver",
    "_mhd_flux_cell!" => "Riemann solver",
    "_divergence_cell!" => "flux divergence and stage update",
    "_emf_cell!" => "EMF and induction",
    "_induction_face!" => "EMF and induction",
    "_bcc_cell!" => "EMF and induction",
    "cons2prim!" => "conversions and floors",
    "enforce_floors!" => "conversions and floors",
    "enforce_scalar_floors!" => "conversions and floors",
    "enforce_state!" => "conversions and floors",
    "amend_rad_state!" => "conversions and floors",
    "floor_rad_energy!" => "conversions and floors",
    "fill_ghosts!" => "ghost cells",
    "mhd_ghost_energy!" => "ghost cells",
    "timestep" => "time step",
)
const ROWS = ["reconstruction (PLM)", "Riemann solver", "flux divergence and stage update",
              "EMF and induction", "radiation transport (HLL)", "matter-radiation coupling",
              "conversions and floors", "ghost cells", "time step", "other"]

function classify(frames)
    names = [String(f.func) for f in frames]       # innermost first
    for (fn, row) in SUBTREE
        fn in names && return row
    end
    for nm in names
        haskey(KERNEL, nm) && return KERNEL[nm]
    end
    return "other"
end

"Seconds per step, and the share of each row, over `NSTEPS` steps each."
function measure(physics)
    sim = blast(N; physics)
    step!(sim)                                       # compile
    t = @elapsed for _ in 1:NSTEPS
        step!(sim)
    end
    Profile.clear()
    Profile.init(n=10^7, delay=0.001)
    @profile for _ in 1:NSTEPS
        step!(sim)
    end
    data = Profile.fetch(include_meta=false)
    cache = Dict{UInt64,Vector{Base.StackTraces.StackFrame}}()
    counts = Dict(r => 0 for r in ROWS)
    sample = UInt64[]
    for ip in data
        if ip != 0
            push!(sample, ip)
            continue
        end
        if !isempty(sample)
            frames = reduce(vcat, (get!(() -> Profile.lookup(x), cache, x) for x in sample))
            if any(f -> f.func === :step!, frames)
                counts[classify(frames)] += 1
            end
            empty!(sample)
        end
    end
    total = sum(values(counts))
    return t / NSTEPS, Dict(r => counts[r] / total for r in ROWS), sim
end

Threads.nthreads() == 1 ||
    @warn "run on one thread (-t 1): the per-CPU numbers assume it"
@printf("Honghu %s, Julia %s, %d thread(s), %d^3 cells, %d steps\n",
        pkgversion(Honghu), VERSION, Threads.nthreads(), N, NSTEPS)

cols = (:hydro, :rhd, :mhd)
res = Dict(p => measure(p) for p in cols)
for p in cols
    sim = res[p][3]
    @printf("%-5s %.3f s/step, %d radiation substeps in %d steps\n", p, res[p][1],
            sim.diag.rad_substeps, sim.diag.steps)
end

cells = N^3
ns(p) = res[p][1] / cells * 1e9 * Threads.nthreads()
cell(x) = x < 0.05 ? "--" : @sprintf("%.1f", x)
println()
println("| | Hydrodynamics | Radiation hydrodynamics | MHD |")
println("|:---|---:|---:|---:|")
@printf("| **Mcell-updates/(s CPU)** | %s |\n",
        join((@sprintf("%.2f", 1e3 / ns(p)) for p in cols), " | "))
@printf("| **ns CPU / cell-update** | %s |\n", join((@sprintf("%.0f", ns(p)) for p in cols), " | "))
for r in ROWS
    @printf("| %s | %s |\n", r, join((cell(res[p][2][r] * ns(p)) for p in cols), " | "))
end
