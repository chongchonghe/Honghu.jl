# Every measurement in `docs/photoionization-findings.md`, in one script.
#
#   julia --project=. -t 8 docs/photoionization-experiments/measure.jl            # all of it
#   julia --project=. -t 8 docs/photoionization-experiments/measure.jl cost       # named parts
#
# Parts: `cost`, `integrators`, `chat`. A run of all three writes `measure.txt` beside this
# file; a run of some of them only prints.
#
#   cost         microseconds per cell of `chemistry_step!`, on a 32^3 octant of each Iliev test,
#                for both integrators, on one thread and on the threads of this session
#   integrators  Test 2 at n = 16 to 100 and 500 Myr with each integrator
#   chat         Test 1 at n = 16 for four reduced speeds of light, against the two references
#
# Nothing here is part of the package. The one-thread numbers need a second Julia process,
# because `threaded_jk` follows `Threads.nthreads()`; the script starts it itself, and `cost-worker`
# is the argument that process runs with.

using Honghu
using Printf
using Dates
const P = Honghu.Problems

const CGS = Honghu.CGS
const MYR = 1e6 * CGS.year
const KPC = 1e3 * CGS.parsec
const TREC = 1 / (2.59e-13 * 1e-3)            # recombination time of the Iliev tests, 122.3 Myr
const CHAT_FACTOR = 3e-4                      # the example's reduced speed of light

# What a run prints, kept so that it can be written to `measure.txt` as well.
const LOG = String[]
function emit(args...)
    s = string(args...)
    println(s)
    push!(LOG, s)
    flush(stdout)
end
emitf(fmt::AbstractString, args...) = emit(Printf.format(Printf.Format(fmt), args...))

median_of(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1]) / 2)

# --- 1. cost per cell -----------------------------------------------------------

# Microseconds per interior cell of one `chemistry_step!`, as the median of `nrep` calls
# that all start from the same state, after 0.25 t_rec so that the front is in the box.
# Returns the cost and what the step did, and the number of cells already ionized.
function cost_per_cell(test, solver; n=32, nrep=5)
    sim, _ = P.stromgren_iliev(; test, n, chat_factor=CHAT_FACTOR, solver)
    run!(sim; tmax=0.25 * TREC, verbose=false)
    dt = timestep(sim)
    t = sim.t
    U0, R0 = copy(sim.U), copy(sim.R)
    Honghu.chemistry_step!(sim, dt, t)           # a first call, for any compilation left
    us = Float64[]
    st = nothing
    for _ in 1:nrep
        copyto!(sim.U, U0)
        copyto!(sim.R, R0)
        t0 = time_ns()
        st = Honghu.chemistry_step!(sim, dt, t)
        push!(us, (time_ns() - t0) * 1e-3 / n^3)
    end
    xH = first(field(sim, :X_2))
    return median_of(us), st, count(>(0.5), xH)
end

# One line per (test, solver), `key=value` so that the parent process can read a worker's.
function cost_lines()
    lines = String[]
    for test in (1, 2), (name, solver) in (("Rosenbrock", RosenbrockChem()),
                                           ("BackwardEuler", BackwardEulerChem()))
        us, st, nion = cost_per_cell(test, solver)
        push!(lines, @sprintf("cost test=%d solver=%s threads=%d us=%.3f steps=%.3f max=%d failures=%d ionized=%d",
                              test, name, Threads.nthreads(), us, st.steps / st.cells,
                              st.max_steps, st.failures, nion))
    end
    return lines
end

function part_cost()
    emit("## 1. Microseconds per cell of chemistry_step!, 32^3 octant, chat = 3e-4 c, after 0.25 t_rec")
    emit("   median of 5 calls from one state; steps = solver steps per cell (Newton iterations for BackwardEuler)")
    lines = cost_lines()
    if Threads.nthreads() != 1
        # The one-thread numbers come from a second process: `threaded_jk` follows nthreads().
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --threads=1 $(@__FILE__) cost-worker`
        append!(lines, filter(l -> startswith(l, "cost "), readlines(cmd)))
    end
    recs = [Dict(String(first(kv)) => String(last(kv)) for kv in split.(split(l)[2:end], '=')) for l in lines]
    sort!(recs; by=r -> (r["test"], r["solver"], parse(Int, r["threads"])))
    emit(rpad("test", 6), rpad("solver", 15), rpad("threads", 9), rpad("us/cell", 10), rpad("steps/cell", 12),
         rpad("max steps", 11), rpad("failures", 10), "cells with x_HII > 0.5")
    for r in recs
        emit(rpad(r["test"], 6), rpad(r["solver"], 15), rpad(r["threads"], 9), rpad(r["us"], 10),
             rpad(r["steps"], 12), rpad(r["max"], 11), rpad(r["failures"], 10), r["ionized"])
    end
    return recs
end

# --- 2. the two integrators on Test 2 ---------------------------------------------

# Mean of `f` over the cells whose centre is in each of `nb` equal bins of radius from the
# source to the box side. Cells beyond that radius (the corners) are left out.
function radial_means(sim, fields; nb=16, centered=false)
    g = sim.grid
    L = P._ILIEV_L
    xs = [coordinates(g, d) for d in 1:3]
    data = [first(field(sim, f)) for f in fields]
    sums = zeros(nb, length(fields))
    cnt = zeros(Int, nb)
    for k in axes(data[1], 3), j in axes(data[1], 2), i in axes(data[1], 1)
        r = sqrt(xs[1][i]^2 + xs[2][j]^2 + xs[3][k]^2)
        b = floor(Int, r / L * nb) + 1
        b > nb && continue
        cnt[b] += 1
        for (m, d) in enumerate(data)
            sums[b, m] += d[i, j, k]
        end
    end
    return sums ./ max.(cnt, 1), cnt
end

# The profile and the radius of a run at its present time. `Tion` is the median temperature
# of the cells with x_HII > 0.9, the number the interior overheating is read from.
function snapshot(sim)
    prof, cnt = radial_means(sim, (:temperature, :X_1))
    T = first(field(sim, :temperature))
    xion = first(field(sim, :X_2))
    return (r=P.ionized_radius(sim), T=prof[:, 1], xHI=prof[:, 2], cnt=cnt,
            Tion=median_of(T[xion.>0.9]), nion=count(>(0.9), xion))
end

function run_to(sim, tmyr)
    run!(sim; tmax=tmyr * MYR, verbose=false)
    return snapshot(sim)
end

# Test 2 built here and not by `stromgren_iliev`, which fixes the network: the same box,
# source, band and radiation, with the heating per ionization as an argument. Passing none
# gives the network's default (photon energy less 13.6 eV, 16.05 eV), which is the factory's
# run, and the replica is checked against it below.
function test2_sim(; heating=nothing, n=16)
    L = P._ILIEV_L
    kw = heating === nothing ? (;) : (; heating_energy=heating * CGS.eV)
    net = HydrogenOTSA(; sigma=1.63e-18, photon_energy=29.65 * CGS.eV, kw...)
    rad = RadiationPhysics(c=CGS.c_light, chat=CHAT_FACTOR * CGS.c_light, a_rad=CGS.a_rad,
                           k_B=CGS.k_B, opacity=ConstantOpacity(0.0), beta_order=0)
    bc = Boundaries((Reflecting(), Outflow()), (Reflecting(), Outflow()), (Reflecting(), Outflow()))
    emission = P._gaussian_source(0.0, L / n, n, P._ILIEV_NDOT / 8)
    sim = Simulation(; n=(n, n, n), xlo=(0.0, 0.0, 0.0), xhi=(L, L, L), hydro=false, radiation=rad,
                     eos=IdealGas(gamma=5 / 3, cv=cv_from_mu(1.0, 5 / 3)), boundaries=bc,
                     radiation_boundaries=bc, chemistry=Photochemistry(net; emission))
    rho = P._ILIEV_NH * CGS.m_H
    set_initial_conditions!(sim) do x, y, z
        (rho=rho, T=100.0, x_HII=0.0, Erad=0.0)
    end
    return sim
end

function part_integrators()
    emit("\n## 2. Test 2, n = 16, chat = 3e-4 c: the chemistry integrators")
    emit("   r_I in kpc; T_ion is the median temperature of the cells with x_HII > 0.9; the profiles are means over the")
    emit("   cells in 16 radial bins of width L/16 = 0.4125 kpc (corner cells beyond L are left out)")
    variants = (("Rosenbrock", RosenbrockChem()),
                ("Rosenbrock reltol 1e-4", RosenbrockChem(reltol=1e-4)),
                ("BackwardEuler", BackwardEulerChem()))
    out = Dict{String,Any}()
    for (name, solver) in variants
        sim, _ = P.stromgren_iliev(; test=2, n=16, chat_factor=CHAT_FACTOR, solver)
        wall0 = 0.0
        res = Any[]
        for tmyr in (100, 500)
            w = @elapsed r = run_to(sim, tmyr)
            wall0 += w
            push!(res, merge(r, (wall=wall0,)))
        end
        d = sim.diag
        out[name] = (res=res, steps=d.chem_steps / d.chem_cells, max=d.chem_max,
                     failures=d.chem_failures, cells=d.chem_cells)
        emitf("%-24s wall %.1f s (to 100 Myr %.1f s), chemistry %.2f steps/cell mean, %d max, %d failed cells of %d solves, substep %.2f Myr",
              name, wall0, res[1].wall, out[name].steps, out[name].max, out[name].failures,
              out[name].cells, timestep(sim) / MYR)
    end
    for (it, tmyr) in enumerate((100, 500))
        emit("\n  t = $tmyr Myr")
        emit(rpad("  solver", 26), rpad("r_I (kpc)", 11), rpad("T_ion (K)", 11),
             "profiles: T (K), then x_HI, bins 1 to 16")
        for (name, _) in variants
            r = out[name].res[it]
            emitf("  %-24s %-10.4f %-10.0f %s", name, r.r / KPC, r.Tion,
                  join((@sprintf("%.0f", v) for v in r.T), " "))
            emitf("  %-24s %-10s %-10s %s", "", "", "", join((@sprintf("%.3e", v) for v in r.xHI), " "))
        end
        ref = out["Rosenbrock"].res[it]
        tight = out["Rosenbrock reltol 1e-4"].res[it]
        be = out["BackwardEuler"].res[it]
        rel(a, b) = maximum(abs.(a .- b) ./ b)
        emitf("  Rosenbrock reltol 1e-2 against 1e-4: r_I %+.3f%%, T_ion %+.3f%%, max |dT|/T over bins %.3f%%, max |dx_HI| %.2e",
              100 * (ref.r / tight.r - 1), 100 * (ref.Tion / tight.Tion - 1), 100 * rel(ref.T, tight.T),
              maximum(abs.(ref.xHI .- tight.xHI)))
        emitf("  BackwardEuler against Rosenbrock reltol 1e-2: r_I %+.3f%%, T_ion %+.3f%%, max |dT|/T over bins %.3f%%, max |dx_HI| %.2e",
              100 * (be.r / ref.r - 1), 100 * (be.Tion / ref.Tion - 1), 100 * rel(be.T, ref.T),
              maximum(abs.(be.xHI .- ref.xHI)))
    end

    # The interior temperature against the heating per ionization: the default and the
    # optically thin value of the 1e5 K blackbody (spec 4.3), 6.3 eV.
    emit("\n  Interior temperature against the heating per ionization (RosenbrockChem, same run to 100 and 500 Myr)")
    replica = test2_sim()
    got = run_to(replica, 100)
    want = out["Rosenbrock"].res[1]
    emitf("  replica of the factory run at 100 Myr: r_I differs by %.2e, T_ion by %.2e (relative)",
          abs(got.r / want.r - 1), abs(got.Tion / want.Tion - 1))
    emitf("  %-22s %-8s %-10s %-10s", "heating per ionization", "t (Myr)", "r_I (kpc)", "T_ion (K)")
    for heating in (nothing, 6.3)
        sim = heating === nothing ? replica : test2_sim(; heating)
        for tmyr in (100, 500)
            r = heating === nothing && tmyr == 100 ? got : run_to(sim, tmyr)
            emitf("  %-22s %-8d %-10.4f %-10.0f", heating === nothing ? "16.05 eV (default)" : "6.3 eV",
                  tmyr, r.r / KPC, r.Tion)
        end
    end
    return out
end

# --- 3. the reduced speed of light on Test 1 ----------------------------------------

# Where x_HII falls through 0.5 along a row of cells, by linear interpolation in the distance
# from the source; NaN if it does not. It shows how round the front is: the row along an axis
# and the row along the body diagonal should give the same radius.
function crossing(x, dist)
    i = findlast(>(0.5), x)
    (i === nothing || i == length(x)) && return NaN
    return dist[i] + (x[i] - 0.5) / (x[i] - x[i+1]) * (dist[i+1] - dist[i])
end

function front_along_axis_and_diagonal(sim)
    g = sim.grid
    xs = coordinates(g, 1)
    x = first(field(sim, :X_2))
    n = g.n[1]
    axis = crossing([x[i, 1, 1] for i in 1:n], [sqrt(xs[i]^2 + 2xs[1]^2) for i in 1:n])
    diag = crossing([x[i, i, i] for i in 1:n], [sqrt(3) * xs[i] for i in 1:n])
    return axis, diag
end

function part_chat()
    emit("\n## 3. Test 1, n = 16: ionized radius (kpc) against the finite-chat front equation and the chat = c curve")
    emit("   error = r_I / r_ref - 1; the Test 1 box is the octant and the background ionization is 1.2e-3")
    times = (10, 30, 100, 500)
    emit(rpad("chat/c", 9), rpad("t (Myr)", 9), rpad("r_I", 9), rpad("finite-chat", 13),
         rpad("error", 9), rpad("analytic", 10), "error vs analytic")
    for cf in (1e-4, 3e-4, 1e-3, 3e-3)
        sim, _ = P.stromgren_iliev(; test=1, n=16, chat_factor=cf)
        wall = 0.0
        rows = String[]
        shape = String[]
        for tmyr in times
            wall += @elapsed run!(sim; tmax=tmyr * MYR, verbose=false)
            ra, rd = front_along_axis_and_diagonal(sim)
            push!(shape, @sprintf("%d Myr %.3f / %.3f", tmyr, ra / KPC, rd / KPC))
            r = P.ionized_radius(sim; x0=1.2e-3)
            fin = P.stromgren_radius_finite_c(tmyr * MYR; chat=cf * CGS.c_light)
            ana = P.stromgren_radius_analytic(tmyr * MYR)
            pct(a) = @sprintf("%+.2f%%", 100 * a)
            push!(rows, @sprintf("%-9.0e %-9d %-9.4f %-13.4f %-9s %-10.4f %s",
                                 cf, tmyr, r / KPC, fin / KPC, pct(r / fin - 1), ana / KPC,
                                 pct(r / ana - 1)))
        end
        foreach(emit, rows)
        emit("  front (x_HII = 0.5, kpc) along the axis row / along the body diagonal: ", join(shape, "; "))
        d = sim.diag
        # What `ionized_radius` leaves out: the sphere's recombinations go as the sum of x^2, and
        # a front narrower than a cell counts a half-ionized cell as 0.5 of a sphere and 0.25
        # of the recombinations. At 500 Myr, 4.1 recombination times, the sphere is within 1% of
        # its final size, so the radius of the recombination volume is the Stromgren radius.
        x = first(field(sim, :X_2))
        rrec = cbrt(3 * 8 * sum(abs2, x) * prod(sim.grid.dx) / (4pi))
        rS = P.stromgren_radius_analytic(500 * MYR)
        emitf("  wall %.1f s, %d radiation substeps, %d chemistry failures; at 500 Myr the radius of the recombination volume 8 sum(x^2) dV is %.4f kpc (%+.2f%% against the chat = c curve)",
              wall, d.rad_substeps, d.chem_failures, rrec / KPC, 100 * (rrec / rS - 1))
    end
end

# --- driver ---------------------------------------------------------------------------

function main(args)
    if "cost-worker" in args
        foreach(println, cost_lines())
        return
    end
    parts = isempty(args) ? ["cost", "integrators", "chat"] : args
    bad = setdiff(parts, ["cost", "integrators", "chat"])
    isempty(bad) || error("unknown part $bad; the parts are cost, integrators, chat")
    emit("# Honghu photoionization measurements, ", Dates.format(Dates.now(), "yyyy-mm-dd"),
         "; julia ", VERSION, ", ", Threads.nthreads(), " threads; ", Sys.CPU_NAME)
    "cost" in parts && part_cost()
    "integrators" in parts && part_integrators()
    "chat" in parts && part_chat()
    if sort(parts) == ["chat", "cost", "integrators"]
        path = joinpath(@__DIR__, "measure.txt")
        write(path, join(LOG, '\n') * '\n')
        println("\nwrote $path")
    end
end

main(ARGS)
