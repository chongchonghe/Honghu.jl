# The variable reduced speed of light, measured.
#
# Every number here is a relative L1 error in the gas temperature against a
# `chat = c` reference computed at the same resolution, and -- except where a row
# says otherwise -- at the same number of steps as the standard RSLA it is
# compared with, so the comparison is at equal cost.
#
#     julia --project=. -t auto examples/rsla_limiter.jl
#     julia --project=. -t auto examples/rsla_limiter.jl --no-plot
#     julia --project=. -t auto examples/rsla_limiter.jl --outdir=docs/figures
#
# The last form is the one that regenerates the figure in `docs/rsla-findings.md`,
# which is where these numbers are written up.
#
# Sections:
#   1. the two test problems, RSLA vs the improved RSLA
#   2. what the safety factor beta buys and costs
#   3. why the bound needs the closure's signal speed, not the P1 one
#   4. resolution: tau_cell scales with dx, so the advantage should move

include("common.jl")
using Hydro3D
using Hydro3D.Problems
using Printf

l1(T, ref) = sum(abs, T .- ref) / sum(abs, ref)

"Run a problem to `tmax` and return its gas temperature profile and diagnostics."
function score(build, tmax)
    sim = first(build())
    run!(sim; tmax, verbose=false)
    (T=vec(gas_temperature(sim)), steps=sim.diag.steps, sim=sim)
end

# `chat_factor = 433` is `chat = c` for this problem: its default reduction is
# already phi0 = 0.023 (Quokka's `10 (v0 + cs0)`).
shock(; kw...) = radhydro_shock(; n=256, kw...)
vaytet(; kw...) = marshak_vaytet(; n=64, ngroups=4, kw...)

println("\n", "="^78)
println("1. RSLA versus the improved RSLA, at equal cost")
println("="^78)

let
    _, tmax = shock()
    ref = score(() -> shock(chat_factor=433.0), tmax)
    @printf("\nLowrie-Edwards Mach 3 radiative shock, n = 256 (reference: %d steps)\n",
            ref.steps)
    @printf("%8s  %18s  %18s  %7s  %s\n", "phi0", "RSLA", "IRSLA", "gain", "steps")
    for cf in (10.0, 3.0, 1.0)
        phi0 = cf * 4 / sqrt(3 * 1e6)
        a = score(() -> shock(chat_factor=cf), tmax)
        b = score(() -> shock(chat_factor=cf, chat_limiter=StabilityBound()), tmax)
        ea, eb = l1(a.T, ref.T), l1(b.T, ref.T)
        @printf("%8.4f  %17.4f%%  %17.4f%%  %6.2fx  %d / %d\n",
                phi0, 100ea, 100eb, ea / eb, a.steps, b.steps)
    end

    _, tv = vaytet()
    refv = score(() -> vaytet(), tv)
    @printf("\nVaytet multigroup Marshak wave, n = 64, 4 groups (reference: %d steps)\n",
            refv.steps)
    @printf("%8s  %18s  %18s  %7s  %s\n", "phi0", "RSLA", "IRSLA", "gain", "steps")
    for phi0 in (0.03, 0.01, 0.003)
        a = score(() -> vaytet(chat_over_c=phi0), tv)
        b = score(() -> vaytet(chat_over_c=phi0, chat_limiter=StabilityBound()), tv)
        ea, eb = l1(a.T, refv.T), l1(b.T, refv.T)
        @printf("%8.4f  %17.3f%%  %17.3f%%  %6.2fx  %d / %d\n",
                phi0, 100ea, 100eb, ea / eb, a.steps, b.steps)
    end

    # ... and the same wave with phi set per group, the published prescription.
    @printf("\n%8s  %18s  %18s\n", "phi0", "IRSLA per-cell", "IRSLA per-group")
    for phi0 in (0.03, 0.01, 0.003)
        b = score(() -> vaytet(chat_over_c=phi0, chat_limiter=StabilityBound()), tv)
        c = score(() -> vaytet(chat_over_c=phi0,
                               chat_limiter=StabilityBound()), tv)
        @printf("%8.4f  %17.3f%%  %17.3f%%\n", phi0, 100l1(b.T, refv.T),
                100l1(c.T, refv.T))
    end
end

println("\n", "="^78)
println("2. the safety factor")
println("="^78)
println("""
beta scales both stability channels. The bound is derived from a scalar model of
one transport stage against one source stage, so it is not exact; beta is what
covers the difference. Larger beta is more accurate until it is not stable.""")

let
    _, tmax = shock()
    ref = score(() -> shock(chat_factor=433.0), tmax)
    @printf("\nradiative shock at phi0 = 0.0069\n%8s  %12s  %8s  %s\n",
            "beta", "L1", "steps", "newton failures")
    for beta in (0.3, 0.6, 0.9, 1.0, 1.5, 2.0)
        b = score(() -> shock(chat_factor=3.0,
                              chat_limiter=StabilityBound(; beta)), tmax)
        @printf("%8.2f  %11.4f%%  %8d  %d\n", beta, 100l1(b.T, ref.T), b.steps,
                b.sim.diag.newton_failures)
    end

    _, tv = vaytet()
    refv = score(() -> vaytet(), tv)
    @printf("\nVaytet wave at phi0 = 0.01\n%8s  %12s  %8s  %s\n",
            "beta", "L1", "steps", "max T (wall is 1000 K)")
    for beta in (0.3, 0.6, 0.9, 1.0, 1.5, 2.0)
        b = score(() -> vaytet(chat_over_c=0.01,
                               chat_limiter=StabilityBound(; beta)), tv)
        @printf("%8.2f  %11.3f%%  %8d  %.1f\n", beta, 100l1(b.T, refv.T), b.steps,
                maximum(b.T))
    end
end

println("\n", "="^78)
println("3. the CFL limit this scheme actually has")
println("="^78)
println("""
The published bound permits nu = sqrt(3) = 1.73 as tau_cell -> 0, which is a
Courant number of one for the P1 signal speed c/sqrt(3). Under M1 a
free-streaming cell carries signals at c, not c/sqrt(3). This table runs the
Vaytet wave with the RSLA switched off entirely -- chat = c, no limiter -- so
nothing but the transport step is under test.""")

let
    n, ng = 64, 4
    ref = score(() -> marshak_vaytet(; n, ngroups=ng, cfl_radiation=0.2), 1.36e-7)
    @printf("\n%-26s %6s %8s %10s %10s\n", "reconstruction", "nu", "steps", "max T", "L1")
    for rec in (PCM(), PLM(), PPM()), nu in (0.8, 1.2, 1.5, sqrt(3))
        s = score(() -> marshak_vaytet(; n, ngroups=ng, cfl_radiation=nu,
                                       radiation_reconstruction=rec), 1.36e-7)
        e = l1(s.T, ref.T)
        @printf("%-26s %6.2f %8d %10.4g %9s\n", string(typeof(rec)), nu, s.steps,
                maximum(s.T), e > 9 ? "blown up" : @sprintf("%.3f%%", 100e))
    end
end

println("\n", "="^78)
println("4. resolution")
println("="^78)
println("""
tau_cell is proportional to dx, so refining the grid moves cells from the thick
side of the bound to the thin side and the limiter has less room. The advantage
should shrink under refinement, and this is where it goes.""")

let
    @printf("\nradiative shock at phi0 = 0.0069\n%6s  %12s  %12s  %7s\n",
            "n", "RSLA", "IRSLA", "gain")
    for n in (64, 128, 256, 512)
        _, tmax = radhydro_shock(; n)
        ref = score(() -> radhydro_shock(; n, chat_factor=433.0), tmax)
        a = score(() -> radhydro_shock(; n, chat_factor=3.0), tmax)
        b = score(() -> radhydro_shock(; n, chat_factor=3.0,
                                       chat_limiter=StabilityBound()), tmax)
        ea, eb = l1(a.T, ref.T), l1(b.T, ref.T)
        @printf("%6d  %11.4f%%  %11.4f%%  %6.2fx\n", n, 100ea, 100eb, ea / eb)
    end
end

if PLOTTING
    using Plots
    _, tmax = shock()
    ref = score(() -> shock(chat_factor=433.0), tmax)
    a = score(() -> shock(chat_factor=3.0), tmax)
    b = score(() -> shock(chat_factor=3.0, chat_limiter=StabilityBound()), tmax)
    x = vec(coordinates(ref.sim.grid, 1))

    p1 = plot(x, ref.T; label="chat = c", lw=2, color=:black,
              xlabel="x", ylabel="T_gas", title="Mach 3 radiative shock, phi0 = 0.0069",
              legend=:topleft)
    plot!(p1, x, a.T; label=@sprintf("RSLA (%.3f%%)", 100l1(a.T, ref.T)), lw=1.5,
          color=:crimson, ls=:dash)
    plot!(p1, x, b.T; label=@sprintf("IRSLA (%.3f%%)", 100l1(b.T, ref.T)), lw=1.5,
          color=:royalblue)

    phi = vec(b.sim.rw.chat[1, b.sim.grid.lo[1]:b.sim.grid.hi[1], 1, 1]) ./
          b.sim.radiation.c
    p2 = plot(x, phi; label="phi", lw=1.5, color=:seagreen, xlabel="x",
              ylabel="phi = chat/c", yscale=:log10, legend=:topleft,
              title="the limiter's choice")
    hline!(p2, [b.sim.radiation.chat / b.sim.radiation.c]; label="phi0", ls=:dot,
           color=:crimson)

    saveplot(plot(p1, p2; layout=(2, 1), size=(760, 720)), "rsla_limiter.png")
end

plotting_note()
