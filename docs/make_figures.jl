#!/usr/bin/env julia
#
# Regenerate the verification figures embedded in README.md and in the physics guides.
#
#   julia --project=. -t auto docs/make_figures.jl              # all of them
#   julia --project=. -t auto docs/make_figures.jl stromgren    # only the named ones
#
# Every panel compares the code against something independent: an exact Riemann
# solution, an analytic diffusion solution, a measured convergence slope, or a
# similarity solution. Takes a couple of minutes, plus a few for the Stromgren spheres.

ENV["GKSwstype"] = "100"          # headless GR

using Honghu
using Plots
using Printf
const P = Honghu.Problems

const OUT = joinpath(@__DIR__, "figures")
mkpath(OUT)
default(fontfamily="sans-serif", framestyle=:box, grid=true, gridalpha=0.15,
        legendfontsize=8, titlefontsize=10, guidefontsize=9, dpi=140)

save(p, name) = (savefig(p, joinpath(OUT, name)); println("  wrote $name"))

# --- 1. Sod shock tube vs the exact Riemann solution ------------------------
function figure_sod(nx=256)
    sim, tmax = P.sod_shock_tube(; nx)
    run!(sim; tmax, verbose=false)
    pr = primitives(sim)
    g = sim.grid
    x = coordinates(g, 1)
    xe = range(0, 1; length=2000)
    ex = [P.sod_exact(xi, sim.t) for xi in xe]
    eint(rho, p) = p / (1.4 - 1) / rho

    panels = []
    for (idx, (num, exact, lbl)) in enumerate((
        (pr[1, :, 1, 1], [e[1] for e in ex], "density"),
        (pr[2, :, 1, 1], [e[2] for e in ex], "velocity"),
        (pr[5, :, 1, 1], [e[3] for e in ex], "pressure"),
        (eint.(pr[1, :, 1, 1], pr[5, :, 1, 1]), [eint(e[1], e[3]) for e in ex],
         "specific internal energy")))
        pl = plot(xe, exact; lw=2, c=:black, label=idx == 1 ? "exact" : "",
                  xlabel="x", ylabel=lbl)
        scatter!(pl, x, num; ms=2, msw=0, c=1, label=idx == 1 ? "Honghu (PPM, $nx cells)" : "")
        push!(panels, pl)
    end
    save(plot(panels...; layout=(2, 2), size=(900, 620),
              plot_title=@sprintf("Sod shock tube at t = %.2f", sim.t),
              plot_titlevspan=0.08), "sod.png")
end

# --- 2. Order of accuracy on smooth advection -------------------------------
function figure_convergence(ns=(16, 32, 64, 128, 256))
    schemes = ((PCM(), "PCM", 1), (PLM(), "PLM", 2), (PPM(), "PPM", 3))
    pl = plot(; xscale=:log10, yscale=:log10, xlabel="cells", ylabel="L1 density error",
              legend=:bottomleft, title="Convergence on smooth advection")
    l1 = default_metrics(P.smooth_advection).l1_density
    for (i, (recon, name, _)) in enumerate(schemes)
        errs = Float64[]
        for nx in ns
            sim, tmax = P.smooth_advection(; nx, reconstruction=recon)
            run!(sim; tmax, verbose=false)
            push!(errs, l1(sim))
        end
        slope = log2(errs[end-1] / errs[end])
        plot!(pl, collect(ns), errs; marker=:circle, ms=4, c=i, lw=2,
              label=@sprintf("%s  (measured %.2f)", name, slope))
    end
    # reference slopes
    for (o, st) in ((1, :dot), (2, :dash), (3, :dashdot))
        ref = 3e-2 .* (first(ns) ./ collect(ns)) .^ o
        plot!(pl, collect(ns), ref; c=:gray, ls=st, lw=1, label="order $o")
    end
    save(plot(pl; size=(620, 480)), "convergence.png")
end

# --- 3. Radiation diffusion vs the analytic Gaussian ------------------------
function figure_diffusion()
    D = 1 / (3 * 300.0)
    xa = range(0, 1; length=1000)
    pl = plot(xa, [P.diffusion_exact(x, 1.0; D=D) for x in xa]; lw=2.5, c=:black,
              label="analytic, sigma^2 = sigma0^2 + 2Dt", xlabel="x",
              ylabel="radiation energy density",
              title="Static diffusion limit at t = 1  (D = c/3(rho kappa))")
    plot!(pl, xa, [P.diffusion_exact(x, 0.0; D=D) for x in xa]; lw=1.5, c=:gray,
          ls=:dash, label="initial condition")
    for (i, nx) in enumerate((32, 64, 128))
        sim, tmax = P.radiation_diffusion(; nx)
        run!(sim; tmax, verbose=false)
        g = sim.grid
        E, _ = field(sim, :Erad)
        scatter!(pl, coordinates(g, 1), vec(E); ms=3, msw=0, c=i, label="$nx cells")
    end
    save(plot(pl; xlims=(0.2, 0.8), size=(700, 500)), "diffusion.png")
end

# --- 4. Non-equilibrium Marshak wave ----------------------------------------
function figure_marshak(nx=256)
    sim, tmax = P.marshak_wave(; nx)
    x = coordinates(sim.grid, 1)
    pl = plot(; xlabel="x", ylabel="temperature", xlims=(0, 0.12),
              title="Marshak wave: radiation runs ahead of the matter",
              legend=:topright)
    for (i, t) in enumerate((0.1, 0.3, 1.0))
        run!(sim; tmax=t, verbose=false)
        Tr, _ = field(sim, :Trad)
        Tg, _ = field(sim, :temperature)
        plot!(pl, x, vec(Tr); lw=2, c=i, label=@sprintf("T_rad, t = %.1f", t))
        plot!(pl, x, vec(Tg); lw=2, c=i, ls=:dash, label=@sprintf("T_gas, t = %.1f", t))
    end
    save(plot(pl; size=(700, 500)), "marshak.png")
end

# --- 4b. Radiative shock (Lowrie & Edwards Mach 3) --------------------------
function figure_radshock(nx=256)
    xe, rhoe, ve, Tme, Tre = P.radhydro_shock_exact()
    sim, tmax = P.radhydro_shock(; nx)
    run!(sim; tmax, verbose=false)
    g = sim.grid
    x = coordinates(g, 1)
    Tg = gas_temperature(sim)[:, 1, 1]
    Tr = radiation_temperature(sim)[:, 1, 1]
    err = default_metrics(P.radhydro_shock).l1_Trad(sim)

    full = plot(xe, Tme; lw=2, c=:black, label="T_matter, semi-analytic",
                xlabel="x", ylabel="temperature", legend=:topleft)
    plot!(full, xe, Tre; lw=2, c=:gray, ls=:dash, label="T_radiation, semi-analytic")
    scatter!(full, x, Tg; ms=2.2, msw=0, c=1, label="T_matter, Honghu")
    scatter!(full, x, Tr; ms=2.2, msw=0, c=2, label="T_radiation, Honghu")

    # the Zel'dovich spike is the reason this test is hard: zoom in on it
    zoom = plot(xe, Tme; lw=2, c=:black, label="", xlabel="x", ylabel="temperature",
                xlims=(0.0120, 0.0142), ylims=(3.2, 4.5),
                title="Zel'dovich spike")
    plot!(zoom, xe, Tre; lw=2, c=:gray, ls=:dash, label="")
    scatter!(zoom, x, Tg; ms=3, msw=0, c=1, label="")
    scatter!(zoom, x, Tr; ms=3, msw=0, c=2, label="")

    save(plot(full, zoom; layout=(1, 2), size=(1050, 450),
              plot_title=@sprintf("Mach 3 radiative shock, %d cells (relative L1 in T_rad = %.3f%%)",
                                  nx, 100err),
              plot_titlevspan=0.09, bottom_margin=5Plots.mm,
              left_margin=5Plots.mm), "radshock.png")
end

# --- 5. Shadow: what M1 buys over flux-limited diffusion --------------------
function figure_shadow()
    sim, tmax = P.radiation_shadow()
    run!(sim; tmax, verbose=false)
    g = sim.grid
    E, _ = field(sim, :Erad)
    z = log10.(max.(E[:, :, 1], 1e-12))
    pl = heatmap(coordinates(g, 1), coordinates(g, 2), permutedims(z);
                 xlabel="x", ylabel="y", colorbar_title="log10 E_rad", right_margin=6Plots.mm,
                 aspect_ratio=:equal, xlims=(0, 1), ylims=(-0.2, 0.2), c=:inferno,
                 title="M1 shadow behind an opaque clump")
    # outline the clump
    th = range(0, 2pi; length=200)
    plot!(pl, 0.25 .+ 0.10 .* cos.(th), 0.06 .* sin.(th); c=:cyan, lw=1.5, label="")

    cut = plot(coordinates(g, 1), E[:, cld(g.n[2], 2), 1]; lw=2, c=1, label="behind clump (y = 0)",
               yscale=:log10, xlabel="x", ylabel="radiation energy", legend=:bottomleft)
    plot!(cut, coordinates(g, 1), E[:, 3, 1]; lw=2, c=2, label="unobstructed beam")
    save(plot(pl, cut; layout=@layout([a{0.6h}; b]), size=(760, 720)), "shadow.png")
end

# --- 6. Kelvin-Helmholtz ----------------------------------------------------
function figure_kh(n=256)
    sim, _ = P.kelvin_helmholtz(; n)
    panels = []
    for t in (0.8, 1.4, 2.0)
        run!(sim; tmax=t, verbose=false)
        push!(panels, plotstate(sim; var=:density, c=:magma, colorbar=false,
                                title=@sprintf("t = %.1f", t), xlabel="x",
                                ylabel=(t == 0.8 ? "y" : "")))
    end
    save(plot(panels...; layout=(1, 3), size=(1050, 380),
              plot_title="Kelvin-Helmholtz instability, $(n)^2, PPM + SSPRK3",
              plot_titlevspan=0.14, bottom_margin=5Plots.mm), "kh.png")
end

# --- 7. Sedov blast: spherical symmetry on a Cartesian grid -----------------
function figure_sedov(n=96)
    sim, tmax = P.sedov_blast(; n)
    run!(sim; tmax, verbose=false)
    g = sim.grid
    rho, _ = field(sim, :density)
    ic = cld(n, 2)
    sl = heatmap(coordinates(g, 1), coordinates(g, 2), permutedims(rho[:, :, ic]);
                 xlabel="x", ylabel="y", aspect_ratio=:equal, c=:viridis,
                 colorbar_title="", title="z = 0 slice", left_margin=5Plots.mm)
    # radial scatter of every cell, against the similarity radius
    r = Float64[]
    d = Float64[]
    xs, ys, zs = coordinates(g, 1), coordinates(g, 2), coordinates(g, 3)
    for k in 1:n, j in 1:n, i in 1:n
        push!(r, sqrt(xs[i]^2 + ys[j]^2 + zs[k]^2))
        push!(d, rho[i, j, k])
    end
    rp = scatter(r, d; ms=0.6, msw=0, alpha=0.15, c=1, label="all cells",
                 xlabel="radius", ylabel="density", xlims=(0, 0.5), legend=:topleft)
    vline!(rp, [1.15 * (tmax^2)^(1 / 5)]; c=:black, lw=2, ls=:dash,
           label="Sedov-Taylor radius")
    save(plot(sl, rp; layout=(1, 2), size=(1000, 430),
              plot_title=@sprintf("Sedov blast, %d^3, t = %.3f", n, tmax),
              plot_titlevspan=0.12, bottom_margin=6Plots.mm), "sedov.png")
end

# --- 8. Stromgren spheres: Iliev et al. (2006) Tests 1 and 2 --------------------
const MYR = 1e6 * Honghu.CGS.year
const KPC = 1e3 * Honghu.CGS.parsec

# The spherical average of the (n, n, n) array `q` about the source at the origin, in `nb`
# radial bins out to the box side. Returns the bin centres in kpc and the averages.
function radial_average(sim, q; nb=32)
    g = sim.grid
    xs = [coordinates(g, d) for d in 1:3]
    L = g.xhi[1] - g.xlo[1]
    s, c = zeros(nb), zeros(Int, nb)
    for k in axes(q, 3), j in axes(q, 2), i in axes(q, 1)
        b = floor(Int, sqrt(xs[1][i]^2 + xs[2][j]^2 + xs[3][k]^2) / L * nb) + 1
        b <= nb || continue
        s[b] += q[i, j, k]
        c[b] += 1
    end
    keep = c .> 0
    return ((1:nb) .- 0.5)[keep] .* (L / nb / KPC), (s./max.(c, 1))[keep]
end

# The radius history against the two references, and spherically averaged profiles. Prints
# the radii at the paper's output times, which the photochemistry guide tabulates; for Test 2
# also the median temperature of the ionized gas (x_HII > 0.9). Test 2 has no analytic
# solution, and its reference curves are Test 1's, for scale.
function figure_stromgren(test; n=32, chat_factor=1e-3, snaps=(10, 100, 500))
    sim, _ = P.stromgren_iliev(; test, n, chat_factor)
    x0 = test == 1 ? 1.2e-3 : 0.0
    chat = chat_factor * Honghu.CGS.c_light
    ht, hr = [0.0], [0.0]
    record(s) = (push!(ht, s.t / MYR); push!(hr, P.ionized_radius(s; x0) / KPC); true)
    profiles = []
    for tm in (10, 30, 100, 200, 500)
        run!(sim; tmax=tm * MYR, verbose=false, callback=record)
        rf = P.stromgren_radius_finite_c(tm * MYR; chat) / KPC
        ra = P.stromgren_radius_analytic(tm * MYR) / KPC
        @printf("  Test %d, %3d Myr: r_I = %.3f kpc, finite chat %.3f (%+.1f%%), chat = c %.3f (%+.1f%%)",
                test, tm, hr[end], rf, 100 * (hr[end] / rf - 1), ra, 100 * (hr[end] / ra - 1))
        T = gas_temperature(sim)
        if test == 2
            Tion = sort(T[first(field(sim, :X_2)).>0.9])
            @printf(", median T(x > 0.9) = %.0f K", Tion[cld(length(Tion), 2)])
        end
        println()
        if tm in snaps
            r, xHI = radial_average(sim, first(field(sim, :X_1)))
            push!(profiles, (tm, r, xHI, radial_average(sim, T)[2]))
        end
    end
    println("  Test $test: $(sim.diag.chem_failures) failed chemistry solves")

    tg = range(0.5, 500; length=200)
    pre = test == 1 ? "" : "Test 1, "
    pr = plot(tg, [P.stromgren_radius_analytic(t * MYR) / KPC for t in tg]; c=:black, lw=1.5,
              label=pre * "analytic", xlabel="t (Myr)", ylabel="ionized radius (kpc)",
              legend=:bottomright)
    plot!(pr, tg, [P.stromgren_radius_finite_c(t * MYR; chat) / KPC for t in tg]; c=:black,
          ls=:dash, label=pre * "finite chat")
    plot!(pr, ht, hr; c=1, lw=2, label="Honghu")
    px = plot(; xlabel="r (kpc)", ylabel="neutral fraction", yscale=:log10, ylims=(1e-5, 2),
              legend=:bottomright)
    for (m, (tm, r, xHI, _)) in enumerate(profiles)
        plot!(px, r, max.(xHI, 1e-5); c=m, lw=1.5, label="$tm Myr")
    end
    panels = Any[pr, px]
    if test == 2
        pT = plot(; xlabel="r (kpc)", ylabel="temperature (K)", yscale=:log10, legend=:topright)
        for (m, (tm, r, _, T)) in enumerate(profiles)
            plot!(pT, r, T; c=m, lw=1.5, label="$tm Myr")
        end
        push!(panels, pT)
    end
    save(plot(panels...; layout=(1, length(panels)), size=(470 * length(panels), 400),
              plot_title=@sprintf("Iliev et al. Test %d, %d^3 octant, chat = %g c", test, n,
                                  chat_factor),
              plot_titlevspan=0.08, bottom_margin=5Plots.mm, left_margin=8Plots.mm),
         "stromgren_test$test.png")
end

# Names on the command line select figures; none selects all of them.
want(name) = isempty(ARGS) || name in ARGS

println("writing figures to $OUT")
want("sod") && figure_sod()
want("convergence") && figure_convergence()
want("diffusion") && figure_diffusion()
want("marshak") && figure_marshak()
want("radshock") && figure_radshock()
want("shadow") && figure_shadow()
want("kh") && figure_kh()
want("sedov") && figure_sedov()
for t in (1, 2)          # `stromgren` makes both, `stromgren1` or `stromgren2` one
    (want("stromgren") || "stromgren$t" in ARGS) && figure_stromgren(t)
end
println("done")
