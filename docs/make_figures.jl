#!/usr/bin/env julia
#
# Regenerate the verification figures embedded in README.md.
#
#   julia --project=. -t auto docs/make_figures.jl
#
# Every panel compares the code against something independent: an exact Riemann
# solution, an analytic diffusion solution, a measured convergence slope, or a
# similarity solution. Takes a couple of minutes.

ENV["GKSwstype"] = "100"          # headless GR

using Hydro3D
using Plots
using Printf
const P = Hydro3D.Problems

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
        scatter!(pl, x, num; ms=2, msw=0, c=1, label=idx == 1 ? "Hydro3D (PPM, $nx cells)" : "")
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
    scatter!(full, x, Tg; ms=2.2, msw=0, c=1, label="T_matter, Hydro3D")
    scatter!(full, x, Tr; ms=2.2, msw=0, c=2, label="T_radiation, Hydro3D")

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

println("writing figures to $OUT")
figure_sod()
figure_convergence()
figure_diffusion()
figure_marshak()
figure_radshock()
figure_shadow()
figure_kh()
figure_sedov()
println("done")
