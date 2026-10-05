# Every number and data file behind the numerical tests of Krumholz, Wibking &
# He, "An Improved Reduced Speed of Light Approximation" (Sec. 5), except the
# linear waves, which are `linear_waves_rhd.jl`.
#
#     julia --project=. -t 4 examples/irsla_paper.jl --outdir=DIR [section ...]
#
# Sections (all by default): checkerboard shock_super shock_sub vaytet pulse
# diffusion2d diffusion_thick timing equal_accuracy ensman ensman_floored
# diffusion_thick_clipped. Each writes CSV files
# into DIR and prints a summary. `phi` in every table is the reduction factor of
# the first group at the end of the run, from the last full step.
#
# Every error is a relative L1 error in the gas temperature against a
# reference run at chat = c at the same resolution, and every IRSLA run takes
# the same steps as the RSLA it is compared with unless a row says otherwise.

include("common.jl")
using Hydro3D
using Hydro3D.Problems
using Printf
using Random: Xoshiro, rand

const SECTIONS = isempty(OPTS) ?
    ["checkerboard", "shock_super", "shock_sub", "vaytet", "pulse", "diffusion2d",
     "diffusion_thick", "timing", "equal_accuracy", "ensman", "ensman_floored",
     "diffusion_thick_clipped"] : OPTS

l1(T, ref) = sum(abs, T .- ref) / sum(abs, ref)

# The RSLA and IRSLA runs take beta = 0.9 times the largest stable step of the
# standard RSLA (paper, Proposition 2), so that the IRSLA is compared with the RSLA
# at its best. In Hydro3D's CFL number, C_r = D nu0, with nu0 <= 1/max(sum_d sigma_d):
# max sum_d sigma_d = sqrt(D) for the M1 closure and D/sqrt(3) for P1 (Eddington).
# The chat = c reference runs keep the problems' own, more conservative steps.
const BETA = 0.9
const CR_P1 = BETA * sqrt(3)          # any D: D * BETA * sqrt(3)/D
cr_m1(D) = BETA * sqrt(D)             # D * BETA/sqrt(D)

function writecsv(name, header, rows)
    open(outpath(name), "w") do io
        println(io, join(header, ","))
        for r in rows
            println(io, join(r, ","))
        end
    end
    println("  wrote ", outpath(name))
end

"""Run to `tmax`; return the gas temperature, the local phi at the end of the run
and the cost. `run!` cuts the last step short to land on `tmax`, and the limiter
sets phi from the step actually taken, so phi is taken from the last full step
(whichever of the last two steps is longer)."""
function score(sim, tmax)
    g = sim.grid
    I = (g.lo[1]:g.hi[1], g.lo[2]:g.hi[2], g.lo[3]:g.hi[3])
    cur = zeros(size(sim.rw.chat[1, I...])); prev = copy(cur)
    dts = [0.0, 0.0]                                   # dt of prev, cur
    cb = s -> (prev .= cur; cur .= view(s.rw.chat, 1, I...); dts .= (dts[2], s.dt); true)
    t = @elapsed run!(sim; tmax, verbose=false, callback=cb)
    phi = (dts[2] >= dts[1] ? cur : prev) ./ sim.radiation.c
    (T=vec(gas_temperature(sim)), Tr=vec(radiation_temperature(sim)), phi=vec(phi),
     steps=sim.diag.steps, sub=sim.diag.rad_substeps, wall=t, sim=sim)
end

# The limiter of the earlier drafts, as corrected in 2026-08: the transport /
# source channel with tau = chi dx_min and the mean signal speed, plus the
# radiation-force cap -- emulated from the local cell data, in the flux form. `D` is the number of
# active dimensions of a square grid (it needs sum_d 1/dx_d = D/dx_min).
struct EarlierBound <: Hydro3D.AbstractChatLimiter
    beta::Float64
    D::Int
    flux::Bool
end
EarlierBound(; beta=0.9, D=1, flux=true) = EarlierBound(beta, D, flux)
Hydro3D._convert_limiter(l::EarlierBound, ::Type{T}) where {T} = l
Hydro3D.flux_form(l::EarlierBound) = l.flux
function Hydro3D.chat_factor(l::EarlierBound, phi0, w::Hydro3D.LocalCell)
    nu0 = phi0 * w.cdt * l.D / w.dxmin
    sig = w.sdx * w.dxmin / l.D                    # the mean of the directions' speeds
    den = 2sig - w.chiE * w.dxmin
    ptau = den > 0 ? (2phi0 / nu0) / den : Inf
    # radiation-force cap phi0/(2 nu0 P_rad/P_gas); for gamma = 5/3 and a gas at
    # radiative equilibrium P_rad/P_gas = a T^4/((gamma-1) rho c_v T) = (3/8) hrat
    prat = 3 * w.hrat / 8
    pP = prat > 0 ? phi0 / (2 * nu0 * prat) : Inf
    return min(1.0, max(phi0, l.beta * min(ptau, pP)))
end

# ---------------------------------------------------------------------------
if "checkerboard" in SECTIONS
    println("\n== checkerboard: the bound is exact in 1, 2 and 3 dimensions ==")
    struct FixedPhi <: Hydro3D.AbstractChatLimiter
        phi::Float64
    end
    Hydro3D.chat_factor(l::FixedPhi, phi0, w::Hydro3D.LocalCell) = l.phi
    Hydro3D._convert_limiter(l::FixedPhi, ::Type{T}) where {T} = l

    function cb_growth(lim; D, n=16, phi0=1e-3, tau, Pcal, nst=60, nmeas=20)
        cv, gamma = 1.0, 5 / 3
        a_rad = Pcal * (gamma - 1) * cv
        dx = 1 / n
        # A tight coupling tolerance: at P = 1e-6 the perturbation is ~1e-10 of the
        # gas energy, below the default, and near the bound its damping is done by
        # the implicit solve alone.
        rad = RadiationPhysics(; c=1.0, chat=phi0, a_rad, beta_order=0,
                               opacity=ConstantOpacity(tau / dx), chat_limiter=lim,
                               tolerance=1e-14)
        sim = Simulation(; n=ntuple(d -> d <= D ? n : 1, 3), eos=IdealGas(; gamma, cv),
                         radiation=rad, hydro=false, boundaries=Boundaries(Periodic()),
                         cfl_radiation=0.4, radiation_reconstruction=PCM())
        cell(x) = floor(Int, x / dx)
        set_initial_conditions!(sim) do x, y, z
            s = iseven(cell(x) + (D > 1) * cell(y) + (D > 2) * cell(z)) ? 1.0 : -1.0
            (rho=1.0, T=1.0, Erad=a_rad * (1 + 1e-4 * s))
        end
        g = sim.grid
        I = (g.lo[1]:g.hi[1], g.lo[2]:g.hi[2], g.lo[3]:g.hi[3])
        sgn = [(-1.0)^((i - g.lo[1]) + (D > 1) * (j - g.lo[2]) + (D > 2) * (k - g.lo[3]))
               for i in I[1], j in I[2], k in I[3]]
        proj() = abs(sum(sim.R[1, I...] .* sgn))
        dt = Hydro3D.radiation_timestep(sim.radiation, g, sim.cfl_radiation)
        a = Float64[proj()]
        for s in 1:nst
            step!(sim, dt)
            push!(a, proj())
            (!isfinite(a[end]) || a[end] > 100 * a[1]) && break
        end
        # A growing mode is measured over the last linear stretch before it has
        # grown a hundredfold; a decaying one over steps 4 to 16, after the
        # initial transient and before it reaches the floor set by the neutral
        # thermal mode.
        if a[end] > 1.5 * a[1] || !isfinite(a[end])
            s2 = findlast(isfinite, a)
            s2 = findlast(x -> x < 100 * a[1], a[1:s2])
            s1 = max(1, s2 - 8)
            return (a[s2] / a[s1])^(1 / (s2 - s1)), dt
        end
        return (a[17] / a[5])^(1 / 12), dt
    end
    rows = []
    for D in (1, 2, 3), (tau, Pcal) in ((0.3, 1e-6), (1.0, 1e-6), (1.0, 1e-1))
        n, phi0 = 16, 1e-3
        dt = 0.4 / (phi0 * D * n)
        chi = tau * n
        K = 4 * Pcal * (2 / 3) * dt * chi
        phib = 1 / (dt * (D * sqrt(1 / 3) * n - chi / (2 + K)))
        for fac in (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2)
            g, _ = cb_growth(FixedPhi(fac * phib); D, n, tau, Pcal)
            push!(rows, (D, tau, Pcal, "fixed", fac, fac * phib, g))
        end
        for (name, lim) in (("new", StabilityBound()), ("new_beta1", StabilityBound(beta=1.0)),
                            ("earlier", EarlierBound(; D)))
            g, dt = cb_growth(lim; D, n, tau, Pcal)
            push!(rows, (D, tau, Pcal, name, NaN, NaN, g))
            @printf("  D=%d tau=%.1f P=%.0e %-10s g = %.5f\n", D, tau, Pcal, name, g)
        end
    end
    writecsv("checkerboard.csv", ("D", "tau", "Pcal", "limiter", "fac", "phi", "g"), rows)
end

# ---------------------------------------------------------------------------
if "shock_super" in SECTIONS
    println("\n== supercritical radiative shock (Mach 5, phi0 = 0.025) ==")
    ref = score(radhydro_shock_super(chat_over_c=1.0)...)
    xs = [x for x in range(-0.04 + 0.05 / 256, 0.01 - 0.05 / 256, length=128)]
    runs = [("rsla", ConstantChat()), ("irsla", StabilityBound()),
            ("irsla_flux", StabilityBound(flux_form=true)),
            ("earlier", EarlierBound())]
    res = Dict(name => score(radhydro_shock_super(chat_limiter=lim, cfl_radiation=CR_P1)...)
               for (name, lim) in runs)
    @printf("  reference: %d substeps\n", ref.sub)
    for (name, _) in runs
        r = res[name]
        @printf("  %-11s L1 = %7.4f%%   phi in [%.4f, %.4f]   %d substeps\n", name,
                100l1(r.T, ref.T), extrema(r.phi)..., r.sub)
    end
    writecsv("shock_super_profiles.csv",
             ("x", "T_ref", "Tr_ref", "T_rsla", "Tr_rsla", "T_irsla", "Tr_irsla", "phi_irsla"),
             [(xs[i], ref.T[i], ref.Tr[i], res["rsla"].T[i], res["rsla"].Tr[i],
               res["irsla"].T[i], res["irsla"].Tr[i], res["irsla"].phi[i]) for i in eachindex(xs)])
    rows = [(name, l1(res[name].T, ref.T), res[name].sub) for (name, _) in runs]
    push!(rows, ("reference", 0.0, ref.sub))
    # beta scan
    for beta in (0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.5)
        r = score(radhydro_shock_super(chat_limiter=StabilityBound(; beta), cfl_radiation=CR_P1)...)
        e = l1(r.T, ref.T)
        push!(rows, ("beta=$beta", isfinite(e) ? e : Inf, r.sub))
        @printf("  beta = %.1f   L1 = %.4f%%\n", beta, 100e)
    end
    writecsv("shock_super_scores.csv", ("run", "L1", "substeps"), rows)
    # resolution
    rows = []
    for n in (64, 128, 256, 512)
        rr = score(radhydro_shock_super(; n, chat_over_c=1.0)...)
        a = score(radhydro_shock_super(; n, cfl_radiation=CR_P1)...)
        b = score(radhydro_shock_super(; n, chat_limiter=StabilityBound(), cfl_radiation=CR_P1)...)
        push!(rows, (n, l1(a.T, rr.T), l1(b.T, rr.T), a.sub, b.sub))
        @printf("  n = %3d   RSLA %.4f%%   IRSLA %.4f%%\n", n, 100l1(a.T, rr.T), 100l1(b.T, rr.T))
    end
    writecsv("shock_super_resolution.csv", ("n", "L1_rsla", "L1_irsla", "sub_rsla", "sub_irsla"), rows)
end

# ---------------------------------------------------------------------------
if "shock_sub" in SECTIONS
    println("\n== subcritical radiative shock (Mach 3) ==")
    rows = []
    ref = score(radhydro_shock(n=256, chat_factor=433.0)...)
    for cf in (10.0, 3.0, 1.0)
        phi0 = cf * 4 / sqrt(3e6)
        a = score(radhydro_shock(n=256, chat_factor=cf, cfl_radiation=CR_P1)...)
        b = score(radhydro_shock(n=256, chat_factor=cf, chat_limiter=StabilityBound(),
                                 cfl_radiation=CR_P1)...)
        push!(rows, (phi0, l1(a.T, ref.T), l1(b.T, ref.T), a.sub, b.sub))
        @printf("  phi0 = %.4f   RSLA %.4f%%   IRSLA %.4f%%\n", phi0, 100l1(a.T, ref.T),
                100l1(b.T, ref.T))
    end
    writecsv("shock_sub.csv", ("phi0", "L1_rsla", "L1_irsla", "sub_rsla", "sub_irsla"), rows)
end

# ---------------------------------------------------------------------------
if "vaytet" in SECTIONS
    println("\n== multigroup Marshak wave (Vaytet et al. 2011), 64 cells, 4 groups ==")
    vay(; kw...) = marshak_vaytet(; n=64, ngroups=4, kw...)
    tmax = last(vay())
    ref = score(first(vay()), tmax)
    rows = []
    prof = Dict{String,Any}()
    for phi0 in (0.003, 0.01, 0.03)
        a = score(first(vay(chat_over_c=phi0, cfl_radiation=cr_m1(1))), tmax)
        b = score(first(vay(chat_over_c=phi0, chat_limiter=StabilityBound(),
                            cfl_radiation=cr_m1(1))), tmax)
        push!(rows, (phi0, l1(a.T, ref.T), l1(b.T, ref.T), maximum(b.T), a.sub, b.sub))
        @printf("  phi0 = %.3f   RSLA %.3f%%   IRSLA %.3f%%\n", phi0,
                100l1(a.T, ref.T), 100l1(b.T, ref.T))
        phi0 == 0.01 && (prof["a"] = a; prof["b"] = b)
    end
    writecsv("vaytet_scores.csv", ("phi0", "L1_rsla", "L1_irsla", "Tmax_irsla", "sub_rsla",
                                   "sub_irsla"), rows)
    xs = collect(range(20 / 128, 20 - 20 / 128, length=64))
    writecsv("vaytet_profiles.csv", ("x", "T_ref", "T_rsla", "T_irsla", "phi_irsla"),
             [(xs[i], ref.T[i], prof["a"].T[i], prof["b"].T[i], prof["b"].phi[i])
              for i in eachindex(xs)])
end

# ---------------------------------------------------------------------------
if "pulse" in SECTIONS
    println("\n== advecting pulse in the dynamic diffusion limit ==")
    lp(T, ref) = sum(abs, T .- ref) / sum(abs, ref .- 1e7)     # relative to the pulse
    ref = score(radhydro_pulse_dyn()...)
    rows = []
    prof = Dict{String,Any}()
    for phi0 in (0.1, 0.01)
        for (name, lim) in (("rsla", ConstantChat()), ("irsla", StabilityBound()),
                            ("irsla_estep", StabilityBound(energy_step=true)))
            r = score(radhydro_pulse_dyn(chat_over_c=phi0, chat_limiter=lim,
                                         cfl_radiation=CR_P1)...)
            push!(rows, (phi0, name, lp(r.T, ref.T), minimum(r.phi), maximum(r.phi), r.sub))
            @printf("  phi0 = %.2f %-12s L1 = %8.3f%%  phi in [%.3f, %.3f]  %d substeps\n",
                    phi0, name, 100lp(r.T, ref.T), extrema(r.phi)..., r.sub)
            phi0 == 0.1 && (prof[name] = r)
        end
    end
    push!(rows, (1.0, "reference", 0.0, 1.0, 1.0, ref.sub))
    writecsv("pulse_scores.csv", ("phi0", "run", "L1", "phi_min", "phi_max", "substeps"), rows)
    xs = collect(range(-512 + 4, 512 - 4, length=128))
    writecsv("pulse_profiles.csv", ("x", "T_ref", "T_rsla", "T_irsla", "T_irsla_estep", "phi_irsla"),
             [(xs[i], ref.T[i], prof["rsla"].T[i], prof["irsla"].T[i], prof["irsla_estep"].T[i],
               prof["irsla"].phi[i]) for i in eachindex(xs)])
end

# ---------------------------------------------------------------------------
if "diffusion2d" in SECTIONS
    println("\n== two-dimensional thermal diffusion at tau_cell = 0.5 ==")
    # A 1e-3 random perturbation of the temperature seeds every mode, as any
    # real flow does; without it a smooth start takes no notice of a grid-scale
    # instability for a long time.
    function blob(; n=64, phi0=0.01, lim=ConstantChat(), tmax=30.0, cfl=cr_m1(2))
        dx = 1 / n
        chi = 0.5 / dx
        rad = RadiationPhysics(; c=1.0, chat=phi0, a_rad=1.0, beta_order=0,
                               opacity=ConstantOpacity(chi), chat_limiter=lim)
        sim = Simulation(; n=(n, n, 1), eos=IdealGas(; gamma=5 / 3, cv=100.0), radiation=rad,
                         hydro=false, boundaries=Boundaries(Periodic()), cfl_radiation=cfl)
        rng = Xoshiro(42)
        set_initial_conditions!(sim) do x, y, z
            T = (1 + exp(-((x - 0.5)^2 + (y - 0.5)^2) / (2 * 0.08^2))) *
                (1 + 1e-3 * (2 * rand(rng) - 1))
            (rho=1.0, T=T, Erad=T^4)
        end
        score(sim, tmax)
    end
    ref = blob(phi0=1.0, cfl=0.4)
    rows = []
    for (name, lim) in (("rsla", ConstantChat()), ("irsla", StabilityBound()),
                        ("earlier", EarlierBound(D=2)))
        r = try
            blob(lim=lim)
        catch err
            (T=fill(NaN, length(ref.T)), phi=[NaN], sub=0)
        end
        e = l1(r.T, ref.T)
        push!(rows, (name, isfinite(e) ? e : Inf, minimum(r.phi), maximum(r.phi), r.sub))
        @printf("  %-8s L1 = %9.4f%%  phi in [%.4f, %.4f]\n", name, 100e, extrema(r.phi)...)
    end
    push!(rows, ("reference", 0.0, 1.0, 1.0, ref.sub))
    writecsv("diffusion2d.csv", ("run", "L1", "phi_min", "phi_max", "substeps"), rows)
end

# A thick (tau_cell = 20), gas-dominated hot spot diffusing in a periodic box, for
# `diffusion_thick` and `diffusion_thick_clipped`.
function thick(; n=128, phi0=0.01, lim=ConstantChat(), tmax=1000.0, cfl=cr_m1(1))
    dx = 1 / n
    chi = 20 / dx
    rad = RadiationPhysics(; c=1.0, chat=phi0, a_rad=1.0, beta_order=0,
                           opacity=ConstantOpacity(chi), chat_limiter=lim)
    sim = Simulation(; n=(n, 1, 1), eos=IdealGas(; gamma=5 / 3, cv=4000.0), radiation=rad,
                     hydro=false, boundaries=Boundaries(Periodic()), cfl_radiation=cfl)
    rng = Xoshiro(42)
    set_initial_conditions!(sim) do x, y, z
        T = (1 + exp(-(x - 0.5)^2 / (2 * 0.08^2))) * (1 + 1e-3 * (2 * rand(rng) - 1))
        (rho=1.0, T=T, Erad=T^4)
    end
    score(sim, tmax)
end

# ---------------------------------------------------------------------------
if "diffusion_thick" in SECTIONS
    println("\n== thermal diffusion in optically thick, gas-dominated cells (tau_cell = 20) ==")
    # The regime the IRSLA is for: every cell is thick and eq:rich holds
    # (4 a T^3/(rho c_v) <= 0.008 < phi0/(nu0 sigma - phi0) = 0.011 for M1 in 1D at
    # nu0 = 0.9), so phi = 1 everywhere, while the RSLA raises the radiation's
    # heat capacity by 1/phi0 and slows the diffusion of the hot spot.
    ref = thick(phi0=1.0, cfl=0.4)
    rows = []
    prof = Dict{String,Any}()
    for (name, lim) in (("rsla", ConstantChat()), ("irsla", StabilityBound()))
        r = thick(lim=lim)
        e = l1(r.T, ref.T)
        prof[name] = r
        push!(rows, (name, e, minimum(r.phi), maximum(r.phi), r.sub))
        @printf("  %-8s L1 = %9.4f%%  phi in [%.4f, %.4f]  %d substeps\n", name, 100e,
                extrema(r.phi)..., r.sub)
    end
    push!(rows, ("reference", 0.0, 1.0, 1.0, ref.sub))
    writecsv("diffusion_thick.csv", ("run", "L1", "phi_min", "phi_max", "substeps"), rows)
    xs = collect(range(0.5 / 128, 1 - 0.5 / 128, length=128))
    T0 = [1 + exp(-(x - 0.5)^2 / (2 * 0.08^2)) for x in xs]
    writecsv("diffusion_thick_profiles.csv", ("x", "T_initial", "T_ref", "T_rsla", "T_irsla"),
             [(xs[i], T0[i], ref.T[i], prof["rsla"].T[i], prof["irsla"].T[i]) for i in eachindex(xs)])
end

# ---------------------------------------------------------------------------
if "timing" in SECTIONS
    println("\n== cost of the limiter ==")
    rows = []
    for (label, build) in (("shock_super_512", n -> radhydro_shock_super(; n=512)),
                           ("vaytet_256_8groups", n -> marshak_vaytet(; n=256, ngroups=8,
                                                                        chat_over_c=0.01)))
        ts = Float64[]
        for lim in (ConstantChat(), StabilityBound())
            sim, _ = label == "shock_super_512" ?
                     radhydro_shock_super(; n=512, chat_limiter=lim, cfl_radiation=CR_P1) :
                     marshak_vaytet(; n=256, ngroups=8, chat_over_c=0.01, chat_limiter=lim,
                                    cfl_radiation=cr_m1(1))
            for _ in 1:5; step!(sim); end          # compile
            nsub0 = sim.diag.rad_substeps
            t = @elapsed for _ in 1:60; step!(sim); end
            push!(ts, t / (sim.diag.rad_substeps - nsub0))
        end
        push!(rows, (label, ts[1], ts[2], ts[2] / ts[1]))
        @printf("  %-20s RSLA %.3g s/substep   IRSLA %.3g s/substep   ratio %.3f\n", label, ts..., ts[2] / ts[1])
    end
    writecsv("timing.csv", ("problem", "s_per_substep_rsla", "s_per_substep_irsla", "ratio"), rows)
end

# ---------------------------------------------------------------------------
if "equal_accuracy" in SECTIONS
    println("\n== the RSLA step count that matches the IRSLA's accuracy ==")
    # The speedup at equal accuracy: raise the RSLA's phi0 until its error equals
    # that of the IRSLA at the nominal phi0, by bisection in log(phi0), and compare
    # the radiation substeps. Both run at the same nu0, so the RSLA's step shrinks
    # as 1/phi0 (in whole substeps per hydrodynamic step).
    function match(run, err_target, lo, hi; iters=14)
        elo = run(lo).e
        ehi = run(hi).e
        (elo >= err_target >= ehi) || return (NaN, NaN, elo, ehi)
        for _ in 1:iters
            mid = sqrt(lo * hi)
            run(mid).e > err_target ? (lo = mid) : (hi = mid)
        end
        r = run(hi)
        return (hi, r.sub, r.e, ehi)
    end
    rows = []
    # supercritical shock
    let ref = score(radhydro_shock_super(chat_over_c=1.0)...)
        run(phi0; lim=ConstantChat()) = (r = score(radhydro_shock_super(chat_over_c=phi0,
                                                   chat_limiter=lim, cfl_radiation=CR_P1)...);
                                         (e=l1(r.T, ref.T), sub=r.sub))
        b = run(0.025; lim=StabilityBound())
        p, sub, e, _ = match(run, b.e, 0.025, 0.5)
        push!(rows, ("shock_super", 0.025, b.e, b.sub, p, sub, e, sub / b.sub))
    end
    # subcritical shock, at the smallest phi0
    let ref = score(radhydro_shock(n=256, chat_factor=433.0)...)
        cf(phi0) = phi0 * sqrt(3e6) / 4
        run(phi0; lim=ConstantChat()) = (r = score(radhydro_shock(n=256, chat_factor=cf(phi0),
                                                   chat_limiter=lim, cfl_radiation=CR_P1)...);
                                         (e=l1(r.T, ref.T), sub=r.sub))
        phi0 = 4 / sqrt(3e6)
        b = run(phi0; lim=StabilityBound())
        p, sub, e, _ = match(run, b.e, phi0, 0.1)
        push!(rows, ("shock_sub", phi0, b.e, b.sub, p, sub, e, sub / b.sub))
    end
    # multigroup Marshak wave
    let vay(; kw...) = marshak_vaytet(; n=64, ngroups=4, kw...)
        tmax = last(vay())
        ref = score(first(vay()), tmax)
        run(phi0; lim=ConstantChat()) = (r = score(first(vay(chat_over_c=phi0, chat_limiter=lim,
                                                         cfl_radiation=cr_m1(1))), tmax);
                                         (e=l1(r.T, ref.T), sub=r.sub))
        for phi0 in (0.003, 0.01, 0.03)
            b = run(phi0; lim=StabilityBound())
            p, sub, e, _ = match(run, b.e, phi0, min(1.0, 30phi0))
            push!(rows, ("vaytet", phi0, b.e, b.sub, p, sub, e, sub / b.sub))
        end
    end
    for r in rows
        @printf("  %-12s phi0 = %.4f  IRSLA %.3f%% in %d substeps; RSLA matches at phi0 = %.4f (%.3f%%) in %d substeps: speedup %.2f\n",
                r[1], r[2], 100r[3], r[4], r[5], 100r[7], r[6], r[8])
    end
    writecsv("equal_accuracy.csv", ("problem", "phi0", "L1_irsla", "sub_irsla", "phi0_rsla",
                                    "sub_rsla", "L1_rsla", "speedup"), rows)
end

# ---------------------------------------------------------------------------
if "ensman" in SECTIONS
    println("\n== Ensman supercritical radiative shock (Melon Fuksman et al. 2021, Sec. 4.1), M1 ==")
    # Melon Fuksman et al.'s setup -- gas at 10 K and radiation in LTE, mu = 1,
    # gamma = 7/5, no scattering, inflow at 20 km/s onto a reflecting wall, to
    # t = 7.5e3 s -- but with a constant mass opacity kappa = kfac * 3.1e-10/7.78e-10
    # cm^2/g, so that the compressed gas is thicker (they fix kappa rho, which keeps
    # every cell as thin as the upstream ones), and on the few cells a production
    # run gives a shock. The reference is chat = c at the same resolution.
    # For each RSLA phi0 the IRSLA phi0 of equal accuracy is found by bisection in
    # log(phi0), down to the phi0 at which the radiation takes one step per
    # hydrodynamic step (below it phi0 no longer changes the step): their ratio is
    # the speedup in nominal steps, and the substep ratio the one realized with
    # whole substeps per hydrodynamic step.
    function ensman(; phi0, lim=ConstantChat(), n, kfac, cfl=cr_m1(1))
        rho0, T0, mu, gamma, alpha = 7.78e-10, 10.0, 1.0, 1.4, 3.1e-10
        cv = Hydro3D.CGS.k_B / (mu * Hydro3D.CGS.m_u * (gamma - 1))
        rad = RadiationPhysics(; c=Hydro3D.CGS.c_light, chat=phi0 * Hydro3D.CGS.c_light,
                               a_rad=Hydro3D.CGS.a_rad, k_B=Hydro3D.CGS.k_B,
                               opacity=ConstantOpacity(kfac * alpha / rho0), chat_limiter=lim)
        sim = Simulation(; n=(n, 1, 1), xlo=(0.0, 0.0, 0.0), xhi=(7e10, 1.0, 1.0),
                         eos=IdealGas(; gamma, cv), radiation=rad,
                         boundaries=Boundaries((Reflecting(), Outflow()), (Periodic(), Periodic()),
                                               (Periodic(), Periodic())),
                         floors=Floors(erad=Hydro3D.CGS.a_rad * 1e-4), cfl=0.4,
                         cfl_radiation=cfl)
        set_initial_conditions!(sim) do x, y, z
            (rho=rho0, vx=-20e5, T=T0, Erad=Hydro3D.CGS.a_rad * T0^4, Fx=0.0)
        end
        return sim
    end
    tmax = 7.5e3
    rows = []
    for (n, kfac, phis) in ((128, 10.0, (0.03, 0.01, 0.001)), (64, 10.0, (0.03, 0.01)))
        local ref = score(ensman(; phi0=1.0, n, kfac, cfl=0.4), tmax)
        run(phi0, lim) = (r = score(ensman(; phi0, lim, n, kfac), tmax);
                          (e=l1(r.T, ref.T), sub=r.sub, steps=r.steps, phi=r.phi, T=r.T, Tr=r.Tr,
                           pre=r.sim.diag.floors_presolve, post=r.sim.diag.repairs_postsolve))
        # A run that fails (a non-finite error) counts as less accurate than the target.
        worse(e, target) = !(e <= target)
        function matched(lim, phi0, target, phimin)
            lo, hi = phimin, phi0
            worse(run(lo, lim).e, target) || return (lo, missing)   # better still at phimin
            for _ in 1:12
                mid = sqrt(lo * hi)
                worse(run(mid, lim).e, target) ? (lo = mid) : (hi = mid)
            end
            return (hi, run(hi, lim).sub)
        end
        for phi0 in phis
            a = run(phi0, ConstantChat())
            # one radiation step per hydrodynamic step below this phi0
            phimin = phi0 * a.steps / a.sub
            for (name, lim) in (("irsla", StabilityBound()),)
                b = run(phi0, lim)
                pm, subm = matched(lim, phi0, a.e, phimin)
                sp = phi0 / pm
                ss = subm === missing ? missing : a.sub / subm
                push!(rows, (n, kfac, phi0, name, a.e, b.e, maximum(b.phi) / phi0, pm, sp,
                             ss === missing ? NaN : ss, b.pre, b.post))
                @printf("  n = %3d  kappa x%g  phi0 = %.3f  RSLA %6.2f%%  %-12s %6.2f%%  max phi/phi0 %5.1f  matched at phi0 = %.2e: speedup %s in phi0, %s in substeps  (pre-solve floors %d, post-solve repairs %d)\n",
                        n, kfac, phi0, 100a.e, name, 100b.e, maximum(b.phi) / phi0, pm,
                        subm === missing ? ">= $(round(sp; digits=1))" : string(round(sp; digits=2)),
                        ss === missing ? "-" : string(round(ss; digits=2)), b.pre, b.post)
                if n == 64 && phi0 == 0.03 && name == "irsla"
                    # the IRSLA at the phi0 that matches the RSLA's accuracy
                    m = run(pm, lim)
                    local xs = collect(range(7e10 / 128, 7e10 - 7e10 / 128, length=64))
                    writecsv("ensman_profiles.csv", ("x", "T_ref", "Tr_ref", "T_rsla", "Tr_rsla",
                                                     "T_irsla", "Tr_irsla", "T_match", "Tr_match",
                                                     "phi_irsla", "phi_match", "phi0_match"),
                             [(xs[i], ref.T[i], ref.Tr[i], a.T[i], a.Tr[i], b.T[i], b.Tr[i],
                               m.T[i], m.Tr[i], b.phi[i], m.phi[i], pm) for i in eachindex(xs)])
                end
            end
        end
    end
    writecsv("ensman.csv", ("n", "kappa_factor", "phi0", "limiter", "L1_rsla", "L1", "max_phi_over_phi0",
                            "phi0_matched", "speedup_phi0", "speedup_substeps",
                            "floors_presolve", "repairs_postsolve"), rows)
end

# ---------------------------------------------------------------------------
if "ensman_floored" in SECTIONS
    println("\n== the Ensman shock on 64 cells with E floored before the coupling solve ==")
    # The earlier algorithm: raise E to the floor after every transport stage, even
    # where the solve could absorb a negative E. Redefines a Hydro3D method for the
    # rest of the process, so it runs after every section but the clipped one,
    # which redefines it again; the reference and the RSLA run first.
    "ensman" in SECTIONS || error("ensman_floored needs the ensman section's set-up")
    ref = score(ensman(; phi0=1.0, n=64, kfac=1.0, cfl=0.4), 7.5e3)
    a = score(ensman(; phi0=0.01, n=64, kfac=1.0), 7.5e3)
    b = score(ensman(; phi0=0.01, n=64, kfac=1.0, lim=StabilityBound()), 7.5e3)
    @eval Hydro3D function floor_rad_energy!(R::AbstractArray{T,4}, U::AbstractArray{T,4},
                                             chat::AbstractArray{T,4},
                                             phys::RadiationPhysics{NG,M,T},
                                             g::CartesianGrid{T},
                                             floors::Floors{T}) where {NG,M,T}
        ef = floors.erad / NG
        n = 0
        for k in g.lo[3]:g.hi[3], j in g.lo[2]:g.hi[2], i in g.lo[1]:g.hi[1], gr in 1:NG
            ib = gbase(gr) + IERAD
            R[ib, i, j, k] > ef || (R[ib, i, j, k] = ef; n += 1)
        end
        return n
    end
    f = Base.invokelatest(score, ensman(; phi0=0.01, n=64, kfac=1.0, lim=StabilityBound()), 7.5e3)
    rows = [("rsla", l1(a.T, ref.T)), ("irsla", l1(b.T, ref.T)), ("irsla_floored", l1(f.T, ref.T))]
    for (name, e) in rows
        @printf("  %-14s L1 = %7.2f%%\n", name, 100e)
    end
    writecsv("ensman_floored.csv", ("run", "L1"), rows)
end

# ---------------------------------------------------------------------------
if "diffusion_thick_clipped" in SECTIONS
    println("\n== the thick hot spot with |F| clipped to cE before the coupling solve ==")
    # What codes commonly do: the full realizability repair between the transport
    # stage and the implicit solve, instead of flooring E only. This redefines a
    # Hydro3D method for the rest of the process, so the section runs last; the
    # reference runs first, unclipped, and the clipped run goes through
    # `invokelatest` because this block was compiled before the redefinition.
    ref = thick(phi0=1.0, cfl=0.4)
    @eval Hydro3D function floor_rad_energy!(R::AbstractArray{T,4}, U::AbstractArray{T,4},
                                             chat::AbstractArray{T,4},
                                             phys::RadiationPhysics{NG,M,T},
                                             g::CartesianGrid{T},
                                             floors::Floors{T}) where {NG,M,T}
        amend_rad_state!(R, phys, g, floors)
    end
    r = Base.invokelatest(thick; lim=StabilityBound())
    e = l1(r.T, ref.T)
    @printf("  irsla, clipped  L1 = %9.4f%%  phi in [%.4f, %.4f]  %d substeps\n", 100e,
            extrema(r.phi)..., r.sub)
    writecsv("diffusion_thick_clipped.csv", ("run", "L1", "phi_min", "phi_max", "substeps"),
             [("irsla_clipped", e, minimum(r.phi), maximum(r.phi), r.sub)])
end
