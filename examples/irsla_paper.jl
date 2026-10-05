# Every number and data file behind the numerical tests of Krumholz, Wibking &
# He, "An Improved Reduced Speed of Light Approximation" (Sec. 5), except the
# linear waves, which are `linear_waves_rhd.jl`.
#
#     julia --project=. -t 4 examples/irsla_paper.jl --outdir=DIR [section ...]
#
# Sections (all by default): checkerboard shock_super shock_sub vaytet pulse
# diffusion2d diffusion_thick timing. Each writes CSV files into DIR and prints a summary.
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
     "diffusion_thick", "timing"] : OPTS

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

"Run to `tmax`; return the gas temperature, the local phi and the cost."
function score(sim, tmax)
    t = @elapsed run!(sim; tmax, verbose=false)
    g = sim.grid
    I = (g.lo[1]:g.hi[1], g.lo[2]:g.hi[2], g.lo[3]:g.hi[3])
    phi = sim.rw.chat[1, I...] ./ sim.radiation.c
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
        c = score(first(vay(chat_over_c=phi0, chat_limiter=StabilityBound(per_group=false),
                            cfl_radiation=cr_m1(1))), tmax)
        push!(rows, (phi0, l1(a.T, ref.T), l1(b.T, ref.T), l1(c.T, ref.T), maximum(b.T),
                     a.sub, b.sub))
        @printf("  phi0 = %.3f   RSLA %.3f%%   IRSLA %.3f%%   IRSLA per cell %.3f%%\n", phi0,
                100l1(a.T, ref.T), 100l1(b.T, ref.T), 100l1(c.T, ref.T))
        phi0 == 0.01 && (prof["a"] = a; prof["b"] = b; prof["c"] = c)
    end
    writecsv("vaytet_scores.csv", ("phi0", "L1_rsla", "L1_irsla", "L1_irsla_percell", "Tmax_irsla",
                                   "sub_rsla", "sub_irsla"), rows)
    xs = collect(range(20 / 128, 20 - 20 / 128, length=64))
    writecsv("vaytet_profiles.csv", ("x", "T_ref", "T_rsla", "T_irsla", "T_irsla_percell", "phi_irsla"),
             [(xs[i], ref.T[i], prof["a"].T[i], prof["b"].T[i], prof["c"].T[i], prof["b"].phi[i])
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
        for (name, lim, wc) in (("rsla", ConstantChat(), false), ("irsla", StabilityBound(), false),
                                ("irsla_wc", StabilityBound(), true),
                                ("irsla_estep", StabilityBound(energy_step=true), false))
            r = score(radhydro_pulse_dyn(chat_over_c=phi0, chat_limiter=lim,
                                         wavespeed_correction=wc, cfl_radiation=CR_P1)...)
            push!(rows, (phi0, name, lp(r.T, ref.T), minimum(r.phi), maximum(r.phi), r.sub))
            @printf("  phi0 = %.2f %-12s L1 = %8.3f%%  phi in [%.3f, %.3f]  %d substeps\n",
                    phi0, name, 100lp(r.T, ref.T), extrema(r.phi)..., r.sub)
            phi0 == 0.1 && (prof[name] = r)
        end
    end
    # chat = c at the RSLA steps of phi0 = 0.1 and 0.01
    for cfl in (CR_P1 / 0.1, CR_P1 / 0.01)
        r = score(radhydro_pulse_dyn(cfl_radiation=cfl, wavespeed_correction=true)...)
        push!(rows, (1.0, "exact_wc_cfl$(cfl)", lp(r.T, ref.T), 1.0, 1.0, r.sub))
        @printf("  chat = c, correction on, cfl = %4.1f   L1 = %.3f%%\n", cfl, 100lp(r.T, ref.T))
    end
    push!(rows, (1.0, "reference", 0.0, 1.0, 1.0, ref.sub))
    writecsv("pulse_scores.csv", ("phi0", "run", "L1", "phi_min", "phi_max", "substeps"), rows)
    xs = collect(range(-512 + 4, 512 - 4, length=128))
    writecsv("pulse_profiles.csv", ("x", "T_ref", "T_rsla", "T_irsla", "T_irsla_wc", "phi_irsla"),
             [(xs[i], ref.T[i], prof["rsla"].T[i], prof["irsla"].T[i], prof["irsla_wc"].T[i],
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

# ---------------------------------------------------------------------------
if "diffusion_thick" in SECTIONS
    println("\n== thermal diffusion in optically thick, gas-dominated cells (tau_cell = 20) ==")
    # The regime the IRSLA is for: every cell is thick and eq:rich holds
    # (4 a T^3/(rho c_v) <= 0.008 < phi0/(nu0 sigma - phi0) = 0.011 for M1 in 1D at
    # nu0 = 0.9), so phi = 1 everywhere, while the RSLA raises the radiation's
    # heat capacity by 1/phi0 and slows the diffusion of the hot spot.
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
