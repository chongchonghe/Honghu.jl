# Linear radiation-hydrodynamic waves, measured in the code and compared with
# linear theory: the phase speed and decay rate of the acoustic and entropy
# modes, for the unmodified equations (chat = c), the standard RSLA and the
# improved RSLA, across optical depth.
#
#     julia --project=. -t 4 examples/linear_waves_rhd.jl --outdir=DIR
#
# Units: rho = T = 1 and an isothermal sound speed of one, so that the
# dimensionless numbers of the linear analysis are read off directly:
# c = calC, a_rad = calP, and a wave of wavenumber k = 2 pi in a unit periodic
# box has optical depth tau = chi / k per radian. The wave is initialised as an
# exact eigenmode of the linearised system, and its complex frequency is
# measured from the evolution of its Fourier amplitude.
#
# Writes `linear_waves_rhd.csv`: one row per (tau, scheme, mode) with the
# measured and predicted Omega = omega/k.

include("common.jl")
using Hydro3D
using Printf
using LinearAlgebra: eigen

const CAL_C = 1.0e3
const CAL_P = 1.0e-2
const GAMMA = 5 / 3
const PHI0 = 0.03
const NCELL = 64

"The linearised system of the paper at M = 0 (Sec. 2.3): A + i tau Stilde."
function wave_matrix(tau, phi; C=CAL_C, P=CAL_P, gamma=GAMMA)
    g1 = gamma - 1
    A = ComplexF64[0 1 0 0 0
                   1 0 1 0 0
                   0 g1 0 0 0
                   0 0 0 0 phi*C
                   0 0 0 phi*C/3 0]
    S = ComplexF64[0 0 0 0 0
                   0 -4P/(3C) 0 0 P
                   0 0 -4C*P*g1 C*P*g1 0
                   0 0 4phi*C -phi*C 0
                   0 4phi/3 0 0 -phi*C]
    return A + im * tau * S
end

"""
Eigenvalue and eigenvector of `mode` (:acoustic, the right-going sound wave, or
:entropy) at optical depth `tau`, followed by continuity from `tau = 1e-4`,
where the modes are unambiguous.
"""
function eigenmode(tau, phi, mode)
    taus = exp.(range(log(1e-4), log(tau), length=400))
    F = eigen(wave_matrix(taus[1], phi))
    target = mode === :acoustic ? sqrt(GAMMA) : 0.0
    # at low tau the matter modes are the three with |Re| of order unity
    k = argmin(abs.(F.values .- target) .+ 1e3 .* (abs.(real.(F.values)) .> 10))
    lam = F.values[k]
    vec = F.vectors[:, k]
    for t in taus[2:end]
        F = eigen(wave_matrix(t, phi))
        k = argmin(abs.(F.values .- lam) ./ max(abs(lam), 1e-12))
        lam, vec = F.values[k], F.vectors[:, k]
    end
    return lam, vec
end

"The local phi the limiter picks in the uniform background at this step."
function background_phi(lim, rad, dt, tau, k)
    chi = tau * k
    dx = 1 / NCELL
    hrat = 4 * rad.a_rad / (1 / (GAMMA - 1))      # 4 a T^3 / (rho c_v)
    w = Hydro3D.LocalCell(rad.c * dt, sqrt(1 / 3) / dx, chi, chi, hrat, dx)
    return Hydro3D.chat_factor(lim, rad.chat / rad.c, w)
end

"Run one wave and return the measured and predicted Omega."
function measure(tau, scheme, mode; amp=1e-5)
    k = 2pi
    chi = tau * k
    lim = scheme === :irsla ? StabilityBound() : ConstantChat()
    phi0 = scheme === :exact ? 1.0 : PHI0
    rad = RadiationPhysics(; c=CAL_C, chat=phi0 * CAL_C, a_rad=CAL_P, beta_order=1,
                           closure=EddingtonClosure(), opacity=ConstantOpacity(chi),
                           chat_limiter=lim,
                           # the per-step change of a 1e-5 wave at the true
                           # speed of light's step is ~1e-9 of the gas energy,
                           # below the default tolerance of the coupling solve
                           tolerance=1e-14)
    # The RSLA and IRSLA take 0.9 times the RSLA's largest stable step for the P1
    # closure, nu0 = 0.9 sqrt(3) (paper, Proposition 2); the chat = c run keeps 0.4.
    cfl = scheme === :exact ? 0.4 : 0.9 * sqrt(3)
    sim = Simulation(; n=(NCELL, 1, 1), eos=IdealGas(; gamma=GAMMA, cv=1 / (GAMMA - 1)),
                     radiation=rad, boundaries=Boundaries(Periodic()), cfl_radiation=cfl)
    dt = Hydro3D.radiation_timestep(sim.radiation, sim.grid, sim.cfl_radiation)
    phi = scheme === :irsla ? background_phi(lim, rad, dt, tau, k) : phi0
    Om, v = eigenmode(tau, phi, mode)
    v = v / v[1]                                   # normalise to the density
    # about three e-folds of decay, or two sound crossings if it decays slowly
    tend = min(2.0, 3 / (k * abs(imag(Om))))
    set_initial_conditions!(sim) do x, y, z
        e = amp * exp(im * k * x)
        (rho=1 + real(v[1] * e), vx=real(v[2] * e), T=1 + real(v[3] * e),
         Erad=CAL_P * (1 + real(v[4] * e)), Fx=CAL_C * CAL_P * real(v[5] * e))
    end
    g = sim.grid
    xs = [(i - g.lo[1] + 0.5) / NCELL for i in g.lo[1]:g.hi[1]]
    coef() = sum((sim.U[1, i, 1, 1] - 1) * exp(-im * k * xs[i-g.lo[1]+1])
                 for i in g.lo[1]:g.hi[1]) / NCELL
    ts = Float64[]
    cs = ComplexF64[]
    nsteps = ceil(Int, tend / dt)
    every = max(1, nsteps ÷ 40)
    for s in 1:nsteps
        step!(sim, dt)                              # one radiation substep per step
        if s % every == 0
            push!(ts, sim.t)
            push!(cs, coef())
        end
    end
    # c(t) = c0 exp(-i Omega k t): fit log|c| and the unwrapped phase linearly
    lr = log.(abs.(cs))
    ph = cs[1] |> angle
    phs = [ph]
    for c in cs[2:end]
        d = angle(c) - phs[end]
        d -= 2pi * round(d / 2pi)
        push!(phs, phs[end] + d)
    end
    slope(y) = (n = length(ts); tm = sum(ts) / n; ym = sum(y) / n;
                sum((ts .- tm) .* (y .- ym)) / sum((ts .- tm) .^ 2))
    Om_meas = (-slope(phs) + im * slope(lr)) / k
    # the chat actually used, to confirm the limiter picked the phi assumed
    phi_used = sim.rw.chat[1, g.lo[1], 1, 1] / CAL_C
    return (; Om_meas, Om_theory=Om, phi, phi_used)
end

rows = String["tau,scheme,mode,phi,Re_meas,Im_meas,Re_theory,Im_theory"]
println("Linear waves: calC = $CAL_C, calP = $CAL_P, phi0 = $PHI0, $NCELL cells per wavelength")
@printf("%8s %6s %9s %8s %22s %22s\n", "tau", "scheme", "mode", "phi", "Omega measured", "Omega theory")
for tau in (0.1, 1.0, 10.0, 100.0, 1000.0), mode in (:acoustic, :entropy),
    scheme in (:exact, :rsla, :irsla)
    r = measure(tau, scheme, mode)
    @printf("%8.1e %6s %9s %8.4f %10.5f%+10.5fi %10.5f%+10.5fi\n", tau, scheme, mode, r.phi,
            real(r.Om_meas), imag(r.Om_meas), real(r.Om_theory), imag(r.Om_theory))
    abs(r.phi_used - r.phi) < 1e-5 * r.phi ||
        @warn "limiter used phi = $(r.phi_used), not the assumed $(r.phi)"
    push!(rows, join((tau, scheme, mode, r.phi, real(r.Om_meas), imag(r.Om_meas),
                      real(r.Om_theory), imag(r.Om_theory)), ","))
end
open(outpath("linear_waves_rhd.csv"), "w") do io
    foreach(l -> println(io, l), rows)
end
println("  wrote ", outpath("linear_waves_rhd.csv"))
