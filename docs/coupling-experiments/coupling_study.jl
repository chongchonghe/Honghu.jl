# Every measurement in `docs/coupling-findings.md`, in one script.
#
#   julia --project=. -t 4 docs/coupling-experiments/coupling_study.jl            # all of it
#   julia --project=. -t 4 docs/coupling-experiments/coupling_study.jl cost runs  # named sections
#
# Sections: conditioning, test, robust, derivation, ode, cost, runs.
#
# Nothing here is part of the package. Two `AbstractCoupling` subtypes are
# defined below that are deliberately *not* in `src/` -- they are the candidate
# steppers the findings document argues about, and they live here so that the
# numbers can be reproduced without adding an unused solver to the tree.

using Hydro3D
using Hydro3D: AbstractCoupling, CouplingState, CouplingResult, SVec, solve_coupling,
               coupling_rhs, coupling_jacobian, gas_scale, gas_temperature_of,
               solve_jacobian, ArrowheadSolve, apply_energy_floors, newton_solve,
               backward_euler, NewtonStepper, ImplicitEulerStepper,
               maxabs, sumvec, dot, mul, divide, ngroups
using Hydro3D.Problems
using LinearAlgebra: cond
using Printf
# Selectively, as `src/Hydro3D.jl` does: a blanket `using OrdinaryDiffEq` would
# shadow this package's own `step!`.
using OrdinaryDiffEq: Rodas5P, FBDF

const P = Hydro3D.Problems

setprecision(BigFloat, 256)

# --- one cell, in code units --------------------------------------------------

function cell_state(::Type{T}=Float64; NG=4, opacity, Tgas=1.3, Trad=0.5, rho=1.0, cv=1.0,
                    dt=1e-3, c=1.0, chat=1.0, arad=1.0, kB=1.0) where {T}
    grp = NG == 1 ? Hydro3D.GREY : RadiationGroups(T(1) / T(10), T(20), NG)
    Erad = (T(arad) * T(Trad)^4) * planck_fractions(grp, T(Trad), T(kB))
    z = zero(SVec{NG,T})
    chatv = SVec{NG,T}(ntuple(_ -> T(chat), NG)...)
    CouplingState{NG,NG + 1,T,typeof(opacity)}(T(cv) * T(Tgas), Erad, z, z, T(rho), T(cv),
                                               opacity, T(dt), T(c), chatv, T(arad), T(kB),
                                               grp, T(1e-10), T(1e-30))
end

big_opacity(o::ConstantOpacity) = ConstantOpacity(BigFloat(o.planck), BigFloat(o.energy),
                                                  BigFloat(o.flux))
big_opacity(o::PowerLawOpacity) = PowerLawOpacity(BigFloat(o.kappa0), BigFloat(o.rho0),
                                                  BigFloat(o.T0), BigFloat(o.m),
                                                  BigFloat(o.n))

"""
The exact root of the backward-Euler system in 256-bit arithmetic, *verified* by
evaluating its own residual there rather than taken on trust from the family of
solvers it is used to judge.
"""
function be_reference(; NG, opacity, kw...)
    sb = cell_state(BigFloat; NG, opacity=big_opacity(opacity), kw...)
    r = solve_coupling(DirectCoupling(tolerance=BigFloat(1e-45), max_iterations=2000,
                                      stepper=NewtonStepper()), sb)
    cs = gas_scale(sb)
    scale = abs(sb.Egas + dot(cs, sb.Erad + sb.src))
    F0, Fg = _residual(sb, r.Egas, r.Erad)
    resid = max(abs(F0), maxabs(Fg)) / scale
    (Egas=r.Egas, Erad=r.Erad, scale=scale, ok=(r.converged && resid < BigFloat(1e-40)))
end

@inline function _residual(s::CouplingState{NG,M,T}, Egas, Erad) where {NG,M,T}
    cs = gas_scale(s)
    fgas, frad = coupling_rhs(s, Egas, Erad)
    ((Egas - (s.Egas - dot(cs, s.work))) - s.dt * fgas,
     (Erad - (s.Erad + s.src + s.work)) - s.dt * frad)
end

"""
Distance from the reference in units of the cell's total energy -- the scale
every solver here states its own tolerance against. A per-component *relative*
error would be dominated by whichever group holds `1e-30` of the spectrum, which
is not what any of them is trying to control.
"""
function graderr(r, ref, NG)
    e = abs(BigFloat(r.Egas) - ref.Egas)
    for g in 1:NG
        e = max(e, abs(BigFloat(r.Erad[g]) - ref.Erad[g]))
    end
    Float64(e / ref.scale)
end

# --- the candidate steppers that are not in the tree --------------------------

@inline function _blocks(s::CouplingState{NG,M,T}, x0, xg) where {NG,M,T}
    F0, Fg = _residual(s, x0, xg)
    J = coupling_jacobian(s, x0, xg)
    (F0, Fg, one(T) - s.dt * J.gas_gas, -s.dt * J.gas_rad, -s.dt * J.rad_gas,
     SVec{NG,T}(ntuple(g -> one(T) - s.dt * J.rad_rad[g], Val(NG))...))
end

"""
Backward Euler in the natural variables `(E_gas, E_g)`, stopping on a *fixed*
residual tolerance. This is `DirectCoupling(stepper = NewtonStepper())` with the
automatic tolerance degradation removed, and it is the formulation whose
round-off behaviour the findings document is about.

`test` selects the convergence criterion:

  - `:residual`  -- `max|F| <= tol * scale`, the naive one.
  - `:scaled`    -- `max|F_i / J_ii| <= tol * scale`, the residual measured in
    the units of the increment it implies.
  - `:increment` -- `max|dx| <= tol * scale`, which is what `ImplicitEuler` does.

`damp = true` adds the only globalisation a one-step Newton can have cheaply:
the gas energy is never allowed to leave `[max(E_floor, x/4), 4x]` in a step.
"""
struct PlainDirect{S} <: AbstractCoupling
    tolerance::Float64
    maxit::Int
    test::S
    damp::Bool
end
PlainDirect(; tolerance=1e-9, maxit=100, test=:residual, damp=false) =
    PlainDirect(tolerance, maxit, test, damp)

function Hydro3D.solve_coupling(m::PlainDirect, s::CouplingState{NG,M,T}) where {NG,M,T}
    atol = m.tolerance * abs(s.Egas + dot(gas_scale(s), s.Erad + s.src))
    Emin = s.cv * s.Tfloor
    x0 = s.Egas
    xg = s.Erad + s.src
    n = 0
    conv = false
    while n < m.maxit
        n += 1
        F0, Fg, J00, J0g, Jg0, Jgg = _blocks(s, x0, xg)
        if m.test === :residual && max(abs(F0), maxabs(Fg)) <= atol
            conv = true
            break
        elseif m.test === :scaled &&
               max(abs(F0 / J00), maxabs(divide(Fg, Jgg))) <= atol
            conv = true
            break
        end
        dx0, dxg = solve_jacobian(ArrowheadSolve(), J00, J0g, Jg0, Jgg, -F0, -Fg)
        lam = one(T)
        if m.damp
            lo = max(Emin, T(0.25) * x0)
            hi = 4 * x0
            while lam > T(1e-8) && (x0 + lam * dx0 < lo || x0 + lam * dx0 > hi)
                lam /= 2
            end
        end
        x0 += lam * dx0
        xg += lam * dxg
        if m.test === :increment && lam == 1 &&
           max(abs(dx0), maxabs(dxg)) <= atol
            conv = true
            break
        end
    end
    Egas, Erad = apply_energy_floors(s, x0, xg)
    CouplingResult{NG,T}(Egas, Erad, n, conv)
end

"""
The natural variables reduced to one scalar unknown, and solved with a bracket.

At a fixed gas temperature the group block of the backward-Euler system is
*linear* in `E_g`, so it can be eliminated in closed form,

    E_g(T) = (rad0_g + tau_g B_g(T)) / (1 + a_g),
    tau_g  = dt chat rho kappa_P,g(T),   a_g = dt chat rho kappa_E,g(T)

and what is left is one scalar equation -- the cell's energy balance,

    G(E_gas) = E_gas - gas0 + sum_g (c/chat_g) (E_g(T) - rad0_g) = 0 .

`G` is negative at the temperature floor and positive once `E_gas` exceeds the
cell's total energy, so a bracket exists for every cell, always. Newton inside
that bracket, replaced by a bisection step whenever it would leave it, therefore
cannot walk away from the root -- which is exactly what the unsafeguarded
`(1 + NG)`-dimensional iteration does when the opacity depends steeply on
temperature.
"""
struct BracketedDirect <: AbstractCoupling
    tolerance::Float64
    maxit::Int
end
BracketedDirect(; tolerance=1e-9, maxit=100) = BracketedDirect(tolerance, maxit)

"Group energies that solve the group block exactly at this gas energy, the
scalar residual `G`, and `dG/dE_gas`."
@inline function _reduced(s::CouplingState{NG,M,T}, Egas, gas0, rad0) where {NG,M,T}
    Tg = gas_temperature_of(s, Egas)
    B = Hydro3D.thermal_emission(s, Tg)
    dB = Hydro3D.thermal_emission_dT(s, Tg)
    kP, kE = Hydro3D.coupling_opacity(s, Tg)
    dkP, dkE = Hydro3D.coupling_opacity_dT(s, Tg)
    kdt = (s.dt * s.chat) * s.rho
    tau = mul(kdt, kP)
    aa = mul(kdt, kE)
    E = SVec{NG,T}(ntuple(g -> (rad0[g] + tau[g] * B[g]) / (one(T) + aa[g]), Val(NG))...)
    cs = gas_scale(s)
    G = Egas - gas0 + dot(cs, E - rad0)
    dtau = mul(kdt, dkP) / s.cv
    daa = mul(kdt, dkE) / s.cv
    dE = SVec{NG,T}(ntuple(Val(NG)) do g
        (dtau[g] * B[g] + tau[g] * dB[g] / s.cv - E[g] * daa[g]) / (one(T) + aa[g])
    end...)
    (E, G, one(T) + dot(cs, dE))
end

function Hydro3D.solve_coupling(m::BracketedDirect, s::CouplingState{NG,M,T}) where {NG,M,T}
    cs = gas_scale(s)
    gas0 = s.Egas - dot(cs, s.work)
    rad0 = s.Erad + s.src + s.work
    Etot = abs(s.Egas + dot(cs, s.Erad + s.src))
    atol = m.tolerance * Etot
    lo = s.cv * s.Tfloor
    hi = max(4 * Etot, 4 * s.Egas)
    x = clamp(s.Egas, lo, hi)
    E, G, dG = _reduced(s, x, gas0, rad0)
    _, Glo, _ = _reduced(s, lo, gas0, rad0)
    _, Ghi, _ = _reduced(s, hi, gas0, rad0)
    k = 0
    while Ghi < 0 && k < 60
        hi *= 4
        _, Ghi, _ = _reduced(s, hi, gas0, rad0)
        k += 1
    end
    (Glo > 0 || Ghi < 0) && return CouplingResult{NG,T}(x, E, 0, false)
    n = 0
    conv = false
    while n < m.maxit
        n += 1
        E, G, dG = _reduced(s, x, gas0, rad0)
        G < 0 ? (lo = x) : (hi = x)
        if abs(G) <= atol
            conv = true
            break
        end
        xn = dG == 0 ? T(Inf) : x - G / dG
        (xn <= lo || xn >= hi) && (xn = (lo + hi) / 2)
        if abs(xn - x) <= eps(T) * abs(x)
            x = xn
            conv = true
            break
        end
        x = xn
    end
    E, _, _ = _reduced(s, x, gas0, rad0)
    Egas, Erad = apply_energy_floors(s, x, E)
    CouplingResult{NG,T}(Egas, Erad, n, conv)
end

# --- the sweep the robustness sections use ------------------------------------

const STATES = [(1.3, 0.5), (0.5, 1.3), (1.0, 0.3), (10.0, 1.0), (1.0, 10.0),
                (1.0, 1.0001), (3.0, 0.1), (0.1, 3.0)]
const TAUS = [1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6, 1e8, 1e9, 1e10, 1e12]
const LAWS = ["constant" => k -> ConstantOpacity(k),
              "T^-3.5" => k -> PowerLawOpacity(kappa0=k, n=-3.5),
              "T^+2" => k -> PowerLawOpacity(kappa0=k, n=2.0)]
const DT = 1e-3

const METHODS = ["Quokka (R base)" => QuokkaCoupling(),
                 "natural, residual" => PlainDirect(test=:residual),
                 "natural, resid+raise" => DirectCoupling(stepper=NewtonStepper()),
                 "natural, increment" => PlainDirect(test=:increment),
                 "natural, incr+damp" => PlainDirect(test=:increment, damp=true),
                 "natural, bracketed" => BracketedDirect(),
                 "natural, ImplicitEuler" => DirectCoupling(),
                 "adaptive 1e-2" => AdaptiveCoupling(),
                 "adaptive 1e-8" => AdaptiveCoupling(reltol=1e-8, abstol=1e-14)]

# --- section: conditioning ----------------------------------------------------

function section_conditioning()
    NG = 4
    println("\n== conditioning of the Newton matrix, and the noise in the residual, NG = $NG ==")
    @printf("%8s | %12s %12s | %12s %12s\n", "tau", "cond natural", "cond Quokka",
            "resid noise", "eps*tau")
    for tau in TAUS
        s = cell_state(; NG, opacity=ConstantOpacity(tau / DT), dt=DT)
        ref = be_reference(; NG, opacity=ConstantOpacity(tau / DT), dt=DT)
        Egas = Float64(ref.Egas)
        Erad = SVec{NG,Float64}(ntuple(g -> Float64(ref.Erad[g]), NG)...)

        J = coupling_jacobian(s, Egas, Erad)
        A = zeros(NG + 1, NG + 1)
        A[1, 1] = 1 - s.dt * J.gas_gas
        for g in 1:NG
            A[1, 1+g] = -s.dt * J.gas_rad[g]
            A[1+g, 1] = -s.dt * J.rad_gas[g]
            A[1+g, 1+g] = 1 - s.dt * J.rad_rad[g]
        end

        Tg = Egas / s.cv
        B = Hydro3D.thermal_emission(s, Tg)
        dB = Hydro3D.thermal_emission_dT(s, Tg)
        kP, kE = Hydro3D.coupling_opacity(s, Tg)
        Q = zeros(NG + 1, NG + 1)
        Q[1, 1] = 1.0
        for g in 1:NG
            Q[1, 1+g] = s.c / s.chat[g]
            Q[1+g, 1] = (kP[g] / kE[g]) * dB[g] / s.cv
            Q[1+g, 1+g] = -(kP[g] / kE[g]) / (s.dt * s.chat[g] * s.rho * kP[g]) - 1
        end

        F0, Fg = _residual(s, Egas, Erad)
        noise = max(abs(F0), maxabs(Fg)) / Float64(ref.scale)
        @printf("%8.0e | %12.3e %12.3e | %12.2e %12.2e\n", tau, cond(A), cond(Q), noise,
                eps() * tau)
    end
end

# --- section: test ------------------------------------------------------------

function section_test()
    NG = 4
    tests = [("residual", PlainDirect(test=:residual)),
             ("scaled", PlainDirect(test=:scaled)),
             ("increment", PlainDirect(test=:increment))]
    println("\n== the convergence test, natural variables, tolerance 1e-9 held fixed, NG = $NG ==")
    @printf("%8s |", "tau")
    for (nm, _) in tests
        @printf(" %12s %3s %s |", nm, "it", "c")
    end
    @printf(" %12s %3s %s\n", "ImplicitEuler", "it", "c")
    for tau in [1e0, 1e2, 1e4, 1e6, 1e7, 1e8, 1e9, 1e10, 1e12]
        op = ConstantOpacity(tau / DT)
        ref = be_reference(; NG, opacity=op, dt=DT)
        s = cell_state(; NG, opacity=op, dt=DT)
        scale = abs(s.Egas + sumvec(s.Erad))
        @printf("%8.0e |", tau)
        for (_, m) in tests
            r = solve_coupling(m, s)
            @printf(" %12.2e %3d %s |", graderr(r, ref, NG), r.iterations,
                    r.converged ? " " : "!")
        end
        eg, er, n, ok = backward_euler(ImplicitEulerStepper(), DirectCoupling(), s,
                                       1e-9, 1e-9 * scale)
        @printf(" %12.2e %3d %s\n", graderr((Egas=eg, Erad=er), ref, NG), n, ok ? " " : "!")
    end

    println("\n== where the residual test gives up, against the tolerance it was asked for ==")
    println("   (failures out of the 8 states; the table above shows what the failing")
    println("    iteration returns -- the root, to 1e-16 of the cell energy, every time)")
    @printf("%9s |", "tau")
    for tol in (1e-7, 1e-9, 1e-11, 1e-13)
        @printf(" %12s", "tol=$tol")
    end
    println()
    for tau in (1e3, 1e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 1e9)
        @printf("%9.0e |", tau)
        for tol in (1e-7, 1e-9, 1e-11, 1e-13)
            nf = 0
            for (Tg, Tr) in STATES
                s = cell_state(; NG, opacity=ConstantOpacity(tau / DT), Tgas=Tg, Trad=Tr,
                               dt=DT)
                solve_coupling(PlainDirect(tolerance=tol), s).converged || (nf += 1)
            end
            @printf(" %12d", nf)
        end
        println()
    end

    println("\n== ... and the same for the replaced base variables ==")
    @printf("%9s |", "tau")
    for tol in (1e-7, 1e-9, 1e-11, 1e-13)
        @printf(" %12s", "tol=$tol")
    end
    println()
    for tau in (1e3, 1e6, 1e9, 1e12)
        @printf("%9.0e |", tau)
        for tol in (1e-7, 1e-9, 1e-11, 1e-13)
            nf = 0
            for (Tg, Tr) in STATES
                s = cell_state(; NG, opacity=ConstantOpacity(tau / DT), Tgas=Tg, Trad=Tr,
                               dt=DT)
                solve_coupling(QuokkaCoupling(tolerance=tol), s).converged || (nf += 1)
            end
            @printf(" %12d", nf)
        end
        println()
    end
end

# --- section: robust ----------------------------------------------------------

function section_robust()
    println("\n== robustness sweep, NG = 4, $(length(STATES)) states x $(length(TAUS)) optical depths ==")
    for (lname, mk) in LAWS
        tot = 0
        skipped = 0
        fails = zeros(Int, length(METHODS))
        worst = zeros(length(METHODS))
        worstat = ["-" for _ in METHODS]
        iters = zeros(Int, length(METHODS))
        nconv = zeros(Int, length(METHODS))
        for (Tg, Tr) in STATES, tau in TAUS
            op = mk(tau / DT)
            ref = be_reference(; NG=4, opacity=op, Tgas=Tg, Trad=Tr, dt=DT)
            if !ref.ok
                skipped += 1
                continue
            end
            tot += 1
            s = cell_state(; NG=4, opacity=op, Tgas=Tg, Trad=Tr, dt=DT)
            for (i, (nm, m)) in enumerate(METHODS)
                r = solve_coupling(m, s)
                r.converged ? (nconv[i] += 1; iters[i] += r.iterations) : (fails[i] += 1)
                startswith(nm, "adaptive") && continue   # scored against the ODE, not this root
                e = graderr(r, ref, 4)
                if e > worst[i]
                    worst[i] = e
                    worstat[i] = @sprintf("T=%.3g/%.3g tau=%.0e", Tg, Tr, tau)
                end
            end
        end
        @printf("\n-- %s opacity: %d cases scored, %d rejected (no verified reference) --\n",
                lname, tot, skipped)
        @printf("  %-24s %10s %12s %9s  %s\n", "method", "fail", "worst err", "mean it",
                "worst at")
        for i in eachindex(METHODS)
            @printf("  %-24s %4d/%-5d %12s %9.1f  %s\n", METHODS[i][1], fails[i], tot,
                    startswith(METHODS[i][1], "adaptive") ? "-" :
                    @sprintf("%.1e", worst[i]),
                    iters[i] / max(nconv[i], 1), worstat[i])
        end
    end

    println("\n== the bracketed scalar solver on every case, including those with no verified reference ==")
    for (lname, mk) in LAWS
        fails = 0
        tot = 0
        worst = 0.0
        worstat = "-"
        its = 0
        bigfail = 0
        for (Tg, Tr) in STATES, tau in TAUS
            op = mk(tau / DT)
            tot += 1
            s = cell_state(; NG=4, opacity=op, Tgas=Tg, Trad=Tr, dt=DT)
            sb = cell_state(BigFloat; NG=4, opacity=big_opacity(op), Tgas=Tg, Trad=Tr, dt=DT)
            rb = solve_coupling(BracketedDirect(tolerance=1e-45, maxit=4000), sb)
            r = solve_coupling(BracketedDirect(), s)
            r.converged || (fails += 1)
            rb.converged || (bigfail += 1)
            its += r.iterations
            e = graderr(r, (Egas=rb.Egas, Erad=rb.Erad,
                            scale=BigFloat(abs(s.Egas + sumvec(s.Erad)))), 4)
            if e > worst
                worst = e
                worstat = @sprintf("T=%.3g/%.3g tau=%.0e", Tg, Tr, tau)
            end
        end
        @printf("  %-10s %d/%d fail (%d in 256-bit)   worst %8.1e at %s   mean %.1f it\n",
                lname, fails, tot, bigfail, worst, worstat, its / tot)
    end
end

# --- section: ode -------------------------------------------------------------

function section_ode()
    NG = 4
    ode_taus = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 1e1, 1e2, 1e3, 1e4, 1e6, 1e9, 1e12]
    function ode_ref(s)
        r1 = solve_coupling(AdaptiveCoupling(reltol=1e-12, abstol=1e-16, alg=Rodas5P()), s)
        r2 = solve_coupling(AdaptiveCoupling(reltol=1e-12, abstol=1e-16, alg=FBDF()), s)
        ref = (Egas=BigFloat(r1.Egas),
               Erad=SVec{NG,BigFloat}(ntuple(g -> BigFloat(r1.Erad[g]), NG)...),
               scale=BigFloat(abs(s.Egas + sumvec(s.Erad))))
        (ref, graderr(r2, ref, NG))
    end

    println("\n== one backward-Euler step against the ODE trajectory it discretises, NG = $NG ==")
    @printf("%8s | %10s %10s | %10s %6s | %10s %6s | %10s\n", "tau", "BE Quokka",
            "BE natural", "Ad(1e-2)", "steps", "Ad(1e-8)", "steps", "ref check")
    for tau in ode_taus
        s = cell_state(; NG, opacity=ConstantOpacity(tau / DT), dt=DT)
        ref, chk = ode_ref(s)
        q = solve_coupling(QuokkaCoupling(), s)
        d = solve_coupling(DirectCoupling(stepper=NewtonStepper()), s)
        a2 = solve_coupling(AdaptiveCoupling(), s)
        a8 = solve_coupling(AdaptiveCoupling(reltol=1e-8, abstol=1e-14), s)
        @printf("%8.0e | %10.2e %10.2e | %10.2e %6d | %10.2e %6d | %10.2e\n", tau,
                graderr(q, ref, NG), graderr(d, ref, NG), graderr(a2, ref, NG),
                a2.iterations, graderr(a8, ref, NG), a8.iterations, chk)
    end

    function substepped(m, s::CouplingState{NG2,M,T}, n) where {NG2,M,T}
        Egas, Erad, tot, ok = s.Egas, s.Erad, 0, true
        for _ in 1:n
            ss = CouplingState{NG2,M,T,typeof(s.opacity)}(Egas, Erad, s.src / n, s.work / n,
                                                          s.rho, s.cv, s.opacity, s.dt / n,
                                                          s.c, s.chat, s.a_rad, s.k_B,
                                                          s.groups, s.Tfloor, s.Efloor)
            r = solve_coupling(m, ss)
            Egas, Erad, tot, ok = r.Egas, r.Erad, tot + r.iterations, ok & r.converged
        end
        CouplingResult{NG2,T}(Egas, Erad, tot, ok)
    end

    println("\n== the same step split into n backward-Euler substeps (Newton iterations in brackets) ==")
    @printf("%8s |", "tau")
    for n in (1, 2, 4, 8, 16, 64)
        @printf(" %12s", "n=$n")
    end
    println()
    for tau in (1e-2, 1e-1, 1.0, 3.0, 1e1, 1e2)
        s = cell_state(; NG, opacity=ConstantOpacity(tau / DT), dt=DT)
        ref, _ = ode_ref(s)
        @printf("%8.0e |", tau)
        for n in (1, 2, 4, 8, 16, 64)
            r = substepped(QuokkaCoupling(), s, n)
            @printf(" %8.2e(%3d)", graderr(r, ref, NG), r.iterations)
        end
        println()
    end
end

# --- section: cost ------------------------------------------------------------

"Minimum wall time and allocation of `f(s)` over `n` samples of `reps` calls."
function bench(f, s; n=200, reps=20)
    f(s)
    best = Inf
    bytes = typemax(Int)
    for _ in 1:n
        bytes = min(bytes, @allocated f(s))
        t0 = time_ns()
        for _ in 1:reps
            f(s)
        end
        best = min(best, (time_ns() - t0) / reps)
    end
    (best, bytes)
end

function section_cost()
    println("\n== cost of one cell solve; constant opacity, tau = 1e4, out of equilibrium ==")
    @printf("%-24s", "method")
    for NG in (1, 2, 4, 8, 16)
        @printf("%20s", "NG=$NG")
    end
    println()
    for (nm, m) in METHODS
        nm == "natural, residual" && continue
        @printf("%-24s", nm)
        for NG in (1, 2, 4, 8, 16)
            s = cell_state(; NG, opacity=ConstantOpacity(1e7), dt=DT)
            t, b = bench(x -> solve_coupling(m, x), s)
            @printf(" %11.0f ns %5.1fkB", t, b / 1024)
        end
        println()
    end
end

# --- section: runs ------------------------------------------------------------

_Tgas(sim) = vec(first(field(sim, :temperature)))
_Trad(sim) = vec(first(field(sim, :Trad)))
_score(a, b) = sum(abs, a - b) / sum(abs, b)

function run_case(name, mk, tmax)
    println("\n===== $name =====")
    out = []
    for (nm, m) in METHODS
        nm == "natural, increment" && continue
        mk(m)                                  # warm up compilation, untimed
        sim = mk(m)
        GC.gc()
        t0 = time()
        run!(sim; tmax, verbose=false)
        wall = time() - t0
        push!(out, (nm, sim))
        d = sim.diag
        @printf("  %-24s %7.2f s  steps=%5d  newton %6.2f mean /%5d max / %d fail\n",
                nm, wall, d.steps, d.newton_iterations / max(d.newton_cells, 1),
                d.newton_max, d.newton_failures)
    end
    ref = out[end][2]
    println("  -- L1 difference against $(out[end][1]) --")
    for (nm, sim) in out
        @printf("  %-24s dT_gas = %9.3e   dT_rad = %9.3e\n", nm,
                _score(_Tgas(sim), _Tgas(ref)), _score(_Trad(sim), _Trad(ref)))
    end
end

"A uniform multigroup box relaxing onto equilibrium: `matter_coupling` with groups."
function mg_relax(m; NG=4, kappa=1e3, T_gas=1.0, T_rad=0.1, n=64)
    groups = RadiationGroups(0.1, 20.0, NG)
    rad = RadiationPhysics(; c=1.0, chat=1.0, a_rad=1.0, k_B=1.0, groups=groups,
                           opacity=ConstantOpacity(kappa), beta_order=0, coupling=m)
    sim = Simulation(; n=(n, 1, 1), radiation=rad, hydro=false,
                     eos=IdealGas(gamma=5 / 3, cv=1.0), boundaries=Boundaries(Periodic()))
    set_initial_conditions!(sim) do x, y, z
        (rho=1.0, T=T_gas * (1 + 0.5sinpi(2x)), Erad=T_rad^4)
    end
    sim
end

"A multigroup Marshak wave whose opacity depends on temperature."
function mg_marshak(m; NG=4, n=64, expo=-3.5, kappa0=300.0, T0=0.1, Erad0=1e-4)
    groups = RadiationGroups(0.01, 40.0, NG)
    rad = RadiationPhysics(; c=1.0, chat=1.0, a_rad=1.0, k_B=1.0, groups=groups,
                           opacity=PowerLawOpacity(kappa0=kappa0, T0=1.0, n=expo),
                           beta_order=0, coupling=m)
    fracL = planck_fractions(groups, 1.0, 1.0)
    wall = Fixed(ntuple(Hydro3D.NRAD * NG) do i
        gr, comp = divrem(i - 1, Hydro3D.NRAD)
        comp == 0 ? fracL[gr+1] : 0.0
    end)
    outflow = (Outflow(), Outflow())
    sim = Simulation(; n=(n, 1, 1), radiation=rad, hydro=false,
                     eos=IdealGas(gamma=5 / 3, cv=1.0),
                     boundaries=Boundaries(Outflow()),
                     radiation_boundaries=Boundaries((wall, Outflow()), outflow, outflow))
    set_initial_conditions!(sim) do x, y, z
        (rho=1.0, T=T0, Erad=Erad0, Fx=0.0)
    end
    sim
end

function section_runs()
    run_case("marshak_vaytet, 4 groups, n = 64",
             m -> first(P.marshak_vaytet(n=64, ngroups=4, coupling=m)),
             last(P.marshak_vaytet(n=64, ngroups=4)))
    run_case("stiff relaxation, kappa = 1e12, 4 groups", m -> mg_relax(m; kappa=1e12), 1e-2)
    run_case("multigroup Marshak, kappa = 300 T^-3.5", m -> mg_marshak(m), 0.3)
    run_case("multigroup Marshak, kappa = 3e5 T^+2",
             m -> mg_marshak(m; expo=2.0, kappa0=3e5), 0.02)
end

# --- section: derivation ------------------------------------------------------

"""
The claim of findings section 5.5, checked against the code: eliminating the
group block does not merely shorten the Newton step, it fixes its *direction*.

Run on the cell section 4 traces -- gas at 0.5, radiation at 1.3, `kappa ~ T^2`,
`tau(T_gas) = 2.5e8` -- this prints the eliminated `dG/dE_gas` and the Schur
complement of the unreduced `(1+NG)` system at the same opening iterate, next to
the closed forms the document derives for each.
"""
function section_derivation()
    op = PowerLawOpacity(kappa0=1e9 / DT, n=2.0)
    s = cell_state(; NG=4, opacity=op, Tgas=0.5, Trad=1.3, dt=DT)
    cs = gas_scale(s)
    gas0 = s.Egas - dot(cs, s.work)
    rad0 = s.Erad + s.src + s.work
    T0 = s.Egas / s.cv
    B = Hydro3D.thermal_emission(s, T0)
    kP, _ = Hydro3D.coupling_opacity(s, T0)
    n_expo = 2.0

    println("\n== the elimination fixes the Newton direction, not just the step length ==")
    @printf("  cell: T_gas = %.3g, T_rad = %.3g, kappa ~ T^%.1f, tau(T_gas) = %.3g, NG = 4\n",
            T0, 1.3, n_expo, DT * s.chat[1] * s.rho * kP[1])
    @printf("  total energy of the cell           %14.8g\n",
            s.Egas + dot(cs, s.Erad + s.src))

    # the eliminated scalar
    E, G, dG = _reduced(s, s.Egas, gas0, rad0)
    # the unreduced (1+NG) system, at the same opening iterate
    F0, Fg, J00, J0g, Jg0, Jgg = _blocks(s, s.Egas, s.Erad + s.src)
    schur = J00 - sum(J0g[g] * Jg0[g] / Jgg[g] for g in 1:4)
    dx0, _ = solve_jacobian(ArrowheadSolve(), J00, J0g, Jg0, Jgg, -F0, -Fg)

    # the two closed forms. Both carry the emission term 4 a T^3 / cv; they differ
    # in whether the opacity term survives at large tau, and in which group
    # energies it is evaluated at.
    emission = (s.c / s.chat[1]) * 4 * s.a_rad * T0^3 / s.cv
    opacity_unreduced = (n_expo / T0) * (sumvec(B) - sumvec(s.Erad))

    @printf("\n  %-34s %16s %16s\n", "", "eliminated G", "unreduced Schur")
    @printf("  %-34s %16.8g %16.8g\n", "group energies it linearises at",
            sumvec(E), sumvec(s.Erad))
    @printf("  %-34s %16.8g %16.8g\n", "emission term 4 a T^3 / cv", emission, emission)
    @printf("  %-34s %16.8g %16.8g\n", "opacity term (n/T) sum (B - .)", 0.0,
            opacity_unreduced)
    @printf("  %-34s %16.8g %16.8g\n", "predicted derivative", 1 + emission,
            1 + emission + opacity_unreduced)
    @printf("  %-34s %16.8g %16.8g\n", "measured derivative", dG, schur)
    @printf("  %-34s %16.8g %16.8g\n", "first Newton step", -G / dG, dx0)

    println()
    for (nm, m) in ("bracketed scalar" => BracketedDirect(),
                    "natural, residual" => PlainDirect(test=:residual),
                    "Quokka (R base)" => QuokkaCoupling())
        r = solve_coupling(m, s)
        @printf("  %-20s E_gas = %.10g   %3d it   converged = %d\n", nm, r.Egas,
                r.iterations, r.converged)
    end
    println("  the root of this cell is 1.210310446")

    # findings section 5.4: the bracket. G is <= 0 at the temperature floor and
    # >= 0 at the cell's total energy, and at that upper probe it reduces exactly
    # to sum_g (c/chat_g) E_g -- the first and last terms of G cancel outright.
    println("\n== the bracket, over every cell of the robustness sweep ==")
    worst_lo = -Inf
    worst_hi = Inf
    identity_dev = 0.0
    bad = 0
    cases = 0
    for (_, mk) in LAWS, (Tg, Tr) in STATES, tau in TAUS
        s = cell_state(; NG=4, opacity=mk(tau / DT), Tgas=Tg, Trad=Tr, dt=DT)
        cs = gas_scale(s)
        gas0 = s.Egas - dot(cs, s.work)
        rad0 = s.Erad + s.src + s.work
        Etot = s.Egas + dot(cs, s.Erad + s.src)
        _, Glo, _ = _reduced(s, s.cv * s.Tfloor, gas0, rad0)
        Ehi, Ghi, _ = _reduced(s, Etot, gas0, rad0)
        sum_hi = dot(cs, Ehi)
        identity_dev = max(identity_dev,
                           abs(Ghi - sum_hi) / max(abs(Ghi), abs(sum_hi), Etot))
        worst_lo = max(worst_lo, Glo / Etot)
        worst_hi = min(worst_hi, Ghi / Etot)
        (Glo > 0 || Ghi < 0) && (bad += 1)
        cases += 1
    end
    @printf("  cases                                            %6d\n", cases)
    @printf("  worst G(cv*Tfloor) / E_tot        (want <= 0)  %10.3e\n", worst_lo)
    @printf("  worst G(E_tot)     / E_tot        (want >= 0)  %10.3e\n", worst_hi)
    @printf("  max |G(E_tot) - sum_g (c/chat) E_g|, relative  %10.3e\n", identity_dev)
    @printf("  cells where [cv*Tfloor, E_tot] fails to bracket  %6d\n", bad)
end

# --- driver -------------------------------------------------------------------

const SECTIONS = ["conditioning" => section_conditioning, "test" => section_test,
                  "robust" => section_robust, "derivation" => section_derivation,
                  "ode" => section_ode, "cost" => section_cost, "runs" => section_runs]

let want = isempty(ARGS) ? first.(SECTIONS) : ARGS
    for (nm, f) in SECTIONS
        nm in want && f()
    end
end
