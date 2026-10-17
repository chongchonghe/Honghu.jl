# Prototype of the one-variable dust-gas coupling solve, and every number in
# `docs/coupling-new-method-dust.md` §6.
#
#   julia --project=. docs/coupling-experiments/dust_prototype.jl            # all
#   julia --project=. docs/coupling-experiments/dust_prototype.jl sweep ode  # named sections
#
# Sections: sweep, ode, limits. Nothing here is part of the package: it builds
# the dust residual out of the package's own pieces (`coupling_coefficients`,
# `_eliminate_groups`, `energy_balance`) so the plan can be checked before it is
# implemented.

using Honghu
using Honghu: CouplingState, SVec, gas_scale, gas_temperature_of, energy_balance,
              coupling_coefficients, _eliminate_groups, dot, Brent, IntervalNonlinearProblem, solve
using Printf

function cell(; NG=1, opacity, Tgas=1.0, Trad=0.0, rho=1.0, cv=1.5, dt=1e-8, c=1e8, chat=1e7,
              arad=1.0, kB=1.0)
    T = Float64
    grp = NG == 1 ? Honghu.GREY : RadiationGroups(0.1, 20.0, NG)
    Erad = (arad * Trad^4) * planck_fractions(grp, max(Trad, 1e-10), kB) + SVec{NG,T}(ntuple(_ -> 1e-20, NG)...)
    z = zero(SVec{NG,T})
    chatv = SVec{NG,T}(ntuple(_ -> chat, NG)...)
    CouplingState{NG,NG + 1,T,typeof(opacity)}(cv * Tgas, Erad, z, z, rho, cv, opacity, dt, c,
                                               chatv, arad, kB, grp, 1e-10, 1e-30)
end

# the one-variable dust residual: x = T_d. K = Lambda0 n^2.
@inline function dustH(Td, s, K)
    emis, absorp = coupling_coefficients(s, Td)
    Erad = first(_eliminate_groups(s, emis, absorp))
    Egas = -energy_balance(s, zero(Td), Erad)       # conservation defines E_gas
    gas0 = s.Egas - dot(gas_scale(s), s.work)
    Tg = gas_temperature_of(s, Egas)
    return Egas - gas0 + s.dt * K * sqrt(Tg) * (Tg - Td)
end
dust_state(Td, s) = (Erad = first(_eliminate_groups(s, coupling_coefficients(s, Td)...));
                     Egas = -energy_balance(s, 0.0, Erad); (Egas, Erad))

function bracket(s, K, x0)
    x = max(x0, s.Tfloor); h = dustH(x, s, K); n = 1
    if h > 0
        while n < 400; y = 2x; n += 1; dustH(y, s, K) <= 0 && return (x, y, n, true); x = y; end
    else
        while n < 400; y = max(x/2, s.Tfloor); n += 1; dustH(y, s, K) >= 0 && return (y, x, n, true)
            y == s.Tfloor && return (y, x, n, false); x = y; end
    end
    (x, x, n, false)
end

function solve_dust(s, K; tol=1e-9)
    Etot = s.Egas + dot(gas_scale(s), s.Erad)
    lo, hi, nb, ok = bracket(s, K, s.Egas / s.cv)
    ok || return (NaN, NaN, nb, false)
    f = (x, p) -> dustH(x, s, K)
    prob = IntervalNonlinearProblem{false}(f, (lo, hi), nothing)
    sol = solve(prob, Brent(); abstol=tol * sol_scale(lo, hi) / 2, maxiters=200)
    width = abs(first(dust_state(sol.left, s)) - first(dust_state(sol.right, s)))
    nsolve = 1
    while width > tol * Etot && nsolve < 4
        # convert: the bracket in T_d maps to E_gas by the local slope
        slope = width / max(sol.right - sol.left, eps(sol.u))
        sol = solve(IntervalNonlinearProblem{false}(f, (sol.left, sol.right), nothing), Brent();
                    abstol=tol * Etot / (4slope), maxiters=200)
        width = abs(first(dust_state(sol.left, s)) - first(dust_state(sol.right, s)))
        nsolve += 1
    end
    global NEVAL += nb + nsolve
    (sol.u, width / Etot, nb, width <= tol * Etot)
end

roots(s,K) = (xs = exp10.(range(-8, 8, length=4001)); hs = [dustH(x, s, K) for x in xs]; [xs[i] for i in 1:length(hs)-1 if sign(hs[i]) != sign(hs[i+1])])
signchanges(s, K) = (xs = exp10.(range(-8, 8, length=4001)); hs = [dustH(x, s, K) for x in xs];
                     count(i -> sign(hs[i]) != sign(hs[i+1]) && hs[i] != 0, 1:length(hs)-1))

sol_scale(lo, hi) = hi
NEVAL = 0
# --- sweep -------------------------------------------------------------------
function sweep()
    nfail = 0; nmulti = 0; ntot = 0; worst = 0.0
    for NG in (1, 4), op in (ConstantOpacity(1.0), PowerLawOpacity(kappa0=1.0, rho0=1.0, T0=1.0, m=0.0, n=2.0)),
        Tg in (0.01, 0.3, 1.0, 10.0), Tr in (0.0, 0.1, 1.0, 5.0), tauc in (1e-4, 1.0, 1e4, 1e8, 1e12),
        kd in (0.0, 1e-6, 1.0, 1e4, 1e8)
        s0 = cell(; NG, opacity=op, Tgas=Tg, Trad=Tr, dt=1.0, c=1.0, chat=1.0, cv=1.0)
        # scale rho*kappa via rho (opacity per mass); K via kd = dt K / cv
        s = cell(; NG, opacity=op, Tgas=Tg, Trad=Tr, dt=1.0, c=1.0, chat=1.0, cv=1.0, rho=tauc)
        K = kd
        ntot += 1
        Td, r, nb, ok = solve_dust(s, K)
        ok || (nfail += 1; @printf("FAIL NG=%d %s Tg=%g Tr=%g tau=%g kd=%g r=%g\n", NG, nameof(typeof(op)), Tg, Tr, tauc, kd, r))
        worst = max(worst, abs(r))
        sc = signchanges(s, K)
        sc > 1 && (println("   Td found = ", Td, "  roots near: ", roots(s,K)))
        sc > 1 && (nmulti += 1; @printf("MULTI(%d) NG=%d %s Tg=%g Tr=%g tau=%g kd=%g\n", sc, NG, nameof(typeof(op)), Tg, Tr, tauc, kd))
    end
    @printf("sweep: %d cells, %d failed, %d with >1 root, worst state error / E_tot = %.2e\n", ntot, nfail, nmulti, worst)
end
const SECTIONS = isempty(ARGS) ? ["sweep", "ode", "limits"] : ARGS
"sweep" in SECTIONS && sweep()
"sweep" in SECTIONS && @printf("march steps + root-finder calls per cell: %.1f\n", NEVAL / 1600)

# --- ode: RadDust (Quokka testRadDust) with B = a T^4, against Rodas5P ------
if "ode" in SECTIONS
using OrdinaryDiffEq: ODEProblem, solve as odesolve, Rodas5P
const ADT = Base.require(Base.PkgId(Base.UUID("47edcb42-4c32-4615-8424-f2b9edc5f35b"), "ADTypes"))
# RadDust parameters (Quokka testRadDust), but with the physical B = a T^4
const c, chat, chi, K, cv, arad = 1e8, 1e7, 1e4, 1e6, 1.5, 1.0
Tdust(T, E) = (h(x) = c * chi * (arad * x^4 - E) - K * sqrt(T) * (T - x);
               solve(IntervalNonlinearProblem{false}((x, p) -> h(x), (1e-12, max(T, (max(E, 0.0)/arad)^0.25) + 1), nothing), Brent(); abstol=0.0).u)
function rhs(u, p, t)
    Eg, E = u; T = max(Eg / cv, 1e-12); Td = Tdust(T, E)
    L = K * sqrt(T) * (T - Td)
    [-L, chat * chi * (arad * Td^4 - E)]
end
prob = ODEProblem(rhs, [cv * 1.0, 1e-20], (0.0, 3e-5))
ref = odesolve(prob, Rodas5P(autodiff=ADT.AutoFiniteDiff()); reltol=1e-11, abstol=1e-14)
Tref = ref.u[end][1] / cv; Eref = ref.u[end][2]
println("reference: T_gas = $Tref, E_rad = $Eref")
for dt in (1e-7, 1e-8, 1e-9)
    Eg, E = cv * 1.0, 1e-20
    n = round(Int, 3e-5 / dt)
    for _ in 1:n
        s = cell(; NG=1, opacity=ConstantOpacity(chi), Tgas=Eg / cv, rho=1.0, cv, dt, c, chat, arad)
        s = Honghu.CouplingState{1,2,Float64,typeof(s.opacity)}(Eg, SVec(E), s.src, s.work, s.rho, s.cv, s.opacity, dt, c, s.chat, arad, 1.0, s.groups, 1e-10, 1e-30)
        Td, r, nb, ok = solve_dust(s, K); ok || error("fail")
        Eg, Erad = dust_state(Td, s); E = Erad[1]
    end
    @printf("dt=%g: T_gas err = %.3e, E_rad err = %.3e, energy err = %.2e\n", dt, Eg/cv/Tref - 1, E/Eref - 1,
            (Eg + (c/chat)*E) / (cv + (c/chat)*1e-20) - 1)
end
end

# --- limits: K -> infinity recovers the gas-coupled solve; K = 0 leaves the gas alone
if "limits" in SECTIONS
for NG in (1, 4), op in (ConstantOpacity(1.0), PowerLawOpacity(kappa0=1.0, rho0=1.0, T0=1.0, m=0.0, n=-1.5)), (Tg, Tr) in ((1.0, 0.1), (0.3, 2.0))
    s = cell(; NG, opacity=op, Tgas=Tg, Trad=Tr, dt=1.0, c=1.0, chat=1.0, cv=1.0, rho=10.0)
    plain = Honghu.solve_coupling(BracketedCoupling(tolerance=1e-12), s)
    Td, _, _, _ = solve_dust(s, 1e12; tol=1e-12); Eg, _ = dust_state(Td, s)
    Td0, _, _, _ = solve_dust(s, 0.0; tol=1e-12); Eg0, Er0 = dust_state(Td0, s)
    @printf("NG=%d %-16s Tg=%.1f Tr=%.1f  K->inf: Egas rel diff %.1e (Td/Tgas-1 = %.1e) | K=0: Egas-Egas0 = %.1e, sum Erad change %.1e\n",
            NG, nameof(typeof(op)), Tg, Tr, Eg / plain.Egas - 1, Td / (Eg / s.cv) - 1, Eg0 - s.Egas, sum(Er0) - sum(s.Erad))
end
end
