#!/usr/bin/env julia
#
# The thermal balance behind the table in dtype/README.md: the equilibrium temperature of
# fully ionized hydrogen in ionization equilibrium, and the cooling time, for Honghu's
# HydrogenIliev2 network (Iliev Tests 2 and 5) and for the network of Quokka's
# DTypeFront_JAFF test; and the KI equilibrium of Quokka's neutral gas. The rates are copied
# from networks/hydrogen_iliev2 and from Quokka's
# src/networks/photoionization_H_caseB_config.
#
#   julia docs/photoionization-experiments/thermal_balance.jl
#
kB = 1.380649e-16; eV = 1.602176634e-12; mp = 1.6726e-24
alphaB(T) = 2.63e-13 * (T / 1e4)^(-0.7)
# Honghu (networks/hydrogen_iliev2): recombination, free-free, Lyman alpha (n_e n_H0)
h_rec(T) = T < 100 ? 0.0 : 6.1e-10 * kB * T^0.11
h_ff(T) = 1.3 * 1.427e-27 * sqrt(T)
h_lya(T) = 7.5e-19 * exp(-118348 / T) / (1 + sqrt(T / 1e5))
# Quokka (photoionization_H_caseB_config): recombination, free-free + CLE (Frazer & Heitsch 2019)
function cle(T)
    T < 1e2 ? 3.47e-29 * T^1.915 : T < 10^2.8 ? 2.34e-26 * T^0.5 : T < 10^3.6 ? 1.11e-24 * T^(-0.099) :
    T < 1e4 ? 1.08e-32 * T^2.127 : T < 10^4.5 ? 2.67e-30 * T^1.529 : T < 1e5 ? 1.74e-24 * T^0.237 :
    T < 1e6 ? 1.10e-21 * T^(-0.323) : 7.49e-21 * T^(-0.462)
end
q_rec(T) = T < 100 ? 0.0 : 6.1e-10 * kB * T * T^(-0.89)
q_ff(T) = 1.3 * 1.427e-27 * sqrt(T) + cle(T)

# In ionization equilibrium photoionizations = recombinations, so heating = alpha_B n_e n_p dE;
# with x_H0 the neutral fraction, the Lyman alpha term is Lambda_lya n_e n_H0 = Lambda_lya n_e n_p x0 / (1 - x0).
net_h(T, dE, x0) = alphaB(T) * dE - (h_rec(T) + h_ff(T)) - h_lya(T) * x0 / (1 - x0)
net_q(T, dE) = alphaB(T) * dE - (q_rec(T) + q_ff(T))
function root(f, lo, hi)
    for _ in 1:200
        m = (lo + hi) / 2
        f(m) > 0 ? (lo = m) : (hi = m)
    end
    (lo + hi) / 2
end
Eavg = 13.6 * 26 * log(26 / 13.6) / (26 - 13.6)
println("Quokka band [13.6, 26] eV flat: mean photon energy ", round(Eavg, digits=2), " eV, heating ", round(Eavg - 13.6, digits=2), " eV")
for x0 in (0.0, 1e-3, 1e-2)
    println("Honghu, dE = 16.05 eV, x_H0 = $x0: T_eq = ", round(root(T -> net_h(T, 16.05eV, x0), 1e3, 1e6)))
end
println("Honghu network, dE = 4.88 eV, x_H0 = 0: T_eq = ", round(root(T -> net_h(T, (Eavg - 13.6)eV, 0.0), 1e3, 1e6)))
println("Quokka network, dE = 4.88 eV: T_eq = ", round(root(T -> net_q(T, (Eavg - 13.6)eV), 1e3, 1e6)))
println("Quokka network, dE = 16.05 eV: T_eq = ", round(root(T -> net_q(T, 16.05eV), 1e3, 1e6)))
# cooling time of ionized gas, t = (3/2)(2n) kB T / (n^2 Lambda) = 3 kB T / (n Lambda)
Myr = 3.156e13
for (lab, n, T, L) in (("Honghu Test 5", 1e-3, 1.5e4, h_rec(1.5e4) + h_ff(1.5e4)),
                       ("Quokka DTypeFront_JAFF", 139.9, 7000.0, q_rec(7000.0) + q_ff(7000.0)))
    t = 3kB * T / (n * L)
    println(lab, ": n = $n, T = $T, cooling time = ", t / Myr, " Myr")
end
# neutral gas: KI heating 2e-26 n, cooling n^2 Lambda_KI (Quokka); Honghu has neither
LKI(T) = 2.0e-26 * (1.0e7 * exp(-118400.0 / (T + 1.0e3)) + 1.4e-2 * sqrt(T) * exp(-92.0 / T))
println("Quokka neutral equilibrium at n = 139.9: ", round(root(T -> 2e-26 * 139.9 - 139.9^2 * LKI(T), 1.0, 1000.0), digits=2), " K")
println("Quokka neutral equilibrium at n = 100: ", round(root(T -> 2e-26 * 100 - 100^2 * LKI(T), 1.0, 1000.0), digits=2), " K")
# sound-crossing time of the Stromgren radius in each
println("Honghu Test 5: R_S / c_II = ", 5.39 * 3.086e21 / sqrt(2kB * 1.5e4 / mp) / Myr, " Myr")
