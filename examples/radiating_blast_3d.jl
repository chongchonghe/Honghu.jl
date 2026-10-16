#!/usr/bin/env julia
#
# A full 3D radiation-hydrodynamics run in CGS units: a hot, over-pressured
# sphere expanding into a marginally optically thick medium, with the radiation
# field coupled to the gas.
#
#   julia --project=. -t auto examples/radiating_blast_3d.jl [n] [--no-plot] [--outdir=DIR]
#
# Exercises everything at once: HLLC hydro, M1 transport, radiation subcycling
# at the reduced speed of light, the implicit coupling, and the radiation force.
# Writes blast.png (a six-panel overview of the final state) and a VTK series to
# blast/. About 2 minutes at n = 64 on 8 threads.
#
# Choosing the parameters of a radiation-hydrodynamics problem takes some care.
# Two ratios decide whether the setup is well posed:
#
#   a_rad T^4 / (rho c_v T)   how much the radiation dominates the energy. The
#                             non-relativistic equations solved here assume the
#                             gas stays slow, so a radiation energy density that
#                             swamps the gas by many orders of magnitude will
#                             accelerate the matter towards c and the model
#                             stops being valid. Keep this of order one or less.
#
#   chat / max(|v| + c_s)     the reduced speed of light must still outrun every
#                             other signal. A factor of ~20 is comfortable.

using Honghu
using Printf
include(joinpath(@__DIR__, "common.jl"))
const CGS = Honghu.CGS

n = isempty(OPTS) ? 64 : parse(Int, OPTS[1])

# --- problem scales ---------------------------------------------------------
L = 1.0e14                  # box size, cm
rho0 = 1.0e-9               # ambient density, g/cm^3
T0 = 1.0e3                  # ambient temperature, K
Tblast = 1.0e4              # temperature inside the initial sphere, K
rblast = 0.1L
kappa = 1.0e-5              # cm^2/g -> optical depth across the box ~ 1

gamma = 5 / 3
eos = IdealGas(; gamma, cv=cv_from_mu(0.6, gamma))
cs = sqrt(gamma * (gamma - 1) * eos.cv * Tblast)

# gas-dominated by design: check before running, not after
beta_rad = CGS.a_rad * Tblast^4 / (rho0 * eos.cv * Tblast)
@printf("radiation/gas energy in the blast : %.3g\n", beta_rad)
@printf("sound speed in the blast          : %.3g cm/s\n", cs)

# The reduced speed of light only has to beat every other signal in the problem.
rad = RadiationPhysics(c=CGS.c_light, chat=CGS.c_light / 1000, a_rad=CGS.a_rad,
                       opacity=ConstantOpacity(kappa), beta_order=1)
@printf("chat / sound speed                : %.3g\n", rad.chat / cs)

sim = Simulation(n=(n, n, n), xlo=(-L / 2, -L / 2, -L / 2), xhi=(L / 2, L / 2, L / 2),
                 eos=eos, radiation=rad,
                 boundaries=Boundaries(Outflow()),
                 cfl=0.4, cfl_radiation=0.3, max_radiation_substeps=10)

set_initial_conditions!(sim) do x, y, z
    hot = x^2 + y^2 + z^2 < rblast^2
    T = hot ? Tblast : T0
    (rho=rho0, T=T, Erad=CGS.a_rad * T^4)     # start in radiative equilibrium
end

println(sim)

tmax = 1.5e7    # s, roughly the time for the blast to cross a quarter of the box
t0 = conserved_totals(sim)
vtkdir = outpath("blast")
run!(sim; tmax, outdir=vtkdir, basename="blast", dtout=tmax / 30)
t1 = conserved_totals(sim)

# With a reduced speed of light the conserved quantity is E_gas + (c/chat) E_rad,
# not E_gas + E_rad. Here the boundaries are open, so most of the radiation that
# started in the hot sphere simply leaves the box: the box crossing time for
# radiation is L/chat, several times shorter than the run. Swap in
# Boundaries(Reflecting()) and this budget closes to the Newton tolerance.
cscale = rad.c / rad.chat
E0 = t0.energy_gas + cscale * t0.energy_rad
E1 = t1.energy_gas + cscale * t1.energy_rad
@printf("\nmass remaining in the box      : %.4f\n", t1.mass / t0.mass)
@printf("E_gas remaining                : %.4f\n", t1.energy_gas / t0.energy_gas)
@printf("E_rad remaining                : %.4f  (radiated away, L/chat = %.2g s)\n",
        t1.energy_rad / t0.energy_rad, L / rad.chat)
@printf("E_gas + (c/chat) E_rad         : %.4f of initial\n", E1 / E0)

Tg = gas_temperature(sim)
Tr = radiation_temperature(sim)
prim = primitives(sim)
@printf("gas temperature       : %.4g .. %.4g K\n", minimum(Tg), maximum(Tg))
@printf("radiation temperature : %.4g .. %.4g K\n", minimum(Tr), maximum(Tr))
@printf("peak gas speed        : %.4g cm/s  (%.3g c)\n",
        maximum(sqrt.(prim[2, :, :, :] .^ 2 .+ prim[3, :, :, :] .^ 2 .+ prim[4, :, :, :] .^ 2)),
        maximum(abs.(prim[2, :, :, :])) / CGS.c_light)
println("  wrote 31 VTK snapshots to $vtkdir/")

PLOTTING && saveplot(plotpanels(sim; size=(1100, 700)), "blast.png")
plotting_note()
