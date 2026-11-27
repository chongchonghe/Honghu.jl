# Athena++'s inputs/mhd/athinput.blast in Honghu, for a like-for-like comparison.
#   julia --project=. -t 1 notes/performance-experiments/athena_blast.jl [n] [nsteps]
using Honghu, Printf
n = parse(Int, get(ARGS, 1, "128"))
nsteps = parse(Int, get(ARGS, 2, "20"))
sim = Simulation(; n=(n, n, n), xlo=(-1.0, -1.0, -1.0), xhi=(1.0, 1.0, 1.0),
                 eos=IdealGas(; gamma=5/3, cv=1.0), boundaries=Boundaries(Periodic()),
                 mhd=true, reconstruction=PLM(), cfl=0.3)
b0 = 1.0; ang = pi / 6
set_initial_conditions!(sim) do x, y, z
    (rho=1.0, p=x^2 + y^2 + z^2 < 0.01 ? 10.0 : 0.1, bx=b0 * cos(ang), by=b0 * sin(ang), bz=0.0)
end
step!(sim)
t = @elapsed for _ in 1:nsteps; step!(sim); end
@printf("Honghu, Athena++ MHD blast %d^3, %d steps: %.2f Mcell-updates/s\n", n, nsteps, nsteps * n^3 / t / 1e6)
