# The blast of test/benchmarks.jl as a function, for the other scripts here.
using Honghu, Printf
function blast(n, physics; recon=PLM(), tile=nothing, T=Float64)
    gamma = 5 / 3
    sim = Simulation(; n=(n, n, n), xlo=(-0.5, -0.5, -0.5), xhi=(0.5, 0.5, 0.5),
                     eos=IdealGas(; gamma, cv=1.0), boundaries=Boundaries(Outflow()),
                     mhd=physics === :mhd, reconstruction=recon, tile, T)
    r0 = 3.5 / n; pin = (gamma - 1) / (4 / 3 * pi * r0^3); pamb = 1e-5; b0 = sqrt(2pamb) / sqrt(2)
    set_initial_conditions!(sim) do x, y, z
        r2 = x * x + y * y + z * z
        s = (rho=1.0, p=r2 < r0^2 ? pin : pamb)
        physics === :mhd ? (; s..., bx=b0, by=b0, bz=0.0) : s
    end
    sim
end
