# Steps per second of the 3D blast, one configuration per run, for notes/performance-findings.md.
#   julia --project=. -t 1 notes/performance-experiments/bench.jl hydro|rhd|mhd n [tile|none] [Float32|Float64] [PCM|PLM|PPM]
# NSTEPS=k sets the steps timed (default 2e7/n^3), NOFLUSH=1 runs with flush_subnormals = false.
using Honghu, Printf
physics = Symbol(ARGS[1]); n = parse(Int, ARGS[2])
tile = length(ARGS) >= 3 && ARGS[3] != "none" ? Tuple(parse.(Int, split(ARGS[3], ","))) : nothing
T = length(ARGS) >= 4 && ARGS[4] == "Float32" ? Float32 : Float64
recon = length(ARGS) >= 5 ? (ARGS[5] == "PCM" ? PCM() : ARGS[5] == "PPM" ? PPM() : PLM()) : PLM()
function blast(n)
    gamma = 5 / 3
    sim = Simulation(; n=(n, n, n), xlo=(-0.5, -0.5, -0.5), xhi=(0.5, 0.5, 0.5),
                     eos=IdealGas(; gamma, cv=1.0), boundaries=Boundaries(Outflow()),
                     mhd=physics === :mhd, reconstruction=recon, max_radiation_substeps=1,
                     radiation=physics === :rhd ? RadiationPhysics(c=1e3, chat=1e2, a_rad=1e-12, opacity=ConstantOpacity(100.0)) : nothing, tile, T, flush_subnormals=get(ENV, "NOFLUSH", "0") != "1")
    r0 = 3.5 / n; pin = (gamma - 1) / (4 / 3 * pi * r0^3); pamb = 1e-5; b0 = sqrt(2pamb) / sqrt(2)
    set_initial_conditions!(sim) do x, y, z
        r2 = x * x + y * y + z * z
        s = (rho=1.0, p=r2 < r0^2 ? pin : pamb)
        physics === :mhd ? (; s..., bx=b0, by=b0, bz=0.0) : s
    end
    sim
end
sim = blast(n); step!(sim)
ns = parse(Int, get(ENV, "NSTEPS", string(max(3, round(Int, 2e7 / n^3)))))
get(ENV, "FTZ", "0") == "1" && set_zero_subnormals(true)
t = @elapsed for _ in 1:ns; step!(sim); end
@printf("%-5s n=%-4d tile=%-10s %-7s %s: %.2f Mcell-updates/s (%.0f ns/cell)\n", physics, n,
        string(tile), T, typeof(recon).name.name, n^3 * ns / t / 1e6, t / ns / n^3 * 1e9)
