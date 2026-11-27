# The MHD Riemann solvers alone, on random states: ns per solve of a scalar loop.
#   julia --project=. -t 1 notes/performance-experiments/micro.jl
using Honghu, Printf
const H = Honghu
function run(N, solver)
    T = Float64; g = T(5/3)
    st = [ (H.HydroState(1.0+0.1rand(), 0.1rand(), 0.1rand(), 0.1rand(), 1.0+0.1rand(), 1.0, 2.0, 0.3rand(), 0.3rand()),
            H.HydroState(1.0+0.1rand(), 0.1rand(), 0.1rand(), 0.1rand(), 1.0+0.1rand(), 1.0, 2.0, 0.3rand(), 0.3rand())) for _ in 1:N]
    out = zeros(N)
    f(st, out) = @inbounds for n in eachindex(st)
        F, a, b = H.mhd_flux(solver, st[n][1], st[n][2], g, 0.5, 0.0)
        out[n] = F[1] + F[5] + a + b
    end
    f(st, out)
    t = minimum(@elapsed(f(st, out)) for _ in 1:20)
    @printf("%s: %.1f ns per solve\n", nameof(typeof(solver)), t / N * 1e9)
end
run(10^5, HLLD()); run(10^5, HLLMHD()); run(10^5, LLFMHD())
