# Whether LLVM vectorises a cell kernel's loop over a row: counts vector types, gathers and
# scatters in the IR of the loop `launch!` runs, with the arguments `launch!` hands the kernel.
#   KERN=mhd|emf|bcc|ind|recon|div julia --project=. docs/performance-experiments/vec.jl  (DUMP=1 writes the IR)
include(joinpath(@__DIR__, "bench_setup.jl"))
using InteractiveUtils
const H = Honghu
function row!(ir, j, k, f::F, args::Vararg{Any,N}) where {F,N}
    @simd ivdep for i in ir
        f(i, j, k, args...)
    end
end
sim = blast(16, Symbol(get(ENV, "PHYS", "mhd")))
g = sim.grid; hw, mw = sim.hw, sim.mw; T = Float64
kern = get(ENV, "KERN", "mhd")
args = if kern == "mhd"
    (H._mhd_flux_cell!, hw.flux, mw.fspd[1], mw.ustar[1], hw.qL, hw.qR, mw.bL, mw.bR, hw.prim, mw.bcc,
     sim.B.b[1], sim.eos, sim.riemann, Val(1), T(5/3), true, true)
elseif kern == "recon"
    (H._reconstruct_cell!, hw.qL, hw.qR, hw.prim, sim.reconstruction, Val(1), Val(5))
elseif kern == "emf"
    (H._emf_cell!, mw.emf[3], sim.B.b[1], sim.B.b[2], mw.ustar[1], mw.ustar[2], mw.fspd[1], mw.fspd[2], sim.reconstruction, sim.B.o[1], sim.B.o[2])
elseif kern == "bcc"
    (H._bcc_cell!, mw.bcc, sim.B)
elseif kern == "ind"
    (H._induction_face!, sim.B.b[1], mw.B0[1], mw.emf[2], mw.emf[3], sim.B.o[2], sim.B.o[3], 0.1, 0.1, 0.5, 0.5, 0.1)
elseif kern == "div"
    (H._divergence_cell!, sim.U, hw.flux, Val(1), 1.0, 5, Val(true), nothing)
end
io = IOBuffer()
args = (args[1], H._static_rows(args[2:end])...)
code_llvm(io, row!, Base.typesof(1:16, 1, 1, args...); debuginfo=:none)
str = String(take!(io))
println(kern, ": vector <8 x double>: ", count("<8 x double>", str), "  <4 x double>: ", count("<4 x double>", str),
        "  <2 x double>: ", count("<2 x double>", str), " fdiv: ", count("fdiv", str))
get(ENV, "DUMP", "0") == "1" && write(joinpath(@__DIR__, "dump_$kern.ll"), str)
println("  vector.body: ", count("vector.body:", str), "  gather: ", count("masked.gather", str), " scatter: ", count("masked.scatter", str), " shufflevector: ", count("shufflevector", str))
function row0!(ir, j, k, f::F, args::Vararg{Any,N}) where {F,N}
    for i in ir
        f(i, j, k, args...)
    end
end
code_llvm(io, row0!, Base.typesof(1:16, 1, 1, args...); debuginfo=:none)
s0 = String(take!(io))
println("  plain loop: vector.body: ", count("vector.body:", s0), " <8 x double>: ", count("<8 x double>", s0))
