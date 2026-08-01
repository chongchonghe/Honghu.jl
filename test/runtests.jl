using Test
using Hydro3D
using Hydro3D: SVec, dot, norm, minmod, limit, ppm_edges, eddington_factor,
               eddington_tensor, hll_radiation, cell_ranges, face_ranges,
               NHYDRO, NRAD, IRHO, IVX, IPRE, IERAD, IFRX, IFRY, cons2prim, prim2cons,
               reconstruct!, required_ghosts, allocate, cellcenter, fill_ghosts!,
               variants, rows
const P = Hydro3D.Problems

@testset "Hydro3D" begin
    include("test_utils.jl")
    include("test_grid.jl")
    include("test_reconstruction.jl")
    include("test_riemann.jl")
    include("test_hydro.jl")
    include("test_radiation.jl")
    include("test_radiation_multigroup.jl")
    include("test_io.jl")
    include("test_plotting.jl")
    include("test_extensions.jl")
    include("test_compare.jl")
    include("test_performance.jl")
end

# `AdaptiveCoupling` -- the third coupling solver, and the three-way agreement
# between the solvers -- lives in a package extension, so testing it means
# `using OrdinaryDiffEq`. That package also exports `step!`, which every other
# test file here calls unqualified, so it cannot be loaded into this session
# without breaking them. Run it in a subprocess instead, and only when the
# package is actually installed.
let ode = Base.PkgId(Base.UUID("1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"), "OrdinaryDiffEq")
    script = joinpath(@__DIR__, "test_coupling_adaptive.jl")
    if Base.locate_package(ode) === nothing
        println("\nOrdinaryDiffEq is not installed; skipping ", basename(script))
    else
        println("\nrunning ", basename(script), " in a subprocess ...")
        run(`$(Base.julia_cmd()) --project=$(Base.active_project()) $script`)
    end
end

# The verification figures are the visual counterpart of the tests above: same
# problems, same reference solutions, drawn instead of asserted. They are off by
# default because they re-run every problem at production resolution (a couple
# of minutes) and write files, neither of which belongs in a default test run.
#
#   julia --project=. test/runtests.jl --figures
#   HYDRO3D_FIGURES=1 julia --project=. -e 'using Pkg; Pkg.test()'
if "--figures" in ARGS || get(ENV, "HYDRO3D_FIGURES", "0") != "0"
    script = normpath(joinpath(@__DIR__, "..", "docs", "make_figures.jl"))
    println("\ngenerating verification figures ...")
    try
        # a subprocess, so the figure script's own globals cannot collide with
        # the test session's
        run(`$(Base.julia_cmd()) --project=$(Base.active_project()) -t $(Threads.nthreads()) $script`)
    catch err
        @warn "could not generate figures (is Plots installed?)" exception = err
    end
else
    println("\nrun with --figures to also write the verification figures to docs/figures/")
end
