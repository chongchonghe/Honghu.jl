using Test
using Hydro3D
using Hydro3D: SVec, dot, norm, minmod, limit, ppm_edges, eddington_factor,
               eddington_tensor, hllc, hll_radiation, cell_ranges, face_ranges,
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
    include("test_io.jl")
    include("test_plotting.jl")
    include("test_extensions.jl")
    include("test_compare.jl")
    include("test_performance.jl")
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
