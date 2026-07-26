using Test
using Hydro3D
using Hydro3D: SVec, dot, norm, minmod, limit, ppm_edges, eddington_factor,
               eddington_tensor, hllc, hll_radiation, cell_ranges, face_ranges,
               NHYDRO, NRAD, IRHO, IVX, IPRE, IERAD, IFRX, IFRY, cons2prim, prim2cons,
               reconstruct!, required_ghosts, allocate, cellcenter, fill_ghosts!
const P = Hydro3D.Problems

@testset "Hydro3D" begin
    include("test_utils.jl")
    include("test_grid.jl")
    include("test_reconstruction.jl")
    include("test_riemann.jl")
    include("test_hydro.jl")
    include("test_radiation.jl")
    include("test_io.jl")
end
