# Uniform Cartesian grid with ghost cells.
#
# The grid is always three-dimensional. A direction with `n[d] == 1` is
# *inactive*: it carries no ghost cells, is skipped by the directional sweeps,
# and needs no boundary condition. Running a 1D or 2D problem is therefore just
# a matter of setting `n = (nx, 1, 1)` or `n = (nx, ny, 1)`.

"""
    CartesianGrid(; n, xlo = (0,0,0), xhi = (1,1,1), nghost = 3, T = Float64)

Uniform Cartesian grid of `n = (nx, ny, nz)` interior cells covering
`[xlo[d], xhi[d]]` in each direction, padded with `nghost` ghost cells in every
*active* direction (a direction with `n[d] == 1` gets none).

`n` may also be given as a single integer (cubic grid).

# Fields
  - `n`      : interior cell counts
  - `ng`     : ghost cells per direction (0 where inactive)
  - `lo`,`hi`: first/last interior index of the padded arrays
  - `dims`   : full padded array size, `n .+ 2 .* ng`
  - `dx`     : cell size per direction
  - `active` : `n[d] > 1`

# Example
```julia
g = CartesianGrid(n = (128, 128, 1), xhi = (1.0, 1.0, 1.0), nghost = 3)
```
"""
struct CartesianGrid{T<:AbstractFloat}
    n::NTuple{3,Int}
    ng::NTuple{3,Int}
    lo::NTuple{3,Int}
    hi::NTuple{3,Int}
    dims::NTuple{3,Int}
    xlo::NTuple{3,T}
    xhi::NTuple{3,T}
    dx::NTuple{3,T}
    active::NTuple{3,Bool}
end

_tuple3(x::Integer) = (Int(x), Int(x), Int(x))
_tuple3(x::NTuple{3,Integer}) = (Int(x[1]), Int(x[2]), Int(x[3]))
_tuple3(x::NTuple{2,Integer}) = (Int(x[1]), Int(x[2]), 1)
_tuple3(x::NTuple{1,Integer}) = (Int(x[1]), 1, 1)
_tuple3f(::Type{T}, x::Real) where {T} = (T(x), T(x), T(x))
_tuple3f(::Type{T}, x::NTuple{3,Real}) where {T} = (T(x[1]), T(x[2]), T(x[3]))

function CartesianGrid(; n, xlo=(0.0, 0.0, 0.0), xhi=(1.0, 1.0, 1.0), nghost::Integer=3,
                       T::Type{<:AbstractFloat}=Float64)
    nn = _tuple3(n)
    all(>(0), nn) || throw(ArgumentError("cell counts must be positive, got n = $nn"))
    nghost >= 1 || throw(ArgumentError("nghost must be >= 1, got $nghost"))

    lohi_lo = _tuple3f(T, xlo)
    lohi_hi = _tuple3f(T, xhi)
    all(lohi_hi .> lohi_lo) || throw(ArgumentError("need xhi > xlo elementwise"))

    active = ntuple(d -> nn[d] > 1, 3)
    ng = ntuple(d -> active[d] ? Int(nghost) : 0, 3)
    lo = ntuple(d -> ng[d] + 1, 3)
    hi = ntuple(d -> ng[d] + nn[d], 3)
    dims = ntuple(d -> nn[d] + 2ng[d], 3)
    dx = ntuple(d -> (lohi_hi[d] - lohi_lo[d]) / nn[d], 3)

    CartesianGrid{T}(nn, ng, lo, hi, dims, lohi_lo, lohi_hi, dx, active)
end

"""
    ndim(grid)

Number of active (non-degenerate) directions, 1, 2, or 3.
"""
@inline ndim(g::CartesianGrid) = count(g.active)

"""
    cellcenter(grid, i, j, k)

Coordinates of the centre of the cell with *padded* indices `(i, j, k)`.
"""
@inline function cellcenter(g::CartesianGrid{T}, i::Integer, j::Integer, k::Integer) where {T}
    idx = (i, j, k)
    ntuple(d -> g.xlo[d] + (idx[d] - g.lo[d] + T(0.5)) * g.dx[d], 3)
end

"""
    cellvolume(grid)

Volume of one cell (inactive directions contribute their full box extent).
"""
@inline cellvolume(g::CartesianGrid) = prod(g.dx)

"""
    interior(grid)

`CartesianIndices` spanning the interior (non-ghost) cells.
"""
@inline function interior(g::CartesianGrid)
    CartesianIndices((g.lo[1]:g.hi[1], g.lo[2]:g.hi[2], g.lo[3]:g.hi[3]))
end

"""
    boxsize(grid)

Physical extent `(Lx, Ly, Lz)` of the domain.
"""
@inline boxsize(g::CartesianGrid) = ntuple(d -> g.xhi[d] - g.xlo[d], 3)

"""
    allocate(grid, nvar, T = eltype(grid))

Zero-initialised `(nvar, dims...)` array sized for `grid`, ghost cells included.

The variable index comes first so that the `nvar` components of a cell are
contiguous in Julia's column-major layout, which is what the Riemann solvers
want.
"""
function allocate(g::CartesianGrid{T}, nvar::Integer) where {T}
    zeros(T, nvar, g.dims[1], g.dims[2], g.dims[3])
end

Base.eltype(::CartesianGrid{T}) where {T} = T

# --- sweep helpers ----------------------------------------------------------

"Unit offset tuple for direction `D`, known at compile time."
@inline _shift(::Val{1}) = (1, 0, 0)
@inline _shift(::Val{2}) = (0, 1, 0)
@inline _shift(::Val{3}) = (0, 0, 1)

"""
    cell_ranges(grid, Val(D))

Index ranges of the cells whose interface states must be reconstructed for a
sweep along direction `D`: the interior extended by one cell in `D`.
"""
@inline function cell_ranges(g::CartesianGrid, ::Val{D}) where {D}
    ntuple(Val(3)) do d
        pad = d == D ? 1 : 0
        (g.lo[d] - pad):(g.hi[d] + pad)
    end
end

"""
    face_ranges(grid, Val(D))

Index ranges of the interfaces at which fluxes are needed for a sweep along
direction `D`. Face `f` is the low-side face of cell `f`, so the interior cells
`lo:hi` are bounded by faces `lo:hi+1`.
"""
@inline function face_ranges(g::CartesianGrid, ::Val{D}) where {D}
    ntuple(Val(3)) do d
        g.lo[d]:(g.hi[d] + (d == D ? 1 : 0))
    end
end

"""
    threaded_jk(f, jrange, krange)

Call `f(j, k)` for every pair in `jrange × krange`, spreading the work over
`Threads.nthreads()` threads. Flattening the two outer loops keeps 2D runs
(where `krange` has length one) parallel.
"""
@inline function threaded_jk(f::F, jrange::UnitRange{Int}, krange::UnitRange{Int}) where {F}
    nj = length(jrange)
    nk = length(krange)
    j0 = first(jrange)
    k0 = first(krange)
    ntot = nj * nk
    if ntot <= 1 || Threads.nthreads() == 1
        for t in 0:(ntot-1)
            f(j0 + t % nj, k0 + t ÷ nj)
        end
    else
        Threads.@threads for t in 0:(ntot-1)
            f(j0 + t % nj, k0 + t ÷ nj)
        end
    end
    return nothing
end

"""
    reduce_max_jk(f, jrange, krange, T)

Threaded `maximum` of `f(j, k)::T` over `jrange × krange`. Partial results are
written to distinct slots rather than accumulated per thread, so the result does
not depend on the scheduler.
"""
function reduce_max_jk(f::F, jrange::UnitRange{Int}, krange::UnitRange{Int},
                       ::Type{T}) where {F,T}
    nj = length(jrange)
    j0 = first(jrange)
    k0 = first(krange)
    parts = Vector{T}(undef, nj * length(krange))
    threaded_jk(jrange, krange) do j, k
        @inbounds parts[(k-k0)*nj+(j-j0)+1] = f(j, k)
    end
    return maximum(parts)
end

function Base.show(io::IO, g::CartesianGrid{T}) where {T}
    print(io, "CartesianGrid{", T, "}(n = ", g.n, ", ng = ", g.ng,
          ", extent = ", ntuple(d -> (g.xlo[d], g.xhi[d]), 3), ")")
end
