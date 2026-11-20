# The tiled hydrodynamic sweep.
#
# The untiled sweep along `D` makes three passes over the whole grid:
# reconstruction writes the interface states `qL` and `qR`, the Riemann solver
# reads them and writes `flux`, and the divergence reads `flux`. At 128^3 each of
# those arrays is 84 MB. Past about eight threads the passes stop scaling,
# because the memory bus, not the arithmetic, is what they wait for.
#
# The tiled sweep runs the same three steps on one box of cells at a time. The
# interface states and fluxes of a box live in a per-task scratch array small
# enough to stay in cache, and only `dU` is written to the whole grid. This is
# AMReX's `MFIter` tiling, which is how Quokka itself does it: box-local
# temporaries, the same pointwise kernels. Here the kernels are literally the
# same, the row bodies `_reconstruct_row!`, `_hydro_flux_row!` and
# `_divergence_row!`, called with a `TileArray` in place of the whole-grid
# array, so the result is bit-identical to the untiled sweep.
#
# A box spans the whole interior along the sweep direction and a block of
# `tile[d]` cells along each transverse direction `d`. A box along the sweep
# would recompute the faces on its two ends, and they are the expensive part.

"""
    DEFAULT_TILE

The transverse block sizes `(b1, b2, b3)` of the tiled hydrodynamic sweep. A
sweep along `D` ignores `b_D` and takes the whole interior along `D`, so with
`(32, 1, 1)` the `x` sweep runs one row of cells at a time and the `y` and `z`
sweeps one 32-cell-wide strip of a plane. The scratch of a box, interface states
and fluxes, is `3 nvar (n_D + 3) b_d1 b_d2` numbers: 0.5 MB at 128^3 and 1 MB at
256^3, about a core's L2. Chosen by measurement on a 32-core node; see
`docs/parallel-findings.md`.
"""
const DEFAULT_TILE = (32, 1, 1)

"""
    TILE_MIN_THREADS

Below this many threads, `Simulation(tile = :auto)` (the default) does not tile,
nor on a grid where a sweep would have fewer boxes than threads (a 2D run below
about 1024^2 at 32 threads).
Tiling pays by relieving the memory bus that many cores share, and costs 4-7% on
one core, so on this node it loses at 2 and 4 threads and wins from 8 on (5% at
8, 8% at 16, 30% at 32). The answer is the same either way.
"""
const TILE_MIN_THREADS = 8

# The tile the `tile` keyword of `Simulation` resolves to.
_resolve_tile(tile::Union{Nothing,NTuple{3,Int}}, g::CartesianGrid) = tile
function _resolve_tile(tile::Symbol, g::CartesianGrid)
    tile === :auto || throw(ArgumentError("tile must be :auto, nothing or a tuple of three Ints, got :$tile"))
    return _auto_tile(g, Threads.nthreads())
end

# `:auto`: `DEFAULT_TILE` with `TILE_MIN_THREADS` threads or more, and only if the
# sweep along every active direction has at least one box per thread. A sweep's
# tasks are its boxes, and a 2D run's transverse sweep has few: at 512^2 the `y`
# sweep has 16 boxes of (32, 1, 1), half of 32 threads idle, and the step was 4%
# slower than untiled; at 1024^2, 32 boxes, it was 18% faster.
function _auto_tile(g::CartesianGrid, nthreads::Integer)
    nthreads >= TILE_MIN_THREADS || return nothing
    for D in 1:3
        g.active[D] || continue
        nbox = prod(d -> d == D ? 1 : cld(g.n[d], DEFAULT_TILE[d]), 1:3)
        nbox >= nthreads || return nothing
    end
    return DEFAULT_TILE
end

"""
    TileArray(buf, dims, offset)

A box-local array indexed with *global* cell indices: `A[n, i, j, k]` is element
`(n, i - offset[1], j - offset[2], k - offset[3])` of the column-major array of
size `dims` stored at the start of `buf`. It is the counterpart of AMReX's
`Array4`, and lets the whole-grid kernels' row bodies write the interface states
and fluxes of one tile without a change. One buffer serves the sweeps in every
direction, each with its own `dims`.

Deliberately not an `AbstractArray`, like [`SVec`](@ref), because its axes
would not be `1:size`. Only scalar `getindex` and `setindex!` are defined, so
nothing else can be done with it by accident.
"""
struct TileArray{T}
    buf::Vector{T}
    dims::NTuple{4,Int}
    offset::NTuple{3,Int}
end

@inline function _tile_index(A::TileArray, n::Int, i::Int, j::Int, k::Int)
    d = A.dims
    return n + d[1] * ((i - A.offset[1] - 1) + d[2] * ((j - A.offset[2] - 1) +
                                                      d[3] * (k - A.offset[3] - 1)))
end
Base.@propagate_inbounds Base.getindex(A::TileArray, n::Int, i::Int, j::Int, k::Int) =
    A.buf[_tile_index(A, n, i, j, k)]
Base.@propagate_inbounds Base.setindex!(A::TileArray, v, n::Int, i::Int, j::Int, k::Int) =
    (A.buf[_tile_index(A, n, i, j, k)] = v)

"""
    has_tiled_method(reconstruction) -> Bool

Whether the tiled sweep can run `reconstruction`. True for `PCM`, `PLM` and
`PPM`, whose row bodies it calls. A reconstruction defined outside the package
owns its whole sweep through `reconstruct!`, so it runs untiled.
"""
has_tiled_method(::AbstractReconstruction) = false
has_tiled_method(::Union{PCM,PLM,PPM}) = true

"""
    tile_boxes(grid, D, tile) -> Vector{NTuple{3,UnitRange{Int}}}

The boxes of interior cells that the tiled sweep along `D` visits: the whole
interior along `D`, blocks of `tile[d]` cells along each other direction (the
last block may be shorter), together exactly the interior. They are ordered with
the first index fastest, so consecutive boxes, which run on the same task, are
neighbours in memory.
"""
function tile_boxes(g::CartesianGrid, D::Integer, tile::NTuple{3,Int})
    blocks = ntuple(Val(3)) do d
        lo, hi = g.lo[d], g.hi[d]
        d == D ? [lo:hi] : [b:min(b + tile[d] - 1, hi) for b in lo:tile[d]:hi]
    end
    return vec([(ir, jr, kr) for ir in blocks[1], jr in blocks[2], kr in blocks[3]])
end

# The scratch dimensions of a sweep along `D`: along `D` the interface states are
# written from cell `lo - 1` to face `hi + 2`, `n_D + 3` entries; across, one box.
function _tile_dims(g::CartesianGrid, D::Integer, tile::NTuple{3,Int}, nvar::Integer)
    (Int(nvar), ntuple(d -> d == D ? g.n[d] + 3 : min(tile[d], g.n[d]), Val(3))...)
end

"""
    HydroTiles{T}

What the tiled sweep needs, built once with the simulation: the boxes and the
scratch shape of each direction, and one `(qL, qR, flux)` set of buffers per
task. A task's scratch is indexed by its chunk number, never by `threadid()`, so
no two tasks share one.
"""
struct HydroTiles{T<:AbstractFloat}
    tile::NTuple{3,Int}
    boxes::NTuple{3,Vector{NTuple{3,UnitRange{Int}}}}
    dims::NTuple{3,NTuple{4,Int}}
    scratch::Vector{NTuple{3,Vector{T}}}          # [task] = (qL, qR, flux)
end

function HydroTiles(g::CartesianGrid{T}, nvar::Integer, tile::NTuple{3,Int}) where {T}
    all(>(0), tile) || throw(ArgumentError("tile sizes must be positive, got $tile"))
    boxes = ntuple(D -> g.active[D] ? tile_boxes(g, D, tile) : NTuple{3,UnitRange{Int}}[], Val(3))
    dims = ntuple(D -> _tile_dims(g, D, tile, nvar), Val(3))
    len = maximum(prod, dims)
    scratch = [ntuple(_ -> Vector{T}(undef, len), Val(3)) for _ in 1:Threads.nthreads()]
    return HydroTiles{T}(tile, boxes, dims, scratch)
end

"""
    hydro_sweep_tiled!(dU, prim, tiles, reconstruction, eos, solver, Val(D), grid,
                       nvar, nscalars, overwrite, update = nothing)

The hydrodynamic sweep along `D`, box by box: for each box of
[`tile_boxes`](@ref), reconstruct the interface states, solve the Riemann
problems and take the flux divergence into `dU`, with the states and fluxes kept
in the task's scratch. Bit-identical to `reconstruct!`, `hydro_fluxes!` and
`flux_divergence!` in turn, because it runs their row bodies.
"""
function hydro_sweep_tiled!(dU::Array{T,4}, prim::Array{T,4}, tiles::HydroTiles{T},
                            recon::AbstractReconstruction, eos::AbstractEOS,
                            solver::AbstractRiemannSolver, ::Val{D}, g::CartesianGrid{T},
                            nvar::Int, nscalars::Int, overwrite::Bool,
                            update=nothing) where {T,D}
    boxes = tiles.boxes[D]
    nbox = length(boxes)
    nch = min(length(tiles.scratch), nbox)
    gam = T(adiabatic_index(eos))
    act1 = g.active[_tdir1(Val(D))]
    act2 = g.active[_tdir2(Val(D))]
    invdx = one(T) / g.dx[D]
    ow = Val(overwrite)                 # a type from here on, so no kernel tests it
    if nch <= 1
        _sweep_chunk!(dU, prim, tiles, 1, 1, recon, eos, solver, Val(D), g, gam, act1, act2,
                      invdx, nvar, nscalars, ow, update)
    else
        # Each task picks its scratch and its boxes from its chunk number `c`, the
        # loop variable, which is a fresh binding per iteration. Destructuring the
        # scratch out here instead would assign the same locals on every iteration
        # and box them, and the tasks would share one scratch.
        @sync for c in 1:nch
            Threads.@spawn _sweep_chunk!(dU, prim, tiles, c, nch, recon, eos, solver, Val(D),
                                         g, gam, act1, act2, invdx, nvar, nscalars, ow, update)
        end
    end
    return nothing
end

function _sweep_chunk!(dU, prim, tiles::HydroTiles, c::Int, nch::Int, recon, eos, solver,
                       ::Val{D}, g, gam, act1, act2, invdx, nvar, nscalars, ow::Val,
                       update) where {D}
    qL, qR, fl = tiles.scratch[c]
    boxes = tiles.boxes[D]
    _sweep_boxes!(dU, prim, qL, qR, fl, tiles.dims[D], boxes, _chunk(length(boxes), nch, c),
                  recon, eos, solver, Val(D), g, gam, act1, act2, invdx, nvar, nscalars,
                  ow, update)
    return nothing
end

function _sweep_boxes!(dU, prim, qL::V, qR::V, fl::V, dims, boxes, r, recon, eos,
                       solver, ::Val{D}, g, gam, act1, act2, invdx, nvar, nscalars,
                       ow::Val, update) where {V,D}
    for b in r
        box = boxes[b]
        off = ntuple(d -> d == D ? g.lo[d] - 2 : first(box[d]) - 1, Val(3))
        tL = TileArray(qL, dims, off)
        tR = TileArray(qR, dims, off)
        tF = TileArray(fl, dims, off)
        # cells extended by one along D, as `cell_ranges` does for the whole grid
        cr = ntuple(d -> d == D ? ((first(box[d]) - 1):(last(box[d]) + 1)) : box[d], Val(3))
        launch!(CPUSerial(), cr, _reconstruct_cell!, tL, tR, prim, recon, Val(D), nvar)
        # faces lo:hi+1 along D, as `face_ranges` does
        fr = ntuple(d -> d == D ? (first(box[d]):(last(box[d]) + 1)) : box[d], Val(3))
        launch!(CPUSerial(), fr, _hydro_flux_cell!, tF, tL, tR, prim, eos, solver, Val(D), gam,
                act1, act2, nscalars)
        launch!(CPUSerial(), box, _divergence_cell!, dU, tF, Val(D), invdx, nvar,
                ow, update)
    end
    return nothing
end
