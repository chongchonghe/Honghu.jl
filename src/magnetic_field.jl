# The face-centred magnetic field of constrained-transport MHD.
#
# The normal component of the field lives at the centre of each cell face, as in
# Quokka's `state_new_fc_`, so that the update of the field by the circulation of
# the edge EMFs keeps its discrete divergence at round-off (`mhd_system.jl`).
# Everything else that needs the field reads the cell-centred value, the average
# of a cell's two faces along each direction.
#
# The face convention is the one of the rest of the code: face `f` along `d` is
# the low face of cell `f`, so `b[d][1, i, j, k]` is the field on the low face of
# cell `(i, j, k)`, and its high face is `(i, j, k) + o[d]`. Along an inactive
# direction there is one face per cell, and the low and the high face are the same
# face: `o[d]` is zero there. That one convention carries every MHD stencil into 1D
# and 2D: a difference along an inactive direction is exactly zero, an average is
# the value itself, and a reconstruction of a constant stencil returns the
# constant, which is what a thin periodic 3D box uniform along that direction
# computes.

"""
    MagneticField{T}

The face-centred magnetic field `b` of a simulation with `mhd = true`, as
`sim.B`: `sim.B[d]` is the `(1, i, j, k)` array of the component normal to the
faces along direction `d`, with one more face than cells along `d` when `d` is
active, and `sim.B[d][1, i, j, k]` is the field on the *low* face of cell
`(i, j, k)` (the convention of Quokka and of [`reconstruct!`](@ref)). Ghost faces
are filled by the boundary conditions, like ghost cells.

The field is in units in which the magnetic pressure is `|b|^2 / 2`, as in Quokka
(Heaviside-Lorentz, `b = B / sqrt(4 pi)` in Gaussian units). The cell-centred
total energy `sim.U[IEGAS]` includes the magnetic energy `|b_cc|^2 / 2`, with
`b_cc` the average of a cell's two faces along each direction
([`magnetic_energy`](@ref)).
"""
struct MagneticField{T<:AbstractFloat}
    b::NTuple{3,Array{T,4}}
    # o[d]: the step from a cell's low face along `d` to its high face, zero where
    # `d` is inactive
    o::NTuple{3,NTuple{3,Int}}
end

function MagneticField(g::CartesianGrid{T}) where {T}
    o = ntuple(d -> g.active[d] ? ntuple(e -> Int(e == d), 3) : (0, 0, 0), 3)
    b = ntuple(d -> zeros(T, 1, ntuple(e -> g.dims[e] + o[d][e], 3)...), 3)
    return MagneticField{T}(b, o)
end

Base.getindex(B::MagneticField, d::Integer) = B.b[d]
Base.eltype(::MagneticField{T}) where {T} = T

"""
    cell_centred_field(B, d, i, j, k)

Component `d` of the cell-centred field of cell `(i, j, k)`: the average of its
low and high face along `d`. Along an inactive direction the two are the same face.
"""
@inline function cell_centred_field(B::MagneticField, d::Integer, i, j, k)
    A = B.b[d]
    o = B.o[d]
    @inbounds return (A[1, i, j, k] + A[1, i+o[1], j+o[2], k+o[3]]) / 2
end

"""
    magnetic_energy(B, i, j, k)

The magnetic energy density `|b_cc|^2 / 2` of cell `(i, j, k)`, from the
cell-centred field ([`cell_centred_field`](@ref)), as Quokka takes it. It is the
part of the cell's total energy that is neither kinetic nor internal, so every
place that takes the internal energy as `E - E_kin` subtracts it as well, and
every place that writes `E = E_int + E_kin` adds it back.

`B === nothing`, a simulation without MHD, gives `false`, a zero of every float
type: `x - false` is `x`, so the hydrodynamic paths keep their numbers.
"""
@inline magnetic_energy(::Nothing, i, j, k) = false

@inline function magnetic_energy(B::MagneticField, i, j, k)
    b1 = cell_centred_field(B, 1, i, j, k)
    b2 = cell_centred_field(B, 2, i, j, k)
    b3 = cell_centred_field(B, 3, i, j, k)
    return (b1 * b1 + b2 * b2 + b3 * b3) / 2
end

"""
    fast_magnetosonic_speed(gamma, rho, P, bn, bt1, bt2)

The fast magnetosonic speed for propagation along the direction of `bn`, with
`bt1`, `bt2` the two transverse components: Quokka's `FastMagnetoSonicSpeed`
(`HydroState.hpp`), with `gamma P` for the square of the sound speed times `rho`.
"""
@inline function fast_magnetosonic_speed(gamma::T, rho::T, P::T, bn::T, bt1::T,
                                         bt2::T) where {T}
    gp = gamma * P
    btsq = bt1 * bt1 + bt2 * bt2
    bsq = bn * bn + btsq
    bpg = bsq + gp
    bmg = bsq - gp
    return sqrt(T(0.5) * (bpg + sqrt(bmg * bmg + 4 * gp * btsq)) / rho)
end

# --- boundary conditions -----------------------------------------------------

"""
    check_mhd_boundaries(boundaries)

Throw an `ArgumentError` unless every face of `boundaries` is [`Periodic`](@ref)
or [`Outflow`](@ref), the two conditions the face-centred field has.
"""
function check_mhd_boundaries(b::Boundaries)
    for dir in 1:3, side in 1:2
        bc = facebc(b, dir, side)
        bc isa Union{Periodic,Outflow} ||
            throw(ArgumentError("MHD supports Periodic and Outflow boundaries only, got " *
                                "$(nameof(typeof(bc))) on side $side of direction $dir"))
    end
    return nothing
end

"""
    fill_ghosts!(B::MagneticField, boundaries, grid)

Fill the ghost faces of all three field components. Across a component's own
direction its faces are laid out like cells, and [`fill_face!`](@ref) fills them
as it fills cells. Along it there is one more face than cells:

  - [`Periodic`](@ref): face `hi + 1` is face `lo`, so ghost face `lo - m` is
    face `hi + 1 - m` and ghost face `hi + 1 + m` is face `lo + m`. Faces `lo` and
    `hi + 1` are both evolved, and stay equal because their updates read the same
    numbers.
  - [`Outflow`](@ref): ghost faces copy the boundary face, `lo` or `hi + 1`.

Only these two are defined; [`check_mhd_boundaries`](@ref) refuses the others
when the simulation is built.
"""
function fill_ghosts!(B::MagneticField, b::Boundaries, g::CartesianGrid)
    for d in 1:3
        A = B.b[d]
        for dir in 1:3
            g.active[dir] || continue
            for side in 1:2
                bc = facebc(b, dir, side)
                dir == d ? _fill_normal_faces!(A, bc, g, dir, side) :
                fill_face!(A, bc, g, dir, side, zero(eltype(A)))
            end
        end
    end
    return B
end

function _fill_normal_faces!(A::AbstractArray{T,4}, ::Periodic, g::CartesianGrid, dir::Int,
                             side::Int) where {T}
    lo, hi = g.lo[dir], g.hi[dir]
    for m in 1:g.ng[dir]
        if side == 1
            copyto!(_slab(A, dir, lo - m), _slab(A, dir, hi + 1 - m))
        else
            copyto!(_slab(A, dir, hi + 1 + m), _slab(A, dir, lo + m))
        end
    end
    return nothing
end

function _fill_normal_faces!(A::AbstractArray{T,4}, ::Outflow, g::CartesianGrid, dir::Int,
                             side::Int) where {T}
    lo, hi = g.lo[dir], g.hi[dir]
    src = side == 1 ? lo : hi + 1
    for m in 1:g.ng[dir]
        copyto!(_slab(A, dir, side == 1 ? lo - m : hi + 1 + m), _slab(A, dir, src))
    end
    return nothing
end

# --- initial conditions and diagnostics ---------------------------------------

# The coordinate of face index `idx` along an active direction `e`, written out
# rather than as a cell centre minus half a cell, so that a face at `x = 0.5`
# is at 0.5 exactly.
@inline _face_coord(g::CartesianGrid, e::Int, idx::Integer) = g.xlo[e] + (idx - g.lo[e]) * g.dx[e]

# The centre of face (i, j, k) normal to `d`, and of edge (i, j, k) along `w`: on
# the low face along every active direction the point is staggered in, and at the
# cell centre along the others.
@inline function _staggered_point(g::CartesianGrid, stag::NTuple{3,Bool}, i, j, k)
    c = cellcenter(g, i, j, k)
    idx = (i, j, k)
    return ntuple(e -> stag[e] && g.active[e] ? _face_coord(g, e, idx[e]) : c[e], 3)
end
@inline face_center(g::CartesianGrid, d::Int, i, j, k) =
    _staggered_point(g, ntuple(e -> e == d, 3), i, j, k)
@inline edge_center(g::CartesianGrid, w::Int, i, j, k) =
    _staggered_point(g, ntuple(e -> e != w, 3), i, j, k)

# Fill the interior faces of `B` from `f(x, y, z)`'s keys `bx`, `by`, `bz`, evaluated
# at each face centre, plus the discrete curl of `A(x, y, z) -> (Ax, Ay, Az)`
# evaluated at the edge centres, then the ghost faces. The curl is the one the
# induction update takes of the EMF, so the divergence of its part is zero to
# round-off; the keys give a field whose divergence is exactly zero when `b_d`
# does not vary along `d`, a uniform field included.
function _set_face_field!(f::F, B::MagneticField{T}, A, g::CartesianGrid{T},
                          b::Boundaries) where {F,T}
    keys = (Val(:bx), Val(:by), Val(:bz))
    for d in 1:3
        d1, d2 = mod1(d + 1, 3), mod1(d + 2, 3)
        o1, o2 = B.o[d1], B.o[d2]
        Bd = B.b[d]
        r = ntuple(e -> g.lo[e]:(g.hi[e] + B.o[d][e]), 3)
        for k in r[3], j in r[2], i in r[1]
            val = T(_field(f(face_center(g, d, i, j, k)...), keys[d], zero(T)))
            if A !== nothing
                a2(ii, jj, kk) = T(A(edge_center(g, d2, ii, jj, kk)...)[d2])
                a1(ii, jj, kk) = T(A(edge_center(g, d1, ii, jj, kk)...)[d1])
                val += (a2(i + o1[1], j + o1[2], k + o1[3]) - a2(i, j, k)) / g.dx[d1] -
                       (a1(i + o2[1], j + o2[2], k + o2[3]) - a1(i, j, k)) / g.dx[d2]
            end
            Bd[1, i, j, k] = val
        end
    end
    fill_ghosts!(B, b, g)
    return B
end

"""
    divergence_b(sim) -> Array{T,3}

The discrete divergence `sum_d (b_d[high face] - b_d[low face]) / dx_d` of every
interior cell, the quantity constrained transport keeps at round-off. Directions
that are inactive contribute nothing.
"""
function divergence_b(sim)
    B = sim.B
    B === nothing && throw(ArgumentError("divergence_b needs a simulation with mhd = true"))
    g = sim.grid
    T = eltype(g)
    out = Array{T,3}(undef, g.n...)
    @inbounds for k in 1:g.n[3], j in 1:g.n[2], i in 1:g.n[1]
        ii, jj, kk = i + g.ng[1], j + g.ng[2], k + g.ng[3]
        s = zero(T)
        for d in 1:3
            o = B.o[d]
            s += (B.b[d][1, ii+o[1], jj+o[2], kk+o[3]] - B.b[d][1, ii, jj, kk]) / g.dx[d]
        end
        out[i, j, k] = s
    end
    return out
end
