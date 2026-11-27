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
    return _sqrt(T(0.5) * (bpg + _sqrt(bmg * bmg + 4 * gp * btsq)) / rho)
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

Fill the ghost faces of all three field components, one boundary at a time. Across
a component's own direction its faces are laid out like cells, and
[`fill_face!`](@ref) fills them as it fills cells. Along it there is one more face
than cells:

  - [`Periodic`](@ref): face `hi + 1` is face `lo`, so ghost face `lo - m` is
    face `hi + 1 - m` and ghost face `hi + 1 + m` is face `lo + m`. Faces `lo` and
    `hi + 1` are both evolved, and stay equal because their updates read the same
    numbers.
  - [`Outflow`](@ref): the transverse components are copied from the boundary
    row, and the normal component is integrated outward from the boundary face so
    that `div b = 0` in every ghost cell, as Quokka's MHD diode condition
    (`applyMHDDiodeBC`) does for an outflowing boundary. A copied normal face would
    leave a ghost cell a divergence wherever the transverse field varies along the
    boundary, and the field of a static equilibrium out of balance there.

Only these two are defined; [`check_mhd_boundaries`](@ref) refuses the others
when the simulation is built. The ghost cells of the gas state then need
[`mhd_ghost_energy!`](@ref).
"""
function fill_ghosts!(B::MagneticField{T}, b::Boundaries, g::CartesianGrid) where {T}
    for dir in 1:3
        g.active[dir] || continue
        for side in 1:2
            bc = facebc(b, dir, side)
            # the components across `dir` first: the normal one is integrated from them
            for d in 1:3
                d == dir || fill_face!(B.b[d], bc, g, dir, side, zero(T))
            end
            _fill_normal_faces!(B, bc, g, dir, side)
        end
    end
    return B
end

function _fill_normal_faces!(B::MagneticField, ::Periodic, g::CartesianGrid, dir::Int, side::Int)
    A = B.b[dir]
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

# Quokka's step 2 of `applyMHDDiodeBC`: from the boundary face outward, each ghost
# cell's new normal face closes `div b = 0` with the transverse faces of that cell,
# `b_new = b_known - s dx (sum_t (b_t[high] - b_t[low]) / dx_t)`, `s` the outward sign.
function _fill_normal_faces!(B::MagneticField{T}, ::Outflow, g::CartesianGrid, dir::Int,
                             side::Int) where {T}
    A = B.b[dir]
    o = B.o[dir]
    sgn = side == 1 ? -1 : 1
    across = ntuple(e -> e == dir ? (1:1) : (1:g.dims[e]), 3)
    @inbounds for m in 1:g.ng[dir]
        c = side == 1 ? g.lo[dir] - m : g.hi[dir] + m
        for I in CartesianIndices(across)
            i, j, k = Base.setindex(Tuple(I), c, dir)
            divt = zero(T)
            for t in 1:3
                (t == dir || !g.active[t]) && continue
                ot = B.o[t]
                Bt = B.b[t]
                divt += (Bt[1, i+ot[1], j+ot[2], k+ot[3]] - Bt[1, i, j, k]) / g.dx[t]
            end
            if side == 1          # the new face is the cell's low face
                A[1, i, j, k] = A[1, i+o[1], j+o[2], k+o[3]] - sgn * g.dx[dir] * divt
            else
                A[1, i+o[1], j+o[2], k+o[3]] = A[1, i, j, k] - sgn * g.dx[dir] * divt
            end
        end
    end
    return nothing
end

"""
    mhd_ghost_energy!(U, B, boundaries, grid)

Give every ghost cell beyond an [`Outflow`](@ref) boundary the internal energy of
the interior cell it was copied from, by replacing that cell's magnetic energy in
the copied total energy with the ghost cell's own: Quokka's step 3 of
`applyMHDDiodeBC`. Its ghost faces differ from the source cell's (the normal one
is integrated, see [`fill_ghosts!`](@ref)), so a plain copy of the total energy
would give the ghost cell another pressure. Call it after the ghost cells of `U`
and the ghost faces of `B` are filled. Without an outflow boundary it does nothing:
a periodic ghost cell's faces are images of its source's, and so is its energy.
"""
function mhd_ghost_energy!(U::AbstractArray{T,4}, B::MagneticField{T}, b::Boundaries,
                           g::CartesianGrid) where {T}
    outflow = ntuple(d -> g.active[d] && (facebc(b, d, 1) isa Outflow || facebc(b, d, 2) isa Outflow), 3)
    any(outflow) || return U
    n1, n2, n3 = g.dims
    threaded_jk(1:n2, 1:n3) do j, k
        @inbounds for i in 1:n1
            si, sj, sk = _ghost_source(g, b, i, j, k)
            (si, sj, sk) == (i, j, k) && continue
            U[IEGAS, i, j, k] += magnetic_energy(B, i, j, k) - magnetic_energy(B, si, sj, sk)
        end
    end
    return U
end

# The interior cell whose state the ghost fill copied into cell (i, j, k): along each
# direction, a periodic ghost index wraps and an outflow one clamps to the boundary
# cell. Interior cells are their own source.
@inline function _ghost_source(g::CartesianGrid, b::Boundaries, i, j, k)
    idx = (i, j, k)
    return ntuple(Val(3)) do d
        x = idx[d]
        lo, hi = g.lo[d], g.hi[d]
        if x < lo
            facebc(b, d, 1) isa Periodic ? x + g.n[d] : lo
        elseif x > hi
            facebc(b, d, 2) isa Periodic ? x - g.n[d] : hi
        else
            x
        end
    end
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
    # Along a periodic direction face hi + 1 is face lo: make the two the same number,
    # which `f` or `A` evaluated at both ends of the box gives only to round-off.
    for d in 1:3
        g.active[d] && facebc(b, d, 1) isa Periodic || continue
        copyto!(_slab(B.b[d], d, g.hi[d] + 1), _slab(B.b[d], d, g.lo[d]))
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
