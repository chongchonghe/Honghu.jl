# Boundary conditions.
#
# A boundary condition only ever has to fill ghost cells. Because the momentum
# component for direction `d` lives at index `d + 1` in the hydro state vector
# and at `d + 1` *within each group* of the radiation state vector, the same
# code serves both systems: see the stride in the reflecting condition below.

"""
    BoundaryCondition

Supertype for boundary conditions. Built-in conditions are [`Periodic`](@ref),
[`Outflow`](@ref), [`Reflecting`](@ref), [`Fixed`](@ref) and [`FunctionBC`](@ref).

To add your own, define

```julia
struct MyBC <: Honghu.BoundaryCondition end
Honghu.fill_face!(A, ::MyBC, grid, dir::Int, side::Int, t) = ...
```

where `side` is `1` for the low face and `2` for the high face, and `A` is the
`(nvar, i, j, k)` state array whose ghost cells must be written.
"""
abstract type BoundaryCondition end

"""
    Periodic()

Wrap around to the opposite side of the domain.
"""
struct Periodic <: BoundaryCondition end

"""
    Outflow()

Zero-gradient (transmissive) condition: copy the adjacent interior cell.
"""
struct Outflow <: BoundaryCondition end

"""
    Reflecting()

Mirror the interior state and flip the sign of the normal momentum (hydro) or
the normal radiation flux of every frequency group (radiation).
"""
struct Reflecting <: BoundaryCondition end

"""
    Fixed(state)

Hold the ghost cells at a constant *conserved* state. `state` is a tuple or
`SVec` covering the whole state vector: `NHYDRO` values on a hydro boundary, and
on a radiation boundary `NRAD * ngroups`, the groups contiguous and four
components each, so a two-group state is

    (E₁, Fx₁, Fy₁, Fz₁, E₂, Fx₂, Fy₂, Fz₂)

This is the one place a multigroup problem has to write its group structure out
by hand -- nothing splits a boundary state the way `set_initial_conditions!`
splits a scalar `Erad` -- and a state of the wrong length is a
`DimensionMismatch` at the first ghost fill, not a silent grey boundary.
`Honghu.Problems.marshak_vaytet` builds one from [`planck_fractions`](@ref) and
is the worked example.

Used e.g. to drive a Marshak wave by imposing an incident radiation energy
density and flux at the low-x face.
"""
struct Fixed{N,T} <: BoundaryCondition
    state::SVec{N,T}
end

Fixed(state::Tuple) = Fixed(SVec(promote(float.(state)...)))
Fixed(state::AbstractVector) = Fixed(Tuple(state))

"""
    FunctionBC(f)

Hold the ghost cells at a *conserved* state computed from position and time,
`f(x, y, z, t) -> NTuple` or `SVec` of length `NHYDRO` on a hydro boundary and
`NRAD * ngroups` on a radiation one, with the same group-contiguous layout
[`Fixed`](@ref) documents.

The time-dependent counterpart of [`Fixed`](@ref), which ignores the `t` it is
handed. Covers a driven inflow, a source that switches on, or a boundary profile
that varies across the face:

```julia
# a beam whose intensity ramps up over one light-crossing time (one group)
FunctionBC((x, y, z, t) -> (min(t, 1.0), 0.999 * min(t, 1.0), 0.0, 0.0))
```

`f` is called once per ghost cell per fill, so it should be cheap, but it is not
in the cell loop and need not be allocation-free.
"""
struct FunctionBC{F} <: BoundaryCondition
    f::F
end

"""
    Boundaries(x, y, z)
    Boundaries(bc)

Boundary conditions for the three directions. Each of `x`, `y`, `z` is a
`(low, high)` tuple of [`BoundaryCondition`](@ref)s; the single-argument form
applies `bc` to all six faces.

```julia
Boundaries(Periodic())
Boundaries((Outflow(), Outflow()), (Periodic(), Periodic()), (Periodic(), Periodic()))
```
"""
struct Boundaries{X,Y,Z}
    x::X
    y::Y
    z::Z
end

Boundaries(bc::BoundaryCondition) = Boundaries((bc, bc), (bc, bc), (bc, bc))
Boundaries(; x=(Outflow(), Outflow()), y=x, z=y) = Boundaries(x, y, z)

@inline facebc(b::Boundaries, dir::Int, side::Int) =
    dir == 1 ? b.x[side] : dir == 2 ? b.y[side] : b.z[side]

# A view of the slab of constant index `idx` along direction `dir`.
@inline function _slab(A::AbstractArray{T,4}, dir::Int, idx::Int) where {T}
    if dir == 1
        return view(A, :, idx, :, :)
    elseif dir == 2
        return view(A, :, :, idx, :)
    else
        return view(A, :, :, :, idx)
    end
end

"""
    fill_ghosts!(A, boundaries, grid, t = 0)

Fill every ghost cell of the state array `A` according to `boundaries`.
Inactive directions are skipped.
"""
function fill_ghosts!(A::AbstractArray{T,4}, b::Boundaries, g::CartesianGrid, t=zero(T)) where {T}
    for dir in 1:3
        g.active[dir] || continue
        fill_face!(A, facebc(b, dir, 1), g, dir, 1, t)
        fill_face!(A, facebc(b, dir, 2), g, dir, 2, t)
    end
    return A
end

"""
    fill_face!(A, bc, grid, dir, side, t)

Fill the `grid.ng[dir]` ghost cells on one face. `side == 1` is the low face,
`side == 2` the high face.
"""
function fill_face! end

function fill_face!(A::AbstractArray{T,4}, ::Periodic, g::CartesianGrid, dir::Int, side::Int, t) where {T}
    ng, lo, hi = g.ng[dir], g.lo[dir], g.hi[dir]
    for m in 1:ng
        if side == 1
            copyto!(_slab(A, dir, lo - m), _slab(A, dir, hi + 1 - m))
        else
            copyto!(_slab(A, dir, hi + m), _slab(A, dir, lo - 1 + m))
        end
    end
    return nothing
end

function fill_face!(A::AbstractArray{T,4}, ::Outflow, g::CartesianGrid, dir::Int, side::Int, t) where {T}
    ng, lo, hi = g.ng[dir], g.lo[dir], g.hi[dir]
    src = side == 1 ? lo : hi
    for m in 1:ng
        dst = side == 1 ? lo - m : hi + m
        copyto!(_slab(A, dir, dst), _slab(A, dir, src))
    end
    return nothing
end

function fill_face!(A::AbstractArray{T,4}, ::Reflecting, g::CartesianGrid, dir::Int, side::Int, t) where {T}
    ng, lo, hi = g.ng[dir], g.lo[dir], g.hi[dir]
    # Normal momentum (hydro) / normal radiation flux of every group. The
    # radiation state stores `NRAD` components per group contiguously, so the
    # normal flux of group `gr` is at `NRAD*(gr-1) + dir + 1`; striding by
    # `NRAD` from `dir + 1` hits exactly those. The five-variable hydro vector
    # is not a multiple of `NRAD`, so the same stride hits `dir + 1` and stops,
    # which is why one implementation still serves both.
    for m in 1:ng
        dst = side == 1 ? lo - m : hi + m
        src = side == 1 ? lo + m - 1 : hi - m + 1
        d = _slab(A, dir, dst)
        copyto!(d, _slab(A, dir, src))
        for nflip in (dir+1):NRAD:size(A, 1)
            @views d[nflip, :, :] .= .-d[nflip, :, :]
        end
    end
    return nothing
end

function fill_face!(A::AbstractArray{T,4}, bc::FunctionBC, g::CartesianGrid, dir::Int,
                    side::Int, t) where {T}
    nv = size(A, 1)
    others = dir == 1 ? (2, 3) : dir == 2 ? (1, 3) : (1, 2)
    for m in 1:g.ng[dir]
        idx = side == 1 ? g.lo[dir] - m : g.hi[dir] + m
        for kk in 1:g.dims[others[2]], jj in 1:g.dims[others[1]]
            sub = ntuple(3) do d
                d == dir ? idx : (d == others[1] ? jj : kk)
            end
            x, y, z = cellcenter(g, sub[1], sub[2], sub[3])
            s = bc.f(x, y, z, t)
            length(s) == nv ||
                throw(DimensionMismatch("FunctionBC returned $(length(s)) components but the state has $nv"))
            @inbounds for n in 1:nv
                A[n, sub[1], sub[2], sub[3]] = T(s[n])
            end
        end
    end
    return nothing
end

function fill_face!(A::AbstractArray{T,4}, bc::Fixed{N}, g::CartesianGrid, dir::Int, side::Int, t) where {T,N}
    size(A, 1) == N ||
        throw(DimensionMismatch("Fixed boundary has $N components but the state has $(size(A,1))"))
    ng, lo, hi = g.ng[dir], g.lo[dir], g.hi[dir]
    for m in 1:ng
        dst = side == 1 ? lo - m : hi + m
        d = _slab(A, dir, dst)
        for n in 1:N
            @views fill!(d[n, :, :], T(bc.state[n]))
        end
    end
    return nothing
end
