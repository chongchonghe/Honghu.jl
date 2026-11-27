# One launch point for the cell kernels.
#
# A kernel is a function of one cell, `f(i, j, k, args...)`, and `launch!` calls
# it for every cell of a box. Which hardware runs the calls is the backend's
# business, not the kernel's. This is AMReX's `ParallelFor`, which is how Quokka
# runs the same kernel on a CPU and on a GPU, and it is what lets Honghu's
# kernels transliterate there and back.
#
# The arguments are passed to `launch!` explicitly rather than captured by a
# closure, for two reasons. A GPU backend has to convert every array argument to
# its device form, which it can do to an argument list but not to the inside of
# a closure. And a closure built inside a spawn loop can box what it captures
# (CLAUDE.md, performance rule 3); a kernel that captures nothing cannot.

"""
    AbstractBackend

Where [`launch!`](@ref) runs a cell kernel. [`CPUThreads`](@ref) and
[`CPUSerial`](@ref) are built in; [`KABackend`](@ref) runs on any device
KernelAbstractions supports, a GPU included, once that package is loaded.
"""
abstract type AbstractBackend end

"""
    CPUThreads()

The default backend: the rows `(j, k)` of a box are spread over
`Threads.nthreads()` threads, and each row is one inner loop over `i`. The inner
loop is what the compiler vectorises, so a row is the CPU's unit of work while
the cell stays the kernel's.
"""
struct CPUThreads <: AbstractBackend end

"""
    CPUSerial()

The whole box on the calling task, `k` outermost and `i` innermost. For work
that is already one task of a parallel loop, such as one tile of the tiled hydro
sweep.
"""
struct CPUSerial <: AbstractBackend end

"""
    KABackend(backend)

Runs a cell kernel as a KernelAbstractions kernel on `backend`, one work-item
per cell: `KABackend(CUDABackend())` on an NVIDIA GPU, with the arrays given as
`CuArray`s, or `KABackend(KernelAbstractions.CPU())` on the CPU, which is how
the test suite checks the device path without a device. Its `launch!` lives in
the extension `HonghuKernelAbstractionsExt`, loaded with KernelAbstractions.
"""
struct KABackend{B} <: AbstractBackend
    backend::B
end

"""
    launch!(backend, box, f, args...)

Call `f(i, j, k, args...)` once for every cell `(i, j, k)` of
`box = (irange, jrange, krange)`, on `backend`. The calls may run in any order
and concurrently, so `f` must write only to cell `(i, j, k)`'s own entries (or
to entries no other cell of the box writes).
"""
function launch!(::CPUThreads, box::NTuple{3,UnitRange{Int}}, f::F,
                 args::Vararg{Any,N}) where {F,N}
    isempty(box[1]) && return nothing
    _launch_threads!(box, f, _static_rows(args)...)
    return nothing
end

# Past the function barrier the arrays carry their row count in their type.
function _launch_threads!(box::NTuple{3,UnitRange{Int}}, f::F, args::Vararg{Any,N}) where {F,N}
    ir = box[1]
    # `ivdep`: the calls are independent, which is this function's contract, but
    # LLVM cannot see it through the kernel's array arguments, which may alias, and
    # without it leaves the loop scalar. With it a straight-line kernel runs over
    # eight cells at a time (AVX-512). Each lane does the same IEEE arithmetic as
    # the scalar loop, so the answer is the same to the bit.
    threaded_jk(box[2], box[3]) do j, k
        @simd ivdep for i in ir
            f(i, j, k, args...)
        end
    end
    return nothing
end

# The arguments go to the kernel as they are: `CPUSerial` runs one small box of a
# task, many times over, and a dynamic dispatch per launch would allocate per box. Its
# caller, the tiled sweep, hands it `StaticRows` it built type-stably instead.
function launch!(::CPUSerial, box::NTuple{3,UnitRange{Int}}, f::F,
                 args::Vararg{Any,N}) where {F,N}
    for k in box[3], j in box[2]
        @simd ivdep for i in box[1]        # see `_launch_threads!`
            f(i, j, k, args...)
        end
    end
    return nothing
end

# The number of variables a kernel loops over, given as an `Int` or as `Val(n)`. The
# launch sites pass `Val(nvar)`: a constant trip count lets LLVM unroll the loop over
# variables, which leaves the loop over `i` innermost, the one it vectorises.
@inline _count(n::Integer) = n
@inline _count(::Val{N}) where {N} = N
@inline _val(n::Integer) = Val(n)
@inline _val(v::Val) = v

"""
    StaticRows{R}(data, dims, offset)

A four-index array `(n, i, j, k)` whose row count `R = dims[1]` is part of its
type: what [`launch!`](@ref) hands a cell kernel on the CPU in place of an
`Array{T,4}` or a [`TileArray`](@ref), at the same memory. An `Array`'s stride
along `i` is `size(A, 1)`, a run-time number, so a vectorised loop over `i` has
to gather every element it reads from a separate address; with the stride a
constant, LLVM loads a face array (`R = 1`) contiguously and a state array
(`R = 5`) with wide loads and shuffles. The kernels index it as they index the
array, and the addresses are the same, so the answer is the same to the bit.
"""
struct StaticRows{R,T,A<:AbstractArray{T}} <: AbstractArray{T,4}
    data::A
    dims::NTuple{4,Int}
    offset::NTuple{3,Int}
end

Base.size(A::StaticRows) = A.dims
Base.axes(A::StaticRows) = (Base.OneTo(A.dims[1]), (A.offset[d-1] .+ (1:A.dims[d]) for d in 2:4)...)

@inline function _rows_index(A::StaticRows{R}, n::Int, i::Int, j::Int, k::Int) where {R}
    d, o = A.dims, A.offset
    return n + R * ((i - o[1] - 1) + d[2] * ((j - o[2] - 1) + d[3] * (k - o[3] - 1)))
end
Base.@propagate_inbounds Base.getindex(A::StaticRows, n::Int, i::Int, j::Int, k::Int) =
    A.data[_rows_index(A, n, i, j, k)]
Base.@propagate_inbounds Base.setindex!(A::StaticRows, v, n::Int, i::Int, j::Int, k::Int) =
    (A.data[_rows_index(A, n, i, j, k)] = v)

# The arguments of a launch with every array of a few rows given its row count as a
# type, and tuples (a Runge-Kutta `update`) walked into. One dynamic dispatch per
# launch, which the barrier `_launch_threads!` / `_launch_serial!` absorbs.
_static_rows(args::Tuple) = map(_static_row, args)
_static_row(x) = x
_static_row(t::Tuple) = map(_static_row, t)
function _static_row(A::Array{T,4}) where {T}
    R = size(A, 1)
    1 <= R <= 16 || return A
    return StaticRows{R,T,Array{T,4}}(A, size(A), (0, 0, 0))
end

# The same, type-stably, for a caller that knows the row count `R` of every array it
# passes: each wrapped, anything else passed through. Returning the bare array on a
# mismatch would make the result a `Union` and allocate, so a mismatch is an error.
_static_row(x, ::Val) = x
_static_row(t::Tuple, r::Val) = map(a -> _static_row(a, r), t)
function _static_row(A::Array{T,4}, ::Val{R}) where {T,R}
    size(A, 1) == R || throw(DimensionMismatch("expected $R rows, got $(size(A, 1))"))
    return StaticRows{R,T,Array{T,4}}(A, size(A), (0, 0, 0))
end

function launch!(b::KABackend, box::NTuple{3,UnitRange{Int}}, f, args...)
    error("KABackend($(b.backend)) needs KernelAbstractions: `using KernelAbstractions` " *
          "(and the package of the device, such as CUDA) loads the extension that runs it")
end
