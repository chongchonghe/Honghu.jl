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
    ir = box[1]
    isempty(ir) && return nothing
    threaded_jk(box[2], box[3]) do j, k
        for i in ir
            f(i, j, k, args...)
        end
    end
    return nothing
end

function launch!(::CPUSerial, box::NTuple{3,UnitRange{Int}}, f::F,
                 args::Vararg{Any,N}) where {F,N}
    for k in box[3], j in box[2], i in box[1]
        f(i, j, k, args...)
    end
    return nothing
end

function launch!(b::KABackend, box::NTuple{3,UnitRange{Int}}, f, args...)
    error("KABackend($(b.backend)) needs KernelAbstractions: `using KernelAbstractions` " *
          "(and the package of the device, such as CUDA) loads the extension that runs it")
end
