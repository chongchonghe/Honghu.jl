# The device path of `launch!`: a cell kernel run as a KernelAbstractions
# kernel, one work-item per cell. Loaded with KernelAbstractions; with the arrays
# on a GPU (a `CuArray` and `CUDABackend()`, say) the same cell kernels the CPU
# runs run there.

module HonghuKernelAbstractionsExt

using Honghu: Honghu, KABackend
using KernelAbstractions: KernelAbstractions, @kernel, @index

# The work-item `I` of an `ndrange` the size of the box is the cell
# `I .+ offset`, with `offset = first.(box) .- 1`.
@kernel function _cell_kernel!(f, offset, args)
    I = @index(Global, NTuple)
    f(I[1] + offset[1], I[2] + offset[2], I[3] + offset[3], args...)
end

function Honghu.launch!(b::KABackend{<:KernelAbstractions.Backend},
                        box::NTuple{3,UnitRange{Int}}, f::F, args::Vararg{Any,N}) where {F,N}
    any(isempty, box) && return nothing
    offset = (first(box[1]) - 1, first(box[2]) - 1, first(box[3]) - 1)
    kernel! = _cell_kernel!(b.backend)
    kernel!(f, offset, args; ndrange=(length(box[1]), length(box[2]), length(box[3])))
    KernelAbstractions.synchronize(b.backend)
    return nothing
end

end
