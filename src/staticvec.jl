# Minimal stack-allocated fixed-size vector.
#
# Hydro3D deliberately has no external dependencies, so instead of StaticArrays
# we carry this ~50-line type. It exists so that the Riemann solvers can be
# written in vector form (the way the papers write them) without allocating.

"""
    SVec{N,T}

Immutable, stack-allocated vector of `N` elements of type `T`.

Supports the arithmetic used by the Riemann solvers: `+`, `-`, scalar `*` and
`/`, indexing, and iteration. Deliberately *not* an `AbstractArray`, so that a
stray broadcast cannot silently allocate a heap array in a hot loop.

```jldoctest
julia> Hydro3D.SVec(1.0, 2.0) + Hydro3D.SVec(3.0, 4.0)
SVec{2, Float64}(4.0, 6.0)
```
"""
struct SVec{N,T}
    data::NTuple{N,T}
end

@inline SVec(xs::Vararg{Number,N}) where {N} = SVec(promote(xs...))
@inline SVec{N,T}(xs::Vararg{Number,N}) where {N,T} = SVec{N,T}(convert(NTuple{N,T}, xs))

@inline Base.length(::SVec{N}) where {N} = N
@inline Base.eltype(::Type{SVec{N,T}}) where {N,T} = T
@inline Base.getindex(v::SVec, i::Int) = v.data[i]
@inline Base.iterate(v::SVec, s::Int=1) = s > length(v) ? nothing : (v.data[s], s + 1)
@inline Base.Tuple(v::SVec) = v.data
@inline Base.zero(::Type{SVec{N,T}}) where {N,T} = SVec(ntuple(_ -> zero(T), N))

@inline Base.:+(a::SVec{N}, b::SVec{N}) where {N} = SVec(map(+, a.data, b.data))
@inline Base.:-(a::SVec{N}, b::SVec{N}) where {N} = SVec(map(-, a.data, b.data))
@inline Base.:-(a::SVec) = SVec(map(-, a.data))
@inline Base.:*(s::Number, a::SVec) = SVec(map(x -> s * x, a.data))
@inline Base.:*(a::SVec, s::Number) = s * a
@inline Base.:/(a::SVec, s::Number) = SVec(map(x -> x / s, a.data))
@inline Base.:+(a::SVec, s::Number) = SVec(map(x -> x + s, a.data))
@inline Base.:+(s::Number, a::SVec) = a + s
@inline Base.:-(a::SVec, s::Number) = SVec(map(x -> x - s, a.data))

# Elementwise products and quotients between two group vectors. Multigroup
# coupling is full of them -- `kappaP .* fourPiB`, `Erad ./ kappaPoverE` -- and
# spelling them with `map` here keeps the call sites readable without pulling in
# broadcasting, which `SVec` deliberately does not support (a stray broadcast on
# a state vector is exactly the allocation this type exists to prevent).
@inline mul(a::SVec{N}, b::SVec{N}) where {N} = SVec(map(*, a.data, b.data))
@inline divide(a::SVec{N}, b::SVec{N}) where {N} = SVec(map(/, a.data, b.data))

"""
    sumvec(a::SVec)

Sum of the components. Named to avoid extending `Base.sum`, which would invite
the generic iterator machinery into a hot loop.
"""
@inline sumvec(a::SVec) = sum(a.data)

"""
    maxabs(a::SVec)

Largest absolute component, for convergence tests on a group vector.
"""
@inline maxabs(a::SVec) = maximum(map(abs, a.data))

"""
    dot(a::SVec, b::SVec)

Euclidean inner product. (Local to Hydro3D; not `LinearAlgebra.dot`.)
"""
@inline dot(a::SVec{N}, b::SVec{N}) where {N} = sum(map(*, a.data, b.data))

"""
    norm(a::SVec)

Euclidean 2-norm.
"""
@inline norm(a::SVec) = sqrt(dot(a, a))

function Base.show(io::IO, v::SVec{N,T}) where {N,T}
    print(io, "SVec{", N, ", ", T, "}", v.data)
end
