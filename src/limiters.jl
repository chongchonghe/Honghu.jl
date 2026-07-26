# Slope limiters for piecewise-linear reconstruction.
#
# Each limiter maps the backward and forward differences (dL, dR) of a cell to a
# single limited slope, so that `q(x) = q_i + slope * (x - x_i)/dx` is monotone.

"""
    AbstractLimiter

Supertype for slope limiters. Implement `limit(lim, dL, dR)`, returning the
limited undivided slope across the cell.
"""
abstract type AbstractLimiter end

"""
    MinMod()

`minmod(dL, dR)`. The most diffusive of the TVD limiters, and the most robust.
"""
struct MinMod <: AbstractLimiter end

"""
    MonotonizedCentral()

The MC limiter of van Leer (1977), `minmod(2dL, (dL+dR)/2, 2dR)`. The default:
sharper than minmod while remaining TVD.
"""
struct MonotonizedCentral <: AbstractLimiter end

"""
    VanLeer()

The harmonic-mean limiter `2 dL dR / (dL + dR)` for `dL dR > 0`, else zero.
Smooth, which helps steady-state convergence.
"""
struct VanLeer <: AbstractLimiter end

"""
    Sweby(beta = 1.5)

Generalised minmod limiter `minmod(beta dL, (dL+dR)/2, beta dR)`. TVD for
`1 <= beta <= 2`; `beta = 1` is minmod and `beta = 2` is MC.
"""
struct Sweby{T<:AbstractFloat} <: AbstractLimiter
    beta::T
end
Sweby(beta::Real=1.5) = Sweby(float(beta))

"""
    limit(limiter, dL, dR)

Limited slope of a cell whose backward and forward differences are `dL`, `dR`.
"""
function limit end

@inline function minmod(a::T, b::T) where {T}
    ifelse(a * b <= zero(T), zero(T), ifelse(abs(a) < abs(b), a, b))
end

@inline function minmod(a::T, b::T, c::T) where {T}
    (a * b <= zero(T) || b * c <= zero(T)) && return zero(T)
    sign(a) * min(abs(a), abs(b), abs(c))
end

@inline limit(::MinMod, dL::T, dR::T) where {T} = minmod(dL, dR)

@inline function limit(::MonotonizedCentral, dL::T, dR::T) where {T}
    minmod(T(2) * dL, T(0.5) * (dL + dR), T(2) * dR)
end

@inline function limit(::VanLeer, dL::T, dR::T) where {T}
    p = dL * dR
    p <= zero(T) ? zero(T) : T(2) * p / (dL + dR)
end

@inline function limit(lim::Sweby, dL::T, dR::T) where {T}
    b = T(lim.beta)
    minmod(b * dL, T(0.5) * (dL + dR), b * dR)
end
