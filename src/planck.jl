# The Planck function integrated over a frequency group.
#
# Multigroup radiation needs the fraction of a blackbody's energy that falls in
# each group, which is a difference of the normalised incomplete Planck integral
#
#   P(x) = (15/pi^4) int_0^x t^3/(e^t - 1) dt,     x = h nu / (k_B T)
#
# evaluated at the group edges. Quokka tabulates P on a log grid and interpolates
# (`planck_integral.hpp`, 265 lines including the table). Two series give it
# exactly instead, in a fraction of the code and to near machine precision:
# the Bernoulli expansion below x = 1 and the exponential tail above it. Neither
# needs a table, which also means nothing to regenerate or get out of step.

const _PI4_15 = pi^4 / 15

"""
    planck_integral(x) -> Real

The normalised incomplete Planck integral `P(x) = (15/pi^4) int_0^x t^3/(e^t-1) dt`,
so `P(0) = 0` and `P(Inf) = 1`. `x = h nu / (k_B T)` is frequency in units of the
thermal frequency.

This is the fraction of a blackbody's energy radiated below `nu`.
"""
function planck_integral(x::Real)
    T = float(typeof(x))
    x > 0 || return zero(T)
    isinf(x) && return one(T)

    if x < 1
        # t^3/(e^t - 1) = t^2 * sum_k B_k t^k / k!, integrated term by term. The
        # series converges for x < 2pi, and the terms fall by ~(x/2pi)^2 each, so
        # eight of them hold the whole interval to a few 1e-13 relative -- the
        # error peaks just below the crossover and is ~1e-14 elsewhere.
        s = x^3 / 3 - x^4 / 8 + x^5 / 60 - x^7 / 5040 + x^9 / 272160 -
            x^11 / 13305600 + x^13 / 622702080 - 691 * x^15 / 19615115520000
        return T(s / _PI4_15)
    end

    # int_x^inf t^3/(e^t-1) dt = sum_{n>=1} e^{-nx} (x^3/n + 3x^2/n^2 + 6x/n^3 + 6/n^4).
    # The ratio between terms is e^{-x} <= e^{-1}, so this converges geometrically
    # and the powers of e^{-x} come from one `exp` and repeated multiplication
    # rather than one `exp` per term.
    q = exp(-T(x))
    qn = q
    tail = zero(T)
    cutoff = eps(T) * _PI4_15 / (x^3 + 3x^2 + 6x + 6)   # term magnitude, not value
    n = 1
    while qn > cutoff && n <= 256
        tail += qn * (x^3 / n + 3x^2 / (n * n) + 6x / (n^3) + 6 / (n^4))
        qn *= q
        n += 1
    end
    return T(1 - tail / _PI4_15)
end

"""
    RadiationGroups{NG}(edges; energy_unit = 1)

`NG` frequency groups defined by `NG + 1` monotonically increasing `edges`.

`energy_unit` converts an edge to an energy, so that `x = energy_unit * edge /
(k_B T)`: pass `CGS.h` to give the edges in Hz, `CGS.eV` to give them in
electron volts, or leave it at `1` to give them directly in units of `k_B T`,
which is what the unit-free test problems do.

The outermost edges are the integration limits, so the first group collects
everything below `edges[1]` and the last everything above `edges[end]` -- the
fractions always sum to exactly one, whatever the edges are. This matches
Quokka's `radBoundaries`, including that convention.
"""
struct RadiationGroups{NG,M,T<:AbstractFloat}
    edges::SVec{M,T}
    energy_unit::T

    function RadiationGroups{NG,M,T}(edges::SVec{M,T}, energy_unit::T) where {NG,M,T}
        M == NG + 1 ||
            throw(ArgumentError("RadiationGroups{$NG} needs $(NG+1) edges, got $M"))
        NG >= 1 || throw(ArgumentError("need at least one group"))
        for g in 1:(M-1)
            edges[g] < edges[g+1] ||
                throw(ArgumentError("group edges must increase; edge $g is $(edges[g]) " *
                                    "and edge $(g+1) is $(edges[g+1])"))
        end
        energy_unit > 0 || throw(ArgumentError("energy_unit must be positive"))
        return new{NG,M,T}(edges, energy_unit)
    end
end

function RadiationGroups(edges; energy_unit=1.0)
    e = Tuple(edges)
    M = length(e)
    M >= 2 || throw(ArgumentError("need at least two edges to define one group"))
    T = float(promote_type(eltype(e), typeof(energy_unit)))
    return RadiationGroups{M - 1,M,T}(SVec{M,T}(convert.(T, e)...), T(energy_unit))
end

"""
    RadiationGroups(lo, hi, ngroups; energy_unit = 1, spacing = :log)

`ngroups` groups between `lo` and `hi`, spaced logarithmically (the usual choice,
since a Planck spectrum spans decades) or linearly.
"""
function RadiationGroups(lo::Real, hi::Real, ngroups::Integer; energy_unit=1.0,
                         spacing::Symbol=:log)
    ngroups >= 1 || throw(ArgumentError("need at least one group"))
    0 < lo < hi || throw(ArgumentError("need 0 < lo < hi, got lo = $lo, hi = $hi"))
    e = if spacing === :log
        exp.(range(log(lo), log(hi); length=ngroups + 1))
    elseif spacing === :linear
        collect(range(lo, hi; length=ngroups + 1))
    else
        throw(ArgumentError("spacing must be :log or :linear, got $spacing"))
    end
    return RadiationGroups(e; energy_unit)
end

"The single group spanning all frequencies, which reproduces grey radiation."
const GREY = RadiationGroups([0.0, Inf])

"""
    ngroups(groups) -> Int

How many frequency groups. A compile-time constant, so kernels specialise on it.
"""
@inline ngroups(::RadiationGroups{NG}) where {NG} = NG

"""
    group_centers(groups) -> SVec

The geometric mean of each group's edges, which is the frequency a
piecewise-constant opacity is evaluated at. Quokka's `ComputeBinCenterOpacity`
uses the same `sqrt(nu_g nu_{g+1})`.

An unbounded outer group has no geometric mean, so it falls back to its finite
edge -- the alternative is an opacity of zero or infinity in the group that
usually carries the least energy.
"""
@inline function group_centers(g::RadiationGroups{NG,M,T}) where {NG,M,T}
    return SVec(ntuple(Val(NG)) do i
        lo, hi = g.edges[i], g.edges[i+1]
        lo <= 0 ? hi : (isinf(hi) ? lo : sqrt(lo * hi))
    end)
end

"""
    planck_fractions(groups, T_rad, k_B) -> SVec{NG}

The fraction of a blackbody at temperature `T_rad` falling in each group, summing
to exactly one.

The last group takes `1 - sum(rest)` rather than its own integral, so the sum is
one by construction no matter where the edges sit -- the same guarantee Quokka's
`ComputePlanckEnergyFractions` makes, and the reason total emitted energy is
conserved to round-off regardless of the group structure.
"""
@inline function planck_fractions(g::RadiationGroups{NG,M,T}, T_rad, k_B) where {NG,M,T}
    NG == 1 && return SVec{1,T}(one(T))
    s = g.energy_unit / (k_B * T_rad)
    # The cumulative integral at each interior edge, computed first so that the
    # differencing below carries no state. A running total updated inside the
    # `ntuple` closure would be captured *and* assigned, which boxes it and
    # heap-allocates on every call -- the trap documented in CLAUDE.md.
    P = ntuple(i -> planck_integral(g.edges[i+1] * s), Val(NG - 1))
    fr = ntuple(Val(NG)) do i
        f = i == 1 ? P[1] : (i < NG ? P[i] - P[i-1] : one(T) - P[NG-1])
        max(f, zero(T))
    end
    return SVec{NG,T}(fr...)
end

"""
    planck_fractions_dT(groups, T_rad, k_B) -> SVec{NG}

`d(fraction)/dT` for each group, by centred differences on a relative step.

The Jacobian of the coupling needs `d(a T^4 f_g)/dT = 4 a T^3 f_g + a T^4 f_g'`.
The second piece is what moves energy between groups as the gas heats, and
dropping it -- as an implementation that assumed fixed fractions would -- slows
the Newton iteration without changing where it converges.
"""
@inline function planck_fractions_dT(g::RadiationGroups{NG,M,T}, T_rad, k_B) where {NG,M,T}
    NG == 1 && return SVec{1,T}(zero(T))
    h = cbrt(eps(T)) * T_rad
    return (planck_fractions(g, T_rad + h, k_B) - planck_fractions(g, T_rad - h, k_B)) / (2h)
end
