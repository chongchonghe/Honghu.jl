# Spatial reconstruction: cell averages -> interface states.
#
# Interface convention (same as Quokka): face `f` along direction `D` is the
# *low-side* face of cell `f`. Hence
#
#     qL[.., f, ..]   left  state at face f, extrapolated from cell f-1
#     qR[.., f, ..]   right state at face f, extrapolated from cell f
#
# and the interior cells `lo:hi` are bounded by the faces `lo:hi+1`.
#
# Reconstruction is always applied to *primitive* variables.

"""
    AbstractReconstruction

Supertype for spatial reconstruction schemes: [`PCM`](@ref), [`PLM`](@ref),
[`PPM`](@ref).
"""
abstract type AbstractReconstruction end

"""
    PCM()

Piecewise-constant (Godunov) reconstruction. First-order accurate; useful as a
robustness fallback and for verifying that higher-order schemes actually help.
"""
struct PCM <: AbstractReconstruction end

"""
    PLM(limiter = MonotonizedCentral())

Piecewise-linear reconstruction with a TVD slope limiter. Second-order accurate
in smooth regions.
"""
struct PLM{L<:AbstractLimiter} <: AbstractReconstruction
    limiter::L
end
PLM() = PLM(MonotonizedCentral())

"""
    PPM()

Piecewise-parabolic method of Colella & Woodward (1984), with the interface
constraint and monotonicity correction used by Athena++ / Quokka (Mignone 2014):
edge values are clamped to the range of the three neighbouring cell averages,
and at a local extremum the parabola collapses to an MC-limited line.

Third-order accurate on smooth data away from extrema; this is the recommended
default.
"""
struct PPM <: AbstractReconstruction end

"""
    required_ghosts(method)

Minimum number of ghost cells needed by a reconstruction method. At least two
are always used so that the reconstruction sweep can write the interface states
of the first ghost cell without leaving the array.
"""
required_ghosts(::PCM) = 2
required_ghosts(::PLM) = 2
required_ghosts(::PPM) = 3

"""
    reconstruct!(qL, qR, q, method, Val(D), grid, nvar)

Fill the interface states `qL`, `qR` along direction `D` from the cell-centred
primitive variables `q`. All three arrays are `(nvar, i, j, k)` and share the
grid layout; only the components `1:nvar` are touched.
"""
function reconstruct! end

function reconstruct!(qL::A, qR::A, q::A, ::PCM, ::Val{D}, g::CartesianGrid,
                      nvar::Int) where {T,A<:AbstractArray{T,4},D}
    d1, d2, d3 = _shift(Val(D))
    ir, jr, kr = cell_ranges(g, Val(D))
    threaded_jk(jr, kr) do j, k
        @inbounds for i in ir, n in 1:nvar
            v = q[n, i, j, k]
            qR[n, i, j, k] = v
            qL[n, i+d1, j+d2, k+d3] = v
        end
    end
    return nothing
end

function reconstruct!(qL::A, qR::A, q::A, m::PLM, ::Val{D}, g::CartesianGrid,
                      nvar::Int) where {T,A<:AbstractArray{T,4},D}
    d1, d2, d3 = _shift(Val(D))
    ir, jr, kr = cell_ranges(g, Val(D))
    lim = m.limiter
    threaded_jk(jr, kr) do j, k
        @inbounds for i in ir, n in 1:nvar
            qc = q[n, i, j, k]
            qm = q[n, i-d1, j-d2, k-d3]
            qp = q[n, i+d1, j+d2, k+d3]
            s = limit(lim, qc - qm, qp - qc) / 2
            qR[n, i, j, k] = qc - s
            qL[n, i+d1, j+d2, k+d3] = qc + s
        end
    end
    return nothing
end

function reconstruct!(qL::A, qR::A, q::A, ::PPM, ::Val{D}, g::CartesianGrid,
                      nvar::Int) where {T,A<:AbstractArray{T,4},D}
    d1, d2, d3 = _shift(Val(D))
    ir, jr, kr = cell_ranges(g, Val(D))
    threaded_jk(jr, kr) do j, k
        @inbounds for i in ir, n in 1:nvar
            qmm = q[n, i-2d1, j-2d2, k-2d3]
            qm = q[n, i-d1, j-d2, k-d3]
            qc = q[n, i, j, k]
            qp = q[n, i+d1, j+d2, k+d3]
            qpp = q[n, i+2d1, j+2d2, k+2d3]
            am, ap = ppm_edges(qmm, qm, qc, qp, qpp)
            qR[n, i, j, k] = am
            qL[n, i+d1, j+d2, k+d3] = ap
        end
    end
    return nothing
end

"""
    ppm_edges(qmm, qm, qc, qp, qpp) -> (a_minus, a_plus)

PPM interface values on the low and high edge of the central cell, given the
five-point stencil of cell averages.
"""
@inline function ppm_edges(qmm::T, qm::T, qc::T, qp::T, qpp::T) where {T}
    # (1) fourth-order interface estimates, C&W eq. (1.9). The terms are grouped
    #     so that the result is exactly symmetric under reflection of the
    #     stencil in floating-point arithmetic (as in Athena++).
    c1 = T(7) / T(12)
    c2 = -one(T) / T(12)
    a_minus = (c1 * qc + c2 * qp) + (c1 * qm + c2 * qmm)
    a_plus = (c1 * qp + c2 * qpp) + (c1 * qc + c2 * qm)

    # (2) constrain the interfaces to the range of the neighbouring averages
    #     (Mignone+ 2005, eq. B8)
    blo = min(qm, qc, qp)
    bhi = max(qm, qc, qp)
    am = clamp(a_minus, blo, bhi)
    ap = clamp(a_plus, blo, bhi)

    # (3) monotonicity correction, C&W eq. (1.10)
    dqm = qc - am
    dqp = ap - qc
    if dqp * dqm <= zero(T)
        # local extremum: fall back to an MC-limited line (Balsara 2017)
        dq0 = limit(MonotonizedCentral(), qc - qm, qp - qc) / 2
        am = qc - dq0
        ap = qc + dq0
    else
        # parabola overshoots on one side: reset the opposite interface
        if abs(dqm) >= 2 * abs(dqp)
            am = qc - 2 * dqp
        end
        if abs(dqp) >= 2 * abs(dqm)
            ap = qc + 2 * dqm
        end
    end
    return am, ap
end
