# Time integration.
#
# Space and time are separated by the method of lines: the spatial operator
#
#     L(U) = -(1/dx) [F_{i+1/2} - F_{i-1/2}]
#
# is assembled once per stage from the reconstruction and the Riemann solver,
# and the stages themselves are pure ODE updates. Hydrodynamics uses an explicit
# strong-stability-preserving Runge-Kutta method; radiation uses the IMEX
# scheme below, which treats transport explicitly and the stiff matter coupling
# implicitly.

"""
    AbstractIntegrator

Supertype of the explicit time integrators: [`ForwardEuler`](@ref),
[`SSPRK2`](@ref), [`SSPRK3`](@ref).
"""
abstract type AbstractIntegrator end

"""
    ForwardEuler()

First-order explicit Euler. Only first-order accurate in time, so it is useful
mainly with [`PCM`](@ref) or for debugging.
"""
struct ForwardEuler <: AbstractIntegrator end

"""
    SSPRK2()

Two-stage, second-order strong-stability-preserving Runge-Kutta (Heun).
"""
struct SSPRK2 <: AbstractIntegrator end

"""
    SSPRK3()

Three-stage, third-order strong-stability-preserving Runge-Kutta of Shu & Osher
(1988). The default: it is the natural partner of [`PPM`](@ref) and has the
largest SSP region of the three.
"""
struct SSPRK3 <: AbstractIntegrator end

"""
    tableau(integrator)

Shu-Osher coefficients `(a, b, c)` per stage, each stage performing

    U <- a U^n + b U + c dt L(U)
"""
tableau(::ForwardEuler) = ((1.0, 0.0, 1.0),)
tableau(::SSPRK2) = ((1.0, 0.0, 1.0), (0.5, 0.5, 0.5))
tableau(::SSPRK3) = ((1.0, 0.0, 1.0), (0.75, 0.25, 0.25), (1 / 3, 2 / 3, 2 / 3))

"""
    nstages(integrator)

Number of right-hand-side evaluations per step.
"""
nstages(i::AbstractIntegrator) = length(tableau(i))

"""
    combine!(dst, a, A, b, B, c, C, grid, nvar)

Interior-cell kernel `dst = a A + b B + c C`. `dst` may alias `A` or `B`.
"""
function combine!(dst::AbstractArray{T,4}, a::T, A::AbstractArray{T,4},
                  b::T, B::AbstractArray{T,4}, c::T, C::AbstractArray{T,4},
                  g::CartesianGrid{T}, nvar::Int) where {T}
    ir = g.lo[1]:g.hi[1]
    threaded_jk(g.lo[2]:g.hi[2], g.lo[3]:g.hi[3]) do j, k
        @inbounds for i in ir, n in 1:nvar
            dst[n, i, j, k] = a * A[n, i, j, k] + b * B[n, i, j, k] + c * C[n, i, j, k]
        end
    end
    return nothing
end

# --- IMEX PD-ARS ------------------------------------------------------------
#
# Three-stage, second-order, asymptotic-preserving IMEX scheme (Chu et al. 2019;
# the variant used by Quokka). Stage 1 is the identity, so only two right-hand
# sides and two implicit solves are actually evaluated. It is written in
# Shu-Osher form so that the explicit part inherits the SSP property of
# midpoint RK2.

"Explicit tableau entries of the IMEX PD-ARS scheme."
const IMEX_AEX_21 = 1.0
const IMEX_AEX_31 = 0.5
const IMEX_AEX_32 = 0.5
"Implicit tableau entries of the IMEX PD-ARS scheme."
const IMEX_AIM_22 = 1.0
const IMEX_AIM_32 = 0.5
const IMEX_AIM_33 = 0.5
"Shu-Osher weight of stage 3, `AIM_32 / AIM_22`."
const IMEX_ALPHA = IMEX_AIM_32 / IMEX_AIM_22

"""
    combine_gas!(U, alpha, Utmp, grid)

`U <- (1 - alpha) U + alpha Utmp` for the gas state, combining density, momentum
and *internal* energy and then rebuilding the total energy.

Combining the total energy directly would leave a spurious kinetic residual
`alpha (1-alpha) |Δm|² / (2 rho)` in the internal energy extracted by the next
implicit solve, which can drive the temperature negative in cells straddling a
radiation front.
"""
function combine_gas!(U::A, alpha::T, Utmp::A, g::CartesianGrid{T}) where {T,A<:AbstractArray{T,4}}
    w = one(T) - alpha
    ir = g.lo[1]:g.hi[1]
    threaded_jk(g.lo[2]:g.hi[2], g.lo[3]:g.hi[3]) do j, k
        @inbounds for i in ir
            u = getstate(U, Val(NHYDRO), i, j, k)
            v = getstate(Utmp, Val(NHYDRO), i, j, k)
            eint = w * internal_energy(u) + alpha * internal_energy(v)
            rho = w * u[IRHO] + alpha * v[IRHO]
            mx = w * u[IMX] + alpha * v[IMX]
            my = w * u[IMY] + alpha * v[IMY]
            mz = w * u[IMZ] + alpha * v[IMZ]
            U[IRHO, i, j, k] = rho
            U[IMX, i, j, k] = mx
            U[IMY, i, j, k] = my
            U[IMZ, i, j, k] = mz
            U[IEGAS, i, j, k] = eint + (mx * mx + my * my + mz * mz) / (2 * rho)
        end
    end
    return nothing
end
