# Local Lax-Friedrichs (Rusanov) solver.
# The Julia counterpart of Quokka's src/hydro/LLF.hpp, Toro (2009) eq. 10.56.
#
# Quokka's version carries a sixth flux component for the dual-energy internal
# energy; there is no dual energy here, so that component is absent.

"""
    LLF()

Local Lax-Friedrichs, also called Rusanov: a single wave at the fastest signal
speed `max(|u| + c_s)`, with the whole Riemann fan replaced by that one
estimate.

The most diffusive and most robust of the solvers, and the useful control in a
comparison: whatever [`HLLC`](@ref) buys is the difference between them. It
smears contact discontinuities badly, because it has no contact wave to resolve
them with, while shocks -- which are self-steepening -- survive comparatively
well. Uses neither `gamma` nor the carbuncle correction.

```julia
compare(Honghu.Problems.sod_shock_tube,
        (HLLC = (riemann = HLLC(),), LLF = (riemann = LLF(),)); sizes = (64, 128, 256))
```
"""
struct LLF <: AbstractRiemannSolver end

@inline function riemann_flux(::LLF, sL::HydroState{T}, sR::HydroState{T},
                              gamma::T, du::T, dw::T) where {T<:AbstractFloat}
    # the fastest signal speed at the interface
    Sp = max(abs(sL.u) + sL.cs, abs(sR.u) + sR.cs)

    z = zero(T)
    o = one(T)
    U_L = SVec(sL.rho, sL.rho * sL.u, sL.rho * sL.v, sL.rho * sL.w, sL.E)
    U_R = SVec(sR.rho, sR.rho * sR.u, sR.rho * sR.v, sR.rho * sR.w, sR.E)
    D_L = SVec(z, o, z, z, sL.u)
    D_R = SVec(z, o, z, z, sR.u)
    F_L = sL.u * U_L + sL.P * D_L
    F_R = sR.u * U_R + sR.P * D_R

    # open the Riemann fan
    return (F_L + F_R) / 2 - (Sp / 2) * (U_R - U_L)
end
