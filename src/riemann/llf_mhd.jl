# Local Lax-Friedrichs solver for ideal MHD.
# The Julia counterpart of Quokka's src/hydro/LLF_mhd.hpp, Toro (2009) eq. 10.56.
#
# As in the hydrodynamic LLF, Quokka's dual-energy component is absent, and so are
# its passive scalars: the sweep carries the mass scalars on the mass flux.

"""
    LLFMHD()

Local Lax-Friedrichs (Rusanov) solver for ideal MHD: one wave at the fastest
signal speed `max(|u| + c_f)` over the two sides, with `c_f` the fast
magnetosonic speed along the face normal. Quokka's `LLF_MHD`.

The control in a comparison, as [`LLF`](@ref) is for [`HLLC`](@ref): it smears
contacts and rotational discontinuities, which [`HLLD`](@ref) resolves. Quokka
uses it for its first-order fallback fluxes. The bounding speeds it returns,
`am = -min(0, uL - c_fL, uR - c_fR)` and `ap = max(0, uL + c_fL, uR + c_fR)`,
are Davis's estimates rather than HLLD's, as in Quokka. Uses neither the carbuncle
input nor `sL.E`: the energy is rebuilt from the pressure, `P/(gamma - 1)`, as
Quokka does.
"""
struct LLFMHD <: AbstractMHDRiemannSolver end

@inline function mhd_flux(::LLFMHD, sL::HydroState{T}, sR::HydroState{T}, gamma::T, bx::T,
                          dw::T) where {T<:AbstractFloat}
    bx_sq = bx * bx
    # magnetic pressure and kinetic energy (the transverse components grouped for
    # floating-point symmetry under reflection, as in Quokka)
    pb_L = (bx_sq + (sL.by * sL.by + sL.bz * sL.bz)) / 2
    pb_R = (bx_sq + (sR.by * sR.by + sR.bz * sR.bz)) / 2
    ke_L = sL.rho * (sL.u * sL.u + (sL.v * sL.v + sL.w * sL.w)) / 2
    ke_R = sR.rho * (sR.u * sR.u + (sR.v * sR.v + sR.w * sR.w)) / 2
    ptot_L = sL.P + pb_L
    ptot_R = sR.P + pb_R

    fspd_L = fast_magnetosonic_speed(gamma, sL.rho, sL.P, bx, sL.by, sL.bz)
    fspd_R = fast_magnetosonic_speed(gamma, sR.rho, sR.P, bx, sR.by, sR.bz)
    z = zero(T)
    fspd_m = -min(z, min(sL.u - fspd_L, sR.u - fspd_R))
    fspd_p = max(z, max(sL.u + fspd_L, sR.u + fspd_R))

    # conserved states
    E_L = ke_L + pb_L + sL.P / (gamma - 1)
    E_R = ke_R + pb_R + sR.P / (gamma - 1)
    U_L = SVec(sL.rho, sL.rho * sL.u, sL.rho * sL.v, sL.rho * sL.w, E_L)
    U_R = SVec(sR.rho, sR.rho * sR.u, sR.rho * sR.v, sR.rho * sR.w, E_R)

    # half the fastest signal speed, Toro eq. 10.43
    a = max(abs(sL.u) + fspd_L, abs(sR.u) + fspd_R) / 2

    # the fluxes of the two sides
    F_L = SVec(U_L[2], U_L[2] * sL.u + ptot_L - bx_sq, U_L[3] * sL.u - bx * sL.by,
               U_L[4] * sL.u - bx * sL.bz,
               sL.u * (E_L + ptot_L - bx_sq) - bx * (sL.v * sL.by + sL.w * sL.bz))
    F_R = SVec(U_R[2], U_R[2] * sR.u + ptot_R - bx_sq, U_R[3] * sR.u - bx * sR.by,
               U_R[4] * sR.u - bx * sR.bz,
               sR.u * (E_R + ptot_R - bx_sq) - bx * (sR.v * sR.by + sR.w * sR.bz))

    # Toro eq. 10.42
    return (F_L + F_R) / 2 - a * (U_R - U_L), fspd_m, fspd_p
end
