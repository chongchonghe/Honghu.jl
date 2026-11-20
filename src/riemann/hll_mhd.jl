# HLL solver for ideal MHD (Harten, Lax & van Leer 1983).
# Quokka has no such solver; this is the HLL flux inside Quokka's src/hydro/HLLD.hpp
# (the target of its MM21 blending), with HLLD's bounding speeds, taken out on its
# own. The arithmetic follows hlld.jl line for line, so HLLD at theta = 0 returns
# this flux.

"""
    HLLMHD()

Two-wave HLL solver for ideal MHD: one intermediate state between the two fast
waves, with HLLD's bounding speeds (Miyoshi & Kusano 2005, eq. 67),
`S_L = min(u_L, u_R) - max(c_fL, c_fR)` and `S_R = max(u_L, u_R) + max(c_fL, c_fR)`,
returned as `am = -min(0, S_L)` and `ap = max(0, S_R)`.

For benchmarking [`HLLD`](@ref): the two share their speeds, so constrained
transport gets the same `alpha^±` from either, and they differ only inside the fan,
where HLL smears the contact and the rotational discontinuities that HLLD resolves.
It is the flux HLLD blends toward with the weight of Minoshima & Miyoshi (2021),
and with LD04 averaging, itself the HLL flux of the transverse field along one
direction, it makes a scheme whose gas and field see the same two-wave model.
Quokka has no MHD HLL solver. Rebuilds the energy of each side from the pressure,
`P / (gamma - 1)`, as HLLD does, and uses neither the carbuncle input nor `sL.E`.
"""
struct HLLMHD <: AbstractMHDRiemannSolver end

@inline function mhd_flux(::HLLMHD, sL::HydroState{T}, sR::HydroState{T}, gamma::T, bx::T,
                          dw::T) where {T<:AbstractFloat}
    z = zero(T)
    half = T(0.5)

    # the L/R states, as HLLD's Step 1
    vel_magn_sq_L = sL.u * sL.u + (sL.v * sL.v + sL.w * sL.w)
    vel_magn_sq_R = sR.u * sR.u + (sR.v * sR.v + sR.w * sR.w)
    bx_sq = bx * bx
    pb_L = half * (bx_sq + (sL.by * sL.by + sL.bz * sL.bz))
    pb_R = half * (bx_sq + (sR.by * sR.by + sR.bz * sR.bz))
    ke_L = half * (sL.rho * vel_magn_sq_L)
    ke_R = half * (sR.rho * vel_magn_sq_R)
    mx_L, mx_R = sL.u * sL.rho, sR.u * sR.rho
    E_L = ke_L + pb_L + sL.P / (gamma - 1)
    E_R = ke_R + pb_R + sR.P / (gamma - 1)

    # the wave speeds, MK5 eq. 67
    spd_fms_max = max(fast_magnetosonic_speed(gamma, sL.rho, sL.P, bx, sL.by, sL.bz),
                      fast_magnetosonic_speed(gamma, sR.rho, sR.P, bx, sR.by, sR.bz))
    S0 = min(z, min(sL.u, sR.u) - spd_fms_max)
    S4 = max(z, max(sL.u, sR.u) + spd_fms_max)

    # the L/R fluxes
    ptot_L = sL.P + pb_L
    ptot_R = sR.P + pb_R
    F_L = SVec(mx_L, mx_L * sL.u + ptot_L - bx_sq, sL.v * sL.rho * sL.u - bx * sL.by,
               sL.w * sL.rho * sL.u - bx * sL.bz,
               sL.u * (E_L + ptot_L - bx_sq) - bx * (sL.v * sL.by + sL.w * sL.bz))
    F_R = SVec(mx_R, mx_R * sR.u + ptot_R - bx_sq, sR.v * sR.rho * sR.u - bx * sR.by,
               sR.w * sR.rho * sR.u - bx * sR.bz,
               sR.u * (E_R + ptot_R - bx_sq) - bx * (sR.v * sR.by + sR.w * sR.bz))

    F = if S0 >= z
        F_L                                # supersonic to the right
    elseif S4 <= z
        F_R                                # supersonic to the left
    else
        U_L = SVec(sL.rho, mx_L, sL.v * sL.rho, sL.w * sL.rho, E_L)
        U_R = SVec(sR.rho, mx_R, sR.v * sR.rho, sR.w * sR.rho, E_R)
        (S4 * F_L - S0 * F_R + S0 * S4 * (U_R - U_L)) / (S4 - S0)
    end
    return F, -S0, S4
end
