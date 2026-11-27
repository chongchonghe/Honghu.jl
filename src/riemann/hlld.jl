# HLLD approximate Riemann solver for ideal MHD.
# The Julia counterpart of Quokka's src/hydro/HLLD.hpp, itself after Athena++:
# Miyoshi & Kusano (2005), hereafter MK5, with the corrections of Minoshima &
# Miyoshi (2021), hereafter MM21. Line for line, with Quokka's grouping of the
# transverse components (for floating-point symmetry under reflection), minus the
# dual-energy and passive-scalar components, which this code does not carry
# through the solver.

"Quokka's `DELTA`: the relative size of `rho (S - u)(S - S_M) - bx^2` below which MK5's star state is degenerate."
const HLLD_DELTA = 1.0e-4

"""
    HLLD()

Five-wave HLLD solver of Miyoshi & Kusano (2005): the two fast waves, the two
rotational (Alfven) waves and the contact, with the low-Mach pressure and the
carbuncle blending of Minoshima & Miyoshi (2021). Quokka's `HLLD`, and the
default solver of a simulation with `mhd = true`.

  - The bounding speeds are MK5's eq. 67, `S_L = min(u_L, u_R) - max(c_fL, c_fR)`
    and `S_R = max(u_L, u_R) + max(c_fL, c_fR)`, returned as
    `am = -min(0, S_L)` and `ap = max(0, S_R)`.
  - The total pressure of the star states carries MM21's eq. 15 correction, a
    factor `phi = chi (2 - chi)` on the velocity-jump term with
    `chi = min(1, c_u / c_f)`, so that pressure perturbations scale as `M^2` at
    low Mach number.
  - Inside the fan the flux is blended toward HLL by MM21's eq. 10 sensor,
    `theta = min(1, (c_f - min(du, 0)) / (c_f - min(dw, 0)))^4`, with `du` the jump
    of the reconstructed normal velocity and `dw` the transverse velocity jump
    the sweep passes in, so that a shock aligned with the grid does not grow a
    carbuncle.

Like Quokka's, it rebuilds the energy of each side from the pressure,
`P / (gamma - 1)`, and does not read `sL.E`.
"""
struct HLLD <: AbstractMHDRiemannSolver end

@inline function mhd_flux(::HLLD, sL::HydroState{T}, sR::HydroState{T}, gamma::T, bx::T,
                          perp_v_jump::T) where {T<:AbstractFloat}
    z = zero(T)
    half = T(0.5)

    #--- Step 1. L/R states
    vel_magn_sq_L = sL.u * sL.u + (sL.v * sL.v + sL.w * sL.w)
    vel_magn_sq_R = sR.u * sR.u + (sR.v * sR.v + sR.w * sR.w)
    bx_sq = bx * bx
    b_magn_sq_L = bx_sq + (sL.by * sL.by + sL.bz * sL.bz)
    b_magn_sq_R = bx_sq + (sR.by * sR.by + sR.bz * sR.bz)
    pb_L = half * b_magn_sq_L
    pb_R = half * b_magn_sq_R
    ke_L = half * (sL.rho * vel_magn_sq_L)
    ke_R = half * (sR.rho * vel_magn_sq_R)

    rho_L, rho_R = sL.rho, sR.rho
    mx_L, mx_R = sL.u * rho_L, sR.u * rho_R
    my_L, my_R = sL.v * rho_L, sR.v * rho_R
    mz_L, mz_R = sL.w * rho_L, sR.w * rho_R
    # Reciprocals formed once and multiplied by, as Athena++'s hlld.cpp does: a
    # division costs several multiplications of throughput, and this solver is
    # bound by its divider, not its arithmetic.
    igm1 = one(T) / (gamma - 1)
    E_L = ke_L + pb_L + sL.P * igm1
    E_R = ke_R + pb_R + sR.P * igm1

    #--- Step 2. L & R wave speeds, MK5 eq. 67
    spd_fms_L = fast_magnetosonic_speed(gamma, sL.rho, sL.P, bx, sL.by, sL.bz)
    spd_fms_R = fast_magnetosonic_speed(gamma, sR.rho, sR.P, bx, sR.by, sR.bz)
    spd_fms_max = max(spd_fms_L, spd_fms_R)
    u_min = min(sL.u, sR.u)
    u_max = max(sL.u, sR.u)
    S0 = min(z, u_min - spd_fms_max)
    S4 = max(z, u_max + spd_fms_max)
    fspd_m = -S0
    fspd_p = S4

    #--- Step 3. L/R fluxes
    ptot_L = sL.P + pb_L
    ptot_R = sR.P + pb_R
    F_L = SVec(mx_L, mx_L * sL.u + ptot_L - bx_sq, my_L * sL.u - bx * sL.by,
               mz_L * sL.u - bx * sL.bz,
               sL.u * (E_L + ptot_L - bx_sq) - bx * (sL.v * sL.by + sL.w * sL.bz))
    F_R = SVec(mx_R, mx_R * sR.u + ptot_R - bx_sq, my_R * sR.u - bx * sR.by,
               mz_R * sR.u - bx * sR.bz,
               sR.u * (E_R + ptot_R - bx_sq) - bx * (sR.v * sR.by + sR.w * sR.bz))

    #--- Step 4. middle and Alfven wave speeds
    siui_L = S0 - sL.u                     # MK5: S_i - u_i
    siui_R = S4 - sR.u
    # MM21 eq. 10 shock sensor, which blends toward HLL in Step 6. The normal
    # total-pressure jump in S_M is not suppressed: transverse compression can
    # also occur next to a genuine pressure-driven flow.
    para_v_jump = sR.u - sL.u              # negative: compression
    denom_tp = max(T(1e-14), spd_fms_max - min(perp_v_jump, z))
    tp = clamp((spd_fms_max - min(para_v_jump, z)) / denom_tp, z, one(T))
    theta = (tp * tp) * (tp * tp)

    # S_M, MK5 eq. 38, with the full total-pressure jump
    sm_denom = siui_R * rho_R - siui_L * rho_L
    sm_denom_inv = one(T) / sm_denom
    S2 = (siui_R * mx_R - siui_L * mx_L + (ptot_L - ptot_R)) * sm_denom_inv
    sism_L = S0 - S2                       # S_i - S_M
    sism_R = S4 - S2
    sism_inv_L = one(T) / sism_L
    sism_inv_R = one(T) / sism_R
    rhos_L = rho_L * siui_L * sism_inv_L   # MK5 eq. 43
    rhos_R = rho_R * siui_R * sism_inv_R

    rhos_inv_L = one(T) / rhos_L
    rhos_inv_R = one(T) / rhos_R
    rho_sqrt_L = _sqrt(rhos_L)
    rho_sqrt_R = _sqrt(rhos_R)
    S1 = S2 - abs(bx) * (rho_sqrt_L * rhos_inv_L)  # MK5 eq. 51; 1/sqrt(r) = sqrt(r)/r
    S3 = S2 + abs(bx) * (rho_sqrt_R * rhos_inv_R)

    #--- Step 5. intermediate states
    # total pressure with the MM21 low-Mach correction (eqs. 8, 14-16)
    rho_inv_L = one(T) / rho_L
    rho_inv_R = one(T) / rho_R
    spd_ca_sq_L = b_magn_sq_L * rho_inv_L
    spd_ca_sq_R = b_magn_sq_R * rho_inv_R
    spd_cax_sq_L = bx_sq * rho_inv_L
    spd_cax_sq_R = bx_sq * rho_inv_R
    cu_arg_L = max(z, (spd_ca_sq_L + vel_magn_sq_L)^2 - 4 * vel_magn_sq_L * spd_cax_sq_L)
    cu_arg_R = max(z, (spd_ca_sq_R + vel_magn_sq_R)^2 - 4 * vel_magn_sq_R * spd_cax_sq_R)
    spd_cu_sq_L = half * ((spd_ca_sq_L + vel_magn_sq_L) + _sqrt(cu_arg_L))
    spd_cu_sq_R = half * ((spd_ca_sq_R + vel_magn_sq_R) + _sqrt(cu_arg_R))
    spd_cu_max = _sqrt(max(z, spd_cu_sq_L, spd_cu_sq_R))  # sqrt is monotonic
    chi = min(one(T), spd_cu_max / max(T(1e-14), spd_fms_max))
    phi = chi * (2 - chi)
    ptot_star = ((siui_R * rho_R) * ptot_L - (siui_L * rho_L) * ptot_R +
                 phi * (rho_L * rho_R) * (siui_R * siui_L) * (sR.u - sL.u)) * sm_denom_inv

    # the star states, MK5 eqs. 39, 44-48
    mxs_L, mys_L, mzs_L, bys_L, bzs_L, Es_L, vbs_L =
        _hlld_star(rhos_L, rhos_inv_L, rho_L, siui_L, sism_L, sism_inv_L, S2, bx, bx_sq,
                   sL.u, sL.v, sL.w, sL.by, sL.bz, E_L, ptot_L, ptot_star)
    mxs_R, mys_R, mzs_R, bys_R, bzs_R, Es_R, vbs_R =
        _hlld_star(rhos_R, rhos_inv_R, rho_R, siui_R, sism_R, sism_inv_R, S2, bx, bx_sq,
                   sR.u, sR.v, sR.w, sR.by, sR.bz, E_R, ptot_R, ptot_star)

    # the double-star states, MK5 eqs. 59-63
    rho_sum_inv = one(T) / (rho_sqrt_L + rho_sqrt_R)
    bx_sign = bx > z ? one(T) : -one(T)
    vy_ss = rho_sum_inv * (rho_sqrt_L * (mys_L * rhos_inv_L) + rho_sqrt_R * (mys_R * rhos_inv_R) +
                           bx_sign * (bys_R - bys_L))
    vz_ss = rho_sum_inv * (rho_sqrt_L * (mzs_L * rhos_inv_L) + rho_sqrt_R * (mzs_R * rhos_inv_R) +
                           bx_sign * (bzs_R - bzs_L))
    myss_L, myss_R = rhos_L * vy_ss, rhos_R * vy_ss
    mzss_L, mzss_R = rhos_L * vz_ss, rhos_R * vz_ss
    by_ss = rho_sum_inv * (rho_sqrt_L * bys_R + rho_sqrt_R * bys_L +
                           bx_sign * rho_sqrt_L * rho_sqrt_R *
                           ((mys_R * rhos_inv_R) - (mys_L * rhos_inv_L)))
    bz_ss = rho_sum_inv * (rho_sqrt_L * bzs_R + rho_sqrt_R * bzs_L +
                           bx_sign * rho_sqrt_L * rho_sqrt_R *
                           ((mzs_R * rhos_inv_R) - (mzs_L * rhos_inv_L)))
    # the shared v . b, averaged over its L and R evaluations for symmetry
    vb_ss = S2 * bx + half * ((myss_L * by_ss + mzss_L * bz_ss) * rhos_inv_L +
                              (myss_R * by_ss + mzss_R * bz_ss) * rhos_inv_R)
    Ess_L = Es_L - rho_sqrt_L * bx_sign * (vbs_L - vb_ss)
    Ess_R = Es_R + rho_sqrt_R * bx_sign * (vbs_R - vb_ss)

    U_L = SVec(rho_L, mx_L, my_L, mz_L, E_L)
    U_R = SVec(rho_R, mx_R, my_R, mz_R, E_R)
    Us_L = SVec(rhos_L, mxs_L, mys_L, mzs_L, Es_L)
    Us_R = SVec(rhos_R, mxs_R, mys_R, mzs_R, Es_R)
    Uss_L = SVec(rhos_L, mxs_L, myss_L, mzss_L, Ess_L)
    Uss_R = SVec(rhos_R, mxs_R, myss_R, mzss_R, Ess_R)

    Dss_L = S1 * (Uss_L - Us_L)
    Ds_L = S0 * (Us_L - U_L)
    Dss_R = S3 * (Uss_R - Us_R)
    Ds_R = S4 * (Us_R - U_R)

    #--- Step 6. the flux
    F = if S0 >= z
        F_L                                # supersonic to the right
    elseif S4 <= z
        F_R                                # supersonic to the left
    elseif S1 >= z
        F_L + Ds_L                         # U*_L
    elseif S2 >= z
        F_L + Ds_L + Dss_L                 # U**_L
    elseif S3 > z
        F_R + Ds_R + Dss_R                 # U**_R
    else
        F_R + Ds_R                         # U*_R
    end

    # inside the fan, blend the whole flux toward HLL with the MM21 weight
    if theta < one(T) && S0 < z && S4 > z
        F_HLL = (S4 * F_L - S0 * F_R + S0 * S4 * (U_R - U_L)) * (one(T) / (S4 - S0))
        F = theta * F + (one(T) - theta) * F_HLL
    end

    return F, fspd_m, fspd_p
end

# One side's star state of MK5: the momentum, the transverse field, the energy and
# v* . b*, with the degenerate case of MK5 (a vanishing denominator in eqs. 44-47,
# where the Alfven and the fast wave coincide) taken as Quokka takes it.
@inline function _hlld_star(rhos::T, rhos_inv::T, rho::T, siui::T, sism::T, sism_inv::T, S2::T,
                            bx::T, bx_sq::T, u::T, v::T, w::T, by::T, bz::T, E::T, ptot::T,
                            ptot_star::T) where {T}
    mxs = rhos * S2
    den = rho * siui * sism - bx_sq
    if abs(den) < T(HLLD_DELTA) * ptot_star
        mys = rhos * v
        mzs = rhos * w
        bys = by
        bzs = bz
    else
        den_inv = one(T) / den
        tmp = bx * (siui - sism) * den_inv             # MK5 eqs. 44, 46
        mys = rhos * (v - by * tmp)
        mzs = rhos * (w - bz * tmp)
        tmp = (rho * (siui * siui) - bx_sq) * den_inv  # MK5 eqs. 45, 47
        bys = by * tmp
        bzs = bz * tmp
    end
    vbs = (mxs * bx + (mys * bys + mzs * bzs)) * rhos_inv
    Es = (siui * E - ptot * u + ptot_star * S2 + bx * (u * bx + (v * by + w * bz) - vbs)) * sism_inv
    return mxs, mys, mzs, bys, bzs, Es, vbs
end
