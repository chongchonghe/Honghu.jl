# HLLC approximate Riemann solver for the Euler equations.
# The Julia counterpart of Quokka's src/hydro/HLLC.hpp.
#
# Follows Toro (2009) and Balsara (2017), with the two low-dissipation /
# anti-carbuncle modifications used by Quokka:
#
#   * the nonlinear wave-speed correction of Rider (1999), which damps
#     slow-moving-shock oscillations,
#   * the carbuncle cure and low-dissipation pressure of Minoshima & Miyoshi
#     (JCP 2021), which switch off the pressure jump in the contact-speed
#     estimate where a shock is aligned with the grid and restore the correct
#     low-Mach pressure scaling.

"""
    HLLC()

Three-wave HLLC solver: left- and right-going acoustic waves plus the contact
discontinuity. Resolving the contact is what keeps a material interface sharp,
which is why this is the default.

Structurally a gamma-law solver -- `gamma` enters the Roe-averaged sound speed
and Rider's correction as a scalar -- so it is not appropriate for a genuinely
non-ideal equation of state. See [`AbstractEOS`](@ref).
"""
struct HLLC <: AbstractRiemannSolver end

@inline function riemann_flux(::HLLC, sL::HydroState{T}, sR::HydroState{T},
                              gamma::T, du::T, dw::T) where {T<:AbstractFloat}
    # --- Roe averages --------------------------------------------------------
    wl = sqrt(sL.rho)
    wr = sqrt(sR.rho)
    nrm = one(T) / (wl + wr)
    ut = (wl * sL.u + wr * sR.u) * nrm
    vt = (wl * sL.v + wr * sR.v) * nrm
    wt = (wl * sL.w + wr * sR.w) * nrm
    vsq = ut * ut + vt * vt + wt * wt
    HL = (sL.E + sL.P) / sL.rho          # specific enthalpy
    HR = (sR.E + sR.P) / sR.rho
    Ht = (wl * HL + wr * HR) * nrm
    csq = (gamma - 1) * (Ht - vsq / 2)
    cst = csq > zero(T) ? sqrt(csq) : (sL.cs + sR.cs) / 2

    # --- wave speeds (Batten+ 1997) with Rider's nonlinear correction --------
    # The correction only acts in compression (uL > uR).
    Gfac = (gamma + 1) / 2
    sN = Gfac * max(sL.u - sR.u, zero(T)) / 2
    S_L = min(sL.u - (sL.cs + sN), ut - (cst + sN))
    S_R = max(sR.u + (sR.cs + sN), ut + (cst + sN))

    # --- carbuncle correction, Minoshima & Miyoshi (2021) eq. 10 -------------
    cs_max = max(sL.cs, sR.cs)
    tp = min(one(T), (cs_max - min(du, zero(T))) / (cs_max - min(dw, zero(T))))
    theta = tp * tp * tp * tp

    # --- contact speed -------------------------------------------------------
    dmL = sL.rho * (S_L - sL.u)
    dmR = sR.rho * (S_R - sR.u)
    S_star = (theta * (sR.P - sL.P) +
              (sL.rho * sL.u * (S_L - sL.u) - sR.rho * sR.u * (S_R - sR.u))) / (dmL - dmR)

    # --- low-dissipation pressure, Minoshima & Miyoshi (2021) eq. 23 ---------
    vmagL = sqrt(sL.u * sL.u + sL.v * sL.v + sL.w * sL.w)
    vmagR = sqrt(sR.u * sR.u + sR.v * sR.v + sR.w * sR.w)
    chi = min(one(T), max(vmagL, vmagR) / cs_max)
    phi = chi * (2 - chi)
    # The linearised star pressure can go negative when two strong rarefactions
    # open up a near vacuum (Toro's "123" problem), where the wave-speed
    # estimates stop being valid. Clamping at zero keeps the star state
    # admissible; for any well-resolved state the clamp is inactive.
    P_LR = max((sL.P + sR.P) / 2 +
               phi * (dmL * (S_star - sL.u) + dmR * (S_star - sR.u)) / 2, zero(T))

    # --- fluxes --------------------------------------------------------------
    z = zero(T)
    o = one(T)
    D_L = SVec(z, o, z, z, sL.u)
    D_R = SVec(z, o, z, z, sR.u)
    D_S = SVec(z, o, z, z, S_star)

    U_L = SVec(sL.rho, sL.rho * sL.u, sL.rho * sL.v, sL.rho * sL.w, sL.E)
    U_R = SVec(sR.rho, sR.rho * sR.u, sR.rho * sR.v, sR.rho * sR.w, sR.E)
    F_L = sL.u * U_L + sL.P * D_L
    F_R = sR.u * U_R + sR.P * D_R

    if S_L > z
        return F_L
    elseif S_R < z
        return F_R
    elseif S_star > z
        return (S_star * (S_L * U_L - F_L) + (S_L * P_LR) * D_S) / (S_L - S_star)
    else
        return (S_star * (S_R * U_R - F_R) + (S_R * P_LR) * D_S) / (S_R - S_star)
    end
end
