# HLLC approximate Riemann solver for the Euler equations.
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
#
# The solver works in *canonical* form: `u` is always the velocity normal to the
# interface, `v` and `w` the two transverse components. The caller is
# responsible for permuting the components.

"""
    hllc(rhoL, uL, vL, wL, pL, csL, EL,
         rhoR, uR, vR, wR, pR, csR, ER, gamma, du, dw) -> SVec{5}

HLLC flux across one interface in canonical form. The returned components are
`(mass, normal momentum, transverse momentum 1, transverse momentum 2, energy)`.

`du` is the jump in normal velocity across the interface and `dw` the smallest
transverse velocity difference in the two cells adjoining it; both feed the
carbuncle correction and may be passed as zero to disable it.
"""
@inline function hllc(rhoL::T, uL::T, vL::T, wL::T, pL::T, csL::T, EL::T,
                      rhoR::T, uR::T, vR::T, wR::T, pR::T, csR::T, ER::T,
                      gamma::T, du::T, dw::T) where {T<:AbstractFloat}
    # --- Roe averages --------------------------------------------------------
    wl = sqrt(rhoL)
    wr = sqrt(rhoR)
    nrm = one(T) / (wl + wr)
    ut = (wl * uL + wr * uR) * nrm
    vt = (wl * vL + wr * vR) * nrm
    wt = (wl * wL + wr * wR) * nrm
    vsq = ut * ut + vt * vt + wt * wt
    HL = (EL + pL) / rhoL          # specific enthalpy
    HR = (ER + pR) / rhoR
    Ht = (wl * HL + wr * HR) * nrm
    csq = (gamma - 1) * (Ht - vsq / 2)
    cst = csq > zero(T) ? sqrt(csq) : (csL + csR) / 2

    # --- wave speeds (Batten+ 1997) with Rider's nonlinear correction --------
    # The correction only acts in compression (uL > uR).
    Gfac = (gamma + 1) / 2
    sN = Gfac * max(uL - uR, zero(T)) / 2
    S_L = min(uL - (csL + sN), ut - (cst + sN))
    S_R = max(uR + (csR + sN), ut + (cst + sN))

    # --- carbuncle correction, Minoshima & Miyoshi (2021) eq. 10 -------------
    cs_max = max(csL, csR)
    tp = min(one(T), (cs_max - min(du, zero(T))) / (cs_max - min(dw, zero(T))))
    theta = tp * tp * tp * tp

    # --- contact speed -------------------------------------------------------
    dmL = rhoL * (S_L - uL)
    dmR = rhoR * (S_R - uR)
    S_star = (theta * (pR - pL) + (rhoL * uL * (S_L - uL) - rhoR * uR * (S_R - uR))) / (dmL - dmR)

    # --- low-dissipation pressure, Minoshima & Miyoshi (2021) eq. 23 ---------
    vmagL = sqrt(uL * uL + vL * vL + wL * wL)
    vmagR = sqrt(uR * uR + vR * vR + wR * wR)
    chi = min(one(T), max(vmagL, vmagR) / cs_max)
    phi = chi * (2 - chi)
    # The linearised star pressure can go negative when two strong rarefactions
    # open up a near vacuum (Toro's "123" problem), where the wave-speed
    # estimates stop being valid. Clamping at zero keeps the star state
    # admissible; for any well-resolved state the clamp is inactive.
    P_LR = max((pL + pR) / 2 + phi * (dmL * (S_star - uL) + dmR * (S_star - uR)) / 2, zero(T))

    # --- fluxes --------------------------------------------------------------
    z = zero(T)
    o = one(T)
    D_L = SVec(z, o, z, z, uL)
    D_R = SVec(z, o, z, z, uR)
    D_S = SVec(z, o, z, z, S_star)

    U_L = SVec(rhoL, rhoL * uL, rhoL * vL, rhoL * wL, EL)
    U_R = SVec(rhoR, rhoR * uR, rhoR * vR, rhoR * wR, ER)
    F_L = uL * U_L + pL * D_L
    F_R = uR * U_R + pR * D_R

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

# Direction bookkeeping: for a sweep along `D`, `_tdir1` and `_tdir2` are the two
# transverse directions in the cyclic order used by the canonical form
# (D=1 -> y,z;  D=2 -> z,x;  D=3 -> x,y).
@inline _tdir1(::Val{1}) = 2
@inline _tdir1(::Val{2}) = 3
@inline _tdir1(::Val{3}) = 1
@inline _tdir2(::Val{1}) = 3
@inline _tdir2(::Val{2}) = 1
@inline _tdir2(::Val{3}) = 2
@inline _tshift1(::Val{1}) = (0, 1, 0)
@inline _tshift1(::Val{2}) = (0, 0, 1)
@inline _tshift1(::Val{3}) = (1, 0, 0)
@inline _tshift2(::Val{1}) = (0, 0, 1)
@inline _tshift2(::Val{2}) = (1, 0, 0)
@inline _tshift2(::Val{3}) = (0, 1, 0)
