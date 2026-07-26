# Exact solution of the Riemann problem for the 1D Euler equations, used to
# verify the hydro solver. Follows Toro (2009), chapter 4.

"""
    sod_exact(x, t; rhoL = 1.0, uL = 0.0, pL = 1.0,
                    rhoR = 0.125, uR = 0.0, pR = 0.1,
                    gamma = 1.4, x0 = 0.5) -> (rho, u, p)

Exact solution of the Riemann problem with the given left and right states,
sampled at position `x` and time `t`. The defaults are the standard Sod shock
tube.

The star pressure is found by Newton iteration on the pressure function, then
the solution is sampled along the ray `S = (x - x0)/t`.
"""
function sod_exact(x::Real, t::Real; rhoL=1.0, uL=0.0, pL=1.0,
                   rhoR=0.125, uR=0.0, pR=0.1, gamma=1.4, x0=0.5)
    g = float(gamma)
    t <= 0 && return x < x0 ? (rhoL, uL, pL) : (rhoR, uR, pR)
    cL = sqrt(g * pL / rhoL)
    cR = sqrt(g * pR / rhoR)
    pstar, ustar = _star_state(rhoL, uL, pL, cL, rhoR, uR, pR, cR, g)
    return _sample((x - x0) / t, pstar, ustar, rhoL, uL, pL, cL, rhoR, uR, pR, cR, g)
end

# pressure function and its derivative for one side
function _pfunc(p, rhoK, pK, cK, g)
    if p > pK   # shock
        A = 2 / ((g + 1) * rhoK)
        B = (g - 1) / (g + 1) * pK
        s = sqrt(A / (B + p))
        return (p - pK) * s, s * (1 - (p - pK) / (2 * (B + p)))
    else        # rarefaction
        pr = p / pK
        return (2 * cK / (g - 1)) * (pr^((g - 1) / (2g)) - 1), pr^(-(g + 1) / (2g)) / (rhoK * cK)
    end
end

function _star_state(rhoL, uL, pL, cL, rhoR, uR, pR, cR, g)
    du = uR - uL
    # two-rarefaction approximation as the initial guess
    p = ((cL + cR - (g - 1) / 2 * du) / (cL / pL^((g - 1) / (2g)) + cR / pR^((g - 1) / (2g))))^(2g / (g - 1))
    p = max(p, 1e-12)
    for _ in 1:100
        fL, dfL = _pfunc(p, rhoL, pL, cL, g)
        fR, dfR = _pfunc(p, rhoR, pR, cR, g)
        f = fL + fR + du
        dp = f / (dfL + dfR)
        pnew = max(p - dp, 1e-12)
        abs(pnew - p) <= 1e-14 * (abs(pnew) + abs(p)) && (p = pnew; break)
        p = pnew
    end
    fL, _ = _pfunc(p, rhoL, pL, cL, g)
    fR, _ = _pfunc(p, rhoR, pR, cR, g)
    return p, (uL + uR + fR - fL) / 2
end

function _sample(S, pstar, ustar, rhoL, uL, pL, cL, rhoR, uR, pR, cR, g)
    if S <= ustar
        # left of the contact
        if pstar > pL   # left shock
            SL = uL - cL * sqrt((g + 1) / (2g) * pstar / pL + (g - 1) / (2g))
            S <= SL && return (rhoL, uL, pL)
            r = pstar / pL
            rho = rhoL * (r + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * r + 1)
            return (rho, ustar, pstar)
        else            # left rarefaction
            SH = uL - cL
            S <= SH && return (rhoL, uL, pL)
            cstar = cL * (pstar / pL)^((g - 1) / (2g))
            ST = ustar - cstar
            if S >= ST
                return (rhoL * (pstar / pL)^(1 / g), ustar, pstar)
            end
            # inside the fan
            u = 2 / (g + 1) * (cL + (g - 1) / 2 * uL + S)
            c = 2 / (g + 1) * (cL + (g - 1) / 2 * (uL - S))
            rho = rhoL * (c / cL)^(2 / (g - 1))
            return (rho, u, pL * (c / cL)^(2g / (g - 1)))
        end
    else
        # right of the contact
        if pstar > pR   # right shock
            SR = uR + cR * sqrt((g + 1) / (2g) * pstar / pR + (g - 1) / (2g))
            S >= SR && return (rhoR, uR, pR)
            r = pstar / pR
            rho = rhoR * (r + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * r + 1)
            return (rho, ustar, pstar)
        else            # right rarefaction
            SH = uR + cR
            S >= SH && return (rhoR, uR, pR)
            cstar = cR * (pstar / pR)^((g - 1) / (2g))
            ST = ustar + cstar
            if S <= ST
                return (rhoR * (pstar / pR)^(1 / g), ustar, pstar)
            end
            u = 2 / (g + 1) * (-cR + (g - 1) / 2 * uR + S)
            c = 2 / (g + 1) * (cR - (g - 1) / 2 * (uR - S))
            rho = rhoR * (c / cR)^(2 / (g - 1))
            return (rho, u, pR * (c / cR)^(2g / (g - 1)))
        end
    end
end
