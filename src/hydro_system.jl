# Hydrodynamics: variable conversion, the HLLC flux sweep, the method-of-lines
# right-hand side, and the hydrodynamic CFL condition.

"""
    Floors(; rho = 1e-30, pressure = 1e-30, erad = 1e-30)

Lower bounds enforced on the state after every stage. Cells that fall below
`rho` are reset to `(rho, v = 0, p = pressure)`; cells whose internal energy
falls below `pressure/(gamma-1)` keep their momentum and have only the thermal
part reset.
"""
struct Floors{T<:AbstractFloat}
    rho::T
    pressure::T
    erad::T
end

function Floors(; rho=1e-30, pressure=1e-30, erad=1e-30)
    Floors(promote(float(rho), float(pressure), float(erad))...)
end

"""
    enforce_floors!(U, eos, grid, floors) -> Int

Project the conserved gas state onto the admissible set, over the whole array,
ghost cells included. Cells below `floors.rho` are reset to a quiescent floor
state; cells whose internal energy is below `floors.pressure/(gamma-1)` keep
their momentum and have only the thermal part reset.

Returns the number of cells repaired, which the driver accumulates so that a
silently unphysical run is visible.

This is deliberately a *separate* operation from [`cons2prim!`](@ref), matching
Quokka's split between `EnforceLimits` and its `const`-qualified
`ComputePrimVars`. The two used to be fused, which meant that reading a
diagnostic field silently mutated the state being diagnosed: results depended on
how often you looked, and the repairs went uncounted. Keep them apart.
"""
function enforce_floors!(U::AbstractArray{T,4}, eos::AbstractEOS, g::CartesianGrid{T},
                         floors::Floors{T}) where {T}
    n1, n2, n3 = g.dims
    nbad = Threads.Atomic{Int}(0)
    eint_min = internal_energy(eos, floors.rho, floors.pressure)
    threaded_jk(1:n2, 1:n3) do j, k
        local bad = 0
        @inbounds for i in 1:n1
            rho = U[IRHO, i, j, k]
            if !(rho > floors.rho)
                # near-vacuum or NaN: reset to a quiescent floor state
                rho = floors.rho
                U[IRHO, i, j, k] = rho
                U[IMX, i, j, k] = zero(T)
                U[IMY, i, j, k] = zero(T)
                U[IMZ, i, j, k] = zero(T)
                U[IEGAS, i, j, k] = eint_min
                bad += 1
            end
            inv_rho = one(T) / rho
            vx = U[IMX, i, j, k] * inv_rho
            vy = U[IMY, i, j, k] * inv_rho
            vz = U[IMZ, i, j, k] * inv_rho
            ekin = rho * (vx * vx + vy * vy + vz * vz) / 2
            emin = internal_energy(eos, rho, floors.pressure)
            if !(U[IEGAS, i, j, k] - ekin > emin)
                U[IEGAS, i, j, k] = ekin + emin
                bad += 1
            end
        end
        bad > 0 && Threads.atomic_add!(nbad, bad)
    end
    return nbad[]
end

"""
    cons2prim!(prim, U, eos, grid)

Fill `prim = (rho, vx, vy, vz, p)` from the conserved state `U` over the whole
array, ghost cells included.

`U` is **only read**. Every caller — the right-hand side, `timestep`, and every
diagnostic and plotting path — gets the same guarantee, so looking at a
simulation cannot change it. Admissibility is the caller's job, via
[`enforce_floors!`](@ref); on a state that has not been floored this returns
whatever the arithmetic gives, including negative pressures.
"""
function cons2prim!(prim::A, U::A, eos::AbstractEOS,
                    g::CartesianGrid{T}) where {T,A<:AbstractArray{T,4}}
    n1, n2, n3 = g.dims
    threaded_jk(1:n2, 1:n3) do j, k
        @inbounds for i in 1:n1
            rho = U[IRHO, i, j, k]
            inv_rho = one(T) / rho
            vx = U[IMX, i, j, k] * inv_rho
            vy = U[IMY, i, j, k] * inv_rho
            vz = U[IMZ, i, j, k] * inv_rho
            eint = U[IEGAS, i, j, k] - rho * (vx * vx + vy * vy + vz * vz) / 2
            prim[IRHO, i, j, k] = rho
            prim[IVX, i, j, k] = vx
            prim[IVY, i, j, k] = vy
            prim[IVZ, i, j, k] = vz
            prim[IPRE, i, j, k] = pressure(eos, rho, eint)
        end
    end
    return prim
end

"""
    prim2cons!(U, prim, eos, grid)

Fill the conserved state from primitives over the whole array.
"""
function prim2cons!(U::A, prim::A, eos::AbstractEOS, g::CartesianGrid{T}) where {T,A<:AbstractArray{T,4}}
    n1, n2, n3 = g.dims
    threaded_jk(1:n2, 1:n3) do j, k
        @inbounds for i in 1:n1
            w = SVec(prim[IRHO, i, j, k], prim[IVX, i, j, k], prim[IVY, i, j, k],
                     prim[IVZ, i, j, k], prim[IPRE, i, j, k])
            setstate!(U, prim2cons(eos, w), i, j, k)
        end
    end
    return U
end

"""
    hydro_fluxes!(flux, qL, qR, prim, eos, solver, Val(D), grid)

Solve the Riemann problem at every interface normal to direction `D` with
`solver` (an [`AbstractRiemannSolver`](@ref)) and store the resulting
conservative flux.

If the reconstruction produced an inadmissible state (non-positive density or
pressure, or a NaN) the interface falls back to the piecewise-constant state of
the adjoining cell, which is guaranteed admissible because the floors have
already been applied.

The carbuncle inputs `du` and `dw` are computed for every solver, not gated on
whether the chosen one consumes them. Quokka does the same
(`hydro_system.hpp:1484-1498`), and reuses the intermediates for its velocity
divergence; the reads are 18 per face in 3D and 2 in 1D, against a reconstruction
and a Riemann solve.
"""
function hydro_fluxes!(flux::A, qL::A, qR::A, prim::A, eos::AbstractEOS,
                       solver::AbstractRiemannSolver,
                       ::Val{D}, g::CartesianGrid{T}) where {T,A<:AbstractArray{T,4},D}
    e1, e2, e3 = _shift(Val(D))
    s1 = _tshift1(Val(D))
    s2 = _tshift2(Val(D))
    act1 = g.active[_tdir1(Val(D))]
    act2 = g.active[_tdir2(Val(D))]
    iu = D + 1                    # normal velocity/momentum slot
    iv = _tdir1(Val(D)) + 1       # first transverse slot
    iw = _tdir2(Val(D)) + 1       # second transverse slot
    gam = T(adiabatic_index(eos))
    ir, jr, kr = face_ranges(g, Val(D))

    threaded_jk(jr, kr) do j, k
        @inbounds for i in ir
            im, jm, km = i - e1, j - e2, k - e3   # cell on the low side of the face

            rhoL = qL[IRHO, i, j, k]
            pL = qL[IPRE, i, j, k]
            uL = qL[iu, i, j, k]
            vL = qL[iv, i, j, k]
            wL = qL[iw, i, j, k]
            if !(rhoL > 0) || !(pL > 0)
                rhoL = prim[IRHO, im, jm, km]
                pL = prim[IPRE, im, jm, km]
                uL = prim[iu, im, jm, km]
                vL = prim[iv, im, jm, km]
                wL = prim[iw, im, jm, km]
            end

            rhoR = qR[IRHO, i, j, k]
            pR = qR[IPRE, i, j, k]
            uR = qR[iu, i, j, k]
            vR = qR[iv, i, j, k]
            wR = qR[iw, i, j, k]
            if !(rhoR > 0) || !(pR > 0)
                rhoR = prim[IRHO, i, j, k]
                pR = prim[IPRE, i, j, k]
                uR = prim[iu, i, j, k]
                vR = prim[iv, i, j, k]
                wR = prim[iw, i, j, k]
            end

            csL = soundspeed(eos, rhoL, pL)
            csR = soundspeed(eos, rhoR, pR)
            EL = internal_energy(eos, rhoL, pL) + rhoL * (uL * uL + vL * vL + wL * wL) / 2
            ER = internal_energy(eos, rhoR, pR) + rhoR * (uR * uR + vR * vR + wR * wR) / 2

            # jump in normal velocity, and the smallest transverse velocity
            # difference in the two adjoining cells (carbuncle correction)
            du = prim[iu, i, j, k] - prim[iu, im, jm, km]
            dw = T(Inf)
            if act1
                a = prim[iv, im+s1[1], jm+s1[2], km+s1[3]] - prim[iv, im, jm, km]
                b = prim[iv, im, jm, km] - prim[iv, im-s1[1], jm-s1[2], km-s1[3]]
                c = prim[iv, i+s1[1], j+s1[2], k+s1[3]] - prim[iv, i, j, k]
                d = prim[iv, i, j, k] - prim[iv, i-s1[1], j-s1[2], k-s1[3]]
                dw = min(dw, min(a, b), min(c, d))
            end
            if act2
                a = prim[iw, im+s2[1], jm+s2[2], km+s2[3]] - prim[iw, im, jm, km]
                b = prim[iw, im, jm, km] - prim[iw, im-s2[1], jm-s2[2], km-s2[3]]
                c = prim[iw, i+s2[1], j+s2[2], k+s2[3]] - prim[iw, i, j, k]
                d = prim[iw, i, j, k] - prim[iw, i-s2[1], j-s2[2], k-s2[3]]
                dw = min(dw, min(a, b), min(c, d))
            end
            isfinite(dw) || (dw = zero(T))

            sL = HydroState(rhoL, uL, vL, wL, pL, csL, EL)
            sR = HydroState(rhoR, uR, vR, wR, pR, csR, ER)
            F = riemann_flux(solver, sL, sR, gam, du, dw)

            flux[IRHO, i, j, k] = F[1]
            flux[iu, i, j, k] = F[2]
            flux[iv, i, j, k] = F[3]
            flux[iw, i, j, k] = F[4]
            flux[IEGAS, i, j, k] = F[5]
        end
    end
    return nothing
end

"""
    flux_divergence!(dU, flux, Val(D), grid, nvar)

Subtract `(F_{i+1/2} - F_{i-1/2}) / dx` from `dU` over the interior cells.
"""
function flux_divergence!(dU::A, flux::A, ::Val{D}, g::CartesianGrid{T},
                          nvar::Int) where {T,A<:AbstractArray{T,4},D}
    e1, e2, e3 = _shift(Val(D))
    invdx = one(T) / g.dx[D]
    ir = g.lo[1]:g.hi[1]
    jr = g.lo[2]:g.hi[2]
    kr = g.lo[3]:g.hi[3]
    threaded_jk(jr, kr) do j, k
        @inbounds for i in ir, n in 1:nvar
            dU[n, i, j, k] -= invdx * (flux[n, i+e1, j+e2, k+e3] - flux[n, i, j, k])
        end
    end
    return nothing
end

"""
    hydro_timestep(prim, eos, grid, cfl)

Largest stable step of the unsplit method-of-lines scheme,

    dt = cfl / max_cells Σ_d (|v_d| + c_s) / dx_d

summed over the active directions. Reduces to `cfl dx / (|v| + c_s)` in 1D.
"""
function hydro_timestep(prim::AbstractArray{T,4}, eos::AbstractEOS, g::CartesianGrid{T},
                        cfl::Real) where {T}
    inv_dx = ntuple(d -> g.active[d] ? one(T) / g.dx[d] : zero(T), 3)
    ir = g.lo[1]:g.hi[1]
    smax = reduce_max_jk(g.lo[2]:g.hi[2], g.lo[3]:g.hi[3], T) do j, k
        local m = zero(T)
        @inbounds for i in ir
            rho = prim[IRHO, i, j, k]
            cs = soundspeed(eos, rho, prim[IPRE, i, j, k])
            s = (abs(prim[IVX, i, j, k]) + cs) * inv_dx[1] +
                (abs(prim[IVY, i, j, k]) + cs) * inv_dx[2] +
                (abs(prim[IVZ, i, j, k]) + cs) * inv_dx[3]
            m = max(m, s)
        end
        m
    end
    smax > 0 || return T(Inf)
    return T(cfl) / smax
end
