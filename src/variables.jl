# State-vector layout and pointwise conversions.
#
# Hydrodynamics (5 variables)
#   conserved   U = (rho, rho*vx, rho*vy, rho*vz, Etot)
#   primitive   W = (rho, vx, vy, vz, p)
#
# Radiation, M1: 4 variables *per frequency group*, groups stored contiguously,
# so a state vector is `NRAD * ngroups` long. See `gbase` in radiation_system.jl.
#   conserved   R = (Erad, Fx, Fy, Fz)
#   primitive   V = (Erad, fx, fy, fz)      with f = F / (c Erad), |f| <= 1
#
# Note that the momentum component for direction `d` is at index `d + 1` in both
# the hydro and the radiation vector. The ghost-filling and reflecting-boundary
# code exploits this.

"Number of hydrodynamic variables."
const NHYDRO = 5
"Number of M1 radiation variables *per frequency group*: energy density and three\nflux components. A grey run has one group, so the state vector is `NRAD` long;\nwith `NG` groups it is `NRAD * NG`."
const NRAD = 4

# conserved hydro
const IRHO = 1
const IMX = 2
const IMY = 3
const IMZ = 4
const IEGAS = 5
# primitive hydro (IRHO shared)
const IVX = 2
const IVY = 3
const IVZ = 4
const IPRE = 5
# radiation (conserved and primitive share the energy slot)
const IERAD = 1
const IFRX = 2
const IFRY = 3
const IFRZ = 4

"""
    cons2prim(eos, u::SVec{5}) -> SVec{5}

Convert a conserved hydro state to primitives `(rho, vx, vy, vz, p)`.
"""
@inline function cons2prim(eos::AbstractEOS, u::SVec{5,T}) where {T}
    rho = u[IRHO]
    inv_rho = one(T) / rho
    vx = u[IMX] * inv_rho
    vy = u[IMY] * inv_rho
    vz = u[IMZ] * inv_rho
    ekin = T(0.5) * rho * (vx * vx + vy * vy + vz * vz)
    eint = u[IEGAS] - ekin
    SVec(rho, vx, vy, vz, pressure(eos, rho, eint))
end

"""
    prim2cons(eos, w::SVec{5}) -> SVec{5}

Convert primitives `(rho, vx, vy, vz, p)` to the conserved hydro state.
"""
@inline function prim2cons(eos::AbstractEOS, w::SVec{5,T}) where {T}
    rho = w[IRHO]
    vx, vy, vz = w[IVX], w[IVY], w[IVZ]
    eint = internal_energy(eos, rho, w[IPRE])
    etot = eint + T(0.5) * rho * (vx * vx + vy * vy + vz * vz)
    SVec(rho, rho * vx, rho * vy, rho * vz, etot)
end

"""
    kinetic_energy(u::SVec{5})

Kinetic energy density `|m|^2 / (2 rho)` of a conserved hydro state.
"""
@inline function kinetic_energy(u::SVec{5,T}) where {T}
    T(0.5) * (u[IMX] * u[IMX] + u[IMY] * u[IMY] + u[IMZ] * u[IMZ]) / u[IRHO]
end

"""
    internal_energy(u::SVec{5})

Internal energy density `Etot - Ekin` of a conserved hydro state.
"""
@inline internal_energy(u::SVec{5}) = u[IEGAS] - kinetic_energy(u)

"""
    rad_cons2prim(c, r::SVec{4}) -> SVec{4}

Convert `(Erad, Fx, Fy, Fz)` to `(Erad, fx, fy, fz)` with reduced flux
`f = F / (c Erad)`.
"""
@inline function rad_cons2prim(c, r::SVec{4,T}) where {T}
    er = r[IERAD]
    s = one(T) / (c * er)
    SVec(er, r[IFRX] * s, r[IFRY] * s, r[IFRZ] * s)
end

"""
    rad_prim2cons(c, v::SVec{4}) -> SVec{4}

Inverse of [`rad_cons2prim`](@ref).
"""
@inline function rad_prim2cons(c, v::SVec{4,T}) where {T}
    er = v[IERAD]
    s = c * er
    SVec(er, v[IFRX] * s, v[IFRY] * s, v[IFRZ] * s)
end

# --- small array <-> SVec helpers -------------------------------------------
#
# `A` is always laid out as (nvar, i, j, k).

@inline function getstate(A::AbstractArray{T,4}, ::Val{N}, i, j, k) where {T,N}
    SVec(ntuple(n -> @inbounds(A[n, i, j, k]), Val(N)))
end

@inline function setstate!(A::AbstractArray{T,4}, u::SVec{N,T}, i, j, k) where {T,N}
    @inbounds for n in 1:N
        A[n, i, j, k] = u[n]
    end
    return nothing
end

@inline function addstate!(A::AbstractArray{T,4}, u::SVec{N,T}, i, j, k) where {T,N}
    @inbounds for n in 1:N
        A[n, i, j, k] += u[n]
    end
    return nothing
end
