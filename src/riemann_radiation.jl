# M1 closure and the HLL Riemann solver for the two-moment radiation system.
#
# The grey two-moment (M1) system evolved here is
#
#     d_t E   + (chat/c)  d_j F_j        = sources
#     d_t F_i + (chat*c)  d_j (T_ij E)   = sources
#
# where `T_ij` is the (dimensionless) Eddington tensor supplied by the M1
# closure and `chat <= c` is the reduced speed of light. Setting `chat = c`
# recovers the untouched M1 system.
#
# Wave speeds are those of Skinner & Ostriker (2013) evaluated in the frozen
# Eddington-tensor approximation of Balsara (1999): the characteristic speed
# normal to the interface is `chat * sqrt(T_nn)`, so the solution of the Riemann
# problem is always the HLL star state.

"""
    AbstractClosure

Supertype for the closure that supplies the Eddington factor `chi(f)`:
[`LevermoreClosure`](@ref) or [`EddingtonClosure`](@ref).
"""
abstract type AbstractClosure end

"""
    LevermoreClosure()

The M1 closure of Levermore (1984),
`chi(f) = (3 + 4f²) / (5 + 2√(4 - 3f²))`. Interpolates between `1/3` in the
diffusion limit and `1` in the free-streaming limit, so radiation travels in
straight lines and casts shadows. The default.
"""
struct LevermoreClosure <: AbstractClosure end

"""
    EddingtonClosure()

The Eddington (P1) approximation, `chi = 1/3` for every `f`, which makes the
radiation pressure isotropic everywhere.

This is *not* a two-moment closure in the usual sense: it cannot free-stream and
will not cast a shadow. It is here because several semi-analytic reference
solutions are derived under it -- notably the Lowrie & Edwards (2008) radiative
shock -- so reproducing them requires matching the assumption.
"""
struct EddingtonClosure <: AbstractClosure end

"""
    eddington_factor([closure = LevermoreClosure()], f)

Eddington factor for the reduced flux `f = |F| / (c E)`.
"""
@inline function eddington_factor(::LevermoreClosure, f_in::T) where {T<:AbstractFloat}
    f = clamp(f_in, zero(T), one(T))
    ffac = sqrt(4 - 3 * f * f)
    (3 + 4 * f * f) / (5 + 2 * ffac)
end

@inline eddington_factor(::EddingtonClosure, f::T) where {T<:AbstractFloat} = one(T) / 3

@inline eddington_factor(f::AbstractFloat) = eddington_factor(LevermoreClosure(), f)

"""
    EddingtonTensor(diag, aniso, n)

The Eddington tensor `T_ij = diag δ_ij + aniso n_i n_j` in factored form, where
`n` is the unit vector along the radiation flux. Index it as `T[i, j]`.
"""
struct EddingtonTensor{T<:AbstractFloat}
    diag::T
    aniso::T
    n::SVec{3,T}
end

"""
    eddington_tensor([closure = LevermoreClosure()], fx, fy, fz) -> EddingtonTensor

Build the Eddington tensor from the reduced flux vector.
"""
@inline function eddington_tensor(cl::AbstractClosure, fx::T, fy::T, fz::T) where {T<:AbstractFloat}
    f = sqrt(fx * fx + fy * fy + fz * fz)
    s = f > zero(T) ? one(T) / f : zero(T)
    chi = eddington_factor(cl, f)
    EddingtonTensor((1 - chi) / 2, (3 * chi - 1) / 2, SVec(fx * s, fy * s, fz * s))
end

@inline eddington_tensor(fx::T, fy::T, fz::T) where {T<:AbstractFloat} =
    eddington_tensor(LevermoreClosure(), fx, fy, fz)

@inline function Base.getindex(T::EddingtonTensor{S}, a::Int, b::Int) where {S}
    d = a == b ? T.diag : zero(S)
    d + T.aniso * T.n[a] * T.n[b]
end

"""
    m1_flux(er, Fx, Fy, Fz, tedd, D) -> (SVec{4}, S)

Physical M1 flux normal to a face with normal direction `D` (before the reduced
speed-of-light rescaling), together with the normalised signal speed
`S = max(0.1, sqrt(T_DD))`.

The `0.1` floor mirrors Quokka: it keeps `S_L = S_R = 0` (and hence a `0/0`
flux) from occurring in cells where the Eddington tensor becomes degenerate.
"""
@inline function m1_flux(er::T, Fx::T, Fy::T, Fz::T, tedd::EddingtonTensor{T}, D::Int) where {T}
    Fn = D == 1 ? Fx : D == 2 ? Fy : Fz
    F = SVec(Fn, tedd[D, 1] * er, tedd[D, 2] * er, tedd[D, 3] * er)
    S = max(T(0.1), sqrt(tedd[D, D]))
    return F, S
end

"""
    hll_radiation(erL, FxL, FyL, FzL, erR, FxR, FyR, FzR, c, chat, Val(D),
                  closure = LevermoreClosure()) -> SVec{4}

HLL flux of the grey M1 system across one interface with normal along direction
`D`. Components are `(energy, x-momentum, y-momentum, z-momentum)`.
"""
@inline function hll_radiation(erL::T, FxL::T, FyL::T, FzL::T,
                               erR::T, FxR::T, FyR::T, FzR::T,
                               c::T, chat::T, ::Val{D},
                               cl::AbstractClosure=LevermoreClosure(),
                               epsE::T=one(T)) where {T<:AbstractFloat,D}
    ci = one(T) / c
    tL = eddington_tensor(cl, FxL * ci / erL, FyL * ci / erL, FzL * ci / erL)
    tR = eddington_tensor(cl, FxR * ci / erR, FyR * ci / erR, FzR * ci / erR)

    GL0, SL0 = m1_flux(erL, FxL, FyL, FzL, tL, D)
    GR0, SR0 = m1_flux(erR, FxR, FyR, FzR, tR, D)

    # rescale for the reduced speed of light
    ce = chat * ci
    cm = chat * c
    GL = SVec(GL0[1] * ce, GL0[2] * cm, GL0[3] * cm, GL0[4] * cm)
    GR = SVec(GR0[1] * ce, GR0[2] * cm, GR0[3] * cm, GR0[4] * cm)

    S_L = -chat * SL0
    S_R = chat * SR0

    U_L = SVec(erL, FxL, FyL, FzL)
    U_R = SVec(erR, FxR, FyR, FzR)

    # frozen Eddington tensor => the interface is always inside the star region.
    # `epsE` scales the dissipation of the energy component only: the optional
    # wave-speed correction of an optically thick face (see RadiationPhysics).
    dU = U_R - U_L
    dU = SVec(epsE * dU[1], dU[2], dU[3], dU[4])
    (S_R * GL - S_L * GR + (S_R * S_L) * dU) / (S_R - S_L)
end
