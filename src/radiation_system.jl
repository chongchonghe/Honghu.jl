# Grey two-moment (M1) radiation transport: physics parameters, variable
# conversion, the HLL flux sweep, and the radiation CFL condition.

"""
    RadiationPhysics(; c, chat = c, a_rad, opacity,
                     closure = LevermoreClosure(),
                     beta_order = 1,
                     tolerance = 1e-9,
                     max_iterations = 100,
                     max_outer_iterations = 5)

Parameters of the grey two-moment radiation module.

  - `c`       : speed of light, in the same units as the rest of the problem.
  - `chat`    : reduced speed of light. Using `chat < c` relaxes the radiation
    CFL condition and leaves the *static* diffusion and free-streaming limits
    intact; it does change dynamic-diffusion problems, so only reduce it when
    the radiation transport is much faster than every other signal in the
    problem.
  - `a_rad`   : radiation constant `4σ/c`, so that the equilibrium radiation
    energy density is `a_rad T⁴`.
  - `opacity` : an [`AbstractOpacity`](@ref).
  - `closure`  : [`LevermoreClosure`](@ref) (the M1 default) or
    [`EddingtonClosure`](@ref), which fixes `chi = 1/3`. Use the latter only to
    reproduce reference solutions derived under the P1 approximation; it cannot
    free-stream.
  - `beta_order` : `1` keeps the `O(v/c)` terms in the matter-radiation coupling
    (radiation force, work, and the comoving-frame corrections); `0` drops them,
    which is appropriate only for a static medium.
  - `tolerance`, `max_iterations`, `max_outer_iterations` : controls of the
    implicit coupling solve (see [`radiation_source_step!`](@ref)).
"""
struct RadiationPhysics{T<:AbstractFloat,O<:AbstractOpacity,C<:AbstractClosure}
    c::T
    chat::T
    a_rad::T
    opacity::O
    closure::C
    beta_order::Int
    tolerance::T
    max_iterations::Int
    max_outer_iterations::Int
end

function RadiationPhysics(; c, chat=c, a_rad, opacity,
                          closure::AbstractClosure=LevermoreClosure(),
                          beta_order::Integer=1,
                          tolerance=1e-9,
                          max_iterations::Integer=100,
                          max_outer_iterations::Integer=5)
    0 <= beta_order <= 1 || throw(ArgumentError("beta_order must be 0 or 1, got $beta_order"))
    chat <= c || throw(ArgumentError("the reduced speed of light must satisfy chat <= c"))
    cc, cch, ar, tol = promote(float(c), float(chat), float(a_rad), float(tolerance))
    RadiationPhysics(cc, cch, ar, opacity, closure, Int(beta_order), tol,
                     Int(max_iterations), Int(max_outer_iterations))
end

"""
    amend_rad(er, Fx, Fy, Fz, c, erad_floor)
        -> (er, Fx, Fy, Fz, fx, fy, fz, repaired)

Project one radiation state onto the admissible set of the M1 closure: energy
density at or above the floor, and reduced flux `|F|/(cE)` strictly below one
(the flux is rescaled, preserving its direction). Also returns the reduced flux
components, which the caller usually wants anyway.
"""
@inline function amend_rad(er::T, Fx::T, Fy::T, Fz::T, c::T, erad_floor::T) where {T}
    repaired = false
    if !(er > erad_floor)
        er = erad_floor
        repaired = true
    end
    ce = c * er
    fx = Fx / ce
    fy = Fy / ce
    fz = Fz / ce
    f = sqrt(fx * fx + fy * fy + fz * fz)
    fmax = one(T) - 20 * eps(T)
    if !(f <= fmax)
        s = (isfinite(f) && f > 0) ? fmax / f : zero(T)
        fx *= s
        fy *= s
        fz *= s
        Fx = fx * ce
        Fy = fy * ce
        Fz = fz * ce
        repaired = true
    else
        Fx = fx * ce
        Fy = fy * ce
        Fz = fz * ce
    end
    return er, Fx, Fy, Fz, fx, fy, fz, repaired
end

"""
    enforce_rad_floors!(R, phys, grid, floors) -> Int

Project the radiation state onto the admissible set of the M1 closure over the
whole array, ghost cells included (see [`amend_rad`](@ref)). Returns the number
of cells repaired.

The radiation counterpart of [`enforce_floors!`](@ref), and separate from
[`rad_cons2prim!`](@ref) for the same reason.
"""
function enforce_rad_floors!(R::AbstractArray{T,4}, phys::RadiationPhysics{T},
                             g::CartesianGrid{T}, floors::Floors{T}) where {T}
    n1, n2, n3 = g.dims
    _amend_range!(R, phys, floors, 1:n1, 1:n2, 1:n3)
end

"""
    amend_rad_state!(R, phys, grid, floors) -> Int

Repair the radiation state of the *interior* cells in place. Used after the
explicit transport stages of the IMEX scheme, whose output must be admissible
before the implicit coupling solve runs; the ghost cells are refilled from the
boundary conditions before they are next read, so there is no point touching
them here.
"""
function amend_rad_state!(R::AbstractArray{T,4}, phys::RadiationPhysics{T},
                          g::CartesianGrid{T}, floors::Floors{T}) where {T}
    _amend_range!(R, phys, floors, g.lo[1]:g.hi[1], g.lo[2]:g.hi[2], g.lo[3]:g.hi[3])
end

function _amend_range!(R::AbstractArray{T,4}, phys::RadiationPhysics{T},
                       floors::Floors{T}, ir, jr, kr) where {T}
    c = phys.c
    nbad = Threads.Atomic{Int}(0)
    threaded_jk(jr, kr) do j, k
        local bad = 0
        @inbounds for i in ir
            er, Fx, Fy, Fz, _, _, _, rep = amend_rad(R[IERAD, i, j, k], R[IFRX, i, j, k],
                                                     R[IFRY, i, j, k], R[IFRZ, i, j, k],
                                                     c, floors.erad)
            if rep
                R[IERAD, i, j, k] = er
                R[IFRX, i, j, k] = Fx
                R[IFRY, i, j, k] = Fy
                R[IFRZ, i, j, k] = Fz
                bad += 1
            end
        end
        bad > 0 && Threads.atomic_add!(nbad, bad)
    end
    return nbad[]
end

"""
    rad_cons2prim!(rprim, R, phys, grid)

Fill the radiation primitives `(Erad, fx, fy, fz)` from the conserved state over
the whole array, ghost cells included.

`R` is **only read**; see [`cons2prim!`](@ref) for why. Run
[`enforce_rad_floors!`](@ref) first if the state may be inadmissible — on an
admissible state the reduced flux this computes automatically satisfies
`|f| < 1`.
"""
function rad_cons2prim!(rprim::A, R::A, phys::RadiationPhysics{T},
                        g::CartesianGrid{T}) where {T,A<:AbstractArray{T,4}}
    n1, n2, n3 = g.dims
    c = phys.c
    threaded_jk(1:n2, 1:n3) do j, k
        @inbounds for i in 1:n1
            er = R[IERAD, i, j, k]
            ce = one(T) / (c * er)
            rprim[IERAD, i, j, k] = er
            rprim[IFRX, i, j, k] = R[IFRX, i, j, k] * ce
            rprim[IFRY, i, j, k] = R[IFRY, i, j, k] * ce
            rprim[IFRZ, i, j, k] = R[IFRZ, i, j, k] * ce
        end
    end
    return rprim
end

"""
    rad_fluxes!(flux, qL, qR, R, phys, Val(D), grid)

Solve the HLL Riemann problem of the M1 system at every interface normal to
direction `D`.

As in the hydro sweep, an inadmissible reconstructed state (non-positive energy
density, or a reduced flux at or beyond the causal limit) falls back to the
piecewise-constant state of the adjoining cells.
"""
function rad_fluxes!(flux::A, qL::A, qR::A, R::A, phys::RadiationPhysics{T},
                     ::Val{D}, g::CartesianGrid{T}) where {T,A<:AbstractArray{T,4},D}
    e1, e2, e3 = _shift(Val(D))
    c = phys.c
    chat = phys.chat
    cl = phys.closure
    ir, jr, kr = face_ranges(g, Val(D))
    threaded_jk(jr, kr) do j, k
        @inbounds for i in ir
            im, jm, km = i - e1, j - e2, k - e3

            erL = qL[IERAD, i, j, k]
            fxL = qL[IFRX, i, j, k]
            fyL = qL[IFRY, i, j, k]
            fzL = qL[IFRZ, i, j, k]
            erR = qR[IERAD, i, j, k]
            fxR = qR[IFRX, i, j, k]
            fyR = qR[IFRY, i, j, k]
            fzR = qR[IFRZ, i, j, k]

            fL = sqrt(fxL * fxL + fyL * fyL + fzL * fzL)
            fR = sqrt(fxR * fxR + fyR * fyR + fzR * fzR)

            ceL = c * erL
            ceR = c * erR
            FxL = fxL * ceL
            FyL = fyL * ceL
            FzL = fzL * ceL
            FxR = fxR * ceR
            FyR = fyR * ceR
            FzR = fzR * ceR

            if !(erL > 0) || !(erR > 0) || !(fL < 1) || !(fR < 1)
                erL = R[IERAD, im, jm, km]
                FxL = R[IFRX, im, jm, km]
                FyL = R[IFRY, im, jm, km]
                FzL = R[IFRZ, im, jm, km]
                erR = R[IERAD, i, j, k]
                FxR = R[IFRX, i, j, k]
                FyR = R[IFRY, i, j, k]
                FzR = R[IFRZ, i, j, k]
            end

            F = hll_radiation(erL, FxL, FyL, FzL, erR, FxR, FyR, FzR, c, chat, Val(D), cl)

            flux[IERAD, i, j, k] = F[1]
            flux[IFRX, i, j, k] = F[2]
            flux[IFRY, i, j, k] = F[3]
            flux[IFRZ, i, j, k] = F[4]
        end
    end
    return nothing
end

"""
    radiation_timestep(phys, grid, cfl)

Radiation CFL condition, `dt = cfl / (chat Σ_d 1/dx_d)` over the active
directions. Independent of the state because the M1 signal speed is bounded by
`chat`.
"""
function radiation_timestep(phys::RadiationPhysics{T}, g::CartesianGrid{T}, cfl::Real) where {T}
    s = zero(T)
    for d in 1:3
        g.active[d] && (s += one(T) / g.dx[d])
    end
    return T(cfl) / (phys.chat * s)
end
