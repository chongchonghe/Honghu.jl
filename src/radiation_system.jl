# Two-moment (M1) radiation transport, grey or multigroup: physics parameters,
# variable conversion, the HLL flux sweep, and the radiation CFL condition.

"""
    RadiationPhysics(; c, chat = c, a_rad, opacity,
                     groups = GREY, k_B = 1,
                     closure = LevermoreClosure(),
                     coupling = BracketedCoupling(),
                     dust = NoDust(),
                     beta_order = 1,
                     tolerance = 1e-9,
                     max_iterations = 100,
                     max_outer_iterations = 5)

Parameters of the two-moment (M1) radiation module, grey or multigroup.

  - `c`       : speed of light, in the same units as the rest of the problem.
  - `chat`    : reduced speed of light. Using `chat < c` relaxes the radiation
    CFL condition and leaves the *static* diffusion and free-streaming limits
    intact; it does change dynamic-diffusion problems, so only reduce it when
    the radiation transport is much faster than every other signal in the
    problem. With a `chat_limiter` other than the default this is the *nominal*
    reduction `phi0 c`: a floor on the local `chat` and the speed the time step
    is still set from, not the speed every cell runs at.
  - `chat_limiter` : an [`AbstractChatLimiter`](@ref) making the reduced speed of
    light local. The default, [`ConstantChat`](@ref), is the standard RSLA and a
    bit-for-bit no-op; [`StabilityBound`](@ref) is the improved RSLA, which
    raises `chat` towards `c` in cells optically thick on the grid scale, where
    the RSLA equilibrates matter and radiation at the wrong temperature and
    where the implicit coupling supplies the damping that makes the full speed
    of light stable anyway.
  - `a_rad`   : radiation constant `4σ/c`, so that the equilibrium radiation
    energy density is `a_rad T⁴`.
  - `opacity` : an [`AbstractOpacity`](@ref).
  - `groups`  : a [`RadiationGroups`](@ref). The default, [`GREY`](@ref), is the
    single group spanning every frequency, and reproduces grey radiation exactly;
    `ngroups > 1` puts the whole radiation module on the multigroup path, from
    the state-vector length down to the coupling solver.
  - `k_B`     : Boltzmann constant. It is read *only* to turn a group edge into
    `x = energy_unit * edge / (k_B T)` for the Planck fractions, so it matters
    only when the group edges are in physical units -- pass `CGS.k_B` alongside
    `energy_unit = CGS.h_planck` for edges in Hz. A grey run computes no
    fractions and never reads it, and unit-free edges given directly in units of
    `k_B T` want the default `1`; wrong in either of those cases it costs
    nothing, wrong in the physical-units case it moves energy between groups.
  - `coupling` : the implicit matter-radiation solver used when `ngroups > 1`, an
    `AbstractCoupling` -- [`BracketedCoupling`](@ref) (the default,
    allocation-free), [`QuokkaCoupling`](@ref) or [`AdaptiveCoupling`](@ref). A
    grey run ignores it and uses [`radiation_source_step!`](@ref)'s own 2×2
    Newton driver.
  - `dust`     : an [`AbstractDustModel`](@ref). [`NoDust`](@ref), the default,
    couples the radiation to the gas. [`BateKetoDust`](@ref) couples it to dust
    at its own temperature instead, and the dust to the gas by collisions; the
    `opacity` is then the dust's, the gas has none, and `coupling` must be
    [`BracketedCoupling`](@ref). A grey run with dust goes through the
    multigroup driver, which handles one group unchanged.
  - `closure`  : [`LevermoreClosure`](@ref) (the M1 default) or
    [`EddingtonClosure`](@ref), which fixes `chi = 1/3`. Use the latter only to
    reproduce reference solutions derived under the P1 approximation; it cannot
    free-stream.
  - `beta_order` : `1` keeps the `O(v/c)` terms in the matter-radiation coupling
    (radiation force, work, and the comoving-frame corrections); `0` drops them,
    which is appropriate only for a static medium.
  - `tolerance`, `max_iterations`, `max_outer_iterations` : controls of the
    implicit coupling solve (see [`radiation_source_step!`](@ref)).
  - `wavespeed_correction = false` : scale the HLL dissipation of the radiation
    *energy* at each face by `epsilon = min(1, 1/tau)`, with `tau` the harmonic
    mean of the two cells' flux-mean optical depths across the face (Skinner et
    al. 2019; optional in Quokka). In optically thick cells the HLL
    dissipation `sigma chat (E_R - E_L)/2` is a numerical diffusion far faster
    than the physical one, `c/(3 tau)`; the correction removes it. Allowed
    with the standard RSLA only: with a local `chat_limiter` the stability bound
    no longer holds. A face's correction is set by both cells beside it, so a
    thick cell bounded with its own correction can run at `phi = 1` behind an
    almost undamped face beside a thin one, and in a uniform medium the lowered
    dissipation lets modes near a quarter of the grid frequency grow wherever
    the radiation's inertia matters, which the checkerboard bound does not see
    (paper Sec. 3.2; `analysis/v2/check_wavespeed.py`).
"""
struct RadiationPhysics{NG,M,T<:AbstractFloat,O<:AbstractOpacity,C<:AbstractClosure,
                        S<:AbstractCoupling,L<:AbstractChatLimiter,D<:AbstractDustModel}
    c::T
    chat::T
    a_rad::T
    k_B::T
    groups::RadiationGroups{NG,M,T}
    opacity::O
    closure::C
    coupling::S
    chat_limiter::L
    beta_order::Int
    tolerance::T
    max_iterations::Int
    max_outer_iterations::Int
    dust::D
    wavespeed_correction::Bool
end

function RadiationPhysics(; c, chat=c, a_rad, opacity,
                          groups::RadiationGroups=GREY,
                          k_B=1.0,
                          closure::AbstractClosure=LevermoreClosure(),
                          coupling::AbstractCoupling=BracketedCoupling(),
                          chat_limiter::AbstractChatLimiter=ConstantChat(),
                          dust::AbstractDustModel=NoDust(),
                          beta_order::Integer=1,
                          tolerance=1e-9,
                          max_iterations::Integer=100,
                          max_outer_iterations::Integer=5,
                          wavespeed_correction::Bool=false)
    0 <= beta_order <= 1 || throw(ArgumentError("beta_order must be 0 or 1, got $beta_order"))
    chat <= c || throw(ArgumentError("the reduced speed of light must satisfy chat <= c"))
    k_B > 0 || throw(ArgumentError("k_B must be positive, got $k_B"))
    wavespeed_correction && !(chat_limiter isa ConstantChat) &&
        throw(ArgumentError("the wave-speed correction cannot be combined with a local " *
                            "chat_limiter: the stability bound does not hold for it"))
    dust isa NoDust || coupling isa BracketedCoupling ||
        throw(ArgumentError("a dust model is supported by BracketedCoupling only, " *
                            "got coupling = $(nameof(typeof(coupling)))"))
    cc, cch, ar, kb, tol =
        promote(float(c), float(chat), float(a_rad), float(k_B), float(tolerance))
    gr = _convert_groups(groups, typeof(cc))
    RadiationPhysics(cc, cch, ar, kb, gr, opacity, closure, coupling,
                     _convert_limiter(chat_limiter, typeof(cc)), Int(beta_order), tol,
                     Int(max_iterations), Int(max_outer_iterations),
                     _convert_dust(dust, typeof(cc)), wavespeed_correction)
end

# A limiter's parameters must be the simulation's float type, for the same
# reason `GREY`'s edges are: `chat_factor` is called with `T` arguments in the
# inner loop and a `Float64` `beta` in a `Float32` run would promote every one.
_convert_limiter(l::ConstantChat, ::Type{T}) where {T} = l
_convert_limiter(l::StabilityBound, ::Type{T}) where {T} =
    StabilityBound(T(l.beta), l.energy_step, T(l.xi), l.flux_form)
_convert_limiter(l::OpticalDepthFit, ::Type{T}) where {T} =
    OpticalDepthFit(T(l.s), T(l.p), l.per_group, l.flux_form)

# `GREY` is a Float64 constant; a Float32 simulation needs its own copy.
_convert_groups(g::RadiationGroups{NG,M,T}, ::Type{T}) where {NG,M,T} = g
_convert_groups(g::RadiationGroups{NG,M}, ::Type{T}) where {NG,M,T} =
    RadiationGroups{NG,M,T}(SVec{M,T}(convert.(T, Tuple(g.edges))...), T(g.energy_unit))

"""
    ngroups(phys) -> Int

The number of frequency groups, a compile-time constant so every radiation
kernel specialises on it. One group is grey radiation and is the default.
"""
@inline ngroups(::RadiationPhysics{NG}) where {NG} = NG

"""
    nradvars(phys) -> Int

Length of the radiation state vector, `4 * ngroups`: an energy density and three
flux components per group.
"""
@inline nradvars(::RadiationPhysics{NG}) where {NG} = NRAD * NG

"""
    gbase(g) -> Int

Index offset of group `g` in the radiation state vector, so that group `g`'s
energy density is at `gbase(g) + IERAD` and its `d` momentum at `gbase(g) + d + 1`.

Groups are stored contiguously, four components at a time, which is Quokka's
`radEnergy_index + numRadVarsPerGroup * g` layout. Keeping a group's components
adjacent means one group's Riemann problem reads one cache line, and it keeps
the momentum-index convention (`d + 1` within a group) that `fill_ghosts!`
relies on to serve hydro and radiation with one implementation.
"""
@inline gbase(g::Integer) = NRAD * (g - 1)

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
function enforce_rad_floors!(R::AbstractArray{T,4}, phys::RadiationPhysics{NG,M,T},
                             g::CartesianGrid{T}, floors::Floors{T}) where {NG,M,T}
    n1, n2, n3 = g.dims
    _amend_range!(R, phys, floors, 1:n1, 1:n2, 1:n3)
end

"""
    amend_rad_state!(R, phys, grid, floors) -> Int

Repair the radiation state of the *interior* cells in place. Used after each
implicit coupling solve of the IMEX scheme, so that every stage hands the next
transport step an admissible state -- not between the transport stage and the
solve, where the flux must be left alone (see [`floor_rad_energy!`](@ref)). The
ghost cells are refilled from the boundary conditions before they are next read,
so there is no point touching them here.
"""
function amend_rad_state!(R::AbstractArray{T,4}, phys::RadiationPhysics{NG,M,T},
                          g::CartesianGrid{T}, floors::Floors{T}) where {NG,M,T}
    _amend_range!(R, phys, floors, g.lo[1]:g.hi[1], g.lo[2]:g.hi[2], g.lo[3]:g.hi[3])
end

"""
    floor_rad_energy!(R, U, chat, phys, grid, floors) -> Int

Hand the transport stage's radiation state to the implicit coupling solve
unchanged where the solve can absorb it, and floor the radiation energy only
where it cannot. Used after the explicit transport stages of the IMEX scheme,
before the solve; returns the number of cells floored.

Neither the flux nor the energy is limited where the solve can take the state.
In an optically thick cell run at a local CFL number far above one -- which is
what a local reduced speed of light does at `phi = 1` -- the transport stage
leaves `|F| >> cE`, and the solve, which divides the flux by
`1 + chat rho kappa_F dt >> 1`, is exactly what turns it into the diffusion flux
`-c grad E / (3 rho kappa_F)`. Clipping `|F|` to `cE` first threw that flux away:
on the thick hot spot of the paper's test it raised the IRSLA error from 0.0003%
to 0.80%, twice that of the standard RSLA. The same stage can also leave `E < 0`
where a thick cell at large `phi` borders cells at small `phi` across a steep
jump; raising `E` to the floor there adds energy every substep, and on an Ensman
shock resolved by 64 cells it made the IRSLA error 17 times that of the RSLA.

The solve conserves `e_int + sum_g E_g/phi_g` (with `phi_g = chat_g/c`) and
returns a positive temperature whenever that is positive, so a negative `E` is
left for it to absorb. Only a cell whose budget is not positive is floored, as
before; the full repair, [`amend_rad_state!`](@ref), runs after the solve.
"""
function floor_rad_energy!(R::AbstractArray{T,4}, U::AbstractArray{T,4},
                           chat::AbstractArray{T,4}, phys::RadiationPhysics{NG,M,T},
                           g::CartesianGrid{T}, floors::Floors{T}) where {NG,M,T}
    ef = floors.erad / NG
    c = phys.c
    nbad = Threads.Atomic{Int}(0)
    threaded_jk(g.lo[2]:g.hi[2], g.lo[3]:g.hi[3]) do j, k
        local bad = 0
        @inbounds for i in g.lo[1]:g.hi[1]
            rho = U[IRHO, i, j, k]
            mx, my, mz = U[IMX, i, j, k], U[IMY, i, j, k], U[IMZ, i, j, k]
            budget = U[IEGAS, i, j, k] - (mx * mx + my * my + mz * mz) / (2 * rho)
            low = false
            for gr in 1:NG
                er = R[gbase(gr)+IERAD, i, j, k]
                budget += er * c / chat[gr, i, j, k]
                low |= !(er > ef)
            end
            (low && !(budget > 0)) || continue
            for gr in 1:NG
                b = gbase(gr)
                R[b+IERAD, i, j, k] > ef || (R[b+IERAD, i, j, k] = ef)
            end
            bad += 1
        end
        bad > 0 && Threads.atomic_add!(nbad, bad)
    end
    return nbad[]
end

function _amend_range!(R::AbstractArray{T,4}, phys::RadiationPhysics{NG,M,T},
                       floors::Floors{T}, ir, jr, kr) where {NG,M,T}
    c = phys.c
    # The floor is per group, so a spectrum split NG ways is not driven below it
    # group by group. Quokka divides its `Erad_floor` by nGroups for the same
    # reason: the floor is a property of the total, not of each bin.
    ef = floors.erad / NG
    nbad = Threads.Atomic{Int}(0)
    threaded_jk(jr, kr) do j, k
        local bad = 0
        @inbounds for i in ir
            local hit = false
            for gr in 1:NG
                b = gbase(gr)
                er, Fx, Fy, Fz, _, _, _, rep = amend_rad(R[b+IERAD, i, j, k], R[b+IFRX, i, j, k],
                                                         R[b+IFRY, i, j, k], R[b+IFRZ, i, j, k],
                                                         c, ef)
                if rep
                    R[b+IERAD, i, j, k] = er
                    R[b+IFRX, i, j, k] = Fx
                    R[b+IFRY, i, j, k] = Fy
                    R[b+IFRZ, i, j, k] = Fz
                    hit = true
                end
            end
            hit && (bad += 1)
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
function rad_cons2prim!(rprim::A, R::A, phys::RadiationPhysics{NG,M,T},
                        g::CartesianGrid{T}) where {NG,M,T,A<:AbstractArray{T,4}}
    n1, n2, n3 = g.dims
    c = phys.c
    threaded_jk(1:n2, 1:n3) do j, k
        @inbounds for i in 1:n1, gr in 1:NG
            b = gbase(gr)
            er = R[b+IERAD, i, j, k]
            # Divide rather than multiply by a reciprocal. The two differ in the
            # last bit, and this kernel is memory-bound, so the two saved
            # divisions buy nothing while the changed bits would make every
            # radiation result differ from the pre-split code for no reason.
            ce = c * er
            rprim[b+IERAD, i, j, k] = er
            rprim[b+IFRX, i, j, k] = R[b+IFRX, i, j, k] / ce
            rprim[b+IFRY, i, j, k] = R[b+IFRY, i, j, k] / ce
            rprim[b+IFRZ, i, j, k] = R[b+IFRZ, i, j, k] / ce
        end
    end
    return rprim
end

"""
    rad_fluxes!(flux, qL, qR, R, chat, phys, Val(D), grid)

Solve the HLL Riemann problem of the M1 system at every interface normal to
direction `D`.

`chat` is the per-cell, per-group reduced speed of light (see
[`update_chat!`](@ref)); the interface uses the **minimum** of the two adjacent
cells. That is the stable choice wherever it varies: an optically thick cell may
locally run at the full speed of light, but letting it set the signal speed of a
flux into a transparent neighbour would impose a CFL-violating speed there,
where no source damping is available to control the resulting amplification.

As in the hydro sweep, an inadmissible reconstructed state (non-positive energy
density, or a reduced flux at or beyond the causal limit) falls back to the
piecewise-constant state of the adjoining cells.
"""
function rad_fluxes!(flux::A, qL::A, qR::A, R::A, chat::Union{A,Nothing},
                     phys::RadiationPhysics{NG,MM,T},
                     ::Val{D}, g::CartesianGrid{T},
                     chiF::Union{A,Nothing}=nothing) where {NG,MM,T,A<:AbstractArray{T,4},D}
    e1, e2, e3 = _shift(Val(D))
    c = phys.c
    cl = phys.closure
    dl = g.dx[D]
    ir, jr, kr = face_ranges(g, Val(D))
    threaded_jk(jr, kr) do j, k
        # Groups are transported independently -- the M1 system couples them only
        # through the implicit source term -- so this is the single-group sweep
        # run NG times at an offset, and `hll_radiation` never learns about groups.
        @inbounds for i in ir, gr in 1:NG
            im, jm, km = i - e1, j - e2, k - e3
            b = gbase(gr)

            erL = qL[b+IERAD, i, j, k]
            fxL = qL[b+IFRX, i, j, k]
            fyL = qL[b+IFRY, i, j, k]
            fzL = qL[b+IFRZ, i, j, k]
            erR = qR[b+IERAD, i, j, k]
            fxR = qR[b+IFRX, i, j, k]
            fyR = qR[b+IFRY, i, j, k]
            fzR = qR[b+IFRZ, i, j, k]

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
                erL = R[b+IERAD, im, jm, km]
                FxL = R[b+IFRX, im, jm, km]
                FyL = R[b+IFRY, im, jm, km]
                FzL = R[b+IFRZ, im, jm, km]
                erR = R[b+IERAD, i, j, k]
                FxR = R[b+IFRX, i, j, k]
                FyR = R[b+IFRY, i, j, k]
                FzR = R[b+IFRZ, i, j, k]
            end

            chatf = _face_chat(chat, c, gr, im, jm, km, i, j, k)

            eps = _face_epsilon(chiF, dl, gr, im, jm, km, i, j, k)
            F = hll_radiation(erL, FxL, FyL, FzL, erR, FxR, FyR, FzR, c, chatf, Val(D), cl,
                              eps)

            flux[b+IERAD, i, j, k] = F[1]
            flux[b+IFRX, i, j, k] = F[2]
            flux[b+IFRY, i, j, k] = F[3]
            flux[b+IFRZ, i, j, k] = F[4]
        end
    end
    return nothing
end

# The reduced speed at a face: the minimum of the two cells it separates, or --
# when `chat` is `nothing` -- the true speed of light, for the time-derivative
# form of a local chat (see `flux_form`), in which the flux is the unreduced one
# and each cell scales its own update afterwards.
@inline _face_chat(chat::AbstractArray, c, gr, im, jm, km, i, j, k) =
    @inbounds min(chat[gr, im, jm, km], chat[gr, i, j, k])
@inline _face_chat(::Nothing, c, gr, im, jm, km, i, j, k) = c

# The wave-speed correction at a face: `min(1, 1/tau)` with `tau` the harmonic
# mean of the two cells' optical depths across it, as Quokka forms it; one
# without the correction.
@inline _face_epsilon(::Nothing, dl::T, gr, im, jm, km, i, j, k) where {T} = one(T)
@inline function _face_epsilon(chiF::AbstractArray{T,4}, dl::T, gr, im, jm, km, i, j,
                               k) where {T}
    @inbounds tL = chiF[gr, im, jm, km] * dl
    @inbounds tR = chiF[gr, i, j, k] * dl
    s = tL + tR
    s > 0 || return one(T)
    return min(one(T), s / (2 * tL * tR))
end

"""
    scale_by_phi!(dR, chat, phys, grid)

Multiply each interior cell's radiation right-hand side by its own reduction
`chat_g/c`, group by group: the time-derivative form of a local reduced speed of
light (see [`flux_form`](@ref)).
"""
function scale_by_phi!(dR::AbstractArray{T,4}, chat::AbstractArray{T,4},
                       phys::RadiationPhysics{NG,M,T}, g::CartesianGrid{T}) where {NG,M,T}
    c = phys.c
    ir = g.lo[1]:g.hi[1]
    threaded_jk(g.lo[2]:g.hi[2], g.lo[3]:g.hi[3]) do j, k
        @inbounds for i in ir, gr in 1:NG
            # A division, not a multiplication by 1/c: it is exactly one where
            # chat = c, so a cell at phi = 1 runs the unmodified scheme bit for bit.
            phi = chat[gr, i, j, k] / c
            b = gbase(gr)
            for n in 0:3
                dR[b+n, i, j, k] *= phi
            end
        end
    end
    return dR
end

"""
    radiation_timestep(phys, grid, cfl)

Radiation CFL condition, `dt = cfl / (chat Σ_d 1/dx_d)` over the active
directions. Independent of the state because the M1 signal speed is bounded by
`chat`.
"""
function radiation_timestep(phys::RadiationPhysics{NG,M,T}, g::CartesianGrid{T},
                            cfl::Real) where {NG,M,T}
    s = zero(T)
    for d in 1:3
        g.active[d] && (s += one(T) / g.dx[d])
    end
    return T(cfl) / (phys.chat * s)
end

# --- the variable reduced speed of light -------------------------------------
#
# The rule itself is in `rsla.jl`; these are the two array-level drivers that
# apply it. They live here because they sweep the radiation state exactly as
# `rad_fluxes!` and `radiation_timestep` above do.

"""
    _grid_scales(grid) -> (dx_min, sum_d 1/dx_d, (1/dx_1, 1/dx_2, 1/dx_3))

The grid numbers the limiter needs, over the *active* directions only: the
smallest cell width (read by the fitting form), the reciprocal-width sum the CFL
number is built from (the same one `radiation_timestep` forms), and the
per-direction reciprocal widths, zero in an inactive direction.

A free function because both callers capture the result in a threaded closure,
and a variable assigned inside a loop in the enclosing scope would be boxed --
every cell would then allocate. That is not hypothetical: written inline it cost
54 bytes a cell.
"""
@inline function _grid_scales(g::CartesianGrid{T}) where {T}
    dxmin = T(Inf)
    invdx = zero(T)
    for d in 1:3
        if g.active[d]
            dxmin = min(dxmin, g.dx[d])
            invdx += one(T) / g.dx[d]
        end
    end
    inv3 = ntuple(d -> g.active[d] ? one(T) / g.dx[d] : zero(T), 3)
    return (dxmin, invdx, inv3)
end

"""
    _signal_sum(closure, R, i, j, k, g, c, inv3) -> sum_d sigma_d / dx_d

The HLL signal speed of group `g` in this cell along each active direction, as a
fraction of `chat`, divided by the cell width and summed: the rate at which one
explicit stage amplifies a grid-scale checkerboard, per unit `chat dt`.

`sigma_d = max(0.1, sqrt(T_dd))` is exactly the speed `m1_flux` hands the Riemann
solver along `d`: `1/sqrt(3)` in every direction where the radiation is
isotropic, and -- where it free-streams -- one along the beam and the `0.1` floor
across it. Using the closure's own speeds rather than the P1 value is what keeps
[`StabilityBound`](@ref) honest in a free-streaming cell, which is exactly the
cell with no source damping to spare; using them *per direction* rather than
the largest of them keeps a beamed cell in two or three dimensions from being
bounded as if it beamed along every axis at once.
"""
@inline function _signal_sum(cl::AbstractClosure, R::AbstractArray{T,4}, i, j, k,
                             gr::Integer, c::T, inv3::NTuple{3,T}) where {T}
    b = gbase(gr)
    iso = sqrt(one(T) / 3) * (inv3[1] + inv3[2] + inv3[3])
    @inbounds er = R[b+IERAD, i, j, k]
    er > 0 || return iso
    ce = c * er
    @inbounds fx = R[b+IFRX, i, j, k] / ce
    @inbounds fy = R[b+IFRY, i, j, k] / ce
    @inbounds fz = R[b+IFRZ, i, j, k] / ce
    # `amend_rad` keeps `f < 1` in the interior, but this also runs in ghost
    # cells filled by a user boundary condition, which nothing has repaired: an
    # inadmissible state gets the fastest speed the closure can produce.
    fx * fx + fy * fy + fz * fz < 1 ||
        return inv3[1] + inv3[2] + inv3[3]
    t = eddington_tensor(cl, fx, fy, fz)
    s = zero(T)
    for d in 1:3
        s += max(T(0.1), sqrt(t[d, d])) * inv3[d]
    end
    return s
end

"""
    update_chiF!(chiF, U, eos, phys, grid, floors)

Fill the per-cell, per-group flux absorption coefficient `rho kappa_F` over the
whole array including ghost cells, for the wave-speed correction of
[`RadiationPhysics`](@ref).
"""
function update_chiF!(chiF::AbstractArray{T,4}, U::AbstractArray{T,4}, eos::AbstractEOS,
                      phys::RadiationPhysics{NG,M,T}, g::CartesianGrid{T},
                      floors::Floors{T}) where {NG,M,T}
    n1, n2, n3 = g.dims
    op = phys.opacity
    groups = phys.groups
    rhomin = floors.rho
    pmin = floors.pressure
    threaded_jk(1:n2, 1:n3) do j, k
        @inbounds for i in 1:n1
            rho, Tgas = _cell_temperature(U, eos, i, j, k, rhomin, pmin)
            kappa = group_opacities(op, groups, rho, Tgas)
            for gr in 1:NG
                kF = last(_limiter_opacities(op, groups, kappa, rho, Tgas, gr))
                chiF[gr, i, j, k] = rho * kF
            end
        end
    end
    return chiF
end

"""
    update_chat!(chat, U, R, eos, phys, grid, floors, dt)

Fill the per-cell, per-group reduced speed of light `chat[g, i, j, k]` from the
current state, over the *whole* array including ghost cells.

Ghost cells are included because the transport step needs the limiter on both
sides of every interface, including the ones bounding the domain; the limiter is
purely local, so filling them costs one extra loop and no communication. `U` and
`R` must already have had their own ghost cells filled.

A [`ConstantChat`](@ref) limiter -- the default -- dispatches to a no-op: the
array was filled with `phys.chat` when the simulation was built and cannot
change.
"""
@inline update_chat!(chat, U, R, eos, phys, g, floors, dt) =
    _fill_chat!(chat, phys.chat_limiter, U, R, eos, phys, g, floors, dt)

@inline _fill_chat!(chat::AbstractArray{T,4}, ::ConstantChat, U::AbstractArray{T,4},
                    R::AbstractArray{T,4}, eos::AbstractEOS,
                    phys::RadiationPhysics{NG,M,T}, g::CartesianGrid{T},
                    floors::Floors{T}, dt::T) where {NG,M,T} = chat

# The gas temperature of cell (i, j, k), with the same floors the coupling uses.
@inline function _cell_temperature(U::AbstractArray{T,4}, eos, i, j, k,
                                   rhomin::T, pmin::T) where {T}
    @inbounds rho = U[IRHO, i, j, k]
    rho < rhomin && (rho = rhomin)
    @inbounds mx = U[IMX, i, j, k]
    @inbounds my = U[IMY, i, j, k]
    @inbounds mz = U[IMZ, i, j, k]
    @inbounds Eint = U[IEGAS, i, j, k] - (mx * mx + my * my + mz * mz) / (2 * rho)
    emin = internal_energy(eos, rho, pmin)
    Eint < emin && (Eint = emin)
    return rho, temperature(eos, rho, Eint)
end

# The three opacities of group `gr`: grey runs distinguish the Planck, energy and
# flux means, multigroup runs use the group opacity for all three, as the
# coupling does.
@inline function _limiter_opacities(op, groups::RadiationGroups{NG}, kappa, rho, Tgas,
                                    gr) where {NG}
    NG == 1 || return (kappa[gr], kappa[gr], kappa[gr])
    return (planck_opacity(op, rho, Tgas), energy_mean_opacity(op, rho, Tgas),
            flux_mean_opacity(op, rho, Tgas))
end

function _fill_chat!(chat::AbstractArray{T,4}, lim::AbstractChatLimiter,
                     U::AbstractArray{T,4}, R::AbstractArray{T,4}, eos::AbstractEOS,
                     phys::RadiationPhysics{NG,M,T}, g::CartesianGrid{T},
                     floors::Floors{T}, dt::T) where {NG,M,T}
    n1, n2, n3 = g.dims
    c = phys.c
    phi0 = phys.chat / c
    op = phys.opacity
    cl = phys.closure
    groups = phys.groups
    arad = phys.a_rad

    # The bound is a statement about the step actually being taken, which is
    # `dt` here -- never the nominal CFL number, which a subcycled or
    # hydro-limited step may undershoot or overshoot.
    dxmin, _, inv3 = _grid_scales(g)
    cdt = c * dt
    rhomin = floors.rho
    pmin = floors.pressure

    threaded_jk(1:n2, 1:n3) do j, k
        @inbounds for i in 1:n1
            rho, Tgas = _cell_temperature(U, eos, i, j, k, rhomin, pmin)
            # 4 a T^3 / (rho c_v): the radiation-to-gas heat capacity ratio that
            # sets how much damping the coupling can supply (see StabilityBound).
            hcap = 4 * arad * Tgas^3 / heat_capacity(eos, rho, Tgas)
            kappa = group_opacities(op, groups, rho, Tgas)
            if NG == 1 || per_group(lim)
                for gr in 1:NG
                    kP, kE, kF = _limiter_opacities(op, groups, kappa, rho, Tgas, gr)
                    w = LocalCell(cdt, _signal_sum(cl, R, i, j, k, gr, c, inv3),
                                  rho * kE, rho * kF, kE > 0 ? hcap * kP / kE : zero(T),
                                  dxmin)
                    chat[gr, i, j, k] = c * chat_factor(lim, phi0, w)
                end
            else
                # The fitting form with `per_group = false`: one `phi` for the whole
                # cell, from the most restrictive group (see [`per_group`](@ref)).
                phi = one(T)
                for gr in 1:NG
                    kP, kE, kF = _limiter_opacities(op, groups, kappa, rho, Tgas, gr)
                    w = LocalCell(cdt, _signal_sum(cl, R, i, j, k, gr, c, inv3),
                                  rho * kE, rho * kF, kE > 0 ? hcap * kP / kE : zero(T),
                                  dxmin)
                    phi = min(phi, chat_factor(lim, phi0, w))
                end
                for gr in 1:NG
                    chat[gr, i, j, k] = c * phi
                end
            end
        end
    end
    return chat
end

"""
    energy_exchange_dt(U, R, eos, phys, grid, xi, floors) -> dt

The longest substep at which every optically thick cell may run at `phi = 1`,
`dt = xi / (c max_i [4 a_rad T^3 / (rho c_v) * sum_d sigma_d/dx_d])` minimised
over interior cells. Requested by `StabilityBound(energy_step = true)`.

**Not a stability condition.** The standard RSLA and the stability-bound limiter
are both stable without it, at any radiation-to-gas pressure ratio. What it buys
is accuracy where radiation carries a large share of the heat capacity: there
the limiter's energy channel caps `phi` near `phi0/(sigma C_r)` at the RSLA step,
because with `K = hrat c dt chi_E` the damping `chi_E/(2 + K)` saturates at
`1/(hrat c dt)` and cannot overcome the transport amplification. Below this
step it can, by a margin `1 - xi`, and cells thick enough -- `chi_E / sum_d(1/dx_d)`
beyond about `2 sigma/(1 - xi)` -- reach `phi = 1` and the exact equations.

With hydrodynamics disabled the gas still exchanges heat with the radiation, so
unlike a radiation-*force* constraint this one does not switch off when the gas
cannot move.
"""
function energy_exchange_dt(U::AbstractArray{T,4}, R::AbstractArray{T,4},
                            eos::AbstractEOS, phys::RadiationPhysics{NG,M,T},
                            g::CartesianGrid{T}, xi::Real,
                            floors::Floors{T}) where {NG,M,T}
    c = phys.c
    arad = phys.a_rad
    cl = phys.closure
    _, _, inv3 = _grid_scales(g)
    rhomin = floors.rho
    pmin = floors.pressure

    rate = reduce_max_jk(g.lo[2]:g.hi[2], g.lo[3]:g.hi[3], T) do j, k
        local m = zero(T)
        @inbounds for i in g.lo[1]:g.hi[1]
            rho, Tgas = _cell_temperature(U, eos, i, j, k, rhomin, pmin)
            hcap = 4 * arad * Tgas^3 / heat_capacity(eos, rho, Tgas)
            s = zero(T)
            for gr in 1:NG
                s = max(s, _signal_sum(cl, R, i, j, k, gr, c, inv3))
            end
            m = max(m, hcap * s)
        end
        return m
    end
    return rate > 0 ? T(xi) / (c * rate) : T(Inf)
end
