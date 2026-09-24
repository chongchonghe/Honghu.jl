# Two-moment (M1) radiation transport, grey or multigroup: physics parameters,
# variable conversion, the HLL flux sweep, and the radiation CFL condition.

"""
    RadiationPhysics(; c, chat = c, a_rad, opacity,
                     groups = GREY, k_B = 1,
                     closure = LevermoreClosure(),
                     coupling = QuokkaCoupling(),
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
struct RadiationPhysics{NG,M,T<:AbstractFloat,O<:AbstractOpacity,C<:AbstractClosure,
                        S<:AbstractCoupling,L<:AbstractChatLimiter}
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
end

function RadiationPhysics(; c, chat=c, a_rad, opacity,
                          groups::RadiationGroups=GREY,
                          k_B=1.0,
                          closure::AbstractClosure=LevermoreClosure(),
                          coupling::AbstractCoupling=BracketedCoupling(),
                          chat_limiter::AbstractChatLimiter=ConstantChat(),
                          beta_order::Integer=1,
                          tolerance=1e-9,
                          max_iterations::Integer=100,
                          max_outer_iterations::Integer=5)
    0 <= beta_order <= 1 || throw(ArgumentError("beta_order must be 0 or 1, got $beta_order"))
    chat <= c || throw(ArgumentError("the reduced speed of light must satisfy chat <= c"))
    k_B > 0 || throw(ArgumentError("k_B must be positive, got $k_B"))
    cc, cch, ar, kb, tol =
        promote(float(c), float(chat), float(a_rad), float(k_B), float(tolerance))
    gr = _convert_groups(groups, typeof(cc))
    RadiationPhysics(cc, cch, ar, kb, gr, opacity, closure, coupling,
                     _convert_limiter(chat_limiter, typeof(cc)), Int(beta_order), tol,
                     Int(max_iterations), Int(max_outer_iterations))
end

# A limiter's parameters must be the simulation's float type, for the same
# reason `GREY`'s edges are: `chat_factor` is called with `T` arguments in the
# inner loop and a `Float64` `beta` in a `Float32` run would promote every one.
_convert_limiter(l::ConstantChat, ::Type{T}) where {T} = l
_convert_limiter(l::StabilityBound, ::Type{T}) where {T} =
    StabilityBound(T(l.beta), l.per_group)
_convert_limiter(l::OpticalDepthFit, ::Type{T}) where {T} =
    OpticalDepthFit(T(l.s), T(l.p), l.per_group)

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

Repair the radiation state of the *interior* cells in place. Used after the
explicit transport stages of the IMEX scheme, whose output must be admissible
before the implicit coupling solve runs; the ghost cells are refilled from the
boundary conditions before they are next read, so there is no point touching
them here.
"""
function amend_rad_state!(R::AbstractArray{T,4}, phys::RadiationPhysics{NG,M,T},
                          g::CartesianGrid{T}, floors::Floors{T}) where {NG,M,T}
    _amend_range!(R, phys, floors, g.lo[1]:g.hi[1], g.lo[2]:g.hi[2], g.lo[3]:g.hi[3])
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
function rad_fluxes!(flux::A, qL::A, qR::A, R::A, chat::A,
                     phys::RadiationPhysics{NG,MM,T},
                     ::Val{D}, g::CartesianGrid{T}) where {NG,MM,T,A<:AbstractArray{T,4},D}
    e1, e2, e3 = _shift(Val(D))
    c = phys.c
    cl = phys.closure
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

            chatf = min(chat[gr, im, jm, km], chat[gr, i, j, k])

            F = hll_radiation(erL, FxL, FyL, FzL, erR, FxR, FyR, FzR, c, chatf, Val(D), cl)

            flux[b+IERAD, i, j, k] = F[1]
            flux[b+IFRX, i, j, k] = F[2]
            flux[b+IFRY, i, j, k] = F[3]
            flux[b+IFRZ, i, j, k] = F[4]
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
    _grid_scales(grid) -> (dx_min, sum_d 1/dx_d)

The two grid numbers the limiter needs, over the *active* directions only: the
cell width its optical depth is measured across, and the reciprocal-width sum the
CFL number is built from (the same one `radiation_timestep` forms).

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
    return (dxmin, invdx)
end

"""
    _signal_speed(closure, R, i, j, k, g, c) -> sig

The fastest signal the M1 system carries in group `g` of this cell, as a fraction
of `chat`: `sqrt(chi(|F|/cE))`, the same wave speed estimate `m1_flux` hands the
HLL solver.

`1/sqrt(3)` where the radiation is isotropic and one where it free-streams. It is
the `sig` of [`chat_factor`](@ref), and computing it locally rather than assuming
the isotropic value is what keeps [`StabilityBound`](@ref) honest in a
free-streaming cell -- which is exactly the cell that has no source damping to
spare.
"""
@inline function _signal_speed(cl::AbstractClosure, R::AbstractArray{T,4}, i, j, k,
                               gr::Integer, c::T) where {T}
    b = gbase(gr)
    @inbounds er = R[b+IERAD, i, j, k]
    er > 0 || return one(T) / sqrt(T(3))
    ce = c * er
    @inbounds fx = R[b+IFRX, i, j, k] / ce
    @inbounds fy = R[b+IFRY, i, j, k] / ce
    @inbounds fz = R[b+IFRZ, i, j, k] / ce
    f = sqrt(fx * fx + fy * fy + fz * fz)
    # `amend_rad` keeps `f < 1` in the interior, but this also runs in ghost
    # cells filled by a user boundary condition, which nothing has repaired.
    f < 1 || return one(T)
    return sqrt(eddington_factor(cl, f))
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

    # `nu0` is the CFL number this step actually runs at -- which is what the
    # stability bound is derived for, and never larger than the nominal `C_r` the
    # paper uses.
    dxmin, invdx = _grid_scales(g)
    nu0 = phi0 * c * dt * invdx

    rhomin = floors.rho
    pmin = floors.pressure

    threaded_jk(1:n2, 1:n3) do j, k
        @inbounds for i in 1:n1
            rho = U[IRHO, i, j, k]
            rho < rhomin && (rho = rhomin)
            mx = U[IMX, i, j, k]
            my = U[IMY, i, j, k]
            mz = U[IMZ, i, j, k]
            Eint = U[IEGAS, i, j, k] - (mx * mx + my * my + mz * mz) / (2 * rho)
            emin = internal_energy(eos, rho, pmin)
            Eint < emin && (Eint = emin)
            Tgas = temperature(eos, rho, Eint)

            # P_rad/P_gas, with `max(a T^4, E_rad)` so that a cell whose gas is
            # hot but whose radiation has not caught up is still recognised as
            # radiation-pressure dominated, and vice versa. Summed over groups:
            # the force on the gas does not care which group carried it.
            Erad = zero(T)
            for gr in 1:NG
                Erad += R[gbase(gr)+IERAD, i, j, k]
            end
            prat = max(arad * Tgas^4, Erad) / pressure(eos, rho, Eint)

            kappa = group_opacities(op, groups, rho, Tgas)
            if NG == 1 || per_group(lim)
                for gr in 1:NG
                    tau_cell = rho * kappa[gr] * dxmin
                    sig = _signal_speed(cl, R, i, j, k, gr, c)
                    chat[gr, i, j, k] = c * chat_factor(lim, phi0, nu0, tau_cell, prat, sig)
                end
            else
                # One `phi` for the whole cell: see [`per_group`](@ref) for why.
                # `phi` is monotone in `tau_cell` and decreasing in `sig`, so the
                # most restrictive group is the most transparent, most strongly
                # beamed one, and the two extremes can be taken separately.
                tau_cell = T(Inf)
                sig = T(1) / sqrt(T(3))
                for gr in 1:NG
                    tau_cell = min(tau_cell, rho * kappa[gr] * dxmin)
                    sig = max(sig, _signal_speed(cl, R, i, j, k, gr, c))
                end
                ch = c * chat_factor(lim, phi0, nu0, tau_cell, prat, sig)
                for gr in 1:NG
                    chat[gr, i, j, k] = ch
                end
            end
        end
    end
    return chat
end

"""
    radiation_pressure_dt(U, R, eos, phys, grid, cfl, floors) -> dt

The time-step constraint that radiation pressure imposes on the transport step,
`dt = cfl / (c max(phi0, 2 P) sum_d 1/dx_d)` minimised over interior cells, with
`P = max(a_rad T^4, E_rad)/P_gas`.

This is the same feedback loop that [`StabilityBound`](@ref)'s `phi_P` channel
caps, in the regime where capping `phi` cannot help: at `2 P > phi0` even
`phi = phi0` -- the standard RSLA itself -- is unstable, and `phi` is already at
its floor, so only the step can give. Returns the ordinary RSLA step wherever
radiation pressure is not dynamically significant, which is almost everywhere.

Evaluated only when a limiter other than [`ConstantChat`](@ref) is in use. Not
because the RSLA is exempt from the instability -- it is not -- but because
adding a step restriction to the default path would silently change every
existing result.
"""
function radiation_pressure_dt(U::AbstractArray{T,4}, R::AbstractArray{T,4},
                               eos::AbstractEOS, phys::RadiationPhysics{NG,M,T},
                               g::CartesianGrid{T}, cfl::Real,
                               floors::Floors{T}) where {NG,M,T}
    c = phys.c
    phi0 = phys.chat / c
    arad = phys.a_rad
    invdx = last(_grid_scales(g))
    rhomin = floors.rho
    pmin = floors.pressure

    # The largest `max(phi0, 2 P)` over interior cells sets the smallest step.
    speed = reduce_max_jk(g.lo[2]:g.hi[2], g.lo[3]:g.hi[3], T) do j, k
        local m = phi0
        @inbounds for i in g.lo[1]:g.hi[1]
            rho = U[IRHO, i, j, k]
            rho < rhomin && (rho = rhomin)
            mx = U[IMX, i, j, k]
            my = U[IMY, i, j, k]
            mz = U[IMZ, i, j, k]
            Eint = U[IEGAS, i, j, k] - (mx * mx + my * my + mz * mz) / (2 * rho)
            emin = internal_energy(eos, rho, pmin)
            Eint < emin && (Eint = emin)
            Tgas = temperature(eos, rho, Eint)
            Erad = zero(T)
            for gr in 1:NG
                Erad += R[gbase(gr)+IERAD, i, j, k]
            end
            m = max(m, 2 * max(arad * Tgas^4, Erad) / pressure(eos, rho, Eint))
        end
        return m
    end
    return T(cfl) / (c * speed * invdx)
end
