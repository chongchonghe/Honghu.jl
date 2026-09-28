# Implicit matter-radiation coupling: the "I" of the IMEX scheme.
#
# The local (zero-dimensional) system solved in every cell is
#
#     d_t E_gas = -c   rho (kappa_P 4piB/c - kappa_E E_rad) - work
#     d_t E_rad = +chat rho (kappa_P 4piB/c - kappa_E E_rad) + S + work chat/c
#     d_t F_rad = -chat rho kappa_F F_rad + O(v/c) terms
#     d_t (rho v) = rho kappa_F F_rad / c
#
# The energy exchange is stiff and is solved with a two-by-two Newton-Raphson
# iteration in the variables `(E_int, R)` where `R` is the energy transferred
# from gas to radiation during the step. Using `R` rather than `E_rad` as the
# second unknown keeps the iteration well conditioned across the whole range
# from optically thin to deep in equilibrium (He, Wibking & Krumholz 2024).
#
# The flux and momentum update is linear once the temperature is known, so it is
# done analytically. The velocity-dependent work term couples the two, and is
# handled by a short lagged outer iteration.
#
# There are two drivers, selected by dispatch on the number of groups.
#
#   NG == 1  the grey driver below, with the 2x2 Newton written out inline. It is
#            the code every existing single-group test measures, so it is left
#            exactly as it was rather than folded into the general case.
#   NG >= 2  `multigroup_source_step!`, which hands each cell to the pluggable
#            `AbstractCoupling` in `coupling.jl` and then does the same
#            analytic flux and momentum update, group by group.

"""
    SourceStats(cells, newton_iterations, max_newton, failures)

Diagnostics returned by [`radiation_source_step!`](@ref).

`newton_iterations` is the total over all cells and `max_newton` the worst any
single cell needed, both counted *per cell* rather than per outer iteration, so
`newton_iterations / cells <= max_newton` always holds. `failures` counts cells
that exhausted `max_iterations` without converging; those cells did not solve
the coupling at all.

Accumulated into [`Diagnostics`](@ref) by `advance_radiation!`.
"""
struct SourceStats
    cells::Int
    newton_iterations::Int
    max_newton::Int
    failures::Int
end

SourceStats() = SourceStats(0, 0, 0, 0)

Base.:+(a::SourceStats, b::SourceStats) = SourceStats(a.cells + b.cells,
                                                      a.newton_iterations + b.newton_iterations,
                                                      max(a.max_newton, b.max_newton),
                                                      a.failures + b.failures)

"""
    radiation_source_step!(U, R, eos, phys, grid, dt, source, t, floors;
                           update_momentum = true)

Apply the implicit matter-radiation coupling to the conserved gas state `U` and
radiation state `R` over a time `dt`, in every interior cell.

`source` is either `nothing` or a callable `source(x, y, z, t)` returning a
radiation energy source density (energy per volume per time), e.g. the
luminosity of a star spread over a cell.

`update_momentum = false` freezes the gas momentum, which is what the static
test problems (Marshak wave, matter-radiation equilibration) assume; it is set
automatically when hydrodynamics is disabled.

This method is the grey (`NG == 1`) driver; a multigroup `phys` selects
[`multigroup_source_step!`](@ref) instead, by dispatch on the group count, and
so does a grey `phys` with a dust model, which this driver does not have.

Returns a [`SourceStats`](@ref).
"""
function radiation_source_step!(U::A, R::A, chatarr::A, eos::AbstractEOS,
                                phys::RadiationPhysics{1,M,T,O,C,SC,L,NoDust},
                                g::CartesianGrid{T}, dt::T, source::S, t::T,
                                floors::Floors{T};
                                update_momentum::Bool=true) where {M,T,A<:AbstractArray{T,4},S,
                                                                   O,C,SC,L}
    c = phys.c
    arad = phys.a_rad
    op = phys.opacity
    cl = phys.closure
    tol = phys.tolerance
    maxit = phys.max_iterations
    keep_beta = phys.beta_order >= 1 && update_momentum
    maxouter = keep_beta ? phys.max_outer_iterations : 1
    lag_tol = T(1e-13)
    big = T(1e100)

    ir = g.lo[1]:g.hi[1]
    nfail = Threads.Atomic{Int}(0)
    nit = Threads.Atomic{Int}(0)
    maxn = Threads.Atomic{Int}(0)
    ncell = 0

    threaded_jk(g.lo[2]:g.hi[2], g.lo[3]:g.hi[3]) do j, k
        local fails = 0
        local iters = 0
        local worst = 0
        @inbounds for i in ir
            rho = U[IRHO, i, j, k]
            mom0 = SVec(U[IMX, i, j, k], U[IMY, i, j, k], U[IMZ, i, j, k])
            Ekin0 = dot(mom0, mom0) / (2 * rho)
            Eint0 = U[IEGAS, i, j, k] - Ekin0
            Erad0 = R[IERAD, i, j, k]
            Frad0 = SVec(R[IFRX, i, j, k], R[IFRY, i, j, k], R[IFRZ, i, j, k])

            # The reduced speed of light is a per-cell quantity under a `chat`
            # limiter; the matter source terms are never scaled by it, which is
            # what `cscale` puts back on the gas side of every exchange.
            chat = chatarr[1, i, j, k]
            cscale = c / chat

            # external radiation source, converted to an energy density added
            # over this step (with the reduced-speed-of-light rescaling)
            Src = zero(T)
            if S !== Nothing
                x, y, z = cellcenter(g, i, j, k)
                Src = T(source(x, y, z, t)) * dt / cscale
            end

            # scale for the relative convergence test
            Eref = Eint0 + cscale * (Erad0 + Src)

            work = zero(T)
            Eint = Eint0
            Erad = Erad0
            Frad = Frad0
            mom1 = mom0
            citers = 0                # Newton iterations spent on *this* cell

            for outer in 1:maxouter
                Eint = Eint0
                Erad = Erad0
                Rex = zero(T)
                Tg = temperature(eos, rho, Eint)
                kP = planck_opacity(op, rho, Tg)
                kE = energy_mean_opacity(op, rho, Tg)
                kF = flux_mean_opacity(op, rho, Tg)
                B = arad * Tg^4
                converged = false
                n = 0

                # --- Newton-Raphson on (Eint, Rex) ---------------------------
                while n < maxit
                    Tg = temperature(eos, rho, Eint)
                    B = arad * Tg^4                       # == 4piB/c
                    kP = planck_opacity(op, rho, Tg)
                    kE = energy_mean_opacity(op, rho, Tg)
                    kPE = kE > 0 ? kP / kE : one(T)
                    tau = dt * rho * kP * chat            # optical depth over chat*dt

                    if n == 0
                        kF = flux_mean_opacity(op, rho, Tg)
                        if keep_beta && outer == 1
                            work = dot(mom0, Frad0) * (2 * kE - kF) * chat / (c * c) * dt
                        end
                        Rex = (B - Erad0 / kPE) * tau + work
                        Erad = Erad0
                    elseif tau > 0
                        Erad = kPE * (B - (Rex - work) / tau)
                    end

                    F_G = Eint - Eint0 + cscale * Rex
                    F_D = Erad - Erad0 - (Rex + Src)
                    F_Dabs = abs(F_D)
                    # Optically thin limit: the coupling term cannot balance the
                    # residual no matter how many iterations we take, so accept
                    # the transport-only radiation update.
                    if tau * max(B, Eref) < tol * Eref
                        Erad = Erad0 + Src
                        F_D = zero(T)
                        F_Dabs = zero(T)
                    end
                    if abs(F_G) < tol * Eref && cscale * F_Dabs < tol * Eref
                        converged = true
                        break
                    end

                    cv = heat_capacity(eos, rho, Tg)
                    dEg_dT = kPE * 4 * arad * Tg^3
                    # Jacobian of (F_G, F_D) w.r.t. (Eint, Rex). kappa_P/kappa_E
                    # is held fixed, which changes only the convergence rate.
                    J00 = one(T)
                    J01 = cscale
                    J10 = dEg_dT / cv
                    J11 = tau > 0 ? (-kPE / tau - one(T)) : -big
                    det = J00 * J11 - J01 * J10
                    y0 = -F_G
                    y1 = -F_D
                    dEint = (J11 * y0 - J01 * y1) / det
                    dRex = (J00 * y1 - J10 * y0) / det

                    # Guard against a Newton step that overshoots past the
                    # radiation temperature; snap to radiative equilibrium
                    # instead, which is the answer the iteration is heading for.
                    Trad = sqrt(sqrt(max(Erad, zero(T)) / arad))
                    if dEint / cv > max(Tg, Trad)
                        Eint = eint_from_temperature(eos, rho, Trad)
                    else
                        Eint += dEint
                        Rex += dRex
                    end
                    n += 1
                end
                citers += n + 1
                converged || (fails += 1)

                # keep the opacities and emission consistent with the solution
                Tg = temperature(eos, rho, Eint)
                B = arad * Tg^4
                kP = planck_opacity(op, rho, Tg)
                kE = energy_mean_opacity(op, rho, Tg)
                kF = flux_mean_opacity(op, rho, Tg)

                # --- flux, momentum ------------------------------------------
                Fcoeff = chat * rho * kF * dt
                if keep_beta
                    # O(v/c) terms: thermal emission dragged along by the gas,
                    # plus the radiation pressure acting on the moving matter.
                    # Written out component by component on purpose: a closure
                    # here (an `ntuple do` block) would capture the loop
                    # variables and force Julia to box them, which turns every
                    # Newton iteration into a heap allocation.
                    ce = c * Erad
                    tedd = eddington_tensor(cl, Frad0[1] / ce, Frad0[2] / ce, Frad0[3] / ce)
                    pl = kP * B
                    kF == kE || (pl += (kF - kE) * Erad)
                    pl *= chat * dt
                    pk = chat * dt * kF * Erad
                    m1, m2, m3 = mom0[1], mom0[2], mom0[3]
                    v1 = pl * m1 + pk * (m1 * tedd[1, 1] + m2 * tedd[1, 2] + m3 * tedd[1, 3])
                    v2 = pl * m2 + pk * (m1 * tedd[2, 1] + m2 * tedd[2, 2] + m3 * tedd[2, 3])
                    v3 = pl * m3 + pk * (m1 * tedd[3, 1] + m2 * tedd[3, 2] + m3 * tedd[3, 3])
                    Frad = (Frad0 + SVec(v1, v2, v3)) / (1 + Fcoeff)
                else
                    Frad = Frad0 / (1 + Fcoeff)
                end

                if update_momentum
                    mom1 = mom0 - (Frad - Frad0) / (c * chat)
                else
                    mom1 = mom0
                end

                keep_beta || break

                # The work done by the radiation force went into the internal
                # energy above; move it to the kinetic energy instead.
                Ekin1 = dot(mom1, mom1) / (2 * rho)
                Eint -= Ekin1 - Ekin0

                work_prev = work
                work = dot(mom1, Frad) * chat / (c * c) * (2 * kE - kF) * dt
                dwork = abs(work - work_prev)
                if work == 0 || cscale * dwork < lag_tol * Eref ||
                   dwork <= T(1e-8) * abs(work)
                    break
                end
            end

            # Counted per cell, not per outer iteration, so that the reported
            # mean can never exceed the reported worst case.
            iters += citers
            worst = max(worst, citers)

            # --- store, enforcing floors while conserving total energy --------
            if Erad < floors.erad
                Eint -= (floors.erad - Erad) * cscale
                Erad = floors.erad
            end
            emin = internal_energy(eos, rho, floors.pressure)
            Eint < emin && (Eint = emin)

            U[IMX, i, j, k] = mom1[1]
            U[IMY, i, j, k] = mom1[2]
            U[IMZ, i, j, k] = mom1[3]
            U[IEGAS, i, j, k] = Eint + dot(mom1, mom1) / (2 * rho)
            R[IERAD, i, j, k] = Erad
            R[IFRX, i, j, k] = Frad[1]
            R[IFRY, i, j, k] = Frad[2]
            R[IFRZ, i, j, k] = Frad[3]
        end
        Threads.atomic_add!(nit, iters)
        fails > 0 && Threads.atomic_add!(nfail, fails)
        Threads.atomic_max!(maxn, worst)
    end

    ncell = prod(g.n)
    return SourceStats(ncell, nit[], maxn[], nfail[])
end

# --- multigroup ---------------------------------------------------------------
#
# The same operator, one frequency group at a time. Two things change:
#
#   * the energy exchange is no longer a 2x2 system but an (NG+1)x(NG+1) one --
#     and it is not solved here at all. The cell state is handed to
#     `solve_coupling`, so which implicit integrator runs is a property of the
#     physics object rather than of this loop.
#   * the gas sees the *sum* over groups. The momentum kick is the sum of the
#     per-group flux changes, while the work term stays a per-group vector,
#     because each group is damped by its own opacity.
#
# The opacity is piecewise constant across a group, so the Planck, energy and
# flux means coincide (see `opacity.jl`) and one `group_opacities` call serves
# all three. That is why `coupling_opacity` returns the same vector twice, and
# why the `(kappa_F - kappa_E) E` term the grey driver carries has no
# counterpart here.

"""
    _group_component(R, i, j, k, Val(NG), Val(C)) -> SVec{NG}

Component `C` of every group at cell `(i, j, k)`, gathered into a group vector.

A free function taking `Val(NG)` so the `ntuple` unrolls, and taking `R, i, j, k`
as arguments so that the closure captures nothing the caller assigns -- the
boxing trap CLAUDE.md records.
"""
@inline _group_component(R::AbstractArray{T,4}, i, j, k, ::Val{NG},
                         ::Val{C}) where {T,NG,C} =
    SVec{NG,T}(ntuple(gr -> @inbounds(R[gbase(gr)+C, i, j, k]), Val(NG))...)

"""
    _group_chat(chat, i, j, k, Val(NG)) -> SVec{NG}

Each group's reduced speed of light at cell `(i, j, k)`.

The same shape as [`_group_component`](@ref), and a free function for the same
reason, but the `chat` array is `(NG, i, j, k)` rather than `(4NG, i, j, k)`:
there is one speed per group, not four.
"""
@inline _group_chat(chat::AbstractArray{T,4}, i, j, k, ::Val{NG}) where {T,NG} =
    SVec{NG,T}(ntuple(gr -> @inbounds(chat[gr, i, j, k]), Val(NG))...)

"""
    _beta_force(closure, Erad, F0x, F0y, F0z, kappa, B, m1, m2, m3, c, chat*dt)
        -> (SVec{NG}, SVec{NG}, SVec{NG})

The `O(v/c)` terms of the per-group flux update: thermal emission dragged along
by the moving gas, plus the radiation pressure acting on it. The grey driver's
same expression, evaluated once per group.

Every value is an argument for the reason given above; written as a closure over
the outer-iteration variables it would box them.
"""
@inline function _beta_force(cl::AbstractClosure, Erad::SVec{NG,T}, F0x::SVec{NG,T},
                             F0y::SVec{NG,T}, F0z::SVec{NG,T}, kap::SVec{NG,T},
                             B::SVec{NG,T}, m1::T, m2::T, m3::T, c::T,
                             cdt::SVec{NG,T}) where {NG,T}
    v = ntuple(Val(NG)) do gr
        ce = c * Erad[gr]
        tedd = eddington_tensor(cl, F0x[gr] / ce, F0y[gr] / ce, F0z[gr] / ce)
        pl = cdt[gr] * kap[gr] * B[gr]
        pk = cdt[gr] * kap[gr] * Erad[gr]
        (pl * m1 + pk * (m1 * tedd[1, 1] + m2 * tedd[1, 2] + m3 * tedd[1, 3]),
         pl * m2 + pk * (m1 * tedd[2, 1] + m2 * tedd[2, 2] + m3 * tedd[2, 3]),
         pl * m3 + pk * (m1 * tedd[3, 1] + m2 * tedd[3, 2] + m3 * tedd[3, 3]))
    end
    return (SVec{NG,T}(ntuple(gr -> v[gr][1], Val(NG))...),
            SVec{NG,T}(ntuple(gr -> v[gr][2], Val(NG))...),
            SVec{NG,T}(ntuple(gr -> v[gr][3], Val(NG))...))
end

"""
    multigroup_source_step!(U, R, eos, phys, grid, dt, source, t, floors;
                            update_momentum = true)

The multigroup matter-radiation coupling: what [`radiation_source_step!`](@ref)
dispatches to when `phys` has more than one frequency group. Same arguments,
same return value, same meaning.

Each interior cell is packed into a [`CouplingState`](@ref) and handed to
`phys.coupling`, so the choice of implicit integrator is a property of the
physics object; `phys.tolerance` and `phys.max_iterations` belong to the grey
driver's own Newton loop and are not consulted here.

`source` returns a *total* radiation energy source density, which is shared out
over the groups by the Planck spectrum at the local gas temperature -- the
spectrum of a thermal emitter, and the only one a scalar can imply. At `NG == 1`
that reduces to the grey source exactly.

Returns a [`SourceStats`](@ref).
"""
function multigroup_source_step!(U::A, R::A, chatarr::A, eos::AbstractEOS,
                                 phys::RadiationPhysics{NG,M,T},
                                 g::CartesianGrid{T}, dt::T, source::S, t::T,
                                 floors::Floors{T};
                                 update_momentum::Bool=true) where {NG,M,T,A<:AbstractArray{T,4},S}
    c = phys.c
    arad = phys.a_rad
    kB = phys.k_B
    op = phys.opacity
    cl = phys.closure
    grp = phys.groups
    method = phys.coupling
    dust = phys.dust
    keep_beta = phys.beta_order >= 1 && update_momentum
    maxouter = keep_beta ? phys.max_outer_iterations : 1
    lag_tol = T(1e-13)
    # The floor bounds the total, not each bin, so it is shared out over the
    # groups exactly as `enforce_rad_floors!` shares it.
    efloor = floors.erad / NG

    ir = g.lo[1]:g.hi[1]
    nfail = Threads.Atomic{Int}(0)
    nit = Threads.Atomic{Int}(0)
    maxn = Threads.Atomic{Int}(0)

    threaded_jk(g.lo[2]:g.hi[2], g.lo[3]:g.hi[3]) do j, k
        local fails = 0
        local iters = 0
        local worst = 0
        @inbounds for i in ir
            rho = U[IRHO, i, j, k]
            mom0 = SVec(U[IMX, i, j, k], U[IMY, i, j, k], U[IMZ, i, j, k])
            Ekin0 = dot(mom0, mom0) / (2 * rho)
            Eint0 = U[IEGAS, i, j, k] - Ekin0
            Erad0 = _group_component(R, i, j, k, Val(NG), Val(IERAD))
            F0x = _group_component(R, i, j, k, Val(NG), Val(IFRX))
            F0y = _group_component(R, i, j, k, Val(NG), Val(IFRY))
            F0z = _group_component(R, i, j, k, Val(NG), Val(IFRZ))

            # The reduced speed of light is per cell *and* per group under a
            # `chat` limiter: the groups form independent hyperbolic subsystems,
            # so each carries its own optical depth and its own bound. The
            # matter side is never scaled, so wherever a per-group radiation
            # quantity lands on the gas it is divided by that group's own
            # `chat/c` -- `c * divide(x, chatv)` below, the vector form of the
            # scalar `cscale` this loop used to hoist.
            chatv = _group_chat(chatarr, i, j, k, Val(NG))
            cdt = chatv * dt

            # The heat capacity is frozen at the old temperature for the whole
            # solve, as in Quokka; that sets how fast the iteration converges,
            # not what it converges to. The *opacity* is not: the law itself
            # goes on the `CouplingState` and every solver re-evaluates it at
            # the temperature it is currently at, which is what Quokka's
            # `ComputeModelDependentKappaEAndKappaP` does inside its Newton loop
            # and what makes an exact Jacobian possible at all.
            Tfloor = temperature(eos, rho, internal_energy(eos, rho, floors.pressure))
            Tg = max(temperature(eos, rho, Eint0), Tfloor)
            cv = heat_capacity(eos, rho, Tg)
            kap = group_opacities(op, grp, rho, Tg)

            # external radiation source, converted to an energy density added
            # over this step (with the reduced-speed-of-light rescaling)
            src = zero(SVec{NG,T})
            if S !== Nothing
                x, y, z = cellcenter(g, i, j, k)
                stot = T(source(x, y, z, t)) / c
                src = stot * mul(planck_fractions(grp, Tg, kB), cdt)
            end

            # scale for the lagged-work convergence test: the total energy of
            # the cell measured on the gas side, group by group.
            Eref = Eint0 + c * sumvec(divide(Erad0 + src, chatv))

            # The first work estimate uses the old momentum and flux, as the
            # grey driver's does, so a single outer iteration already carries it.
            work = keep_beta ?
                   mul(mom0[1] * F0x + mom0[2] * F0y + mom0[3] * F0z,
                       mul(kap, cdt / (c * c))) : zero(SVec{NG,T})
            Eint = Eint0
            Erad = Erad0
            Fx = F0x
            Fy = F0y
            Fz = F0z
            mom1 = mom0
            citers = 0                # coupling iterations spent on *this* cell

            for _ in 1:maxouter
                res = solve_coupling(method,
                                     CouplingState{NG,M,T,typeof(op),typeof(dust)}(Eint0,
                                                                      Erad0, src, work, rho,
                                                                      cv, op, dt, c, chatv,
                                                                      arad, kB, grp, Tfloor,
                                                                      efloor, dust))
                Eint = res.Egas
                Erad = res.Erad
                citers += res.iterations
                res.converged || (fails += 1)

                # keep the opacities and the emission consistent with the
                # solution, at the temperature of whatever the radiation couples
                # to: the gas, or with a dust model the dust
                Tnew = max(temperature(eos, rho, Eint), Tfloor)
                Tnew = dust isa NoDust ? Tnew : res.Tdust
                kapF = group_opacities(op, grp, rho, Tnew)

                # --- flux, momentum ------------------------------------------
                denom = mul(rho * cdt, kapF) + one(T)
                if keep_beta
                    B = (arad * Tnew^4) * planck_fractions(grp, Tnew, kB)
                    vx, vy, vz = _beta_force(cl, Erad, F0x, F0y, F0z, kapF, B,
                                             mom0[1], mom0[2], mom0[3], c, cdt)
                    Fx = divide(F0x + vx, denom)
                    Fy = divide(F0y + vy, denom)
                    Fz = divide(F0z + vz, denom)
                else
                    Fx = divide(F0x, denom)
                    Fy = divide(F0y, denom)
                    Fz = divide(F0z, denom)
                end

                # the gas feels the sum of the per-group flux changes, each
                # converted back at its own group's reduced speed
                mom1 = update_momentum ?
                       mom0 - SVec(sumvec(divide(Fx - F0x, chatv)),
                                   sumvec(divide(Fy - F0y, chatv)),
                                   sumvec(divide(Fz - F0z, chatv))) / c : mom0

                keep_beta || break

                # The work done by the radiation force went into the internal
                # energy above; move it to the kinetic energy instead.
                Ekin1 = dot(mom1, mom1) / (2 * rho)
                Eint -= Ekin1 - Ekin0

                # The convergence test is on the work as the *gas* sees it, so
                # that a per-group `chat` cannot make one group's contribution
                # look larger or smaller than the energy it actually moves.
                gwork_prev = c * sumvec(divide(work, chatv))
                work = mul(mom1[1] * Fx + mom1[2] * Fy + mom1[3] * Fz,
                            mul(kapF, cdt / (c * c)))
                gwork = c * sumvec(divide(work, chatv))
                dwork = abs(gwork - gwork_prev)
                if gwork == 0 || dwork < lag_tol * Eref ||
                   dwork <= T(1e-8) * abs(gwork)
                    break
                end
            end

            # Counted per cell, not per outer iteration, so that the reported
            # mean can never exceed the reported worst case.
            iters += citers
            worst = max(worst, citers)

            # --- store --------------------------------------------------------
            # `solve_coupling` left the pair admissible, but the kinetic-energy
            # bookkeeping above runs after it, so the gas floor is re-checked.
            emin = internal_energy(eos, rho, floors.pressure)
            Eint < emin && (Eint = emin)

            U[IMX, i, j, k] = mom1[1]
            U[IMY, i, j, k] = mom1[2]
            U[IMZ, i, j, k] = mom1[3]
            U[IEGAS, i, j, k] = Eint + dot(mom1, mom1) / (2 * rho)
            for gr in 1:NG
                b = gbase(gr)
                R[b+IERAD, i, j, k] = Erad[gr]
                R[b+IFRX, i, j, k] = Fx[gr]
                R[b+IFRY, i, j, k] = Fy[gr]
                R[b+IFRZ, i, j, k] = Fz[gr]
            end
        end
        Threads.atomic_add!(nit, iters)
        fails > 0 && Threads.atomic_add!(nfail, fails)
        Threads.atomic_max!(maxn, worst)
    end

    return SourceStats(prod(g.n), nit[], maxn[], nfail[])
end

# `RadiationPhysics{1,…,NoDust}` is the more specific method above, so a grey run
# without dust never reaches this one and its numbers are exactly what they were
# before multigroup. A grey run with dust does, and is solved as one group.
radiation_source_step!(U::A, R::A, chatarr::A, eos::AbstractEOS,
                       phys::RadiationPhysics{NG,M,T},
                       g::CartesianGrid{T}, dt::T, source::S, t::T, floors::Floors{T};
                       update_momentum::Bool=true) where {NG,M,T,A<:AbstractArray{T,4},S} =
    multigroup_source_step!(U, R, chatarr, eos, phys, g, dt, source, t, floors;
                            update_momentum)
