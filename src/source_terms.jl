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

Returns a [`SourceStats`](@ref).
"""
function radiation_source_step!(U::A, R::A, eos::AbstractEOS,
                                phys::RadiationPhysics{NG,M,T},
                                g::CartesianGrid{T}, dt::T, source::S, t::T,
                                floors::Floors{T};
                                update_momentum::Bool=true) where {NG,M,T,A<:AbstractArray{T,4},S}
    NG == 1 || throw(ArgumentError(
        "radiation_source_step!: this driver is the single-group coupling; a " *
        "$(NG)-group run needs the multigroup driver"))
    c = phys.c
    chat = phys.chat
    cscale = c / chat
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
