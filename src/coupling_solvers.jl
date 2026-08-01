# The two in-tree coupling solvers. Both take one backward-Euler step of the ODE
# defined by `coupling_rhs`; they differ in what they call the second unknown,
# and that difference is the whole point of keeping them both.
#
# `DirectCoupling` is generated: its residual is `u - u0 - dt f(u)` and its
# Jacobian `I - dt J`, both straight from `coupling_rhs` / `coupling_jacobian`,
# so it has no physics of its own to keep in step.
#
# `QuokkaCoupling` is a separate formulation, kept separate on purpose. It takes
# the *integrated transfer* `R_g` as its unknown, which does not exist until a
# step length has been chosen, so its Jacobian cannot be generated from a
# Jacobian of the right-hand side; see the derivation in the middle of its Newton
# loop. It shares the right-hand side (its opening guess is one explicit step of
# it) and nothing else.
#
# `QuokkaCoupling` also carries the two globalisations that are properties of the
# *iteration* and not of the equations -- the optically-thin shortcut and the
# equilibrium-overshoot guard -- and both solvers finish with
# `apply_energy_floors`. The note at the top of `coupling.jl` says why those
# three live here rather than in the right-hand side.

"""
    QuokkaCoupling(; tolerance = 1e-9, max_iterations = 100)

Backward Euler in Quokka's *replaced* base variables `(E_gas, R_g)`, where `R_g`
is the energy handed from gas to radiation group `g` over the step. The default.

Why not the obvious variables: with `(E_gas, E_g)` the residual for a group is a
difference of two large, nearly-cancelling terms once `tau = chat rho kappa dt`
is big, and its Jacobian entry scales like `tau`, so the linear solve is
ill-conditioned exactly where the physics is stiffest. Writing the unknown as
the *transfer* `R_g` moves `tau` from the residual into the definition, and the
condition number stops tracking it. That is what lets one step cross a coupling
time a hundred million times shorter than `dt`.

The Jacobian is an arrowhead -- diagonal in the groups, plus one dense row and
one dense column for the gas -- so it inverts in `O(NG)` by elimination rather
than `O(NG^3)`, with no matrix ever formed. See [`solve_arrowhead`](@ref).

It is also hand-derived rather than taken from [`coupling_jacobian`](@ref), and
has to be: `R_g` is not a state variable, so there is no `dR_g/dt` to
differentiate. The derivation is written out in the Newton loop.
"""
struct QuokkaCoupling{T<:AbstractFloat} <: AbstractCoupling
    tolerance::T
    max_iterations::Int
end

QuokkaCoupling(; tolerance=1e-9, max_iterations::Integer=100) =
    QuokkaCoupling(float(tolerance), Int(max_iterations))

"""
    DirectCoupling(; tolerance = 1e-9, max_iterations = 100)

Backward Euler in the natural variables `(E_gas, E_g)`, with the same arrowhead
Jacobian structure and the same Newton iteration.

Nothing about it is hand-derived: the residual is `u - u0 - dt f(u)` with `f` =
[`coupling_rhs`](@ref), and the Newton matrix is `I - dt * J` with `J` =
[`coupling_jacobian`](@ref). The arrowhead survives because `J`'s group block is
diagonal.

This is the textbook formulation, and it exists here to be compared against
[`QuokkaCoupling`](@ref): the two solve the same equations to the same
tolerance, so they must agree wherever both converge, and where the direct form
struggles is a measurement of what the base-variable replacement buys.
"""
struct DirectCoupling{T<:AbstractFloat} <: AbstractCoupling
    tolerance::T
    max_iterations::Int
end

DirectCoupling(; tolerance=1e-9, max_iterations::Integer=100) =
    DirectCoupling(float(tolerance), Int(max_iterations))

"""
    solve_arrowhead(J00, J0g, Jg0, Jgg, F0, Fg) -> (dx0, dxg)

Solve the bordered ("arrowhead") linear system

    [ J00   J0g^T ] [dx0]   [F0]
    [ Jg0   diag  ] [dxg] = [Fg]

for the Newton update, by eliminating the group block into the gas row. Exact,
`O(NG)`, and forms no matrix -- which matters because this runs per cell, per
Newton step, per timestep.

A group with `Jgg[g] == 0` is decoupled from the gas (zero optical depth), and
contributes nothing to the elimination; its update is zero.
"""
@inline function solve_arrowhead(J00::T, J0g::SVec{NG,T}, Jg0::SVec{NG,T},
                                 Jgg::SVec{NG,T}, F0::T, Fg::SVec{NG,T}) where {NG,T}
    # dxg = (Fg - Jg0 dx0) / Jgg, substituted into J00 dx0 + sum(J0g dxg) = F0
    inv_d = SVec{NG,T}(ntuple(g -> iszero(Jgg[g]) ? zero(T) : one(T) / Jgg[g], Val(NG))...)
    a = J00 - sumvec(mul(J0g, mul(Jg0, inv_d)))
    b = F0 - sumvec(mul(J0g, mul(Fg, inv_d)))
    dx0 = iszero(a) ? zero(T) : b / a
    dxg = mul(Fg - dx0 * Jg0, inv_d)
    return (dx0, dxg)
end

# The group kernels below take every value they need as an *argument*. Written
# as closures inside the Newton loop they would capture `Rvec` and `Erad`, which
# the loop assigns, and Julia would box them: one heap allocation per group per
# iteration per cell. That is the trap CLAUDE.md records from `source_terms.jl`,
# and it costs 24 kB per cell here when it bites. Passing arguments keeps the
# unrolled `ntuple` free of captured state.

# Group energies implied by a transfer vector R -- the closed-form inversion the
# replaced base variables buy.
@inline _erad_from_R(tau::SVec{NG,T}, kPoverE, B, Rvec, work, Erad0, src,
                     ::Val{NG}) where {NG,T} =
    SVec{NG,T}(ntuple(g -> tau[g] <= 0 ? Erad0[g] + src[g] + Rvec[g] :
                           kPoverE[g] * (B[g] - (Rvec[g] - work[g]) / tau[g]), Val(NG))...)

# 1 where the group is too thin to balance its residual, 0 otherwise.
@inline _thin_mask(tau::SVec{NG,T}, B, scale, tol, ::Val{NG}) where {NG,T} =
    SVec{NG,T}(ntuple(g -> tau[g] * max(B[g], scale) < tol * scale ? one(T) : zero(T),
                      Val(NG))...)

@inline _blend(mask::SVec{NG,T}, thinval, thickval, ::Val{NG}) where {NG,T} =
    SVec{NG,T}(ntuple(g -> mask[g] > 0 ? thinval[g] : thickval[g], Val(NG))...)

@inline _residual_g(mask::SVec{NG,T}, Erad, Erad0, Rvec, src, ::Val{NG}) where {NG,T} =
    SVec{NG,T}(ntuple(g -> mask[g] > 0 ? zero(T) :
                           (Erad[g] - Erad0[g]) - (Rvec[g] + src[g]), Val(NG))...)

@inline _jgg_quokka(tau::SVec{NG,T}, kPoverE, big, ::Val{NG}) where {NG,T} =
    SVec{NG,T}(ntuple(g -> tau[g] <= 0 ? big : -kPoverE[g] / tau[g] - one(T), Val(NG))...)

@inline one_svec(::Val{NG}, ::Type{T}) where {NG,T} =
    SVec{NG,T}(ntuple(_ -> one(T), Val(NG))...)

@inline _ratio(a::SVec{NG,T}, b::SVec{NG,T}, ::Val{NG}) where {NG,T} =
    SVec{NG,T}(ntuple(g -> iszero(b[g]) ? one(T) : a[g] / b[g], Val(NG))...)

# The opacity's own temperature dependence, as it enters QuokkaCoupling's
# `dE_g/dE_gas`: `(dkappaP B - dkappaE E) / kappaE`, and zero for a group with no
# energy-mean opacity, which cannot absorb and so has no such term.
@inline _dkappa_term(dkP::SVec{NG,T}, dkE, kE, B, Erad, ::Val{NG}) where {NG,T} =
    SVec{NG,T}(ntuple(g -> iszero(kE[g]) ? zero(T) :
                           (dkP[g] * B[g] - dkE[g] * Erad[g]) / kE[g], Val(NG))...)

# --- Quokka's replaced base variables ----------------------------------------

function solve_coupling(m::QuokkaCoupling, s::CouplingState{NG,M,T}) where {NG,M,T}
    cscale = s.c / s.chat
    kdt = s.dt * s.chat * s.rho

    Egas = s.Egas
    Erad = s.Erad
    Etot0 = s.Egas + cscale * (sumvec(s.Erad) + sumvec(s.src))

    # Open from the *linearised* transfer at the old state, not from R = 0.
    # `R_g = dt rate_g + work_g` evaluated at the old state is exactly the shared
    # right-hand side taken explicitly -- one forward Euler transfer -- so this
    # opening guess is the physics of `coupling_rhs` and not a second copy of it.
    #
    # Starting at zero instead makes the first implied group energy
    # kappaPoverE (B + work/tau), which goes negative whenever the lagged v.F
    # work term is negative and outweighs the cell's thermal emission. The
    # equilibrium guard below then snaps the gas to its floor and the iteration
    # never recovers, emptying the cell. The grey driver in `source_terms.jl`
    # has always opened this way; the multigroup path had not, and every
    # negative-velocity cell of a beta_order = 1 run drained to the floor.
    Rvec = s.dt * last(coupling_rhs(s, Egas, s.Erad)) + s.work

    n = 0
    converged = false
    while n < m.max_iterations
        n += 1
        Tgas = gas_temperature_of(s, Egas)
        fourPiBoverC = thermal_emission(s, Tgas)
        dB_dT = thermal_emission_dT(s, Tgas)
        kappaP, kappaE = coupling_opacity(s, Tgas)
        dkappaP, dkappaE = coupling_opacity_dT(s, Tgas)

        # tau_g is the optical depth across chat*dt: the number of absorption
        # times the step spans, and the scaling that makes this formulation
        # work. Recomputed here rather than frozen at the old temperature,
        # because the opacity is a function of the iterate -- Quokka's
        # `tau = dt * rho * kappaP * chat` sits inside its loop for the same
        # reason.
        tau = kdt * kappaP
        kPoverE = _ratio(kappaP, kappaE, Val(NG))

        # Given R, the group energies follow in closed form -- this is the
        # inversion that the replacement buys, and why no group energy ever
        # appears as an independent unknown. The opening pass keeps the old
        # energies, which is what makes the linearised R above self-consistent.
        Erad = n == 1 ? s.Erad :
               _erad_from_R(tau, kPoverE, fourPiBoverC, Rvec, s.work, s.Erad, s.src, Val(NG))

        scale = max(abs(Etot0), eps(T))

        # Optically thin groups. When tau is negligible the coupling term cannot
        # balance the group's residual however many iterations we spend, so the
        # group takes the transport-only answer and drops out of the test.
        # Without this the iteration burns all `max_iterations` in every
        # transparent cell -- and a multigroup run is *mostly* transparent
        # groups, since the Wien tail of any spectrum is optically thin.
        thin = _thin_mask(tau, fourPiBoverC, scale, m.tolerance, Val(NG))
        Erad = _blend(thin, s.Erad + s.src, Erad, Val(NG))

        F0 = Egas - s.Egas + cscale * sumvec(Rvec)
        Fg = _residual_g(thin, Erad, s.Erad, Rvec, s.src, Val(NG))

        # Convergence on the residual, scaled by the total energy in the cell so
        # the test means the same thing in code units and in CGS.
        if max(abs(F0), maxabs(Fg)) <= m.tolerance * scale
            converged = true
            break
        end

        # --- the hand-derived Jacobian, and why it has to be hand-derived -----
        #
        # `R_g` is not a state variable, so there is no `dR_g/dt` and
        # `coupling_jacobian` cannot produce these entries. `R_g` is the
        # *integrated* transfer over the step and carries `dt` inside `tau`;
        # this is a different discretisation of the same ODE, not a change of
        # variables applied to `DirectCoupling`'s. (A change of variables would
        # give an invertible map between unknowns at fixed `dt`; here the second
        # unknown does not exist until a step length is chosen.)
        #
        # Chain rule through the closed-form `Erad(R)`:
        #
        #   dF_g/dR_g = (dF_g/dE_g)(dE_g/dR_g) + explicit
        #             = 1 * (-kappaPoverE/tau)  + (-1)
        #   dF_g/dE_gas = dE_g/dE_gas
        #             = [dkappaP_g B_g + kappaP_g dB_g/dT - dkappaE_g E_g]
        #               / (kappaE_g cv)
        #
        # The second is the shared `rad_gas` block divided by `chat rho kappaE`,
        # which is what differentiating `R_g = dt rate_g + work_g` at fixed `R`
        # gives; written out here because the division is singular at
        # `kappaE = 0` and the branch belongs with the rest of this formulation.
        J00 = one(T)
        J0g = cscale * one_svec(Val(NG), T)
        Jg0 = (mul(kPoverE, dB_dT) +
               _dkappa_term(dkappaP, dkappaE, kappaE, fourPiBoverC, Erad,
                            Val(NG))) / s.cv
        # tau <= 0 means the group cannot exchange: a large finite number rather
        # than -Inf, because Inf arithmetic in the elimination makes NaN.
        Jgg = _jgg_quokka(tau, kPoverE, -one(T) / eps(T), Val(NG))

        dEgas, dR = solve_arrowhead(J00, J0g, Jg0, Jgg, -F0, -Fg)

        # Guard against a Newton step that overshoots past the radiation
        # temperature: snap to radiative equilibrium, which is where the
        # iteration is heading anyway. The total radiation energy sets that
        # temperature, since the Planck fractions sum to one.
        Trad = sqrt(sqrt(max(sumvec(Erad), zero(T)) / s.a_rad))
        if dEgas / s.cv > max(Tgas, Trad)
            Egas = s.cv * Trad
        else
            Egas += dEgas
            Rvec += dR
        end

        # Keep the gas on the admissible side during the iteration; a Newton
        # step that overshoots to negative energy would evaluate T^4 on garbage.
        Egas = max(Egas, s.cv * s.Tfloor)
    end

    Egas, Erad = apply_energy_floors(s, Egas, Erad)
    return CouplingResult{NG,T}(Egas, Erad, n, converged)
end

# --- the natural variables ----------------------------------------------------

function solve_coupling(m::DirectCoupling, s::CouplingState{NG,M,T}) where {NG,M,T}
    cscale = s.c / s.chat
    kdt = (s.dt * s.chat * s.rho)
    Etot0 = s.Egas + cscale * (sumvec(s.Erad) + sumvec(s.src))

    # The residual of this formulation cannot be *evaluated* below a floor set by
    # the optical depth. Its group term is tau (B - E), and at the solution B and
    # E agree to round-off, so cancellation gives B - E an absolute error of
    # eps*B and the residual an error of eps*tau*B. Asking for 1e-9 at tau = 1e9
    # asks for a number smaller than the noise in computing it, and the iteration
    # then runs to `max_iterations` while sitting on the right answer.
    #
    # So the achievable tolerance degrades in proportion to tau. That is not a
    # deficiency of this implementation; it is the conditioning that Quokka's
    # replaced base variables are designed to avoid, and stating it here is what
    # makes the comparison against `QuokkaCoupling` an honest one.
    tau_max = maxabs(kdt * first(coupling_opacity(s, gas_temperature_of(s, s.Egas))))
    tol = max(m.tolerance, 4 * eps(T) * max(tau_max, one(T)))

    Egas = s.Egas
    Erad = s.Erad + s.src

    n = 0
    converged = false
    while n < m.max_iterations
        n += 1

        # Backward Euler, written out as `u - u0 - dt f(u)` and nothing else: the
        # constant increments `src` and `work` are the whole-step contributions
        # of terms the right-hand side carries as `(src + work)/dt`, so they
        # integrate to themselves and appear undivided here.
        fgas, frad = coupling_rhs(s, Egas, Erad)
        F0 = (Egas - s.Egas) - s.dt * fgas + cscale * sumvec(s.work)
        Fg = (Erad - s.Erad) - s.dt * frad - s.src - s.work

        scale = max(abs(Etot0), eps(T))
        if max(abs(F0), maxabs(Fg)) <= tol * scale
            converged = true
            break
        end

        # ... and its Jacobian is `I - dt J` with nothing hand-derived, which is
        # the point: this solver has no formulation of its own to keep in step
        # with the physics. The arrowhead survives because `J`'s group block is
        # diagonal (see `CouplingJacobian`).
        J = coupling_jacobian(s, Egas, Erad)
        J00 = one(T) - s.dt * J.gas_gas
        J0g = -s.dt * J.gas_rad
        Jg0 = -s.dt * J.rad_gas
        Jgg = one(T) - s.dt * J.rad_rad

        dEgas, dErad = solve_arrowhead(J00, J0g, Jg0, Jgg, -F0, -Fg)
        Egas = max(Egas + dEgas, s.cv * s.Tfloor)
        Erad += dErad
    end

    Egas, Erad = apply_energy_floors(s, Egas, Erad)
    return CouplingResult{NG,T}(Egas, Erad, n, converged)
end
