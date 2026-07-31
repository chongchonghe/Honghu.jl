# The two in-tree coupling solvers. Both take one backward-Euler step; they
# differ only in what they call the unknowns, and that difference is the whole
# point of keeping them both.

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

# --- Quokka's replaced base variables ----------------------------------------

function solve_coupling(m::QuokkaCoupling, s::CouplingState{NG,M,T}) where {NG,M,T}
    cscale = s.c / s.chat
    # tau_g is the optical depth across chat*dt: the number of absorption times
    # the step spans. It is the scaling that makes this formulation work, and it
    # is held at its old-state value throughout, as in Quokka.
    tau = (s.dt * s.chat * s.rho) * s.kappaP
    kPoverE = _ratio(s.kappaP, s.kappaE, Val(NG))

    Egas = s.Egas
    Erad = s.Erad
    Rvec = zero(SVec{NG,T})
    Etot0 = s.Egas + cscale * (sumvec(s.Erad) + sumvec(s.src))

    n = 0
    converged = false
    while n < m.max_iterations
        n += 1
        Tgas = gas_temperature_of(s, Egas)
        fourPiBoverC = thermal_emission(s, Tgas)
        dB_dT = thermal_emission_dT(s, Tgas)

        # Given R, the group energies follow in closed form -- this is the
        # inversion that the replacement buys, and why no group energy ever
        # appears as an independent unknown.
        Erad = _erad_from_R(tau, kPoverE, fourPiBoverC, Rvec, s.work, s.Erad, s.src, Val(NG))

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

        J00 = one(T)
        J0g = cscale * one_svec(Val(NG), T)
        Jg0 = mul(kPoverE, dB_dT) / s.cv
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
    tau_max = maxabs(kdt * s.kappaP)
    tol = max(m.tolerance, 4 * eps(T) * max(tau_max, one(T)))

    Egas = s.Egas
    Erad = s.Erad + s.src

    n = 0
    converged = false
    while n < m.max_iterations
        n += 1
        Tgas = gas_temperature_of(s, Egas)
        fourPiBoverC = thermal_emission(s, Tgas)
        dB_dT = thermal_emission_dT(s, Tgas)

        # Backward Euler residuals in (Egas, Erad) directly.
        exch = kdt * (mul(s.kappaP, fourPiBoverC) - mul(s.kappaE, Erad))
        F0 = Egas - s.Egas + cscale * (sumvec(exch) + sumvec(s.work))
        Fg = Erad - s.Erad - s.src - (exch + s.work)

        scale = max(abs(Etot0), eps(T))
        if max(abs(F0), maxabs(Fg)) <= tol * scale
            converged = true
            break
        end

        dexch_dEgas = (kdt / s.cv) * mul(s.kappaP, dB_dT)
        J00 = one(T) + cscale * sumvec(dexch_dEgas)
        J0g = -cscale * kdt * s.kappaE
        Jg0 = -dexch_dEgas
        Jgg = one(T) + kdt * s.kappaE

        dEgas, dErad = solve_arrowhead(J00, J0g, Jg0, Jgg, -F0, -Fg)
        Egas = max(Egas + dEgas, s.cv * s.Tfloor)
        Erad += dErad
    end

    Egas, Erad = apply_energy_floors(s, Egas, Erad)
    return CouplingResult{NG,T}(Egas, Erad, n, converged)
end
