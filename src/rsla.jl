# The variable reduced speed of light: a local, per-cell and per-group `chat`.
#
# The standard reduced speed of light approximation (RSLA) multiplies the
# *radiation* source terms by a global `phi0 = chat/c` while leaving the *matter*
# source terms alone, so the two sides equilibrate at the wrong temperature. The
# error is worst at high optical depth -- which is exactly where no speed
# reduction is needed, because matter coupling damps radiation modes strongly
# enough that the implicit source update stabilises a CFL-violating transport
# step. Making `phi` local removes the error where it is large and keeps the
# reduction where it pays: this is the improved RSLA (IRSLA) of Krumholz,
# Wibking & He.
#
# This file holds the rule and nothing else: `AbstractChatLimiter` and the
# pointwise `chat_factor`. The array drivers that apply it -- `update_chat!` and
# `radiation_pressure_dt` -- live in `radiation_system.jl` alongside the other
# kernels that sweep the radiation state, for the same reason `coupling.jl`
# holds the coupling physics and `source_terms.jl` the loop over cells.
#
# What reads the resulting `(NG, i, j, k)` array: `rad_fluxes!` (interface value
# = min of the two adjacent cells) and the source step (the cell's own value, per
# group). Nothing else needs to know that `chat` is not a constant.

"""
    AbstractChatLimiter

The rule that sets the local reduced speed of light `chat_i = phi_i * c`.

A limiter is a pure function of *local* cell quantities -- the cell optical
depth, the ratio of radiation to gas pressure, the nominal reduction `phi0` and
the CFL number of the step -- so it needs no communication and can be evaluated
in ghost cells like any other pointwise quantity. Implement one by adding a
method to [`chat_factor`](@ref).

Three are provided:

  - [`ConstantChat`](@ref)     -- `phi = phi0` everywhere: the standard RSLA, and
    the default;
  - [`StabilityBound`](@ref)   -- the largest `phi` that is stable at the current
    step, in closed form and with no tunable physics parameters;
  - [`OpticalDepthFit`](@ref)  -- the two-parameter fitting form, for
    experimentation.
"""
abstract type AbstractChatLimiter end

"""
    ConstantChat()

The standard RSLA: `phi = phi0` in every cell, i.e. `chat` is the global constant
given to [`RadiationPhysics`](@ref).

The default, and a genuine no-op -- the `chat` array is filled once at
construction and never touched again, and every kernel reads the same number it
read before variable `chat` existed, bit for bit.
"""
struct ConstantChat <: AbstractChatLimiter end

"""
    StabilityBound(; beta = 0.9)

The IRSLA in its parameter-free form: in each cell take the *largest* `phi` that
the step is still stable at.

Accuracy increases monotonically with `phi` up to the stability limit, so the
optimal local choice is the limit itself. Two channels bound it, both derived
from the von Neumann analysis of an upwind transport step followed by an
implicit source update (Krumholz, Wibking & He, §4.3):

  - *transport versus damping.* A grid-scale radiation mode is amplified by
    `2 nu_phi/sqrt(3) - 1` by the transport substep and damped by
    `1/(1 + nu_phi tau_cell)` by the source substep that immediately follows it,
    where `nu_phi = phi c dt / dx` is the local CFL number. Requiring the product
    not to exceed one gives `nu_phi (2/sqrt(3) - tau_cell) <= 2`, hence

        phi_tau = (2 phi0 / nu_0) / max(2/sqrt(3) - tau_cell, 0),

    which is the ordinary CFL bound `sqrt(3) phi0/nu_0` as `tau_cell -> 0` and
    unbounded -- no reduction needed at all -- for `tau_cell >= 2/sqrt(3)`.
  - *radiation force feedback.* Transport-amplified flux perturbations push on
    the gas, and the velocity they generate re-seeds the radiation field on the
    next step. The loop closes unless `nu_phi (P_rad/P_gas) <= 1/2`, hence

        phi_P = phi0 / (2 nu_0 (P_rad/P_gas)).

`beta <= 1` is a safety factor on the two bounds; `phi` is then clamped into
`[phi0, 1]`, so the limiter can never ask for *less* than the RSLA the user
requested. There are no other parameters: `phi0` and the CFL number are already
the user's choices.

Note that `phi > phi0` is normal even in transparent cells whenever the step is
below the radiation CFL limit -- `sqrt(3) phi0/nu_0` with `nu_0 < sqrt(3)` is
larger than `phi0`. That slack is real and the limiter is entitled to it.
"""
struct StabilityBound{T<:AbstractFloat} <: AbstractChatLimiter
    beta::T
end

StabilityBound(; beta=0.9) = StabilityBound(float(beta))

"""
    OpticalDepthFit(; s, p)

The IRSLA in its original fitting form,

    phi = (phi0 + (s tau_cell)^p) / (1 + (s tau_cell)^p),

which rises from `phi0` in transparent cells to one in cells of optical depth
`>> 1/s`, with `p` setting how sharply. The linear analysis of the paper
corresponds to `s = 1/3, p = 4` applied to the optical depth of the *structure*;
because `tau_cell` underestimates that by the number of cells across the
feature, smaller `s` and gentler `p` recover more accuracy in practice.

**This form carries no stability bound.** Values of `(s, p)` that raise `phi`
well above what [`StabilityBound`](@ref) permits do not necessarily crash -- they
can run to completion and be badly wrong, which is the failure mode
`StabilityBound` exists to exclude. Provided for experimentation and for
reproducing published results; prefer `StabilityBound` for anything else.
"""
struct OpticalDepthFit{T<:AbstractFloat} <: AbstractChatLimiter
    s::T
    p::T
end

OpticalDepthFit(; s, p) = OpticalDepthFit(promote(float(s), float(p))...)

"""
    chat_factor(limiter, phi0, nu0, tau_cell, prat) -> phi

The local speed of light reduction factor, in `[phi0, 1]`.

  - `phi0`     : the global nominal reduction `chat/c` the user asked for.
  - `nu0`      : the CFL number of this step at the *nominal* reduced speed,
    `phi0 c dt sum_d 1/dx_d`. The local CFL number is `(phi/phi0) nu0`.
  - `tau_cell` : optical depth across one cell, `rho kappa dx_min`, in this group.
  - `prat`     : `max(a_rad T^4, E_rad) / P_gas` in this cell.

Pointwise and allocation-free: it is called once per cell per group per stage.
"""
function chat_factor end

@inline chat_factor(::ConstantChat, phi0, nu0, tau_cell, prat) = phi0

# 2/sqrt(3): the cell optical depth beyond which the source damping wins at any
# time step, so that phi is unbounded above.
const _TAU_UNBOUNDED = 2 / sqrt(3)

@inline function chat_factor(lim::StabilityBound{T}, phi0::T, nu0::T,
                             tau_cell::T, prat::T) where {T}
    # Transport amplification versus per-stage source damping. The denominator
    # is <= 0 in cells thick enough that no reduction is needed; `Inf` rather
    # than a branch, because `min` below disposes of it.
    den = T(_TAU_UNBOUNDED) - tau_cell
    phi_tau = den > 0 ? (2 * phi0 / nu0) / den : T(Inf)
    # Radiation force feedback. Only bites where radiation pressure is
    # dynamically significant, and `prat = 0` (no radiation) must not divide.
    phi_P = prat > 0 ? phi0 / (2 * nu0 * prat) : T(Inf)
    return min(one(T), max(phi0, lim.beta * min(phi_tau, phi_P)))
end

@inline function chat_factor(lim::OpticalDepthFit{T}, phi0::T, nu0::T,
                             tau_cell::T, prat::T) where {T}
    x = (lim.s * tau_cell)^lim.p
    # `x` overflows to Inf for a large tau_cell and a stiff p; the limit is 1.
    isfinite(x) || return one(T)
    return (phi0 + x) / (1 + x)
end
