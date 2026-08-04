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
    `2 sig nu_phi - 1` by the transport substep and damped by
    `1/(1 + nu_phi tau_cell)` by the source substep that immediately follows it,
    where `nu_phi = phi c dt / dx` is the local CFL number and `sig` is the
    largest signal speed the closure produces, as a fraction of `c`. Requiring
    the product not to exceed one gives `nu_phi (2 sig - tau_cell) <= 2`, hence

        phi_tau = (2 phi0 / nu_0) / max(2 sig - tau_cell, 0),

    which is the ordinary CFL bound `phi0/(sig nu_0)` as `tau_cell -> 0` and
    unbounded -- no reduction needed at all -- for `tau_cell >= 2 sig`.

    **`sig` is not `1/sqrt(3)`.** Krumholz, Wibking & He write this bound with
    `2/sqrt(3)` in place of `2 sig`, which is the P1 signal speed `c/sqrt(3)` --
    correct under the Eddington closure, and correct under M1 wherever the
    radiation is isotropic. It is *not* correct where M1 free-streams: there the
    Eddington factor goes to one, the HLL wave speeds go to `+/- chat` rather
    than `+/- chat/sqrt(3)`, and the bound is too generous by exactly
    `sqrt(3)`. That is the worst place to be wrong, because a free-streaming cell
    is an optically thin one, with no source damping to fall back on.

    Measured here, with the RSLA off entirely (`chat = c`) on the Vaytet Marshak
    wave, whose top group free-streams: `nu = 1.2` is stable and accurate for
    PCM, PLM and PPM alike; `nu = 1.5` corrupts all three (a 129 to 140 per cent
    error in the gas temperature, and a slab hotter than the wall heating it);
    `nu = sqrt(3)`, which the published bound permits, overflows to `Inf` for PLM
    and PPM. So `sig` is evaluated from the local reduced flux, through the same
    `eddington_factor` the Riemann solver uses. In an optically thick cell it is
    `1/sqrt(3)` and the published bound is recovered exactly.
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

`per_group` is discussed under [`per_group`](@ref); the short version is that the
default, `false`, is the one that conserves energy.
"""
struct StabilityBound{T<:AbstractFloat} <: AbstractChatLimiter
    beta::T
    per_group::Bool
end

StabilityBound(; beta=0.9, per_group::Bool=false) = StabilityBound(float(beta), per_group)

"""
    OpticalDepthFit(; s, p, per_group = false)

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
    per_group::Bool
end

OpticalDepthFit(; s, p, per_group::Bool=false) =
    OpticalDepthFit(promote(float(s), float(p))..., per_group)

"""
    per_group(limiter) -> Bool

Whether each frequency group is allowed its own `phi`, or the cell takes one
value for all of them (the default).

Krumholz, Wibking & He set `phi` per group, on the grounds that "the groups form
independent hyperbolic subsystems coupled only through the common matter state".
They are independent in *transport*. They are not independent in the **coupling**,
and there a per-group `phi` costs a property the RSLA otherwise keeps. From

    d/dt [e + sum_g E_g] = c rho sum_g (phi_g - 1) (kappa_P B_g - kappa_E E_g)

with `phi` uniform the sum collapses onto the gas equation and leaves the exact
identity

    Delta [e + sum_g E_g] = (1 - phi) Delta e.

The RSLA does not conserve energy -- nobody claims it does -- but its error is
*determined by how much energy the gas gained or lost*, and by nothing else. It
does not depend on the spectrum, on how many absorption-emission cycles the run
performed, or on which groups they went through. Measured here, in a single cell
with no transport, over 400 coupling steps: the identity holds to five decimals
at `phi = 0.1` and at `phi = 0.01`.

With `phi` varying between groups the sum does not collapse and the identity
fails outright. At `phi = (1, 0.1)` it predicts no drift at all -- the gas
equation is unmodified, `phi_1 = 1` -- and the actual drift is -44.5 per cent of
the cell's initial energy; at `(1, 0.01)`, -56.7 per cent. Swapping which group
is the fast one changes the answer again. What is left is an error set by the
path energy took through the spectrum, which is not a quantity anyone can
account for.

Whether it *hurts* is a separate question, and on the Vaytet Marshak wave it does
not: `per_group = true` scores slightly better there than the per-cell default at
two of three reduction factors. This is a latent hazard rather than an observed
failure, and the default is `false` because a conservation property one can state
is worth more than a few per cent of L1 error. `phi` is then taken from the most
restrictive group -- the largest value every group is simultaneously stable at.
Pass `per_group = true` to reproduce the published prescription.
"""
per_group(::AbstractChatLimiter) = false
per_group(l::StabilityBound) = l.per_group
per_group(l::OpticalDepthFit) = l.per_group

"""
    chat_factor(limiter, phi0, nu0, tau_cell, prat, sig) -> phi

The local speed of light reduction factor, in `[phi0, 1]`.

  - `phi0`     : the global nominal reduction `chat/c` the user asked for.
  - `nu0`      : the CFL number of this step at the *nominal* reduced speed,
    `phi0 c dt sum_d 1/dx_d`. The local CFL number is `(phi/phi0) nu0`.
  - `tau_cell` : optical depth across one cell, `rho kappa dx_min`, in this group.
  - `prat`     : `max(a_rad T^4, E_rad) / P_gas` in this cell.
  - `sig`      : the closure's signal speed as a fraction of `c`,
    `sqrt(eddington_factor(|F|/cE))` -- `1/sqrt(3)` where the radiation is
    isotropic, rising to one where it free-streams.

Pointwise and allocation-free: it is called once per cell per group per stage.
"""
function chat_factor end

@inline chat_factor(::ConstantChat, phi0, nu0, tau_cell, prat, sig) = phi0

@inline function chat_factor(lim::StabilityBound{T}, phi0::T, nu0::T,
                             tau_cell::T, prat::T, sig::T) where {T}
    # Transport amplification versus per-stage source damping. The denominator
    # is <= 0 in cells thick enough that no reduction is needed; `Inf` rather
    # than a branch, because `min` below disposes of it.
    den = 2 * sig - tau_cell
    phi_tau = den > 0 ? (2 * phi0 / nu0) / den : T(Inf)
    # Radiation force feedback. Only bites where radiation pressure is
    # dynamically significant, and `prat = 0` (no radiation) must not divide.
    phi_P = prat > 0 ? phi0 / (2 * nu0 * prat) : T(Inf)
    return min(one(T), max(phi0, lim.beta * min(phi_tau, phi_P)))
end

@inline function chat_factor(lim::OpticalDepthFit{T}, phi0::T, nu0::T,
                             tau_cell::T, prat::T, sig::T) where {T}
    x = (lim.s * tau_cell)^lim.p
    # `x` overflows to Inf for a large tau_cell and a stiff p; the limit is 1.
    isfinite(x) || return one(T)
    return (phi0 + x) / (1 + x)
end
