# The variable reduced speed of light: a local, per-cell and per-group `chat`.
#
# The standard reduced speed of light approximation (RSLA) multiplies the
# *radiation* source terms by a global `phi0 = chat/c` while leaving the *matter*
# source terms alone, so the two sides equilibrate at the wrong temperature. The
# error is worst at high optical depth -- which is exactly where no speed
# reduction is needed, because matter coupling damps radiation modes strongly
# enough that the implicit source update stabilises a CFL-violating transport
# step. Making `phi` local removes the error where it is large and keeps the
# reduction where it pays: this is the adaptive RSLA (ARSA; IRSLA in this code)
# of He, Krumholz & Wibking.
#
# This file holds the rule and nothing else: `AbstractChatLimiter`, the local
# quantities it reads (`LocalCell`), and the pointwise `chat_factor`. The array
# drivers that apply it -- `update_chat!` and `energy_exchange_dt` -- live in
# `radiation_system.jl` alongside the other kernels that sweep the radiation
# state, for the same reason `coupling.jl` holds the coupling physics and
# `source_terms.jl` the loop over cells.
#
# What reads the resulting `(NG, i, j, k)` array, as the cell's own value per
# group: `scale_by_phi!`, which multiplies each cell's transport update by its
# `phi` (the default time-derivative form, see `flux_form`), the pre-solve floor
# `floor_rad_energy!`, and the source step. `rad_fluxes!` reads it only under the
# standard RSLA or with `flux_form = true`, and then takes the minimum of the two
# cells at each interface. Nothing else needs to know that `chat` is not a
# constant.

"""
    AbstractChatLimiter

The rule that sets the local reduced speed of light `chat_i = phi_i * c`.

A limiter is a pure function of *local* cell quantities, collected in a
[`LocalCell`](@ref), so it needs no communication and can be evaluated in ghost
cells like any other pointwise quantity. Implement one by adding a method to
[`chat_factor`](@ref).

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
    LocalCell(cdt, sdx, chiE, chiF, hrat, dxmin)

What a limiter may read about one cell and one group, all local:

  - `cdt`   : `c dt`, the true speed of light times the substep actually taken;
  - `sdx`   : `sum_d sigma_d / dx_d` over the active directions, where
    `sigma_d = max(0.1, sqrt(T_dd))` is the HLL signal speed along `d` as a
    fraction of `chat` -- exactly the speed `m1_flux` hands the Riemann solver;
  - `chiE`  : `rho kappa_E`, the absorption coefficient that damps `E_rad`;
  - `chiF`  : `rho kappa_F`, the coefficient that damps the flux;
  - `hrat`  : `4 a_rad T^3 kappa_P/kappa_E / (rho c_v)`, the ratio of the
    radiation to the gas heat capacity, weighted by how strongly the cell emits
    relative to how strongly it absorbs (see [`StabilityBound`](@ref));
  - `dxmin` : the smallest active cell width, which only the fitting form reads.
"""
struct LocalCell{T<:AbstractFloat}
    cdt::T
    sdx::T
    chiE::T
    chiF::T
    hrat::T
    dxmin::T
end

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
    StabilityBound(; beta = 0.9, energy_step = false, xi = 0.5,
                   flux_form = false)

The IRSLA in its parameter-free form: in each cell take the *largest* `phi` that
the step is still stable at.

Accuracy increases monotonically with `phi` up to the stability limit, so the
optimal local choice is the limit itself -- and for the IMEX PD-ARS update used
here, with the HLL transport flux, that limit is known exactly. A von Neumann
analysis of one substep (He, Krumholz & Wibking, Sec. 3) finds the worst mode is
always the grid-scale checkerboard, for which the transport symbol is a pure
dissipation and the coupling splits into an energy block `(T, E)` and a momentum
block `(v, F)`. Each block is a 2×2 problem with a closed-form boundary:

    phi c dt [ sum_d sigma_d/dx_d - chi_E / (2 + K) ] <= 1,
    phi c dt [ sum_d sigma_d/dx_d - chi_F / 2       ] <= 1,

with `K = hrat * c dt * chi_E = 4 a_rad T^3 kappa_P c dt / c_v`, which is the
substep divided by the gas cooling time at the *true* speed of light.

  - The first term is the transport amplification: the checkerboard is amplified
    by `2 phi c dt sum_d sigma_d/dx_d - 1` in each explicit stage.
  - `chi_E/(2 + K)` is the damping the implicit coupling supplies. While the gas
    heat capacity dominates (`K << 2`) it is half the absorption per step, and the
    first line reduces to `nu (2 sigma - tau) <= 2` in one dimension. Once the gas
    exchanges its heat with the radiation within a step (`K >> 2`) the damping
    saturates at `1/(hrat c dt)`: the radiation cannot be damped faster than the
    gas can absorb what it deposits.
  - The momentum block has the same form with `K` replaced by the ratio of
    radiation to gas *inertia*, which is smaller by `(a/c)^2` and dropped.

Both are *exact* for this scheme, not estimates: they agree with the full
five-variable amplification matrix to the precision of a bisection, for any
Mach number, and a checkerboard run in this code flips from decaying to growing
between 0.9 and 1.1 of the bound in one, two and three dimensions.

Three things follow, and each corrects an earlier form of the limiter.

  - The optical depth enters as `chi / sum_d(1/dx_d)`, i.e. `rho kappa dx/D` for a
    cube in `D` dimensions -- not `rho kappa dx_min`, which is `D` times too
    generous and is unstable in two and three dimensions.
  - The channel that bites when radiation matters energetically is the
    *heat-capacity* one above. It depends on `(gamma - 1) P_rad/P_gas`, not on
    the radiation force, and it never makes the standard RSLA itself unstable:
    at `phi = phi0` the left sides are at most the CFL number times `sigma`. So
    no step restriction is needed for stability, only a cap on `phi`.
  - `sigma_d` is the closure's own signal speed. Under M1 it reaches one where
    the radiation free-streams, and a bound written with the P1 value
    `1/sqrt(3)` is then `sqrt(3)` too generous in exactly the cell with no
    damping to spare.

`beta <= 1` is a safety factor on the bound; `phi` is then clamped into
`[phi0, 1]`, so the limiter can never ask for *less* than the RSLA the user
requested. Where both brackets are `<= 0` there is no bound at all and `phi = 1`
exactly, so optically thick interiors run the unmodified, conservative scheme.

**The cap in radiation-rich cells.** Where `sigma hrat c dt / dx >~ 1` -- which at
the RSLA step means `4 (gamma - 1) P_rad/P_gas >~ phi0/(sigma C_r)` -- the
bound cannot reach one however thick the cell, and `phi` saturates near
`phi0/(sigma C_r)`. That is a limit of the step, not of the limiter: removing it
takes a shorter step. `energy_step = true` asks for one, capping the substep at
`xi / (hrat c sum_d sigma_d/dx_d)` so that thick cells can run at `phi = 1`
(see [`energy_exchange_dt`](@ref)), but never below the CFL step of the unreduced
scheme, at which `phi = 1` is stable everywhere anyway. It is off by default
because it costs steps, by a factor of about
`4 (gamma - 1) P_rad/P_gas * sigma C_r / (xi phi0)` wherever it binds, and at
most `1/phi0`.

Each frequency group always gets its own `phi`; see [`per_group`](@ref) for why
one `phi` per cell has nothing to recommend it.
"""
struct StabilityBound{T<:AbstractFloat} <: AbstractChatLimiter
    beta::T
    energy_step::Bool
    xi::T
    flux_form::Bool
end

function StabilityBound(; beta=0.9, energy_step::Bool=false, xi=0.5, flux_form::Bool=false)
    b, x = promote(float(beta), float(xi))
    return StabilityBound(b, energy_step, x, flux_form)
end

"""
    OpticalDepthFit(; s, p, per_group = false)

The IRSLA in its original fitting form,

    phi = (phi0 + (s tau_cell)^p) / (1 + (s tau_cell)^p),   tau_cell = chi_E dx_min,

which rises from `phi0` in transparent cells to one in cells of optical depth
`>> 1/s`, with `p` setting how sharply.

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
    flux_form::Bool
end

OpticalDepthFit(; s, p, per_group::Bool=false, flux_form::Bool=false) =
    OpticalDepthFit(promote(float(s), float(p))..., per_group, flux_form)

"""
    per_group(limiter) -> Bool

Whether each frequency group is allowed its own `phi`, or the cell takes one
value for all of them, from its most restrictive group.

[`StabilityBound`](@ref) always uses one `phi` per group; only the fitting form
[`OpticalDepthFit`](@ref) keeps the switch, to reproduce earlier results. The groups are
independent in transport; in the coupling they are not, and summing the source
terms gives

    d/dt [e + sum_g E_g] = c rho sum_g (phi_g - 1) (kappa_P B_g - kappa_E E_g),

so with `phi` uniform over groups *and cells* the RSLA obeys the identity
`Delta [e + sum_g E_g] = (1 - phi) Delta e`, and with a per-group `phi` it does
not (measured in one cell, no transport, 400 coupling steps: a drift of -44.5
per cent of the cell's energy at `phi = (1, 0.1)` where the identity predicts
none). That identity, however, is a property of a *global* reduction and is lost
by any local limiter, per-cell or per-group. What a local `phi` keeps, in the
time-derivative form of [`flux_form`](@ref), is exact conservation of
`e + sum_g E_g/phi_g`: the coupling conserves it cell by cell, and the flux
differences telescope in `E_g/phi_g`. That holds with a per-group `phi` exactly
as with a per-cell one. The von Neumann analysis finds both stable for the
coupled multigroup system, and a per-group `phi` is the more accurate of the
two, since a transparent group no longer holds the thick ones back (on the Vaytet
Marshak wave, 21.2, 7.8 and 2.7 per cent against 26.0, 11.5 and 4.3 per cent at
`phi0 = 0.003, 0.01, 0.03`), at the same cost. One `phi` per cell therefore has
no advantage, and the stability bound does not offer it.
"""
per_group(::AbstractChatLimiter) = false
per_group(::StabilityBound) = true
per_group(l::OpticalDepthFit) = l.per_group

"""
    flux_form(limiter) -> Bool

Where a local `phi` enters the transport step. `false`, the default, puts it
where the RSLA puts `chat`: on the time derivative,

    (1/phi_i) dU_i/dt = -(G_{i+1/2} - G_{i-1/2})/dx + S_i,

so each cell's update is `phi_i` times the flux difference of the *unreduced*
HLL flux (wave speeds `+/- sigma c`). `true` puts it inside the divergence,
`dU_i/dt = -(phi G)_{i+1/2} + (phi G)_{i-1/2} + ...`, with the interface value
the minimum of the two cells -- the form of the earlier drafts, kept for
comparison.

The two agree wherever `phi` is uniform, and for [`ConstantChat`](@ref) they are
the same scheme: `F_HLL(chat) = (chat/c) F_HLL(c)` exactly, term by term. They
differ where `phi` varies, and there only the default is right. The RSLA
changes how fast the radiation relaxes, not where it relaxes to: its steady
states are those of the unmodified equations. With `phi` on the time derivative
that stays true cell by cell, for any `phi(x)`. With `phi` inside the divergence
the steady state instead solves `div F + F . grad(ln phi) = S`, and the spurious
term costs real accuracy: on the Mach 5 supercritical shock a 5 per cent ripple
in an otherwise uniform `phi` takes the error from 0.19 to 2.3 per cent, worse
than the RSLA it replaces. The default also keeps a stateable conservation law:
the flux differences telescope in `E_g/phi_g`, which is exactly the radiation
energy the coupling conserves together with the gas (`e + sum_g E_g/phi_g`).

The minimum at interfaces is not needed by the default form: each cell's update
is scaled by its own `phi`, so the stability of a transparent cell is set by its
own speed whatever its neighbours run at.
"""
flux_form(::AbstractChatLimiter) = false
flux_form(l::StabilityBound) = l.flux_form
flux_form(l::OpticalDepthFit) = l.flux_form

"""
    chat_factor(limiter, phi0, cell::LocalCell) -> phi

The local speed of light reduction factor, in `[phi0, 1]`, for one cell and one
group. `phi0` is the global nominal reduction `chat/c` the user asked for.

Pointwise and allocation-free: it is called once per cell per group per substep.
"""
function chat_factor end

@inline chat_factor(::ConstantChat, phi0, cell::LocalCell) = phi0

"""
    stability_denominator(cell) -> D

`D = max(sdx - chiE/(2 + K), sdx - chiF/2)` with `K = hrat cdt chiE`, so that
the bound of [`StabilityBound`](@ref) is `phi cdt D <= 1`. `D <= 0` means no
bound.
"""
@inline function stability_denominator(w::LocalCell{T}) where {T}
    K = w.hrat * w.cdt * w.chiE
    # `K` overflows to Inf in a cell whose gas has no heat capacity to speak of;
    # the limit of chiE/(2 + K) is then zero, which is what IEEE gives anyway.
    dE = w.sdx - w.chiE / (2 + K)
    dF = w.sdx - w.chiF / 2
    return max(dE, dF)
end

@inline function chat_factor(lim::StabilityBound{T}, phi0::T, w::LocalCell{T}) where {T}
    D = stability_denominator(w)
    # No bound at all: the cell is thick enough that the implicit coupling damps
    # the checkerboard at any step. `beta` multiplies a bound, and there is none.
    D > 0 || return one(T)
    return min(one(T), max(phi0, lim.beta / (w.cdt * D)))
end

@inline function chat_factor(lim::OpticalDepthFit{T}, phi0::T, w::LocalCell{T}) where {T}
    x = (lim.s * w.chiE * w.dxmin)^lim.p
    # `x` overflows to Inf for a large tau_cell and a stiff p; the limit is 1.
    isfinite(x) || return one(T)
    return (phi0 + x) / (1 + x)
end
