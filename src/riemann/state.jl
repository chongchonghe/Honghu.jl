# The interface state a Riemann solver sees, and the solver extension point.
#
# Both mirror Quokka: `HydroState` has the field names of `quokka::HydroState`
# (src/hydro/HydroState.hpp), and the solver is selected by a type the flux
# sweep dispatches on, which is the Julia form of Quokka's
# `enum class RiemannSolver` template parameter dispatched with `if constexpr`
# (src/hydro/hydro_system.hpp:57). A solver written here transliterates to a
# Quokka header, and back, close to line by line.

"""
    HydroState(rho, u, v, w, P, cs, E, by = 0, bz = 0)

One side of an interface, in *canonical* form: `u` is the velocity normal to the
interface and `v`, `w` the two transverse components, and likewise `by`, `bz`
for the magnetic field. The caller permutes the components; the solvers never
think about direction.

Field names match `quokka::HydroState`, the transverse field included: one state
type serves both the hydrodynamic and the MHD solvers, as in Quokka, so a solver
transliterates field for field. The hydrodynamic solvers ignore `by` and `bz`, and
the seven-argument constructor sets them to zero. It is an immutable struct of
concrete floats, so it lives on the stack and costs exactly what passing the
numbers separately costs -- `test_performance.jl` holds that claim to account.
The normal field is not a field of the state: it is single-valued at the face and
is passed to [`mhd_flux`](@ref) on its own, as Quokka passes `bx`.

| field | meaning |
|:---|:---|
| `rho` | mass density |
| `u` | normal velocity |
| `v`, `w` | transverse velocities |
| `P` | pressure |
| `cs` | adiabatic sound speed |
| `E` | total energy density, the magnetic energy included |
| `by`, `bz` | transverse magnetic field |
"""
struct HydroState{T<:AbstractFloat}
    rho::T
    u::T
    v::T
    w::T
    P::T
    cs::T
    E::T
    by::T
    bz::T
end

HydroState(rho::T, u::T, v::T, w::T, P::T, cs::T, E::T) where {T<:AbstractFloat} =
    HydroState(rho, u, v, w, P, cs, E, zero(T), zero(T))
HydroState(rho, u, v, w, P, cs, E) = HydroState(promote(rho, u, v, w, P, cs, E)...)
HydroState(rho, u, v, w, P, cs, E, by, bz) =
    HydroState(promote(rho, u, v, w, P, cs, E, by, bz)...)

Base.eltype(::HydroState{T}) where {T} = T

"""
    AbstractRiemannSolver

Supertype for the interface flux solvers: [`HLLC`](@ref) and [`LLF`](@ref).
Pass one to [`Simulation`](@ref) as `riemann = HLLC()`.

A concrete subtype implements

    riemann_flux(solver, sL::HydroState, sR::HydroState, gamma, du, dw) -> SVec{5}

returning the conservative flux `(mass, normal momentum, transverse momentum 1,
transverse momentum 2, energy)` in canonical form.

Every solver takes the same arguments and ignores what it does not need --
`LLF` uses neither `gamma` nor the carbuncle inputs. That is a small departure
from Quokka, whose per-solver argument lists differ, and it is deliberate: the
sweep computes `du` and `dw` unconditionally in both codes, so a uniform
signature costs nothing and keeps one call site instead of one per solver.

`test/test_extensions.jl` pins this contract with a toy solver.
"""
abstract type AbstractRiemannSolver end

"""
    AbstractMHDRiemannSolver <: AbstractRiemannSolver

Supertype of the MHD interface solvers, [`LLFMHD`](@ref) and [`HLLD`](@ref),
which a simulation with `mhd = true` takes as `riemann`. A concrete subtype
implements

    mhd_flux(solver, sL::HydroState, sR::HydroState, gamma, bx, dw) -> (SVec{5}, am, ap)

the counterpart of Quokka's `HLLD(sL, sR, gamma, bx, perp_v_jump)`. It returns the
flux of `(mass, normal momentum, transverse momentum 1, transverse momentum 2,
energy)` in canonical form, and the two non-negative bounding speeds of the
Riemann fan, `am` to the left and `ap` to the right, which constrained transport
needs at every face: for the face velocity and the averaging of the EMFs. `bx`
is the normal field at the face and `dw` the carbuncle input of
[`riemann_flux`](@ref). The flux of the transverse field is not returned: the
field is advanced by the EMFs, never by a flux.
"""
abstract type AbstractMHDRiemannSolver <: AbstractRiemannSolver end

"""
    mhd_flux(solver, sL, sR, gamma, bx, dw) -> (F::SVec{5}, am, ap)

The conservative flux across one interface in canonical form, for an
[`AbstractMHDRiemannSolver`](@ref), and the bounding speeds of the fan.
"""
function mhd_flux end

"""
    riemann_flux(solver, sL, sR, gamma, du, dw) -> SVec{5}

The conservative flux across one interface in canonical form.

`du` is the jump in normal velocity across the interface and `dw` the smallest
transverse velocity difference in the two adjoining cells. Both feed the
carbuncle correction and may be passed as zero to disable it; solvers without
one ignore them.
"""
function riemann_flux end

# Direction bookkeeping: for a sweep along `D`, `_tdir1` and `_tdir2` are the two
# transverse directions in the cyclic order used by the canonical form
# (D=1 -> y,z;  D=2 -> z,x;  D=3 -> x,y). Written out as literal methods rather
# than computed, so they constant-fold.
@inline _tdir1(::Val{1}) = 2
@inline _tdir1(::Val{2}) = 3
@inline _tdir1(::Val{3}) = 1
@inline _tdir2(::Val{1}) = 3
@inline _tdir2(::Val{2}) = 1
@inline _tdir2(::Val{3}) = 2
@inline _tshift1(::Val{1}) = (0, 1, 0)
@inline _tshift1(::Val{2}) = (0, 0, 1)
@inline _tshift1(::Val{3}) = (1, 0, 0)
@inline _tshift2(::Val{1}) = (0, 0, 1)
@inline _tshift2(::Val{2}) = (1, 0, 0)
@inline _tshift2(::Val{3}) = (0, 1, 0)
