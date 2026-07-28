# The interface state a Riemann solver sees, and the solver extension point.
#
# Both mirror Quokka: `HydroState` has the field names of `quokka::HydroState`
# (src/hydro/HydroState.hpp), and the solver is selected by a type the flux
# sweep dispatches on, which is the Julia form of Quokka's
# `enum class RiemannSolver` template parameter dispatched with `if constexpr`
# (src/hydro/hydro_system.hpp:57). A solver written here transliterates to a
# Quokka header, and back, close to line by line.

"""
    HydroState(rho, u, v, w, P, cs, E)

One side of an interface, in *canonical* form: `u` is the velocity normal to the
interface and `v`, `w` the two transverse components. The caller permutes the
components; the solvers never think about direction.

Field names match `quokka::HydroState`. It is an immutable struct of concrete
floats, so it lives on the stack and costs exactly what passing the seven
numbers separately costs -- `test_performance.jl` holds that claim to account.

| field | meaning |
|:---|:---|
| `rho` | mass density |
| `u` | normal velocity |
| `v`, `w` | transverse velocities |
| `P` | pressure |
| `cs` | adiabatic sound speed |
| `E` | total energy density |
"""
struct HydroState{T<:AbstractFloat}
    rho::T
    u::T
    v::T
    w::T
    P::T
    cs::T
    E::T
end

HydroState(rho, u, v, w, P, cs, E) = HydroState(promote(rho, u, v, w, P, cs, E)...)

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
