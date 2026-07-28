"""
    Hydro3D.Problems

Ready-made test problems. Each function returns a fully initialised
[`Simulation`](@ref) together with a suggested `tmax`, so that a complete run is

```julia
sim, tmax = Hydro3D.Problems.sod_shock_tube(nx = 512)
run!(sim; tmax)
```

The hydrodynamic problems are the standard verification suite (Sod, smooth
advection, Kelvin-Helmholtz, Sedov); the radiation problems exercise each limit
of the two-moment system in turn: free streaming, static diffusion, stiff
matter coupling, a Marshak wave, a shadow, and the Mach 3 radiative shock.

# Varying a problem

Every factory forwards unrecognised keywords to [`Simulation`](@ref), so the
numerical choices are already free:

```julia
sod_shock_tube(nx = 512, reconstruction = PLM(), integrator = SSPRK2())
```

The *physics* inside a radiation problem is not reachable that way, because the
factory builds its own [`RadiationPhysics`](@ref) from constants that only exist
in its body. Overriding the whole object means re-deriving those constants and
getting a running simulation with different physics if you get one wrong. The
knobs worth varying are therefore named keywords in their own right --
`closure`, `beta_order`, and `chat_factor` on `radhydro_shock`:

```julia
radhydro_shock(nx = 256, closure = LevermoreClosure())   # M1 instead of P1
```

which is also the shape [`compare`](@ref) expects: one flat `NamedTuple` per
variant, splatted into the factory.
"""
module Problems

using ..Hydro3D
using ..Hydro3D: SVec, AbstractClosure

include("exact.jl")
include("hydro.jl")
include("radiation.jl")

export sod_shock_tube, smooth_advection, kelvin_helmholtz, sedov_blast
export radiation_streaming, radiation_diffusion, matter_coupling
export marshak_wave, radiation_shadow, radhydro_shock
export sod_exact, diffusion_exact, radhydro_shock_exact, interpolate_to

end # module Problems
