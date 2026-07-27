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
matter coupling, a Marshak wave, and a shadow.
"""
module Problems

using ..Hydro3D
using ..Hydro3D: SVec

include("exact.jl")
include("hydro.jl")
include("radiation.jl")

export sod_shock_tube, smooth_advection, kelvin_helmholtz, sedov_blast
export radiation_streaming, radiation_diffusion, matter_coupling
export marshak_wave, radiation_shadow, radhydro_shock
export sod_exact, diffusion_exact, radhydro_shock_exact, interpolate_to

end # module Problems
