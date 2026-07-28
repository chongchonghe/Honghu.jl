# Every `AbstractX` in the package is advertised as an extension point. This file
# is what makes that claim testable: for each one it defines a toy subtype
# implementing *only* the methods the interface docstring says are required, and
# then runs a simulation with it.
#
# Two assertions per interface, and both are needed:
#
#   equivalence  the toy, configured to match a built-in, gives the same answer
#                -> the interface is wired up and complete
#   sensitivity  the toy, configured differently, gives a different answer
#                -> the interface is actually consulted, not silently ignored
#
# When one of these fails after a refactor, the interface docstring is wrong.
# Fix the docstring or restore the contract; do not weaken the test.
#
# These toy types double as copy-paste documentation, so every method below is
# written with its `Hydro3D.` prefix, exactly as a user outside the package
# would have to write it.

using Hydro3D
using Hydro3D: SVec

# --- the interfaces, implemented from their docstrings alone -----------------

# AbstractEOS: seven methods (see src/eos.jl). A gamma-law gas, written out
# independently of `IdealGas`.
struct ToyGamma <: Hydro3D.AbstractEOS
    gamma::Float64
    cv::Float64
end
Hydro3D.pressure(e::ToyGamma, rho, eint) = (e.gamma - 1) * eint
Hydro3D.internal_energy(e::ToyGamma, rho, p) = p / (e.gamma - 1)
Hydro3D.soundspeed(e::ToyGamma, rho, p) = sqrt(e.gamma * p / rho)
Hydro3D.adiabatic_index(e::ToyGamma) = e.gamma
Hydro3D.temperature(e::ToyGamma, rho, eint) = eint / (rho * e.cv)
Hydro3D.eint_from_temperature(e::ToyGamma, rho, T) = rho * e.cv * T
Hydro3D.heat_capacity(e::ToyGamma, rho, T) = rho * e.cv

# AbstractOpacity: three methods.
struct ToyOpacity <: Hydro3D.AbstractOpacity
    kappa::Float64
end
Hydro3D.planck_opacity(o::ToyOpacity, rho, T) = o.kappa
Hydro3D.energy_mean_opacity(o::ToyOpacity, rho, T) = o.kappa
Hydro3D.flux_mean_opacity(o::ToyOpacity, rho, T) = o.kappa

# AbstractLimiter: one method. This is minmod.
struct ToyLimiter <: Hydro3D.AbstractLimiter end
function Hydro3D.limit(::ToyLimiter, dL, dR)
    dL * dR <= 0 ? zero(dL) : (abs(dL) < abs(dR) ? dL : dR)
end

# AbstractReconstruction: two methods. This is PCM. Note that a custom
# reconstruction is the one extension point that needs package internals --
# `_shift`, `cell_ranges` and `threaded_jk` -- because it owns its own sweep.
struct ToyRecon <: Hydro3D.AbstractReconstruction end
Hydro3D.required_ghosts(::ToyRecon) = 2
function Hydro3D.reconstruct!(qL, qR, q, ::ToyRecon, ::Val{D}, g, nvar) where {D}
    d1, d2, d3 = Hydro3D._shift(Val(D))
    ir, jr, kr = Hydro3D.cell_ranges(g, Val(D))
    Hydro3D.threaded_jk(jr, kr) do j, k
        @inbounds for i in ir, n in 1:nvar
            v = q[n, i, j, k]
            qR[n, i, j, k] = v
            qL[n, i+d1, j+d2, k+d3] = v
        end
    end
    return nothing
end

# AbstractIntegrator: one method. This is SSPRK2 (Heun).
struct ToyIntegrator <: Hydro3D.AbstractIntegrator end
Hydro3D.tableau(::ToyIntegrator) = ((1.0, 0.0, 1.0), (0.5, 0.5, 0.5))

# BoundaryCondition: one method. This is Outflow.
struct ToyBC <: Hydro3D.BoundaryCondition end
function Hydro3D.fill_face!(A::AbstractArray{T,4}, ::ToyBC, g, dir::Int,
                            side::Int, t) where {T}
    src = side == 1 ? g.lo[dir] : g.hi[dir]
    for m in 1:g.ng[dir]
        dst = side == 1 ? g.lo[dir] - m : g.hi[dir] + m
        copyto!(Hydro3D._slab(A, dir, dst), Hydro3D._slab(A, dir, src))
    end
    return nothing
end

# AbstractClosure: one method. Kershaw (1976), a genuine third closure, so
# there is no built-in to compare against -- only invariants.
struct ToyClosure <: Hydro3D.AbstractClosure end
Hydro3D.eddington_factor(::ToyClosure, f) = (1 + 2 * f^2) / 3

# AbstractRiemannSolver: one method. This is LLF.
struct ToySolver <: Hydro3D.AbstractRiemannSolver end
function Hydro3D.riemann_flux(::ToySolver, sL::HydroState, sR::HydroState, gamma, du, dw)
    Sp = max(abs(sL.u) + sL.cs, abs(sR.u) + sR.cs)
    z, o = zero(Sp), one(Sp)
    U_L = Hydro3D.SVec(sL.rho, sL.rho * sL.u, sL.rho * sL.v, sL.rho * sL.w, sL.E)
    U_R = Hydro3D.SVec(sR.rho, sR.rho * sR.u, sR.rho * sR.v, sR.rho * sR.w, sR.E)
    F_L = sL.u * U_L + sL.P * Hydro3D.SVec(z, o, z, z, sL.u)
    F_R = sR.u * U_R + sR.P * Hydro3D.SVec(z, o, z, z, sR.u)
    (F_L + F_R) / 2 - (Sp / 2) * (U_R - U_L)
end

# --- harnesses ---------------------------------------------------------------

# A short shock tube. Three steps is enough: the state has been through the
# ghost fill, the reconstruction, the Riemann solver and the integrator.
#
# The uniform drift `vx` is what makes the boundary condition observable this
# early: with `vx = 0` the wave has not reached either face in three steps, and
# an outflow ghost and a reflecting ghost are then bit-identical.
function _hydro3(; nsteps=3, kwargs...)
    sim = Simulation(; n=(32, 1, 1), boundaries=Boundaries(Outflow()), kwargs...)
    set_initial_conditions!(sim) do x, y, z
        x < 0.5 ? (rho=1.0, vx=0.3, p=1.0) : (rho=0.125, vx=0.3, p=0.1)
    end
    for _ in 1:nsteps
        step!(sim)
    end
    copy(sim.U)
end

# A gas that starts out of equilibrium with the radiation field, so the implicit
# coupling, the opacity and the closure all have work to do.
function _rad3(; nsteps=3, closure=LevermoreClosure(), opacity=ConstantOpacity(1.0),
               eos=IdealGas(; gamma=5 / 3, cv=1.0), kwargs...)
    rad = RadiationPhysics(; c=10.0, chat=10.0, a_rad=1.0, opacity, closure,
                           beta_order=0)
    sim = Simulation(; n=(16, 1, 1), eos, radiation=rad, hydro=false,
                     boundaries=Boundaries(Periodic()), kwargs...)
    set_initial_conditions!(sim) do x, y, z
        (rho=1.0, T=1.0, Trad=0.3, Fx=0.1 * exp(-((x - 0.5) / 0.1)^2))
    end
    for _ in 1:nsteps
        step!(sim)
    end
    (copy(sim.U), copy(sim.R))
end

@testset "extension points" begin

    @testset "AbstractEOS" begin
        @test _hydro3(eos=ToyGamma(1.4, 1.0)) ≈ _hydro3(eos=IdealGas(gamma=1.4, cv=1.0))
        @test _hydro3(eos=ToyGamma(1.6, 1.0)) != _hydro3(eos=ToyGamma(1.4, 1.0))
        # the radiation coupling needs the other three methods
        U, R = _rad3(eos=ToyGamma(5 / 3, 1.0))
        Ui, Ri = _rad3(eos=IdealGas(gamma=5 / 3, cv=1.0))
        @test U ≈ Ui && R ≈ Ri
        @test all(isfinite, U) && all(isfinite, R)
    end

    @testset "AbstractOpacity" begin
        @test _rad3(opacity=ToyOpacity(1.0)) == _rad3(opacity=ConstantOpacity(1.0))
        @test _rad3(opacity=ToyOpacity(5.0)) != _rad3(opacity=ToyOpacity(1.0))
    end

    @testset "AbstractLimiter" begin
        @test _hydro3(reconstruction=PLM(ToyLimiter())) ==
              _hydro3(reconstruction=PLM(MinMod()))
        @test _hydro3(reconstruction=PLM(ToyLimiter())) !=
              _hydro3(reconstruction=PLM(MonotonizedCentral()))
    end

    @testset "AbstractReconstruction" begin
        @test _hydro3(reconstruction=ToyRecon()) == _hydro3(reconstruction=PCM())
        @test _hydro3(reconstruction=ToyRecon()) != _hydro3(reconstruction=PLM())
        # the ghost width really is taken from the method
        sim = Simulation(n=(8, 1, 1), reconstruction=ToyRecon())
        @test sim.grid.ng[1] == 2
    end

    @testset "AbstractIntegrator" begin
        @test _hydro3(integrator=ToyIntegrator()) == _hydro3(integrator=SSPRK2())
        @test _hydro3(integrator=ToyIntegrator()) != _hydro3(integrator=SSPRK3())
    end

    @testset "BoundaryCondition" begin
        @test _hydro3(boundaries=Boundaries(ToyBC())) ==
              _hydro3(boundaries=Boundaries(Outflow()))
        @test _hydro3(boundaries=Boundaries(ToyBC())) !=
              _hydro3(boundaries=Boundaries(Reflecting()))
    end

    @testset "AbstractRiemannSolver" begin
        @test _hydro3(riemann=ToySolver()) == _hydro3(riemann=LLF())
        @test _hydro3(riemann=ToySolver()) != _hydro3(riemann=HLLC())
        @test _hydro3() == _hydro3(riemann=HLLC())          # HLLC is the default
        # the solver is a compile-time property of the simulation type
        @test Simulation(n=(8, 1, 1), riemann=LLF()).riemann isa LLF
    end

    @testset "AbstractClosure" begin
        chi = f -> Hydro3D.eddington_factor(ToyClosure(), f)
        @test chi(0.0) ≈ 1 / 3
        @test chi(1.0) ≈ 1.0
        @test all(chi(f) >= chi(f - 0.05) for f in 0.05:0.05:1.0)
        U, R = _rad3(closure=ToyClosure())
        @test all(isfinite, R)
        # Kershaw is neither Levermore nor Eddington, and the transport says so
        @test R != last(_rad3(closure=LevermoreClosure()))
        @test R != last(_rad3(closure=EddingtonClosure()))
    end

end
