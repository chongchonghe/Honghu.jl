# Allocation regression tests for the hot kernels.
#
# These exist because of a real bug: an inner closure in `radiation_source_step!`
# captured variables that were assigned in several places, so Julia boxed them
# and every Newton iteration heap-allocated -- 12 kB per cell, 406 MB and 200 ms
# per source step at 32^3. Nothing in the physics tests noticed, because the
# answers were right; only the run time was destroyed.
#
# The kernels are not literally allocation-free once threading is on:
# `Threads.@threads` allocates task objects, and the ghost-cell views are
# heap-allocated SubArrays. Both are O(threads) or O(faces), not O(cells). So
# the assertion is on allocation *per cell*, which stays flat as the grid grows
# and blows up the moment something is boxed.

# Base's `@allocated` after a warm-up call is the right tool here; BenchmarkTools
# is used for the timing table in `benchmarks.jl`, where its warm-up and
# minimum-over-samples statistics actually matter.

"Bytes allocated per interior cell by `f()`, after warm-up."
function bytes_per_cell(f, sim)
    f()                                     # compile and warm up
    ncells = prod(sim.grid.n)
    minimum(@allocated(f()) for _ in 1:3) / ncells
end

@testset "hot kernels do not allocate per cell" begin
    # A generous bound: the boxing bug was 12416 B/cell, threading overhead on a
    # 24^3 grid is well under 1 B/cell.
    LIMIT = 8.0

    sim = Simulation(n=(24, 24, 24), boundaries=Boundaries(Periodic()))
    set_initial_conditions!(sim) do x, y, z
        (rho=1 + 0.2 * sinpi(2x), vx=0.3, vy=-0.1, vz=0.2, p=1.0)
    end
    hw = sim.hw
    g = sim.grid
    Honghu.cons2prim!(hw.prim, sim.U, sim.eos, g)

    @test bytes_per_cell(() -> Honghu.cons2prim!(hw.prim, sim.U, sim.eos, g),
                         sim) < LIMIT
    @test bytes_per_cell(() -> Honghu.enforce_floors!(sim.U, sim.eos, g, sim.floors),
                         sim) < LIMIT
    for recon in (PCM(), PLM(), PPM())
        @test bytes_per_cell(() -> Honghu.reconstruct!(hw.qL, hw.qR, hw.prim, recon,
                                                       Val(1), g, Honghu.NHYDRO), sim) < LIMIT
    end
    @test bytes_per_cell(() -> Honghu.hydro_fluxes!(hw.flux, hw.qL, hw.qR, hw.prim,
                                                    sim.eos, sim.riemann, Val(1), g), sim) < LIMIT
    @test bytes_per_cell(() -> Honghu.flux_divergence!(hw.dU, hw.flux, Val(1), g,
                                                       Honghu.NHYDRO), sim) < LIMIT
    @test bytes_per_cell(() -> Honghu.combine!(sim.U, 0.5, hw.U0, 0.5, sim.U, 0.1,
                                               hw.dU, g, Honghu.NHYDRO), sim) < LIMIT
    @test bytes_per_cell(() -> Honghu.hydro_timestep(hw.prim, sim.eos, g, 0.4), sim) < LIMIT
end

@testset "scalar kernels do not allocate per cell" begin
    LIMIT = 8.0
    sim = Simulation(n=(24, 24, 24), nscalars=2, boundaries=Boundaries(Periodic()))
    set_initial_conditions!((x, y, z) -> (rho=1 + 0.2sinpi(2x), vx=0.3, p=1.0, X=(0.4, 0.6)), sim)
    hw, g, nv = sim.hw, sim.grid, Honghu.nhydrovars(sim)
    Honghu.cons2prim!(hw.prim, sim.U, sim.eos, g, 2)
    Honghu.reconstruct!(hw.qL, hw.qR, hw.prim, PPM(), Val(1), g, nv)
    @test bytes_per_cell(() -> Honghu.cons2prim!(hw.prim, sim.U, sim.eos, g, 2), sim) < LIMIT
    @test bytes_per_cell(() -> Honghu.hydro_fluxes!(hw.flux, hw.qL, hw.qR, hw.prim, sim.eos,
                                                    sim.riemann, Val(1), g, 2), sim) < LIMIT
    @test bytes_per_cell(() -> Honghu.enforce_floors!(sim.U, sim.eos, g, sim.floors, 2), sim) < LIMIT
    @test bytes_per_cell(() -> Honghu.enforce_scalar_floors!(sim.U, g, 2), sim) < LIMIT
    @test bytes_per_cell(() -> Honghu.enforce_scalar_floors!(sim.U, g, 2, true), sim) < LIMIT
end

@testset "radiation kernels do not allocate per cell" begin
    LIMIT = 8.0

    rad = RadiationPhysics(c=10.0, chat=1.0, a_rad=1.0, opacity=ConstantOpacity(1.0),
                           beta_order=1)
    sim = Simulation(n=(24, 24, 24), radiation=rad, boundaries=Boundaries(Periodic()))
    set_initial_conditions!(sim) do x, y, z
        (rho=1 + 0.2 * sinpi(2x), vx=0.3, p=1.0, Erad=1.0, Fx=0.5)
    end
    rw = sim.rw
    g = sim.grid
    Honghu.rad_cons2prim!(rw.prim, sim.R, rad, g)

    @test bytes_per_cell(() -> Honghu.rad_cons2prim!(rw.prim, sim.R, rad, g),
                         sim) < LIMIT
    @test bytes_per_cell(() -> Honghu.enforce_rad_floors!(sim.R, rad, g, sim.floors),
                         sim) < LIMIT
    @test bytes_per_cell(() -> Honghu.reconstruct!(rw.qL, rw.qR, rw.prim, PPM(), Val(1),
                                                   g, Honghu.NRAD), sim) < LIMIT
    @test bytes_per_cell(() -> Honghu.rad_fluxes!(rw.flux, rw.qL, rw.qR, sim.R,
                                                  rw.chat, rad, Val(1), g), sim) < LIMIT
    # The `chat` limiter runs once per IMEX substep over the whole array, ghost
    # cells included. `group_opacities` and `_signal_sum` (through
    # `eddington_tensor`) both build `SVec`s inside the cell loop, which is
    # exactly the shape that boxed and allocated before.
    lrad = RadiationPhysics(c=10.0, chat=1.0, a_rad=1.0, opacity=ConstantOpacity(1.0),
                            chat_limiter=StabilityBound())
    @test bytes_per_cell(() -> Honghu.update_chat!(rw.chat, sim.U, sim.R, sim.eos, lrad,
                                                   g, sim.floors, 1e-3), sim) < LIMIT

    # the one that regressed
    @test bytes_per_cell(() -> Honghu.radiation_source_step!(sim.U, sim.R, rw.chat,
                                                             sim.eos, rad, g, 1e-3,
                                                             nothing, 0.0, sim.floors), sim) < LIMIT
    # and with beta_order = 0, which takes the other branch
    rad0 = RadiationPhysics(c=10.0, chat=1.0, a_rad=1.0, opacity=ConstantOpacity(1.0),
                            beta_order=0)
    @test bytes_per_cell(() -> Honghu.radiation_source_step!(sim.U, sim.R, rw.chat,
                                                             sim.eos, rad0, g, 1e-3,
                                                             nothing, 0.0, sim.floors), sim) < LIMIT
    # and with dust, which sends even a grey run through the multigroup driver
    # and the dust-temperature solve
    radd = RadiationPhysics(c=10.0, chat=1.0, a_rad=1.0, opacity=ConstantOpacity(1.0),
                            beta_order=1, dust=BateKetoDust(coefficient=1.0))
    @test bytes_per_cell(() -> Honghu.radiation_source_step!(sim.U, sim.R, rw.chat,
                                                             sim.eos, radd, g, 1e-3,
                                                             nothing, 0.0, sim.floors), sim) < LIMIT
end

@testset "a full step scales linearly in cells" begin
    # Allocation per step must not grow with the grid: if it does, something in
    # the cell loop is allocating.
    function step_bytes(n)
        sim = Simulation(n=(n, n, n), boundaries=Boundaries(Periodic()))
        set_initial_conditions!(sim) do x, y, z
            (rho=1 + 0.2 * sinpi(2x), vx=0.3, p=1.0)
        end
        step!(sim)
        minimum(@allocated(step!(sim)) for _ in 1:3)
    end
    b16 = step_bytes(16)
    b32 = step_bytes(32)
    # 32^3 has 8x the cells; allocation must stay within a small constant factor
    @test b32 < 4 * b16 + 200_000
end

@testset "chemistry step does not allocate per cell (backward Euler)" begin
    net = HydrogenIliev2(photon_energy=29.65 * Honghu.CGS.eV)
    rad = RadiationPhysics(c=Honghu.CGS.c_light, chat=1e-3 * Honghu.CGS.c_light,
                           a_rad=Honghu.CGS.a_rad, k_B=Honghu.CGS.k_B,
                           opacity=ConstantOpacity(0.0), beta_order=0)
    sim = Simulation(n=(16, 16, 16), xhi=(1e21, 1e21, 1e21), hydro=false, radiation=rad,
                     chemistry=Photochemistry(net; solver=BackwardEulerChem()))
    m = species_masses(net)
    set_initial_conditions!((x, y, z) -> (rho=1e-3 * m[1], T=100.0,
                                          X=(0.99, 0.01m[2] / m[1], 0.01m[3] / m[1]), Erad=1e-12), sim)
    @test bytes_per_cell(() -> Honghu.chemistry_step!(sim, 1e10, 0.0), sim) < 8.0
end

@testset "chemistry step with the default RosenbrockChem: allocation is bounded" begin
    # The one hot path that is allowed to allocate per cell: the reused OrdinaryDiffEq
    # integrator allocates inside `reinit!` and `solve!` (~350 B per cell measured on this
    # box), where `BackwardEulerChem` above allocates nothing. The bound is an upper limit,
    # so a boxed closure or an integrator rebuilt per cell (~8 kB) still fails it.
    net = HydrogenIliev2(photon_energy=29.65 * Honghu.CGS.eV)
    rad = RadiationPhysics(c=Honghu.CGS.c_light, chat=1e-3 * Honghu.CGS.c_light,
                           a_rad=Honghu.CGS.a_rad, k_B=Honghu.CGS.k_B,
                           opacity=ConstantOpacity(0.0), beta_order=0)
    sim = Simulation(n=(16, 16, 16), xhi=(1e21, 1e21, 1e21), hydro=false, radiation=rad,
                     chemistry=Photochemistry(net))
    m = species_masses(net)
    set_initial_conditions!((x, y, z) -> (rho=1e-3 * m[1], T=100.0,
                                          X=(0.99, 0.01m[2] / m[1], 0.01m[3] / m[1]), Erad=1e-12), sim)
    @test bytes_per_cell(() -> Honghu.chemistry_step!(sim, 1e10, 0.0), sim) < 2048.0
end
