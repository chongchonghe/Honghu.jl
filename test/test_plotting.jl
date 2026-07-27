# Field extraction always works; the plotting methods only exist once Plots is
# loaded, so those tests are skipped when it is not available.

@testset "field extraction" begin
    sim, tmax = P.sod_shock_tube(nx=32)
    run!(sim; tmax=0.05, verbose=false)
    g = sim.grid

    @test coordinates(g, 1) ≈ [(i - 0.5) / 32 for i in 1:32]
    @test length(coordinates(g, 2)) == 1
    @test :density in available_fields(sim)
    @test :Erad ∉ available_fields(sim)          # radiation disabled

    rho, lbl = field(sim, :density)
    @test size(rho) == (32, 1, 1)
    @test lbl == "density"
    @test rho ≈ primitives(sim)[IRHO, :, :, :]

    v, _ = field(sim, :speed)
    @test all(v .>= 0)
    @test v ≈ abs.(primitives(sim)[IVX, :, :, :])
    @test first(field(sim, :temperature)) ≈ gas_temperature(sim)

    @test_throws ArgumentError field(sim, :nonsense)
    @test_throws ArgumentError field(sim, :Erad)  # radiation disabled

    rsim, _ = P.radiation_diffusion(nx=16)
    run!(rsim; tmax=0.01, verbose=false)
    @test :Trad in available_fields(rsim)
    @test first(field(rsim, :Erad)) ≈ rsim.R[IERAD, rsim.grid.lo[1]:rsim.grid.hi[1], :, :]
    @test first(field(rsim, :Trad)) ≈ radiation_temperature(rsim)
    f, _ = field(rsim, :reduced_flux)
    @test all(f .<= 1 + 1e-9)                    # causality
end

@testset "plotting requires Plots" begin
    # Without Plots loaded the stubs must fail with a message that says why.
    if !haskey(Base.loaded_modules, Base.PkgId(Base.UUID("91a5bcdd-55d7-5caf-9e0b-520d859cae80"), "Plots"))
        sim, _ = P.sod_shock_tube(nx=16)
        e = try
            plotstate(sim)
        catch err
            sprint(showerror, err)
        end
        @test occursin("using Plots", e)
    else
        @test true
    end
end
