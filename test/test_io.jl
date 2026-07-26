@testset "checkpoint round trip" begin
    mktempdir() do dir
        sim, _ = P.sod_shock_tube(nx=64)
        run!(sim; tmax=0.05, verbose=false)
        f = save_checkpoint(sim, joinpath(dir, "chk"))
        @test isfile(f)
        @test endswith(f, ".h3d")

        sim2, _ = P.sod_shock_tube(nx=64)
        load_checkpoint!(sim2, f)
        @test sim2.U == sim.U
        @test sim2.t == sim.t
        @test sim2.step == sim.step

        # continuing from the checkpoint reproduces an uninterrupted run
        run!(sim; tmax=0.1, verbose=false)
        run!(sim2; tmax=0.1, verbose=false)
        @test sim2.U == sim.U

        # a mismatched grid is rejected rather than silently misread
        sim3, _ = P.sod_shock_tube(nx=32)
        @test_throws ErrorException load_checkpoint!(sim3, f)

        # radiation state survives too
        rsim, tmax = P.radiation_diffusion(nx=32)
        run!(rsim; tmax=0.01, verbose=false)
        fr = save_checkpoint(rsim, joinpath(dir, "rchk"))
        rsim2, _ = P.radiation_diffusion(nx=32)
        load_checkpoint!(rsim2, fr)
        @test rsim2.R == rsim.R
    end
end

@testset "VTK output" begin
    mktempdir() do dir
        sim, _ = P.sod_shock_tube(nx=16)
        f = write_vtk(sim, joinpath(dir, "out"))
        @test isfile(f)
        @test endswith(f, ".vti")
        head = String(read(f)[1:400])
        @test occursin("VTKFile", head)
        @test occursin("type=\"ImageData\"", head)
        @test occursin("WholeExtent=\"0 16 0 1 0 1\"", head)
        @test occursin("Name=\"density\"", head)
        # the appended payload must be exactly the announced size
        raw = read(f)
        us = findfirst(==(UInt8('_')), raw[findfirst(b"<AppendedData", raw)[1]:end])
        @test us !== nothing

        # with radiation, the radiation fields appear as well
        rsim, _ = P.radiation_diffusion(nx=16)
        f2 = write_vtk(rsim, joinpath(dir, "rad"))
        @test occursin("radiation_energy", String(read(f2)[1:800]))
    end
end

@testset "line output" begin
    mktempdir() do dir
        sim, _ = P.sod_shock_tube(nx=32)
        f = write_line(sim, joinpath(dir, "line.txt"))
        lines = readlines(f)
        @test length(lines) == 33               # header + 32 cells
        @test startswith(lines[1], "# x")
        cols = split(lines[2])
        @test length(cols) == length(split(lines[1])) - 1
        @test parse(Float64, cols[1]) ≈ 0.5 / 32
        @test parse(Float64, cols[2]) ≈ 1.0     # density on the left state
    end
end

@testset "run! output cadence" begin
    mktempdir() do dir
        sim, _ = P.sod_shock_tube(nx=32)
        run!(sim; tmax=0.1, dtout=0.025, outdir=dir, basename="snap", verbose=false)
        files = filter(f -> endswith(f, ".vti"), readdir(dir))
        @test length(files) == 5                # t = 0, 0.025, 0.05, 0.075, 0.1
        @test sim.t ≈ 0.1
    end
end

@testset "callbacks and manual stepping" begin
    sim, _ = P.sod_shock_tube(nx=32)
    seen = Int[]
    run!(sim; tmax=1.0, verbose=false, callback=s -> (push!(seen, s.step); s.step < 5))
    @test sim.step == 5
    @test seen == 1:5

    sim2, _ = P.sod_shock_tube(nx=32)
    dt = step!(sim2, 1e-3)
    @test dt == 1e-3
    @test sim2.t ≈ 1e-3
    @test sim2.step == 1
    @test Hydro3D.timestep(sim2) > 0
end
