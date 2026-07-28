# The comparison harness. Two properties matter beyond "it runs": the stored
# table must recover the numbers exactly, so a figure redrawn from it is the
# same figure, and the harness must not become a second, quieter definition of
# what a variant means.

@testset "compare" begin

    @testset "sweeps variants against resolutions" begin
        t = compare(P.smooth_advection,
                    (PCM=(reconstruction=PCM(),), PLM=(reconstruction=PLM(),));
                    sizes=(32, 64), path=nothing, verbose=false)
        @test size(t) == (4, 9)
        @test t.problem == "smooth_advection"
        @test variants(t) == ["PCM", "PLM"]
        @test t[:size] == [32.0, 64.0, 32.0, 64.0]
        @test t[:cells] == [32.0, 64.0, 32.0, 64.0]
        @test all(t[:steps] .> 0)
        @test all(t[:floors] .== 0)
        @test :l1_density in Symbol.(t.columns)

        # the variants must actually differ, and in the expected direction
        e = t[:l1_density]
        @test e[3] < e[1] && e[4] < e[2]           # PLM beats PCM at both sizes
        @test convergence_rate(t, :l1_density; variant="PCM") ≈ 1 atol = 0.25
        @test convergence_rate(t, :l1_density; variant="PLM") ≈ 2 atol = 0.35

        @test length(rows(t)) == 4
        @test first(rows(t)).variant == "PCM"
    end

    @testset "stored tables recover exactly" begin
        t = compare(P.sod_shock_tube, (base=(;),); sizes=(32,), path=nothing, verbose=false)
        mktempdir() do dir
            p = write_comparison(t, joinpath(dir, "t.txt"))
            u = read_comparison(p)
            # bit-identical, not merely close: a redrawn figure must be the same
            # figure, and a 10-significant-digit format is not enough for that
            @test u.values == t.values
            @test u.variant == t.variant
            @test u.columns == t.columns
            @test u.problem == t.problem
            @test u.meta == t.meta
        end
    end

    @testset "provenance travels with the numbers" begin
        t = compare(P.sod_shock_tube, (a=(;), b=(reconstruction=PCM(),));
                    sizes=(32,), path=nothing, verbose=false)
        keys_ = first.(t.meta)
        for k in ("variants", "variant_kwargs", "sizes", "metrics", "date", "code",
                  "julia", "threads")
            @test k in keys_
        end
        kw = t.meta[findfirst(==("variant_kwargs"), keys_)][2]
        @test occursin("PCM", kw)                  # what "b" meant is recorded
    end

    @testset "writes a table by default" begin
        mktempdir() do dir
            t = compare(P.sod_shock_tube, (base=(;),); sizes=(32,), outdir=dir,
                        verbose=false)
            p = joinpath(dir, "sod_shock_tube_compare.txt")
            @test isfile(p)
            @test read_comparison(p).values == t.values
        end
    end

    @testset "finds the resolution keyword" begin
        # 1D problems spell it `nx`, 2D/3D ones `n`; the caller should not have
        # to remember which
        @test Hydro3D._size_key(P.sod_shock_tube) === :nx
        @test Hydro3D._size_key(P.sedov_blast) === :n
        @test Hydro3D._size_key(P.kelvin_helmholtz) === :n
        t = compare(P.sedov_blast, (base=(;),); sizes=(16,), path=nothing, verbose=false)
        @test t[:cells] == [16.0^3]
    end

    @testset "rejects what it cannot score" begin
        @test_throws ArgumentError compare(P.kelvin_helmholtz, (a=(;),); sizes=(16,),
                                           path=nothing, verbose=false)
        # ... but takes explicit metrics for the same problem
        t = compare(P.kelvin_helmholtz, (a=(;),); sizes=(16,), path=nothing, verbose=false,
                    metrics=(; peak_density=sim -> maximum(first(field(sim, :density)))))
        @test t[:peak_density][1] > 1
        @test_throws ArgumentError compare(P.sod_shock_tube, (); sizes=(16,), path=nothing)
        @test_throws ArgumentError t[:no_such_column]
        @test_throws ArgumentError convergence_rate(t, :peak_density)   # one size only
    end

    @testset "variants may be labelled pairs" begin
        t = compare(P.sod_shock_tube, ["ppm" => (;), "pcm" => (reconstruction=PCM(),)];
                    sizes=(32,), path=nothing, verbose=false)
        @test variants(t) == ["ppm", "pcm"]
        @test t[:l1_density][1] < t[:l1_density][2]
    end

    @testset "records how the run went, not just how good it was" begin
        t = compare(P.radhydro_shock, (P1=(;),); sizes=(64,), path=nothing, verbose=false)
        @test t[:newton_max][1] >= 1
        @test t[:newton_mean][1] <= t[:newton_max][1]
        @test t[:newton_failures][1] == 0
        @test t[:wall_s][1] > 0
        @test t[:spike][1] > 3.9
    end

end
