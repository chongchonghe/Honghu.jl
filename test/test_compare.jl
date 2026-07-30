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

    @testset "every problem takes `n` for resolution" begin
        # The harness builds the simulation, so it has to pass resolution by
        # name. It used to guess between `nx` and `n` by introspecting the
        # factory with `Base.kwarg_decl`, a Julia internal -- and guessed wrong
        # for `matter_coupling` (no size keyword at all) and `radiation_shadow`
        # (`n` is a tuple). One name across all ten problems replaces the guess.
        for f in (P.sod_shock_tube, P.smooth_advection, P.kelvin_helmholtz, P.sedov_blast,
                  P.radiation_streaming, P.radiation_diffusion, P.matter_coupling,
                  P.marshak_wave, P.radhydro_shock, P.radiation_shadow)
            sim, _ = f(; n=32)
            @test maximum(sim.grid.n) == 32
        end
        # `nx` survives as a human-facing alias, so existing scripts still read well
        @test first(P.sod_shock_tube(; nx=77)).grid.n == (77, 1, 1)
        # the shadow's cell counts are not independent: a 1 x 0.4 domain needs
        # ny = 2nx/5 for square cells, and the default must be untouched
        @test first(P.radiation_shadow()).grid.n == (280, 112, 1)
        @test first(P.radiation_shadow(; n=100)).grid.n == (100, 40, 1)

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

    @testset "sweeping one keyword" begin
        t = compare(P.sod_shock_tube, :riemann, (HLLC(), LLF());
                    sizes=(32, 64), path=nothing, verbose=false)
        # the scheme names itself; no label written by hand
        @test variants(t) == ["HLLC", "LLF"]
        @test all(t[:l1_density, "HLLC"] .< t[:l1_density, "LLF"])   # at both sizes

        # equivalent to spelling the variants out
        g = compare(P.sod_shock_tube,
                    (HLLC=(riemann=HLLC(),), LLF=(riemann=LLF(),));
                    sizes=(32, 64), path=nothing, verbose=false)
        @test t[:l1_density] == g[:l1_density]

        # a value with fields still labels itself distinguishably
        u = compare(P.smooth_advection, :reconstruction, (PLM(MinMod()), PLM(VanLeer()));
                    sizes=(32,), path=nothing, verbose=false)
        @test length(variants(u)) == 2
        @test all(startswith("PLM"), variants(u))
        # numbers label themselves rather than their type
        v = compare(P.smooth_advection, :cfl, (0.2, 0.4); sizes=(32,), path=nothing,
                    verbose=false)
        @test variants(v) == ["0.2", "0.4"]
    end

    @testset "two sweeps of one problem do not overwrite each other" begin
        mktempdir() do dir
            compare(P.sod_shock_tube, :riemann, (HLLC(), LLF()); sizes=(32,),
                    outdir=dir, verbose=false)
            compare(P.sod_shock_tube, :reconstruction, (PCM(), PLM()); sizes=(32,),
                    outdir=dir, verbose=false)
            @test isfile(joinpath(dir, "sod_shock_tube_riemann_compare.txt"))
            @test isfile(joinpath(dir, "sod_shock_tube_reconstruction_compare.txt"))
        end
    end

    @testset "duplicate labels are refused" begin
        # two variants sharing a label would merge silently when the table is
        # sliced, and the merged rows would then be fitted as a size sweep
        @test_throws ArgumentError compare(P.sod_shock_tube,
                                           ["a" => (;), "a" => (reconstruction=PCM(),)];
                                           sizes=(32,), path=nothing, verbose=false)
    end

    @testset "wall_s is not a compilation timing" begin
        # Without a warm-up the first run of each variant carries a few hundred
        # milliseconds of Julia compilation, which made a 64-cell run look 550x
        # slower than it is and slower than the same variant at 512 cells --
        # inverting the one column that answers "what did this cost". Eight
        # times the cells and eight times the steps is a wide enough margin
        # that millisecond timer noise cannot flip it.
        t = compare(P.sod_shock_tube, :riemann, (HLLC(),); sizes=(64, 512),
                    path=nothing, verbose=false)
        w = t[:wall_s, "HLLC"]
        @test w[2] > 4 * w[1]
    end

    @testset "scores are told apart from bookkeeping" begin
        t = compare(P.radhydro_shock, (P1=(;),); sizes=(32, 64), path=nothing,
                    verbose=false)
        @test Hydro3D.metric_columns(t) == ["l1_Trad", "spike"]
        @test !any(in(Hydro3D.BOOKKEEPING), Hydro3D.metric_columns(t))
        # `l1_Trad` falls under refinement; `spike` converges to 4.26, so a
        # log-log slope through it would be meaningless
        @test Hydro3D.is_error_metric(t, "l1_Trad")
        @test !Hydro3D.is_error_metric(t, "spike")

        # the rate table prints for the error and not for the spike
        s = sprint(show, MIME("text/plain"), t)
        @test occursin("convergence rate", s)
        @test occursin("l1_Trad", s)
        # ... and all-zero bookkeeping columns collapse to a note
        h = compare(P.sod_shock_tube, (base=(;),); sizes=(32,), path=nothing,
                    verbose=false)
        @test all(h[:newton_max] .== 0)
        @test occursin("zero for every run", sprint(show, MIME("text/plain"), h))
    end

    @testset "a variant may be an arbitrary builder" begin
        # The point of the builder core: a variant can differ in something that
        # is not a keyword of `Simulation` at all. Here it is the initial
        # condition, which lives inside the factory's closure and was
        # unreachable when a variant had to be a set of keywords.
        ref(x) = 1 + 0.5 * sinpi(2x)
        function setup(nx, amp)
            sim, tmax = P.smooth_advection(; n=nx, amplitude=amp)
            (sim, tmax)
        end
        t = compare(["flat" => nx -> setup(nx, 0.1), "steep" => nx -> setup(nx, 0.9)];
                    sizes=(32, 64), path=nothing, verbose=false,
                    metrics=(; peak=sim -> maximum(first(field(sim, :density)))))
        @test variants(t) == ["flat", "steep"]
        @test all(t[:peak, "steep"] .> t[:peak, "flat"])
        @test t[:size] == [32.0, 64.0, 32.0, 64.0]

        # a builder needs metrics: there is no problem to take defaults from
        @test_throws ArgumentError compare(["a" => nx -> setup(nx, 0.1)];
                                           sizes=(32,), path=nothing, verbose=false)
        # ... and bare keywords need a problem to be passed to
        @test_throws ArgumentError compare(["a" => (amplitude=0.1,)]; sizes=(32,),
                                           path=nothing, verbose=false,
                                           metrics=(; peak=sim -> 1.0))
    end

    @testset "prebuilt simulations, without a sweep" begin
        # no problem argument, so the metrics must be named -- but they are the
        # same ones the problem already defines
        m = default_metrics(P.sod_shock_tube)
        a, _ = P.sod_shock_tube(; n=128, riemann=HLLC())
        b, _ = P.sod_shock_tube(; n=128, riemann=LLF())
        t = compare("HLLC" => a, "LLF" => b; tmax=0.2, metrics=m, path=nothing,
                    verbose=false)
        @test variants(t) == ["HLLC", "LLF"]
        @test t[:size] == [128.0, 128.0]              # read off the grid, not requested
        @test t[:l1_density, "HLLC"][1] < t[:l1_density, "LLF"][1]
        # the simulations are run in place and left finished, ready to plot
        @test a.t ≈ 0.2 && b.t ≈ 0.2
        # one resolution, so no rate is claimed
        @test !occursin("convergence rate", sprint(show, MIME("text/plain"), t))

        # unlabelled instances get positional names rather than a guess at what differs
        c, _ = P.sod_shock_tube(; n=64)
        d, _ = P.sod_shock_tube(; n=64, reconstruction=PCM())
        @test variants(compare(c, d; tmax=0.05, metrics=m, path=nothing,
                               verbose=false)) == ["sim1", "sim2"]

        # a Simulation carries no stopping time of its own
        e, _ = P.sod_shock_tube(; n=32)
        @test_throws ArgumentError compare(e; metrics=m, path=nothing, verbose=false)

        # Warm-up cannot run the simulation itself -- that would consume the run
        # being measured -- so it runs a throwaway copy. Without it the first
        # variant carries its type's compilation and the 64-cell run would
        # measure slower than the 512-cell one, which is the same distortion the
        # builder path guards against.
        small, _ = P.sod_shock_tube(; n=64)
        big, _ = P.sod_shock_tube(; n=512)
        u = compare("small" => small, "big" => big; tmax=0.2, metrics=m, path=nothing,
                    verbose=false)
        @test u[:wall_s, "big"][1] > 4 * u[:wall_s, "small"][1]
    end

    @testset "a column can be sliced by variant" begin
        t = compare(P.smooth_advection, :reconstruction, (PCM(), PLM());
                    sizes=(64, 32), path=nothing, verbose=false)      # out of order
        @test t[:size, "PCM"] == [32.0, 64.0]                         # sorted by size
        @test all(t[:l1_density, "PLM"] .< t[:l1_density, "PCM"])
        @test_throws ArgumentError t[:l1_density, "PPM"]
    end

end
