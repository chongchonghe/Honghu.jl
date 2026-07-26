@testset "grid geometry" begin
    g = CartesianGrid(n=(8, 4, 1), xlo=(0.0, -1.0, 0.0), xhi=(2.0, 1.0, 1.0), nghost=3)
    @test g.n == (8, 4, 1)
    @test g.active == (true, true, false)
    @test g.ng == (3, 3, 0)                 # degenerate directions carry no ghosts
    @test g.dims == (14, 10, 1)
    @test g.lo == (4, 4, 1)
    @test g.hi == (11, 7, 1)
    @test Hydro3D.ndim(g) == 2
    @test all(g.dx .≈ (0.25, 0.5, 1.0))

    # cell centres line up with the domain
    @test all(cellcenter(g, g.lo...) .≈ (0.125, -0.75, 0.5))
    @test all(cellcenter(g, g.hi...) .≈ (1.875, 0.75, 0.5))
    @test Hydro3D.cellvolume(g) ≈ 0.25 * 0.5 * 1.0
    @test length(Hydro3D.interior(g)) == 8 * 4 * 1

    @test_throws ArgumentError CartesianGrid(n=(0, 1, 1))
    @test_throws ArgumentError CartesianGrid(n=(4, 1, 1), xlo=(1.0, 0.0, 0.0), xhi=(0.0, 1.0, 1.0))
end

@testset "sweep ranges" begin
    g = CartesianGrid(n=(8, 4, 1), nghost=3)
    # the reconstruction sweep covers the interior extended by one cell in D
    @test cell_ranges(g, Val(1)) == (3:12, 4:7, 1:1)
    @test cell_ranges(g, Val(2)) == (4:11, 3:8, 1:1)
    # faces bound the interior cells
    @test face_ranges(g, Val(1)) == (4:12, 4:7, 1:1)
    @test face_ranges(g, Val(2)) == (4:11, 4:8, 1:1)
    # writing qL at cell hi+1 must stay inside the array
    @test last(cell_ranges(g, Val(1))[1]) + 1 <= g.dims[1]
end

@testset "boundary conditions" begin
    nv = NHYDRO
    for (recon, ng) in ((PCM(), 2), (PLM(), 2), (PPM(), 3))
        @test required_ghosts(recon) == ng
    end

    g = CartesianGrid(n=(4, 1, 1), nghost=2)
    A = allocate(g, nv)
    for i in g.lo[1]:g.hi[1], n in 1:nv
        A[n, i, 1, 1] = 10n + (i - g.lo[1])
    end

    B = copy(A)
    fill_ghosts!(B, Boundaries(Periodic()), g)
    @test B[1, g.lo[1]-1, 1, 1] == A[1, g.hi[1], 1, 1]
    @test B[1, g.lo[1]-2, 1, 1] == A[1, g.hi[1]-1, 1, 1]
    @test B[1, g.hi[1]+1, 1, 1] == A[1, g.lo[1], 1, 1]

    B = copy(A)
    fill_ghosts!(B, Boundaries(Outflow()), g)
    @test B[1, g.lo[1]-1, 1, 1] == B[1, g.lo[1]-2, 1, 1] == A[1, g.lo[1], 1, 1]
    @test B[1, g.hi[1]+2, 1, 1] == A[1, g.hi[1], 1, 1]

    B = copy(A)
    fill_ghosts!(B, Boundaries(Reflecting()), g)
    @test B[1, g.lo[1]-1, 1, 1] == A[1, g.lo[1], 1, 1]        # scalar mirrored
    @test B[2, g.lo[1]-1, 1, 1] == -A[2, g.lo[1], 1, 1]       # normal momentum flipped
    @test B[3, g.lo[1]-1, 1, 1] == A[3, g.lo[1], 1, 1]        # transverse untouched
    @test B[2, g.lo[1]-2, 1, 1] == -A[2, g.lo[1]+1, 1, 1]

    B = copy(A)
    fill_ghosts!(B, Boundaries((Fixed((1.0, 2.0, 3.0, 4.0, 5.0)), Outflow()),
                               (Outflow(), Outflow()), (Outflow(), Outflow())), g)
    @test B[:, g.lo[1]-1, 1, 1] == [1.0, 2.0, 3.0, 4.0, 5.0]

    # a reflecting wall must leave a uniform state untouched apart from the sign
    @test_throws DimensionMismatch fill_ghosts!(allocate(g, NRAD),
                                                Boundaries(Fixed((1.0, 2.0, 3.0, 4.0, 5.0))), g)
end
