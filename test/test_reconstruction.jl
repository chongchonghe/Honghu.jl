@testset "ppm_edges" begin
    # exact on a linear profile
    am, ap = ppm_edges(-2.0, -1.0, 0.0, 1.0, 2.0)
    @test am ≈ -0.5
    @test ap ≈ 0.5
    # constant data stays constant
    @test all(ppm_edges(3.0, 3.0, 3.0, 3.0, 3.0) .≈ (3.0, 3.0))
    # interfaces stay inside the range of the neighbouring averages
    for q in ((0.0, 0.0, 1.0, 1.0, 1.0), (1.0, 0.0, 5.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0, 0.0))
        am, ap = ppm_edges(q...)
        lo, hi = min(q[2], q[3], q[4]), max(q[2], q[3], q[4])
        @test lo - 1e-12 <= am <= hi + 1e-12
        @test lo - 1e-12 <= ap <= hi + 1e-12
    end
    # at a local maximum the parabola collapses to a limited line
    am, ap = ppm_edges(0.0, 1.0, 2.0, 1.0, 0.0)
    @test am ≈ ap ≈ 2.0
end

@testset "reconstruction exactness" begin
    # PLM must be exact on linear data, PPM on quadratic data
    g = CartesianGrid(n=(16, 1, 1), nghost=3)
    q = allocate(g, 1)
    qL = allocate(g, 1)
    qR = allocate(g, 1)
    x(i) = cellcenter(g, i, 1, 1)[1]

    # linear field: cell average equals the centre value
    for i in 1:g.dims[1]
        q[1, i, 1, 1] = 2.0 + 3.0 * x(i)
    end
    reconstruct!(qL, qR, q, PLM(), Val(1), g, 1)
    for i in g.lo[1]:g.hi[1]+1
        xf = g.xlo[1] + (i - g.lo[1]) * g.dx[1]
        @test qL[1, i, 1, 1] ≈ 2.0 + 3.0 * xf
        @test qR[1, i, 1, 1] ≈ 2.0 + 3.0 * xf
    end

    # quadratic field: use the exact cell average so PPM should be exact
    f(x) = 1.0 + 2.0 * x + 3.0 * x^2
    Favg(a, b) = ((b - a) + (b^2 - a^2) + (b^3 - a^3)) / (b - a)
    for i in 1:g.dims[1]
        a = x(i) - g.dx[1] / 2
        b = x(i) + g.dx[1] / 2
        q[1, i, 1, 1] = Favg(a, b)
    end
    reconstruct!(qL, qR, q, PPM(), Val(1), g, 1)
    # only check where the monotonicity constraint is inactive (monotone region)
    for i in g.lo[1]+1:g.hi[1]
        xf = g.xlo[1] + (i - g.lo[1]) * g.dx[1]
        @test qL[1, i, 1, 1] ≈ f(xf) atol = 1e-12
        @test qR[1, i, 1, 1] ≈ f(xf) atol = 1e-12
    end
end

@testset "reconstruction is direction symmetric" begin
    # the same 1D profile laid out along x, y and z must reconstruct identically
    vals = [1.0, 1.0, 1.2, 2.0, 3.5, 3.6, 3.6, 2.0, 1.0, 1.0, 1.0, 1.0]
    out = Vector{Vector{Float64}}()
    for D in 1:3
        n = ntuple(d -> d == D ? length(vals) : 1, 3)
        g = CartesianGrid(n=n, nghost=3)
        q = allocate(g, 1)
        qL = allocate(g, 1)
        qR = allocate(g, 1)
        for (m, v) in enumerate(vals)
            idx = ntuple(d -> d == D ? m + g.ng[d] : 1, 3)
            q[1, idx...] = v
        end
        fill_ghosts!(q, Boundaries(Outflow()), g)
        reconstruct!(qL, qR, q, PPM(), Val(D), g, 1)
        push!(out, [qL[1, ntuple(d -> d == D ? i : 1, 3)...] for i in g.lo[D]:g.hi[D]+1])
    end
    @test out[1] ≈ out[2]
    @test out[1] ≈ out[3]
end

@testset "convergence order on smooth advection" begin
    function l1(nx, recon)
        sim, tmax = P.smooth_advection(nx=nx, reconstruction=recon)
        run!(sim; tmax, verbose=false)
        pr = primitives(sim)
        g = sim.grid
        e = 0.0
        for i in 1:g.n[1]
            xc = g.xlo[1] + (i - 0.5) * g.dx[1]
            e += abs(pr[IRHO, i, 1, 1] - (1 + 0.5 * sinpi(2xc)))
        end
        e / g.n[1]
    end
    # PPM tops out near 2.3 rather than 3 because the Colella-Woodward
    # monotonicity constraint drops to a limited line at the two extrema of the
    # sine wave; this matches the behaviour of the same limiter in Athena++ and
    # Quokka.
    for (recon, expected) in ((PCM(), 0.75), (PLM(), 1.7), (PPM(), 2.0))
        e1 = l1(32, recon)
        e2 = l1(64, recon)
        e3 = l1(128, recon)
        @test log2(e1 / e2) > expected
        @test log2(e2 / e3) > expected
        @test e3 < e2 < e1
    end
end
