# Iliev et al. (2006) Stromgren Tests 1 and 2, in the corner-source octant geometry of the
# paper, at a resolution and a reduced speed of light that fit in a test. Test 1 is compared
# with the front equation of Shapiro et al. (2006) at the reduced speed of light. That
# reference is approximate, not exact: it counts the photons in flight as Ndot R / chat and
# ignores their depletion by recombinations inside R. Test 2 has no analytic solution and
# gets shape checks.

const PC = Honghu.CGS
_median(v) = (s = sort(vec(v)); s[cld(length(s), 2)])

@testset "Stromgren sphere: Iliev Test 1" begin
    trec = 1 / (2.59e-13 * 1e-3)
    @test P.stromgren_radius_finite_c(2trec; chat=PC.c_light) ≈ P.stromgren_radius_analytic(2trec) rtol = 1e-3
    @test P.stromgren_radius_analytic(500e6 * PC.year) / (1e3 * PC.parsec) ≈ 5.36 rtol = 2e-3

    # Measured with this code: the radius is above the finite-chat reference of Shapiro et al.
    # (2006) by 2.68%, 2.68% and 2.86% at 0.5, 1 and 2 t_rec, so the bound is about 1.5 times
    # the worst of them (0.045 is 1.57 times). It is not a time-step error: with
    # `cfl_radiation` 0.3, 0.1 and 0.03 the three numbers do not move in the third digit. It
    # falls slowly with resolution (n = 12, 16, 24 give 3.8%, 2.7%, 2.1% at 0.5 t_rec and
    # 3.0%, 2.9%, 2.7% at 2 t_rec) and grows as chat falls (1.3% to 1.9% at chat = 1e-3 c, and
    # 24% at 0.5 t_rec at 1e-4 c, where the front outruns the light at first).
    # The reference is itself approximate (see the header), and I did not take the difference
    # apart. The causes not examined, the first one the most likely to matter: (1) the
    # reference's in-flight term Ndot R / chat ignores the depletion of those photons by
    # recombinations inside R; a reviewer's estimate with a sharp-front in-flight term
    # (Ndot - 4 pi R^3 alpha_B n^2 / 3) / chat, kept out of the code, moves R by +1.5% to
    # +1.9% at chat = 3e-4 c and +0.6% at 1e-3 c, the sign and the chat trend measured here;
    # that is a candidate, not a finding; (2) the Gaussian source against the point source
    # of the reference; (3) the diffusion of the HLL transport; (4) the unresolved front.
    sim, _ = P.stromgren_iliev(test=1, n=16, chat_factor=3e-4)
    for tt in (0.5trec, 1.0trec, 2.0trec)
        run!(sim, tmax=tt, verbose=false)
        r = P.ionized_radius(sim; x0=1.2e-3)
        @test abs(r / P.stromgren_radius_finite_c(tt; chat=3e-4 * PC.c_light) - 1) < 0.045
    end
    @test sim.diag.chem_failures == 0

    # behind the front the neutral fraction does not depend on chat
    xHI = map((3e-4, 1e-3)) do f
        s, _ = P.stromgren_iliev(test=1, n=16, chat_factor=f)
        run!(s, tmax=0.5trec, verbose=false)
        field(s, :X_1)[1]
    end
    inside = (xHI[1] .< 0.1) .& (xHI[2] .< 0.1)
    @test count(inside) > 20
    @test abs(_median(xHI[1][inside] ./ xHI[2][inside]) - 1) < 0.02      # measured 0.002
end

@testset "Stromgren sphere: Iliev Test 2 (qualitative)" begin
    trec = 1 / (2.59e-13 * 1e-3)
    sim, _ = P.stromgren_iliev(test=2, n=16, chat_factor=3e-4)
    rprev = 0.0
    for tt in (0.25trec, 0.5trec, 1.0trec)
        run!(sim, tmax=tt, verbose=false)
        r = P.ionized_radius(sim)
        @test r > rprev
        rprev = r
    end
    T = gas_temperature(sim)
    x = field(sim, :X_2)[1]
    @test _median(T[x .> 0.9]) > 1e4                  # hot behind the front
    @test _median(T[x .< 0.01]) < 2e3                 # cold ahead of it
    @test sim.diag.chem_failures == 0
end

@testset "Stromgren sphere: the source and the arguments" begin
    # the cells receive exactly Ndot / 8 photons per second in the octant and Ndot in the
    # whole box, whatever the resolution
    for (centered, Ndot) in ((false, 5e48 / 8), (true, 5e48)), n in (8, 16)
        s, tmax = P.stromgren_iliev(n=n, centered=centered)
        g = s.grid
        total = sum(s.chemistry.emission(Honghu.cellcenter(g, i, j, k)..., 0.0) * prod(g.dx)
                    for k in g.lo[3]:g.hi[3], j in g.lo[2]:g.hi[2], i in g.lo[1]:g.hi[1])
        @test total ≈ Ndot rtol = 1e-12
        @test tmax == 500e6 * PC.year
        @test s.R[1, g.lo[1], g.lo[2], g.lo[3]] < 1e-20      # no ionizing photons to start with (the floor, not a_r T^4)
    end
    @test_throws ArgumentError P.stromgren_iliev(test=3)
    @test P.stromgren_radius_finite_c(0.0; chat=1e9) == 0.0
    # the front is slower than light and ahead of nothing: a smaller chat is behind
    @test P.stromgren_radius_finite_c(1e15; chat=1e8) < P.stromgren_radius_finite_c(1e15; chat=1e9)
end
