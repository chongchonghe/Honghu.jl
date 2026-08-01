@testset "SVec" begin
    a = SVec(1.0, 2.0, 3.0)
    b = SVec(4.0, 5.0, 6.0)
    @test length(a) == 3
    @test a[2] == 2.0
    @test Tuple(a + b) == (5.0, 7.0, 9.0)
    @test Tuple(b - a) == (3.0, 3.0, 3.0)
    @test Tuple(2 * a) == (2.0, 4.0, 6.0)
    @test Tuple(a * 2) == (2.0, 4.0, 6.0)
    @test Tuple(a / 2) == (0.5, 1.0, 1.5)
    @test Tuple(-a) == (-1.0, -2.0, -3.0)
    @test dot(a, b) == 32.0
    @test norm(SVec(3.0, 4.0)) == 5.0
    @test Tuple(zero(SVec{3,Float64})) == (0.0, 0.0, 0.0)
    # arithmetic must not allocate
    f(x, y) = dot(x + y, x - y)
    f(a, b)
    @test (@allocated f(a, b)) == 0
end

@testset "equation of state" begin
    eos = IdealGas(gamma=1.4, cv=2.5)
    @test Hydro3D.pressure(eos, 1.0, 2.5) ≈ 1.0
    @test Hydro3D.internal_energy(eos, 1.0, 1.0) ≈ 2.5
    @test Hydro3D.soundspeed(eos, 1.0, 1.0) ≈ sqrt(1.4)
    # round trip through temperature
    rho, eint = 3.0, 7.0
    T = Hydro3D.temperature(eos, rho, eint)
    @test Hydro3D.eint_from_temperature(eos, rho, T) ≈ eint
    @test Hydro3D.heat_capacity(eos, rho, T) ≈ rho * 2.5
    # cv_from_mu is the ideal-gas relation
    @test cv_from_mu(1.0, 5 / 3) ≈ Hydro3D.CGS.k_B / (Hydro3D.CGS.m_u * (5 / 3 - 1))
end

@testset "conserved <-> primitive" begin
    eos = IdealGas(gamma=5 / 3)
    w = SVec(2.0, 0.3, -0.4, 0.5, 1.7)
    u = prim2cons(eos, w)
    @test u[IRHO] ≈ 2.0
    @test u[2] ≈ 2.0 * 0.3
    w2 = cons2prim(eos, u)
    @test all(Tuple(w2) .≈ Tuple(w))
    @test Hydro3D.internal_energy(u) ≈ 1.7 / (5 / 3 - 1)
    @test Hydro3D.kinetic_energy(u) ≈ 0.5 * 2.0 * (0.09 + 0.16 + 0.25)
end

@testset "opacity" begin
    op = ConstantOpacity(planck=1.0, energy=2.0, flux=3.0)
    @test Hydro3D.planck_opacity(op, 1.0, 100.0) == 1.0
    @test Hydro3D.energy_mean_opacity(op, 1.0, 100.0) == 2.0
    @test Hydro3D.flux_mean_opacity(op, 1.0, 100.0) == 3.0

    pl = PowerLawOpacity(kappa0=5.0, rho0=2.0, T0=10.0, m=1.0, n=-3.0)
    @test Hydro3D.planck_opacity(pl, 4.0, 20.0) ≈ 5.0 * 2.0 * (1 / 8)

    u = UserOpacity((rho, T) -> rho * T)
    @test Hydro3D.flux_mean_opacity(u, 3.0, 4.0) == 12.0
end

@testset "opacity temperature derivatives" begin
    # `dkappa_dT` is what makes the coupling's Newton Jacobian exact. Zero is the
    # documented default, and for a law with no temperature dependence it is not
    # a default but the right answer.
    @test Hydro3D.dkappa_dT(ConstantOpacity(planck=1.0, energy=2.0), 1.0, 100.0) ==
          (0.0, 0.0)

    # a power law differentiates to n kappa / T, and that had better match a
    # difference of the law itself
    pl = PowerLawOpacity(kappa0=5.0, rho0=2.0, T0=10.0, m=1.0, n=-3.5)
    T, h = 20.0, 1e-6
    numeric = (Hydro3D.planck_opacity(pl, 4.0, T + h) -
               Hydro3D.planck_opacity(pl, 4.0, T - h)) / (2h)
    dP, dE = Hydro3D.dkappa_dT(pl, 4.0, T)
    @test dP ≈ numeric rtol = 1e-8
    @test dE == dP
    @test Hydro3D.dkappa_dT(PowerLawOpacity(kappa0=5.0, n=0.0), 1.0, 3.0) == (0.0, 0.0)
    @test Hydro3D.dkappa_dT(pl, 4.0, 0.0) == (0.0, 0.0)     # the law is clamped there

    # user-supplied derivatives, and zero when they are not supplied
    @test Hydro3D.dkappa_dT(UserOpacity((rho, T) -> rho * T), 3.0, 4.0) == (0.0, 0.0)
    ud = UserOpacity(planck=(rho, T) -> rho * T, dplanck_dT=(rho, T) -> rho)
    @test Hydro3D.dkappa_dT(ud, 3.0, 4.0) == (3.0, 3.0)     # denergy_dT defaults to it
    ud2 = UserOpacity(planck=(rho, T) -> rho * T, energy=(rho, T) -> 2rho * T,
                      dplanck_dT=(rho, T) -> rho, denergy_dT=(rho, T) -> 2rho)
    @test Hydro3D.dkappa_dT(ud2, 3.0, 4.0) == (3.0, 6.0)

    # per group, and the frequency-dependent hook
    grp = RadiationGroups(1.0, 100.0, 3)
    @test all(Tuple(Hydro3D.group_opacities_dT(ConstantOpacity(1.0), grp, 1.0, 5.0)) .== 0)
    dpl = Hydro3D.group_opacities_dT(pl, grp, 4.0, T)
    @test all(Tuple(dpl) .≈ dP)                              # grey law, same in every group
    fo = FrequencyOpacity((nu, rho, T) -> nu * T; dkappa_dT=(nu, rho, T) -> nu)
    @test Tuple(Hydro3D.group_opacities_dT(fo, grp, 1.0, 5.0)) ==
          Tuple(Hydro3D.group_centers(grp))
    # ... and zero when the law declines to give one
    @test all(Tuple(Hydro3D.group_opacities_dT(FrequencyOpacity((nu, rho, T) -> nu * T),
                                               grp, 1.0, 5.0)) .== 0)
    # one group takes the grey derivative
    @test Hydro3D.group_opacities_dT(pl, RadiationGroups([0.0, Inf]), 4.0, T)[1] ≈ dP
end

@testset "limiters" begin
    for lim in (MinMod(), MonotonizedCentral(), VanLeer(), Sweby(1.5))
        # monotonicity: zero slope at an extremum, sign preserved otherwise
        @test limit(lim, 1.0, -1.0) == 0.0
        @test limit(lim, -2.0, 3.0) == 0.0
        @test limit(lim, 1.0, 1.0) ≈ 1.0
        @test limit(lim, -1.0, -1.0) ≈ -1.0
        # TVD bound: |slope| never exceeds twice the smaller neighbour difference
        @test abs(limit(lim, 1.0, 4.0)) <= 2 * 1.0 + 1e-12
    end
    @test minmod(2.0, 3.0) == 2.0
    @test minmod(-2.0, -3.0) == -2.0
    @test minmod(2.0, -3.0) == 0.0
    @test minmod(1.0, 2.0, 3.0) == 1.0
end

@testset "metrics" begin
    @test relative_l1([1.0, 2.0], [1.0, 2.0]) == 0.0
    @test relative_l1([2.0, 3.0], [1.0, 2.0]) ≈ 2 / 3
    @test relative_l2([1.0, 2.0], [1.0, 2.0]) == 0.0
    @test relative_l2([0.0, 0.0], [3.0, 4.0]) ≈ 1.0
    @test_throws DimensionMismatch relative_l1([1.0], [1.0, 2.0])
    @test_throws ArgumentError relative_l1([1.0], [0.0])

    # Every problem with a reference solution scores itself, and the numbers are
    # the ones the test suite and the figures already quote.
    scored = ((P.sod_shock_tube, (nx=64,)), (P.smooth_advection, (nx=32,)),
              (P.radiation_diffusion, (nx=32,)), (P.radiation_streaming, (nx=64,)),
              (P.radhydro_shock, (nx=64,)), (P.sedov_blast, (n=16,)))
    for (f, kw) in scored
        @test Hydro3D.has_default_metrics(f)
        sim, tmax = f(; kw...)
        run!(sim; tmax, verbose=false)
        m = default_metrics(f)
        @test !isempty(m)
        @test all(isfinite(g(sim)) && g(sim) >= 0 for g in values(m))
    end

    # a problem with no reference solution says so rather than inventing one
    @test !Hydro3D.has_default_metrics(P.kelvin_helmholtz)
    @test_throws ArgumentError default_metrics(P.kelvin_helmholtz)

    # the metric must fall with resolution, or it is not measuring the error
    l1 = default_metrics(P.smooth_advection).l1_density
    errs = map((32, 64, 128)) do nx
        sim, tmax = P.smooth_advection(; nx, reconstruction=PLM())
        run!(sim; tmax, verbose=false)
        l1(sim)
    end
    @test errs[1] > errs[2] > errs[3]
    @test log2(errs[2] / errs[3]) > 1.8            # PLM is second order

    # the vendored reference table is parsed once and shared
    @test P.radhydro_shock_exact()[1] === P.radhydro_shock_exact()[1]
end
