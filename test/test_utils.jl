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
