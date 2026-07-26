@testset "HLLC consistency" begin
    gam = 1.4
    # identical left and right states must return the physical flux F(U)
    for (rho, u, v, w, p) in ((1.0, 0.0, 0.0, 0.0, 1.0),
                              (2.0, 0.7, -0.3, 0.2, 3.0),
                              (0.5, -1.3, 0.4, -0.9, 0.25))
        cs = sqrt(gam * p / rho)
        E = p / (gam - 1) + rho * (u^2 + v^2 + w^2) / 2
        F = hllc(rho, u, v, w, p, cs, E, rho, u, v, w, p, cs, E, gam, 0.0, 0.0)
        @test F[1] ≈ rho * u
        @test F[2] ≈ rho * u * u + p
        @test F[3] ≈ rho * u * v
        @test F[4] ≈ rho * u * w
        @test F[5] ≈ (E + p) * u
    end
end

@testset "HLLC supersonic upwinding" begin
    gam = 1.4
    # a strongly supersonic left state: the whole fan moves right, so F = F_L
    rhoL, uL, pL = 1.0, 10.0, 1.0
    rhoR, uR, pR = 0.5, 10.0, 0.4
    csL = sqrt(gam * pL / rhoL)
    csR = sqrt(gam * pR / rhoR)
    EL = pL / (gam - 1) + rhoL * uL^2 / 2
    ER = pR / (gam - 1) + rhoR * uR^2 / 2
    F = hllc(rhoL, uL, 0.0, 0.0, pL, csL, EL, rhoR, uR, 0.0, 0.0, pR, csR, ER, gam, 0.0, 0.0)
    @test F[1] ≈ rhoL * uL
    @test F[2] ≈ rhoL * uL^2 + pL
    # mirror image: everything moves left, so F = F_R
    F = hllc(rhoL, -uL, 0.0, 0.0, pL, csL, EL, rhoR, -uR, 0.0, 0.0, pR, csR, ER, gam, 0.0, 0.0)
    @test F[1] ≈ -rhoR * uR
end

@testset "HLLC star pressure stays admissible" begin
    # Toro's "123" problem: two strong rarefactions open a near vacuum. The
    # linearised star pressure would go negative here, so the solver clamps it.
    gam = 1.4
    rho, p = 1.0, 0.4
    cs = sqrt(gam * p / rho)
    for u in (2.0, 5.0, 20.0)
        E = p / (gam - 1) + rho * u^2 / 2
        F = hllc(rho, -u, 0.0, 0.0, p, cs, E, rho, u, 0.0, 0.0, p, cs, E, gam, 2u, 0.0)
        @test F[1] ≈ 0.0 atol = 1e-12       # symmetric: no mass crosses
        @test F[2] >= 0.0                    # momentum flux is a pressure
        @test F[5] ≈ 0.0 atol = 1e-12       # and no energy crosses
    end
end

@testset "HLLC contact resolution" begin
    # HLLC is exact for an isolated contact discontinuity: equal pressure and
    # velocity, jump in density only.
    gam = 1.4
    u, p = 0.7, 2.0
    rhoL, rhoR = 1.0, 0.125
    csL = sqrt(gam * p / rhoL)
    csR = sqrt(gam * p / rhoR)
    EL = p / (gam - 1) + rhoL * u^2 / 2
    ER = p / (gam - 1) + rhoR * u^2 / 2
    F = hllc(rhoL, u, 0.0, 0.0, p, csL, EL, rhoR, u, 0.0, 0.0, p, csR, ER, gam, 0.0, 0.0)
    # the contact moves right, so the upwind (left) state must be picked exactly
    @test F[1] ≈ rhoL * u
    @test F[2] ≈ rhoL * u^2 + p
    @test F[5] ≈ (EL + p) * u
end

@testset "M1 closure" begin
    @test eddington_factor(0.0) ≈ 1 / 3
    @test eddington_factor(1.0) ≈ 1.0
    @test eddington_factor(-0.5) ≈ eddington_factor(0.0)   # clamped
    @test eddington_factor(2.0) ≈ 1.0                      # clamped
    @test 1 / 3 <= eddington_factor(0.7) <= 1.0

    # isotropic limit: T = I/3
    T = eddington_tensor(0.0, 0.0, 0.0)
    for a in 1:3, b in 1:3
        @test T[a, b] ≈ (a == b ? 1 / 3 : 0.0)
    end
    # free-streaming limit along x: T = diag(1, 0, 0)
    T = eddington_tensor(1.0, 0.0, 0.0)
    @test T[1, 1] ≈ 1.0 atol = 1e-12
    @test T[2, 2] ≈ 0.0 atol = 1e-12
    @test T[1, 2] ≈ 0.0 atol = 1e-12
    # the trace is always one, and the tensor is symmetric
    for f in ((0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.2, -0.5, 0.1), (0.0, 0.0, 0.99))
        T = eddington_tensor(f...)
        @test T[1, 1] + T[2, 2] + T[3, 3] ≈ 1.0
        @test T[1, 2] ≈ T[2, 1]
        @test T[1, 3] ≈ T[3, 1]
        @test T[2, 3] ≈ T[3, 2]
    end
end

@testset "radiation HLL consistency" begin
    c = 3.0
    er = 2.0
    Fx, Fy, Fz = 0.5 * c * er, 0.1 * c * er, 0.0
    for D in 1:3
        F = hll_radiation(er, Fx, Fy, Fz, er, Fx, Fy, Fz, c, c, Val(D))
        T = eddington_tensor(Fx / (c * er), Fy / (c * er), Fz / (c * er))
        Fn = (Fx, Fy, Fz)[D]
        @test F[1] ≈ Fn
        @test F[2] ≈ c^2 * T[D, 1] * er
        @test F[3] ≈ c^2 * T[D, 2] * er
        @test F[4] ≈ c^2 * T[D, 3] * er
    end
    # reduced speed of light rescales energy flux by chat/c and pressure by chat*c
    chat = 1.0
    F = hll_radiation(er, Fx, Fy, Fz, er, Fx, Fy, Fz, c, chat, Val(1))
    @test F[1] ≈ (chat / c) * Fx
    @test F[2] ≈ chat * c * eddington_tensor(Fx / (c * er), Fy / (c * er), 0.0)[1, 1] * er
end
