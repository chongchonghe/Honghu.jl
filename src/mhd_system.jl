# Constrained-transport MHD: the sweep that solves the MHD Riemann problems and
# keeps what constrained transport needs at every face, the edge EMFs, and the
# induction update of the face-centred field. The Julia counterpart of the MHD
# path of Quokka's `computeHydroFluxes` and of `src/hydro/mhd_system.hpp`, with
# one EMF scheme, Quokka2026 (Q26), and one averaging, Londrillo & Del Zanna
# (2004) (LD04).
#
# Sign convention, Quokka's (mhd_system.hpp:34-36): the EMF here is
# eps = u x b, the opposite of Ohm's law eps = -u x b in the papers, and every
# formula is transcribed with the flip, so that the induction update is
# d b / dt = curl eps and a line here matches a line of Quokka.
#
# One stage, in the order of Quokka's `advanceHydroAtLevel`:
#
#   1. ghost cells of U, ghost faces of B;
#   2. floors and cons2prim!, both with the magnetic energy; the cell-centred field;
#   3. for each active direction D, the sweep (`_mhd_sweep!`): reconstruct the
#      primitives and the cell-centred field, solve the Riemann problem on the
#      faces of the interior *and* `mhd_riemann_ghosts` rows into the ghosts across
#      D, keeping the flux, the bounding speeds alpha^-, alpha^+ and the face
#      velocity u* of every face, and take the flux divergence over the interior;
#   4. the EMF on every edge that bounds an interior face (`_emf_cell!`);
#   5. the induction update `B = a B0 + b B + c dt curl eps` (`_induction_face!`).
#
# The sweep is untiled: its faces reach into the ghost rows across D, which the
# boxes of `tile_boxes` do not cover. A direction that is inactive has no sweep:
# its faces carry the cell's velocity as u* (the Riemann problem there has L = R)
# and speeds of one, which cancel out of LD04 exactly (see `MHDWorkspace`).

"""
    mhd_riemann_ghosts(reconstruction) -> Int

How many rows of faces beyond the interior, across the sweep direction, the MHD
sweep solves Riemann problems on: the Q26 EMF reconstructs the face velocity
`u*` along each transverse direction to the edges, and a reconstruction of the
edges of the interior reads that many faces past them (paper Sec. 2.3.1: 1 for
PCM, 2 for PLM, 3 for PPM). It is Quokka's `MinimumHydroRiemannGhost`, which
takes 3 for Q26 whatever the order. The simulation's ghost width is one more,
for the carbuncle input of the outermost of those faces.
"""
mhd_riemann_ghosts(::PCM) = 1
mhd_riemann_ghosts(::PLM) = 2
mhd_riemann_ghosts(::PPM) = 3

"""
    MHDWorkspace{T}

Scratch arrays of the MHD step, built with the simulation:

  - `bcc`: the cell-centred field, `(3, i, j, k)`, every cell, ghosts included;
  - `bL`, `bR`: its interface states along the current sweep;
  - `fspd[d]`: `(2, faces of d)`, the bounding speeds `alpha^-`, `alpha^+` the
    Riemann solver returned on each face normal to `d` (Quokka's
    `fast_mhd_wavespeeds`);
  - `ustar[d]`: `(1, faces of d)`, the Q26 face velocity of each face,
    `(alpha^+ u_L + alpha^- u_R) / (alpha^+ + alpha^-)` (Quokka's `facevel`);
  - `emf[w]`: `(1, edges along w)`, the EMF on each edge along `w`, with one more
    edge than cells along each active direction across `w`;
  - `B0[d]`: the field at the start of the step, for the Runge-Kutta stages.

The face arrays are whole-grid and kept between the sweeps, because the EMF of an
edge reads the faces of two directions. Along an inactive direction `d`,
`fspd[d]` holds ones and is never written: LD04 weights the corner EMFs by these
speeds, and when the two sides of every face along `d` are equal the corners pair
up and the speeds along `d` cancel, exactly for one.
"""
struct MHDWorkspace{T<:AbstractFloat}
    bcc::Array{T,4}
    bL::Array{T,4}
    bR::Array{T,4}
    fspd::NTuple{3,Array{T,4}}
    ustar::NTuple{3,Array{T,4}}
    emf::NTuple{3,Array{T,4}}
    B0::NTuple{3,Array{T,4}}
end

function MHDWorkspace(g::CartesianGrid{T}, B::MagneticField{T}) where {T}
    faces(d, n) = (n, size(B.b[d])[2:4]...)
    fspd = ntuple(d -> g.active[d] ? zeros(T, faces(d, 2)...) : ones(T, faces(d, 2)...), 3)
    ustar = ntuple(d -> zeros(T, faces(d, 1)...), 3)
    emf = ntuple(w -> zeros(T, 1, ntuple(e -> g.dims[e] + (e != w && g.active[e]), 3)...), 3)
    B0 = ntuple(d -> zero(B.b[d]), 3)
    return MHDWorkspace{T}(allocate(g, 3), allocate(g, 3), allocate(g, 3), fspd, ustar, emf, B0)
end

# --- cell-centred quantities ------------------------------------------------------

@inline function _bcc_cell!(i, j, k, bcc, B::MagneticField)
    @inbounds for d in 1:3
        bcc[d, i, j, k] = cell_centred_field(B, d, i, j, k)
    end
    return nothing
end

# The face velocity along an inactive direction `d`: the cell's own velocity,
# what the Riemann problem with equal sides gives.
@inline function _inactive_ustar_cell!(i, j, k, ustar, prim, d)
    @inbounds ustar[1, i, j, k] = prim[d+1, i, j, k]
    return nothing
end

# --- the sweep ------------------------------------------------------------------------

# The body of the MHD sweep for the face `(i, j, k)` along `D`, a kernel for
# `launch!`: the counterpart of `_hydro_flux_cell!`, and of the MHD branch of
# Quokka's `ComputeFluxes`. The normal field comes from the face, `Bn = B[D]`,
# never from a reconstruction; the transverse field is the cell-centred field
# reconstructed with the gas. An inadmissible reconstructed state falls back to
# the adjoining cell, field included.
@inline function _mhd_flux_cell!(i, j, k, flux, fspd, ustar, qL, qR, bL, bR, prim, bcc, Bn,
                                 eos::AbstractEOS, solver::AbstractMHDRiemannSolver, ::Val{D},
                                 gam::T, act1::Bool, act2::Bool) where {D,T}
    e1, e2, e3 = _shift(Val(D))
    iu = D + 1
    iv = _tdir1(Val(D)) + 1
    iw = _tdir2(Val(D)) + 1
    kv = _tdir1(Val(D))           # rows of the transverse field in bcc, bL, bR
    kw = _tdir2(Val(D))
    @inbounds begin
        im, jm, km = i - e1, j - e2, k - e3
        bx = Bn[1, i, j, k]

        # The fallback is a select, not a branch: both candidates are loaded, so the
        # loop over a row is straight-line code that LLVM can vectorise. Every index
        # read is inside the arrays whichever side is taken.
        okL = (qL[IRHO, i, j, k] > 0) & (qL[IPRE, i, j, k] > 0)
        rhoL = ifelse(okL, qL[IRHO, i, j, k], prim[IRHO, im, jm, km])
        pL = ifelse(okL, qL[IPRE, i, j, k], prim[IPRE, im, jm, km])
        uL = ifelse(okL, qL[iu, i, j, k], prim[iu, im, jm, km])
        vL = ifelse(okL, qL[iv, i, j, k], prim[iv, im, jm, km])
        wL = ifelse(okL, qL[iw, i, j, k], prim[iw, im, jm, km])
        byL = ifelse(okL, bL[kv, i, j, k], bcc[kv, im, jm, km])
        bzL = ifelse(okL, bL[kw, i, j, k], bcc[kw, im, jm, km])

        okR = (qR[IRHO, i, j, k] > 0) & (qR[IPRE, i, j, k] > 0)
        rhoR = ifelse(okR, qR[IRHO, i, j, k], prim[IRHO, i, j, k])
        pR = ifelse(okR, qR[IPRE, i, j, k], prim[IPRE, i, j, k])
        uR = ifelse(okR, qR[iu, i, j, k], prim[iu, i, j, k])
        vR = ifelse(okR, qR[iv, i, j, k], prim[iv, i, j, k])
        wR = ifelse(okR, qR[iw, i, j, k], prim[iw, i, j, k])
        byR = ifelse(okR, bR[kv, i, j, k], bcc[kv, i, j, k])
        bzR = ifelse(okR, bR[kw, i, j, k], bcc[kw, i, j, k])

        csL = soundspeed(eos, rhoL, pL)
        csR = soundspeed(eos, rhoR, pR)
        EL = internal_energy(eos, rhoL, pL) + rhoL * (uL * uL + vL * vL + wL * wL) / 2 +
             (bx * bx + byL * byL + bzL * bzL) / 2
        ER = internal_energy(eos, rhoR, pR) + rhoR * (uR * uR + vR * vR + wR * wR) / 2 +
             (bx * bx + byR * byR + bzR * bzR) / 2

        dw = _transverse_jump(prim, i, j, k, im, jm, km, iv, iw, _tshift1(Val(D)),
                              _tshift2(Val(D)), act1, act2)

        sL = HydroState(rhoL, uL, vL, wL, pL, csL, EL, byL, bzL)
        sR = HydroState(rhoR, uR, vR, wR, pR, csR, ER, byR, bzR)
        F, am, ap = mhd_flux(solver, sL, sR, gam, bx, dw)

        flux[IRHO, i, j, k] = F[1]
        flux[iu, i, j, k] = F[2]
        flux[iv, i, j, k] = F[3]
        flux[iw, i, j, k] = F[4]
        flux[IEGAS, i, j, k] = F[5]

        # the bounding speeds, and the Q26 face velocity from them (Quokka
        # hydro_system.hpp, the face velocity of the MHD branch; paper eq. for u*)
        fspd[1, i, j, k] = am
        fspd[2, i, j, k] = ap
        ustar[1, i, j, k] = ifelse(ap + am > zero(T), (ap * uL + am * uR) / (ap + am),
                                   (uL + uR) / 2)
    end
    return nothing
end

# The sweep along `D`: the Riemann problems on the interior faces along `D` and
# `G` rows of ghost faces across it, and the flux divergence over the interior.
function _mhd_sweep!(dU, sim, ::Val{D}, ow::Val, update) where {D}
    g = sim.grid
    T = eltype(g)
    hw, mw = sim.hw, sim.mw
    recon = sim.reconstruction
    nvar = nhydrovars(sim)
    G = mhd_riemann_ghosts(recon)
    across = ntuple(d -> d != D && g.active[d] ? ((g.lo[d] - G):(g.hi[d] + G)) : (g.lo[d]:g.hi[d]),
                    Val(3))
    cells = ntuple(d -> d == D ? ((g.lo[d] - 1):(g.hi[d] + 1)) : across[d], Val(3))
    faces = ntuple(d -> d == D ? (g.lo[d]:(g.hi[d] + 1)) : across[d], Val(3))
    be = CPUThreads()
    launch!(be, cells, _reconstruct_cell!, hw.qL, hw.qR, hw.prim, recon, Val(D), Val(nvar))
    launch!(be, cells, _reconstruct_cell!, mw.bL, mw.bR, mw.bcc, recon, Val(D), Val(3))
    launch!(be, faces, _mhd_flux_cell!, hw.flux, mw.fspd[D], mw.ustar[D], hw.qL, hw.qR, mw.bL,
            mw.bR, hw.prim, mw.bcc, sim.B.b[D], sim.eos, sim.riemann, Val(D),
            T(adiabatic_index(sim.eos)), g.active[_tdir1(Val(D))], g.active[_tdir2(Val(D))])
    # The passive scalars in a kernel of their own: an inner loop over them would
    # keep LLVM from vectorising the Riemann solver's loop over the row.
    sim.nscalars > 0 &&
        launch!(be, faces, _scalar_flux_cell!, hw.flux, hw.qL, hw.qR, hw.prim, Val(D),
                sim.nscalars)
    launch!(be, interior_box(g), _divergence_cell!, dU, hw.flux, Val(D), one(T) / g.dx[D], Val(nvar),
            ow, update)
    return nothing
end

# --- the EMF ---------------------------------------------------------------------------

# The two values a reconstruction gives at the low and the high side of "cell"
# `(i, j, k)` of the one-row array `A`, along the step `o`: the arithmetic of
# `_reconstruct_cell!`, applied to faces, as Quokka's `ReconstructTo`. With
# `o = 0`, an inactive direction, every method returns the value itself.
@inline function _recon_pair(::PCM, A, i, j, k, o)
    @inbounds q = A[1, i, j, k]
    return q, q
end

@inline function _recon_pair(m::PLM, A, i, j, k, o)
    @inbounds begin
        qc = A[1, i, j, k]
        qm = A[1, i-o[1], j-o[2], k-o[3]]
        qp = A[1, i+o[1], j+o[2], k+o[3]]
    end
    s = limit(m.limiter, qc - qm, qp - qc) / 2
    return qc - s, qc + s
end

@inline function _recon_pair(::PPM, A, i, j, k, o)
    @inbounds return ppm_edges(A[1, i-2o[1], j-2o[2], k-2o[3]], A[1, i-o[1], j-o[2], k-o[3]],
                               A[1, i, j, k], A[1, i+o[1], j+o[2], k+o[3]],
                               A[1, i+2o[1], j+2o[2], k+2o[3]])
end

# The EMF on the edge `(i, j, k)` along `w`, a kernel for `launch!`; `(w, d0, d1)` is
# cyclic, and the edge is the low corner of cell `(i, j, k)` across `d0` and `d1`.
# Quokka's `ComputeEMF_Quokka2026` and `EMFAverage_LondrilloDelZanna2004`, fused:
#
#   - Q26 (paper Sec. 2.3.1): `b0` and `u*0` of the faces normal to `d0`
#     (`Ba`, `ua`) are reconstructed along `d1` to the edge, from below (B) and
#     above (T); `b1` and `u*1` of the faces normal to `d1` (`Bb`, `ub`) along `d0`,
#     from the left (L) and the right (R); the four corner EMFs are
#     `eps = u*0 b1 - u*1 b0`.
#   - LD04 (paper eq. for LD04, FelkerStone2018 eq. 41): the corners weighted by
#     the bounding speeds at the edge, each the larger of the two faces that meet
#     there, plus the dissipative terms in the jumps of `b0` and `b1`.
#
# `oa` and `ob` are the steps along `d0` and `d1`, zero where they are inactive.
@inline function _emf_cell!(i, j, k, E, Ba, Bb, ua, ub, fa, fb, recon, oa::NTuple{3,Int},
                            ob::NTuple{3,Int})
    @inbounds begin
        i0, j0, k0 = i - ob[1], j - ob[2], k - ob[3]     # the d0-face below the edge
        i1, j1, k1 = i - oa[1], j - oa[2], k - oa[3]     # the d1-face left of the edge

        b0_B = last(_recon_pair(recon, Ba, i0, j0, k0, ob))
        b0_T = first(_recon_pair(recon, Ba, i, j, k, ob))
        u0_B = last(_recon_pair(recon, ua, i0, j0, k0, ob))
        u0_T = first(_recon_pair(recon, ua, i, j, k, ob))
        b1_L = last(_recon_pair(recon, Bb, i1, j1, k1, oa))
        b1_R = first(_recon_pair(recon, Bb, i, j, k, oa))
        u1_L = last(_recon_pair(recon, ub, i1, j1, k1, oa))
        u1_R = first(_recon_pair(recon, ub, i, j, k, oa))

        # the corners, Quokka's quadrants 0 (BL), 1 (TL), 2 (TR), 3 (BR)
        e0 = u0_B * b1_L - u1_L * b0_B
        e1 = u0_T * b1_L - u1_L * b0_T
        e2 = u0_T * b1_R - u1_R * b0_T
        e3 = u0_B * b1_R - u1_R * b0_B

        # the bounding speeds at the edge
        a0m = max(fa[1, i, j, k], fa[1, i0, j0, k0])
        a0p = max(fa[2, i, j, k], fa[2, i0, j0, k0])
        a1m = max(fb[1, i, j, k], fb[1, i1, j1, k1])
        a1p = max(fb[2, i, j, k], fb[2, i1, j1, k1])
    end

    # LD04 eq. 56: two orderings of the same sum, averaged for exact symmetry
    num1 = ((a0p * a1p) * e0 + (a0m * a1p) * e3) + ((a0p * a1m) * e1 + (a0m * a1m) * e2)
    num2 = ((a0p * a1p) * e0 + (a0p * a1m) * e1) + ((a0m * a1p) * e3 + (a0m * a1m) * e2)
    numerator = (num1 + num2) / 2
    denominator = (a0m + a0p) * (a1m + a1p)
    # the dissipative terms, with the opposite sign to the paper's, as eps = u x b
    term2 = ((a1m * a1p) / (a1m + a1p)) * (b0_T - b0_B) +
            ((a0m * a0p) / (a0m + a0p)) * (b1_L - b1_R)
    @inbounds E[1, i, j, k] = numerator / denominator + term2
    return nothing
end

# --- the induction update ----------------------------------------------------------------

# The Runge-Kutta stage `B = a B0 + b B + c curl(eps)` of the face `(i, j, k)` normal to
# `d`, a kernel for `launch!`; `(d, d1, d2)` is cyclic, `E1`, `E2` are the EMFs along
# `d1` and `d2`, and `o1`, `o2` the steps along them. Quokka's `SolveInductionEqn`,
# with `c = c_RK dt`. Along an inactive direction the step is zero and the
# difference vanishes.
@inline function _induction_face!(i, j, k, Bd, B0d, E1, E2, o1::NTuple{3,Int}, o2::NTuple{3,Int},
                                  dx1, dx2, a, b, c)
    @inbounds begin
        m1 = E1[1, i, j, k]
        p1 = E1[1, i+o2[1], j+o2[2], k+o2[3]]
        m2 = E2[1, i, j, k]
        p2 = E2[1, i+o1[1], j+o1[2], k+o1[3]]
        dbdt = (dx1 * (m1 - p1) + dx2 * (p2 - m2)) / (dx1 * dx2)
        Bd[1, i, j, k] = a * B0d[1, i, j, k] + b * Bd[1, i, j, k] + c * dbdt
    end
    return nothing
end

"""
    constrained_transport!(sim, update)

The EMF on every edge that bounds an interior face ([`MHDWorkspace`](@ref)), from
the face arrays the sweeps left, and then, with `update = (a, B0, b, c dt)`, the
Runge-Kutta stage of the face-centred field, `B = a B0 + b B + c dt curl(eps)` on
every interior face. Any linear combination of divergence-free fields is
divergence-free and `curl` is a discrete curl, so every stage of every tableau
keeps `div b` where it was.
"""
function constrained_transport!(sim, update)
    g = sim.grid
    T = eltype(g)
    mw, B = sim.mw, sim.B
    recon = sim.reconstruction
    be = CPUThreads()
    for w in 1:3
        d0, d1 = mod1(w + 1, 3), mod1(w + 2, 3)
        edges = ntuple(e -> e == w ? (g.lo[e]:g.hi[e]) : (g.lo[e]:(g.hi[e] + B.o[e][e])), Val(3))
        launch!(be, edges, _emf_cell!, mw.emf[w], B.b[d0], B.b[d1], mw.ustar[d0], mw.ustar[d1],
                mw.fspd[d0], mw.fspd[d1], recon, B.o[d0], B.o[d1])
    end
    update === nothing && return nothing
    a, _, b, c = update
    for d in 1:3
        d1, d2 = mod1(d + 1, 3), mod1(d + 2, 3)
        faces = ntuple(e -> g.lo[e]:(g.hi[e] + B.o[d][e]), Val(3))
        launch!(be, faces, _induction_face!, B.b[d], mw.B0[d], mw.emf[d1], mw.emf[d2], B.o[d1],
                B.o[d2], g.dx[d1], g.dx[d2], T(a), T(b), T(c))
    end
    return nothing
end

"""
    mhd_prepare!(sim)

What the MHD stage needs before its sweeps, after `cons2prim!`: the cell-centred
field of every cell, and the face velocity of every face normal to an inactive
direction, the cell's own velocity.
"""
function mhd_prepare!(sim)
    g = sim.grid
    mw = sim.mw
    whole = (1:g.dims[1], 1:g.dims[2], 1:g.dims[3])
    launch!(CPUThreads(), whole, _bcc_cell!, mw.bcc, sim.B)
    for d in 1:3
        g.active[d] || launch!(CPUThreads(), whole, _inactive_ustar_cell!, mw.ustar[d], sim.hw.prim, d)
    end
    return nothing
end
