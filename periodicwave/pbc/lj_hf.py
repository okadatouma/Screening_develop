# -*- coding: utf-8 -*-
"""λ-jellium の下のバンドに射影した Hartree–Fock（結晶、単位胞に 1 電子）。

VMC と同じコード単位（長さ r_s a_B、エネルギー Ry r_s^2/2、H = sum[-1/2 lap + Delta h] + r_s sum 1/r）。
下のバンドの平面波スピノル |q> = e^{iq.r} u_q / sqrt(A)、u_q = (1, lam q_+)/sqrt(1 + lam^2 q^2)（成分の順は s = +1, -1）、
エネルギー q^2/2。結晶の格子は a1 = (a, 0)、a2 = (-a/2, sqrt3 a/2)、a = sqrt(2 pi/sqrt3)（電子 1 個あたり面積 pi）、
スーパーセルは L x L（run_lambda_jellium.py の lattices と同じ向き）。k は 6x6 の網、G は |G| <= gcut |b| の逆格子。
占有：各 k で HF 行列のいちばん低い固有ベクトル 1 本（絶縁的な結晶）。

  D_k(G, G') = <c^dag_{k+G'} c_{k+G}>,  F(q, q') = <u_q|u_q'>,  v(Q) = 2 pi r_s / |Q|（Q = 0 は除く）
  Hartree : h_k(G, G') = (1/A) v(G-G') rho(G-G') F(k+G, k+G'),  rho(g) = sum_k sum_G F(k+G, k+G+g) D_k(G+g, G)
  Fock    : h_k'(Ga, Gc) = -(1/A) sum_k sum_Gp v(k'+Ga-k-Gp) F(k'+Ga, k+Gp) D_k(Gp, Gp+Gc-Ga) F(k+Gp+Gc-Ga, k'+Gc)
  E = sum_k [ sum_G eps D_k(G,G) + 1/2 tr((h^H + h^F)_k D_k) ]
VMC（Ewald）の Slater 行列式のエネルギーとは、自己像の Madelung 項 N xi/2 だけずれる（check で確かめる）。
"""

import numpy as np


def geometry(L=6):
  a = np.sqrt(2 * np.pi / np.sqrt(3))
  A = np.array([[a, -a / 2], [0.0, a * np.sqrt(3) / 2]])        # 列が a1, a2
  B = 2 * np.pi * np.linalg.inv(A).T                              # 列が b1, b2（A^T B = 2 pi）
  return a, A, B


def gvectors(B, gcut=3.01):
  bl = np.linalg.norm(B[:, 0])
  m = int(np.ceil(gcut * 2)) + 1
  out = []
  for i in range(-m, m + 1):
    for j in range(-m, m + 1):
      g = i * B[:, 0] + j * B[:, 1]
      if np.linalg.norm(g) <= gcut * bl + 1e-9:
        out.append((i, j))
  out.sort(key=lambda t: (np.linalg.norm(t[0] * B[:, 0] + t[1] * B[:, 1]), t))
  return np.array(out, dtype=int)


def kgrid(B, L=6):
  """k = (n1 b1 + n2 b2)/L を第 1 ブリルアン域（|k| 最小の代表）へ。返り値：整数 (n1, n2)、代表の整数のずらし、k。"""
  ns, shifts, ks = [], [], []
  for n1 in range(L):
    for n2 in range(L):
      best = None
      for m1 in range(-2, 3):
        for m2 in range(-2, 3):
          k = (n1 / L - m1) * B[:, 0] + (n2 / L - m2) * B[:, 1]
          d = np.linalg.norm(k)
          if best is None or d < best[0] - 1e-9:
            best = (d, (m1, m2), k)
      ns.append((n1, n2)); shifts.append(best[1]); ks.append(best[2])
  return np.array(ns), np.array(shifts), np.array(ks)


def spinor(q, lam):
  """下のバンドのスピノル u_q（..., 2）。成分 0 が s = +1、1 が s = -1。"""
  qp = q[..., 0] + 1j * q[..., 1]
  nrm = 1.0 / np.sqrt(1.0 + lam ** 2 * (q[..., 0] ** 2 + q[..., 1] ** 2))
  return np.stack([nrm + 0j, lam * qp * nrm], axis=-1)


class LJHF:
  def __init__(self, lam, r_s, L=6, gcut=3.01):
    self.lam, self.r_s, self.L = float(lam), float(r_s), L
    self.a, self.A, self.B = geometry(L)
    self.N = L * L
    self.area = self.N * np.pi
    self.gint = gvectors(self.B, gcut)                 # (NG, 2) 整数
    self.G = self.gint @ self.B.T                      # (NG, 2)
    self.NG = len(self.G)
    self.kn, self.kshift, self.k = kgrid(self.B, L)    # (Nk, 2)
    self.Nk = len(self.k)
    self.q = self.k[:, None, :] + self.G[None, :, :]   # (Nk, NG, 2)
    self.eps = 0.5 * np.sum(self.q ** 2, axis=-1)      # (Nk, NG)
    self.U = spinor(self.q, self.lam)                  # (Nk, NG, 2)
    lut = {tuple(t): i for i, t in enumerate(self.gint)}
    self._lut = lut
    # J[p, a, c] = index(Gp + Gc - Ga)（なければ -1）
    gi = self.gint
    J = -np.ones((self.NG, self.NG, self.NG), dtype=int)
    for p in range(self.NG):
      for a_ in range(self.NG):
        for c in range(self.NG):
          t = tuple(gi[p] + gi[c] - gi[a_])
          J[p, a_, c] = lut.get(t, -1)
    self.J = J
    self.Jm = J >= 0
    self.Jc = np.where(self.Jm, J, 0)
    # Hartree の差 g = G - G'（ずらし表）：H[g_idx] は G-G' の整数を直接使う
    self.dint = gi[:, None, :] - gi[None, :, :]        # (NG, NG, 2) = G - G'
    # Coulomb の重み（Fock 用）：V1[k', a, k, p] = v(k'+Ga-k-Gp)/A
    Q = self.q[:, :, None, None, :] - self.q[None, None, :, :, :]   # (Nk, NG, Nk, NG, 2)
    Qn = np.linalg.norm(Q, axis=-1)
    with np.errstate(divide='ignore'):
      V1 = np.where(Qn > 1e-9, 2 * np.pi * self.r_s / np.maximum(Qn, 1e-12), 0.0) / self.area
    self.V1 = V1
    # F1[k', a, k, p] = <u_{k'+Ga}|u_{k+Gp}>
    self.F1 = np.einsum('xas,kps->xakp', self.U.conj(), self.U)

  # ---------------------------------------------------------------- 平均場
  def rho(self, D):
    """rho(g) を G - G' の整数 (i, j) ごとに：sum_k sum_G F(k+G, k+G+g) D_k(G+g, G)。返り値は dict。"""
    out = {}
    gi = self.gint
    for gidx in range(self.NG):     # g は G の集合の差で出てくるものだけ要るが、Hartree は G-G' で使う
      pass
    # 直接 G-G' の全組で計算
    dset = {}
    for a_ in range(self.NG):
      for b_ in range(self.NG):
        t = tuple(gi[a_] - gi[b_])
        if t == (0, 0) or t in dset:
          continue
        # sum over G'' with G''+g in set
        val = 0j
        idx_from, idx_to = [], []
        for p in range(self.NG):
          j = self._lut.get(tuple(gi[p] + np.array(t)), -1)
          if j >= 0:
            idx_from.append(p); idx_to.append(j)
        idx_from = np.array(idx_from); idx_to = np.array(idx_to)
        if len(idx_from):
          Fk = np.sum(self.U[:, idx_from, :].conj() * self.U[:, idx_to, :], axis=-1)   # (Nk, n)
          val = np.sum(Fk * D[:, idx_to, idx_from])
        dset[t] = val
    return dset

  def hartree(self, D):
    rho = self.rho(D)
    H = np.zeros((self.Nk, self.NG, self.NG), complex)
    gi = self.gint
    for a_ in range(self.NG):
      for b_ in range(self.NG):
        t = tuple(gi[a_] - gi[b_])
        if t == (0, 0):
          continue
        g = gi[a_] - gi[b_]
        gv = g @ self.B.T
        v = 2 * np.pi * self.r_s / np.linalg.norm(gv) / self.area
        F = np.sum(self.U[:, a_, :].conj() * self.U[:, b_, :], axis=-1)    # (Nk,)
        H[:, a_, b_] = v * rho[t] * F
    return H

  def fock(self, D):
    H = np.zeros((self.Nk, self.NG, self.NG), complex)
    # Dsh[k, p, a, c] = D_k(Gp, Gp+Gc-Ga)、F2 の左のスピノル U[k, J(p,a,c)]
    Dg = D[:, np.arange(self.NG)[:, None, None], self.Jc] * self.Jm[None]      # (Nk, p, a, c)
    UJ = self.U[:, self.Jc, :] * self.Jm[None, :, :, :, None]                  # (Nk, p, a, c, 2)
    for kp in range(self.Nk):
      # F2[k, p, a, c] = <u_{k+G_J}|u_{k'+Gc}>
      F2 = np.einsum('kpacs,cs->kpac', UJ.conj(), self.U[kp])
      W = self.V1[kp] * self.F1[kp]                                            # (a, k, p)
      H[kp] = -np.einsum('akp,kpac,kpac->ac', W, Dg, F2, optimize=True)
    return H

  def energy(self, D, hH, hF):
    e1 = np.sum(self.eps * np.real(np.einsum('kgg->kg', D)))
    e2 = 0.5 * np.real(np.einsum('kab,kba->', hH + hF, D))
    return e1 + e2

  # ---------------------------------------------------------------- 解く
  def solve(self, D0, maxit=300, mix=0.5, tol=1e-7, verbose=False):
    D = D0.copy()
    hist = []
    for it in range(maxit):
      hH, hF = self.hartree(D), self.fock(D)
      Hk = hH + hF
      Hk[:, np.arange(self.NG), np.arange(self.NG)] += self.eps
      Hk = 0.5 * (Hk + np.conj(np.transpose(Hk, (0, 2, 1))))
      w, v = np.linalg.eigh(Hk)
      c = v[:, :, 0]
      Dn = np.einsum('ka,kb->kab', c, c.conj())
      E = self.energy(D, hH, hF)
      dD = np.max(np.abs(Dn - D))
      hist.append((E, dD, float(np.min(w[:, 1] - w[:, 0]))))
      if verbose and (it % 10 == 0):
        print(f"  it {it:3d}  E = {E:.6f}  |dD| = {dD:.2e}  gap_min = {hist[-1][2]:.4f}")
      if dD < tol:
        D = Dn
        break
      D = (1 - mix) * D + mix * Dn
    hH, hF = self.hartree(D), self.fock(D)
    Hk = hH + hF
    Hk[:, np.arange(self.NG), np.arange(self.NG)] += self.eps
    Hk = 0.5 * (Hk + np.conj(np.transpose(Hk, (0, 2, 1))))
    w, v = np.linalg.eigh(Hk)
    c = v[:, :, 0]
    Dn = np.einsum('ka,kb->kab', c, c.conj())
    E = self.energy(Dn, self.hartree(Dn), self.fock(Dn))
    return dict(c=c, D=Dn, E=E, w=w, hist=hist)

  # ---------------------------------------------------------------- 初期値
  def seed_wc(self, sigma=0.5, center=(0.0, 0.0)):
    """格子点のガウス（s = +1 成分、幅 sigma）を下のバンドに射影した Bloch 和。"""
    q2 = np.sum(self.q ** 2, axis=-1)
    r0 = np.asarray(center)
    c = np.exp(-0.5 * sigma ** 2 * q2) * self.U[..., 0].conj() * np.exp(-1j * self.q @ r0)
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    return np.einsum('ka,kb->kab', c, c.conj())

  def seed_random(self, rng, sigma=0.5):
    q2 = np.sum(self.q ** 2, axis=-1)
    c = (rng.normal(size=q2.shape) + 1j * rng.normal(size=q2.shape)) * np.exp(-0.25 * sigma ** 2 * q2)
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    return np.einsum('ka,kb->kab', c, c.conj())

  # ---------------------------------------------------------------- チャーン数
  def chern(self, c):
    """Fukui–Hatsugai–Suzuki。セル周期部分の重なり <u_k|u_{k+d}> を、代表のずらしを入れて取る。
    符号：下のバンドそのもののベリー曲率 Omega = 2 lam^2/(1+lam^2 q^2)^2 > 0 の向きで正。"""
    L = self.L
    idx = {tuple(n): i for i, n in enumerate(self.kn)}
    def link(i, j, dn):
      # k_j の代表 = k_i + d - dG  →  dG = k_i + d - k_j（逆格子）
      d = (dn[0] * self.B[:, 0] + dn[1] * self.B[:, 1]) / L
      dG = self.k[i] + d - self.k[j]
      dgi = np.rint(np.linalg.solve(self.B, dG)).astype(int)
      tot = 0j
      for g in range(self.NG):
        t = tuple(self.gint[g] + dgi)
        h = self._lut.get(t, -1)
        if h < 0:
          continue
        F = np.sum(self.U[i, g].conj() * self.U[j, h])
        tot += np.conj(c[i, g]) * c[j, h] * F
      return tot
    tot = 0.0
    for (n1, n2), i in idx.items():
      i10 = idx[((n1 + 1) % L, n2)]; i01 = idx[(n1, (n2 + 1) % L)]; i11 = idx[((n1 + 1) % L, (n2 + 1) % L)]
      u1 = link(i, i10, (1, 0)); u2 = link(i10, i11, (0, 1))
      u3 = link(i01, i11, (1, 0)); u4 = link(i, i01, (0, 1))
      tot += np.angle(u1 * u2 * np.conj(u3) * np.conj(u4))
    # 反時計回りのループで、重なりの積の位相 = +(Omega の束)（下のバンドのスピノル (1, lam q_+) で確認）。
    # この符号の付け方では下のバンドそのものの束は +2 pi（A = i<u|grad u> の向きの付け方とは符号が逆）。
    orient = np.sign(self.B[0, 0] * self.B[1, 1] - self.B[1, 0] * self.B[0, 1])
    return orient * tot / (2 * np.pi)

  def density_fourier(self, c):
    """|<rho(g)>|/N（最短の逆格子 6 本の平均）：結晶の秩序の目安。"""
    D = np.einsum('ka,kb->kab', c, c.conj())
    rho = self.rho(D)
    shell = [tuple(t) for t in self.gint[1:7]]
    return float(np.mean([abs(rho[t]) for t in shell]) / self.N)

  def orbitals_text(self, c):
    """VMC 用の軌道の表：各行 k_x k_y G_x G_y Re c Im c（36 x NG 行）。"""
    rows = []
    for kk in range(self.Nk):
      for g in range(self.NG):
        rows.append([self.k[kk, 0], self.k[kk, 1], self.G[g, 0], self.G[g, 1], c[kk, g].real, c[kk, g].imag])
    return np.array(rows)
