# -*- coding: utf-8 -*-
"""ガウスでぼかした輪切り（輪の指示関数 * ガウス核）の設計値を調べる（doc/gauss_rings.pdf の数値）。

    m_h(d) = T_h(d) - T_{h+1}(d),   T_k(d) = Phi((d - b_k)/sigma) + Phi((-d - b_k)/sigma)   （d = 0 で折り返し）
    T_0 = 1、T_H = 0（いちばん外の輪は外へ延ばし、g が切る）

sigma / Delta を振って、半値全幅・山の高さ・max|m''|・直線（c_h = h）での f'' の振幅・cos 形・乱数での f'' を、
いまの輪切り（tau = 1）と 3 次 B スプラインに並べる。直線での振幅はポアソン和の最低次
(4 pi / Delta^2) exp(-2 pi^2 sigma^2 / Delta^2) と比べる。

    python screening_analysis/check_gauss_rings.py
"""
import numpy as np
from scipy.special import ndtr
from check_ring_kernels import DREF, A, DL, H, kernels, step, d2, fwhm


def gauss_windows(d, sigma, hs):
    """輪が両側に続く場合（折り返しなし）の m_h。"""
    S = lambda x: ndtr(x / sigma)
    return np.array([S(d - h * DL) - S(d - (h + 1) * DL) for h in hs])


def gauss_windows_folded(d, sigma):
    """実装と同じ形：b_0 = 0 で折り返し、いちばん外の輪は外へ延ばす。d >= 0。"""
    b = np.arange(H + 1) * DL
    T = [np.ones_like(d)]
    for k in range(1, H):
        T.append(ndtr((d - b[k]) / sigma) + ndtr((-d - b[k]) / sigma))
    T.append(np.zeros_like(d))
    return np.array([T[h] - T[h + 1] for h in range(H)])


def stats(m, m2, hs, d, cr):
    i3 = list(hs).index(3)
    xc = (hs + 0.5) * DL
    e = slice(5, -5)                                   # 端の数点は片側差分の誤差なので除く
    m2 = m2[:, e]
    f_lin = np.abs((hs[:, None] * m2).sum(0)).max()
    f_cos = np.abs((np.cos(2 * np.pi * xc / A)[:, None] * m2).sum(0)).max()
    f_q = np.abs((xc[:, None] ** 2 * m2).sum(0) - 2).max()
    fr = cr @ m2
    return (fwhm(m[i3], d) / DL, m[i3].max(), np.abs(m2).max(), f_lin, f_cos, f_q,
            np.sqrt((fr ** 2).mean()))


def main():
    d = np.linspace(0, DREF, 16001)
    hs = np.arange(-4, 12)
    rng = np.random.default_rng(2)
    cr = rng.normal(size=(4000, len(hs)))
    print(f"Delta = {DL:.4f}")
    print(f"{'window':22s} {'FWHM/D':>7s} {'peak':>6s} {'max|m2|':>8s} {'lin':>7s} {'Poisson':>8s} {'cos':>7s}"
          f" {'quad':>7s} {'rand':>7s}")
    ks = kernels()
    for name in ("halfsine tau=1", "B2 (cubic B)"):
        St = step(ks[name])
        m = np.array([St(d - h * DL) - St(d - (h + 1) * DL) for h in hs])
        s = stats(m, d2(m, d), hs, d, cr)
        print(f"{name:22s} {s[0]:7.2f} {s[1]:6.3f} {s[2]:8.2f} {s[3]:7.2f} {'':>8s} {s[4]:7.2f} {s[5]:7.2f} {s[6]:7.2f}")
    for sf in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
        m = gauss_windows(d, sf * DL, hs)
        s = stats(m, d2(m, d), hs, d, cr)
        pois = 4 * np.pi / DL ** 2 * np.exp(-2 * np.pi ** 2 * sf ** 2)
        print(f"{'gauss sigma=%.2f D' % sf:22s} {s[0]:7.2f} {s[1]:6.3f} {s[2]:8.2f} {s[3]:7.2f} {pois:8.2f}"
              f" {s[4]:7.2f} {s[5]:7.2f} {s[6]:7.2f}")
    print(f"真の cos：{(2*np.pi/A)**2:.2f}、放物線の真の f'' = 2（quad はそこからのずれの最大）")

    print("実装の形（折り返し・外へ延ばす）での確認、sigma = 0.5 Delta")
    dd = np.linspace(0, DREF, 64001)
    m = gauss_windows_folded(dd, 0.5 * DL)
    print(f"   1 の分割のずれ max|sum m - 1| = {np.abs(m.sum(0) - 1).max():.1e}")
    m1 = np.gradient(m, dd, axis=-1)
    print(f"   d = 0 での傾き max|m_h'(0)| = {np.abs(m1[:, 0]).max():.1e}（折り返しで 0）")
    print(f"   最小値 min m_h = {m.min():.1e}（負にならない）")
    # 窓の外への裾：輪 h の中心から 2 Delta 離れたところの値
    xc = (np.arange(H) + 0.5) * DL
    tail = max(m[h][np.abs(dd - (xc[h] + 2 * DL)).argmin()] for h in range(1, H - 2))
    print(f"   中心から 2 Delta 離れた裾の値 {tail:.1e}")


if __name__ == "__main__":
    main()
