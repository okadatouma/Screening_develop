# -*- coding: utf-8 -*-
"""輪切り窓を「輪の指示関数を核 k でぼかしたもの」とみて、核ごとに比べる（doc/ring_kernels.pdf の数値）。

    m_h(d) = S(d - b_h) - S(d - b_{h+1}),   S' = k,   b_h = h * Delta,   Delta = d_ref / 8
    f(d)   = sum_h c_h m_h(d)   （ln psi に入る和。g は 1 とおく）

核：いまの半周期 sin の階段（tau = 0.5, 1）、三角核（2 次 B スプラインの窓）、raised cosine 核（幅 2 Delta）、
2 次 B スプラインの核（3 次 B スプラインの窓）。調べること：
  1. 1 の分割 sum_h m_h = 1
  2. 直線の再現 sum_h x_h m_h = d（x_h は輪の中心）と、Poisson 和での条件 sum_h k(d - h Delta) = 1/Delta
  3. 窓の max|m''| と半値全幅
  4. c_h = h（直線）、c_h = cos(2 pi x_h / a)（周期 a のなめらかな形）、c_h 独立な乱数、それぞれの f'' の大きさ

    python screening_analysis/check_ring_kernels.py
"""
import numpy as np

DREF, A = 3.2152, 1.9046
DL = DREF / 8                                   # 輪の幅 Delta
H = 8


def b2(t):
    """2 次 B スプライン（台 [0, 3]、積分 1）。"""
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    m = (t >= 0) & (t < 1); out[m] = 0.5 * t[m] ** 2
    m = (t >= 1) & (t < 2); out[m] = 0.5 * (-2 * t[m] ** 2 + 6 * t[m] - 3)
    m = (t >= 2) & (t <= 3); out[m] = 0.5 * (3 - t[m]) ** 2
    return out


def kernels():
    """名前 -> 核 k(u)（積分 1）。階段 S は k の積分。"""
    def halfsine(tau):
        l = tau * DL
        return lambda u: np.where(np.abs(u) <= l / 2, np.pi / (2 * l) * np.cos(np.pi * u / l), 0.0)
    return {
        "halfsine tau=0.5": halfsine(0.5),
        "halfsine tau=1": halfsine(1.0),
        "hat (quad B)": lambda u: np.where(np.abs(u) <= DL, (1 - np.abs(u) / DL) / DL, 0.0),
        "raised-cos 2D": lambda u: np.where(np.abs(u) <= DL, (1 + np.cos(np.pi * u / DL)) / (2 * DL), 0.0),
        "B2 (cubic B)": lambda u: b2(u / DL + 1.5) / DL,
    }


def step(k):
    """S(u) = int_{-inf}^u k。"""
    u = np.linspace(-3 * DL, 3 * DL, 60001)
    su = np.concatenate([[0.0], np.cumsum(0.5 * (k(u[1:]) + k(u[:-1])) * np.diff(u))])
    return lambda x: np.interp(x, u, su, left=0.0, right=1.0)


def windows(k, d):
    """m_h = (chi_h * k)(d) + (chi_h * k)(-d)。輪 h = [b_h, b_{h+1}]、b_0 = 0、いちばん外の輪は外へ無限に延ばす
    （外側は g が切る）。d = 0 で折り返すので m_h'(0) = 0。核の半幅が Delta 以下なら折り返しは効かず、
    コードの「端の輪は片側だけ切る」と同じになる。"""
    S = step(k)
    m = []
    for h in range(H):
        lo, hi = h * DL, (np.inf if h == H - 1 else (h + 1) * DL)
        ring = lambda x: S(x - lo) - (0.0 if np.isinf(hi) else S(x - hi))
        m.append(ring(d) + ring(-d))
    return np.array(m)


def d2(y, x):
    return np.gradient(np.gradient(y, x, axis=-1), x, axis=-1)


def fwhm(y, x):
    top = y.max(); above = x[y >= top / 2]
    return above.max() - above.min()


def main():
    d = np.linspace(0, DREF, 32001)
    inner = (d > 2.5 * DL) & (d < 5.5 * DL)           # 端の輪の影響を受けない範囲
    xc = (np.arange(H) + 0.5) * DL                       # 輪の中心
    rng = np.random.default_rng(1)
    cr = rng.normal(size=(4000, H))
    print(f"Delta = {DL:.4f}")
    print(f"{'kernel':18s} {'PU err':>8s} {'lin err':>8s} {'k-periodic':>11s} {'max|m''|':>9s} {'FWHM/D':>7s}"
          f" {'lin f''':>8s} {'cos f''':>8s} {'rand rms':>9s} {'rand max':>9s}")
    for name, k in kernels().items():
        m = windows(k, d)
        m2 = d2(m, d)
        pu = np.abs(m.sum(0) - 1)[inner].max()
        lin = np.abs((xc[:, None] * m).sum(0) - d)[inner].max()
        per = sum(k(d - h * DL) for h in range(-3, H + 4)) * DL      # Delta sum_h k(d - h Delta)
        kp = (per[inner].max() - per[inner].min())
        mm = np.abs(m2[:, inner]).max()
        fw = fwhm(m[3], d) / DL
        f_lin = np.abs((np.arange(H)[:, None] * m2).sum(0))[inner].max()
        f_cos = np.abs((np.cos(2 * np.pi * xc / A)[:, None] * m2).sum(0))[inner].max()
        f_q = np.abs((xc[:, None] ** 2 * m2).sum(0) - 2)[inner].max()          # 放物線 c_h = x_h^2 なら f'' = 2
        d0 = np.abs(np.gradient(m, d, axis=-1)[:, 0]).max()                      # d = 0 での傾き
        fr = cr @ m2[:, inner]
        print(f"{name:18s} {pu:8.1e} {lin:8.3f} {kp:11.3f} {mm:9.2f} {fw:7.2f} {f_lin:8.2f} {f_cos:8.2f}"
              f" {np.sqrt((fr**2).mean()):9.2f} {np.abs(fr).max(1).mean():9.2f}   quad {f_q:6.2f}  m'(0) {d0:.1e}")
    print(f"参考：真の cos(2 pi d / a) の max|f''| = (2 pi / a)^2 = {(2*np.pi/A)**2:.2f}")
    print(f"式：pi^2/(2 Delta^2) = {np.pi**2/(2*DL**2):.2f},  pi/Delta^2 = {np.pi/DL**2:.2f},  2/Delta^2 = {2/DL**2:.2f}")


def table():
    """表の数値：輪が両側に続くとした場合（図と同じ）。d は [0, d_ref] 全体。"""
    d = np.linspace(0, DREF, 16001)
    hs = np.arange(-4, 12)
    xc = (hs + 0.5) * DL
    rng = np.random.default_rng(2)
    cr = rng.normal(size=(4000, len(hs)))
    print("表（輪が両側に続く場合、d in [0, d_ref]）")
    print(f"{'kernel':18s} {'max|m2|':>8s} {'FWHM/D':>7s} {'peak':>6s} {'lin':>7s} {'cos':>7s} {'rand rms':>9s}")
    for name, k in kernels().items():
        St = step(k)
        m = np.array([St(d - h * DL) - St(d - (h + 1) * DL) for h in hs])
        m2 = d2(m, d)
        f_lin = np.abs((hs[:, None] * m2).sum(0)).max()
        f_cos = np.abs((np.cos(2 * np.pi * xc / A)[:, None] * m2).sum(0)).max()
        fr = cr @ m2
        i3 = list(hs).index(3)
        print(f"{name:18s} {np.abs(m2).max():8.2f} {fwhm(m[i3], d)/DL:7.2f} {m[i3].max():6.3f} {f_lin:7.2f}"
              f" {f_cos:7.2f} {np.sqrt((fr**2).mean()):9.2f}")
    print(f"真の cos：{(2*np.pi/A)**2:.2f}")


if __name__ == "__main__":
    main()
    table()
