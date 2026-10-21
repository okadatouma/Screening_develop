# -*- coding: utf-8 -*-
"""窓の 2 階微分と局所エネルギーの関係を数値で確かめる（doc/window_derivatives.pdf の数値）。

1. 実際の窓（CustomScreenformer と同じ式）の max|m''| と、目安 pi^2/(2 l^2) の比較
2. 受け渡しの恒等式 c_h m_h + c_{h+1} m_{h+1} = c_h g + (c_{h+1}-c_h) g Theta
3. 幅 l で 0 -> 1 に上がる C^1 関数の下限 max|m''| >= 4/l^2
4. 1 次元の例：ln psi に高さ s の段差（幅 l）。運動エネルギー 1/2 int|psi'|^2 / int|psi|^2 は 1/l で増え、
   l = 0 では点ごとの推定 <-psi''/(2 psi)> が段差を見落とす

    python screening_analysis/check_window_derivs.py
"""
import numpy as np

DREF, A = 3.2152, 1.9046
DR = DREF / 8


def theta(u, l):
    """幅 l で 0 から 1 へ上がる階段（l = 0 なら段差）。"""
    if l == 0:
        return (u >= 0).astype(float)
    return 0.5 * (1 + np.sin(np.pi * np.clip(u / l, -0.5, 0.5)))


def g(d):
    return 0.5 * (1 + np.cos(np.pi * np.clip(d / DREF, 0, 1) ** 3))


def ring(d, h, tau):
    l = tau * DR
    lo = 1.0 if h == 0 else theta(d - h * DR, l)            # 端の輪は片側だけ切る（コードと同じ）
    hi = 0.0 if h == 7 else theta(d - (h + 1) * DR, l)
    return g(d) * (lo - hi)


def nested(d, w):
    return 0.5 * (1 + np.cos(np.pi * np.minimum(d / w, 1.0)))


def shell(d, c, w):
    return 0.5 * (1 + np.cos(np.pi * np.clip((d - c) / w, -1, 1)))


def d2(y, x):
    return np.gradient(np.gradient(y, x), x)


def main():
    d = np.linspace(0, DREF, 200001)
    inner = d < DREF * 0.999                                 # d_ref の端点の差分を除く
    print("1. max|m''|（数値）と pi^2/(2 l^2)")
    wn = A * (DREF / A) ** (np.arange(16) / 15)                # 入れ子窓の幅（a から d_ref まで等比）
    mn = max(np.abs(d2(nested(d, w), d))[inner].max() for w in wn)
    print(f"   入れ子窓（16 本, w = {wn[0]:.3f}..{wn[-1]:.3f}）  {mn:8.2f}   目安 {np.pi**2/(2*A**2):8.2f}")
    for tau in (1.0, 0.5):
        l = tau * DR
        per = [np.abs(d2(ring(d, h, tau), d))[inner].max() for h in range(8)]
        print(f"   輪切り tau={tau}（l = {l:.3f}）  全ヘッド最大 {max(per):8.2f}   目安 {np.pi**2/(2*l**2):8.2f}"
              f"   輪 1.61-2.01 だけ {per[4]:7.2f}")
    ms = np.abs(d2(shell(d, 1.8, 0.114), d)).max()
    print(f"   殻窓（半幅 0.114）  {ms:8.2f}   目安 {np.pi**2/(2*0.114**2):8.2f}")

    print("2. 受け渡しの恒等式（輪 4 と 5 の境界 b = 5 DR の近く）")
    rng = np.random.default_rng(0)
    ch, ch1 = rng.normal(size=2)
    b = 5 * DR
    z = (d > b - 0.5 * DR) & (d < b + 0.5 * DR)
    lhs = ch * ring(d, 4, 0.5) + ch1 * ring(d, 5, 0.5)
    rhs = ch * g(d) + (ch1 - ch) * g(d) * theta(d - b, 0.5 * DR)
    print(f"   max|lhs - rhs| = {np.abs(lhs - rhs)[z].max():.2e}")

    print("3. C^1 で幅 l に 0 -> 1：max|m''| l^2 の下限 4 と raised cosine")
    print(f"   raised cosine: pi^2/2 = {np.pi**2/2:.3f}（下限 4 の {np.pi**2/8:.3f} 倍）")

    print("4. 1 次元の例：周期 2 pi、ln psi = s [Theta_l(x - pi/2) - Theta_l(x - 3pi/2)]、s = 0.5")
    s, L = 0.5, 2 * np.pi
    x = np.linspace(0, L, 2_000_001)[:-1]
    dx = x[1] - x[0]
    for l in (0.4, 0.2, 0.1, 0.05, 0.0):
        u = s * (theta(x - np.pi / 2, l) - theta(x - 3 * np.pi / 2, l))
        psi = np.exp(u)
        w = psi ** 2 / np.sum(psi ** 2)
        if l > 0:
            u1 = np.gradient(u, dx)
            u2 = np.gradient(u1, dx)
            el = -0.5 * (u2 + u1 ** 2)                     # 点ごとの局所エネルギー（運動のみ）
            t_true = 0.5 * np.sum(w * u1 ** 2)             # 1/2 int|psi'|^2 / int|psi|^2
            print(f"   l = {l:4.2f}:  1/2<|psi'|^2> = {t_true:7.4f}   <E_L> = {np.sum(w*el):7.4f}"
                  f"   sigma(E_L) = {np.sqrt(np.sum(w*el**2) - np.sum(w*el)**2):7.3f}   l * T = {l*t_true:.4f}")
        else:
            el = np.zeros_like(x)                           # 段差以外では u' = u'' = 0
            print(f"   l = 0   :  1/2<|psi'|^2> = 無限大   点ごとの <E_L> = {np.sum(w*el):7.4f}（段差を見落とす）")


if __name__ == "__main__":
    main()
