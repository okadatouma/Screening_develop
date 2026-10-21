"""4 つの追加検証。

    python screening_analysis/followups.py

(A) 実効レンジ r_eff — 「窓の名目幅 w」ではなく「実際に重みが乗っている距離」。
        r_eff = Σ_{i≠j} W_ij d_ij / Σ_{i≠j} W_ij
    w が相転移で動かなくても、r_eff は動くかもしれない（窓の中身が変わる）。
(B) s の分散分解を「実際に使われる対」に限定して測り直す。
    Trim は s ≤ 1−r を厳密に 0 に潰すので、s 全体の分散には
    「そもそも波動関数に効かない領域」が混ざっている。α(s)>0 の対だけで測る。
(C) softmax（Psiformer）での R²_距離 — 対照。
    「窓が距離を担当するから s は距離を学ばない」なら、窓のない softmax では R² が高いはず。
(D) ヘッド寄与の冗長度 — Σ_h‖c_h‖ と ‖Σ_h c_h‖ の比。
    ベクトルが打ち消し合っていれば比は 1 より大きい。
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import jax                                                     # noqa: E402
from attn_probe import probe as sprobe                          # noqa: E402
from psi_probe import probe as pprobe                           # noqa: E402

ROOT = os.path.dirname(HERE)
SF = os.path.join(ROOT, "results/2deg-Coulomb/CustomScreenformer/%s")
PF = os.path.join(ROOT, "results/2deg-Coulomb/CustomPsiformer/%s")
GS = os.path.join(ROOT, "gpu/results/2deg-Coulomb/CustomScreenformer/%s")
GP = os.path.join(ROOT, "gpu/results/2deg-Coulomb/CustomPsiformer/%s")


def mininum_image(pos, L, NE):
    R = pos.reshape(pos.shape[0], NE, 2)
    dr = R[:, :, None, :] - R[:, None, :, :]
    f = dr @ np.linalg.inv(L).T
    return np.linalg.norm((f - np.round(f)) @ L.T, axis=-1)


def r_eff(a):
    """重み付き平均距離（非対角のみ）。周期ノルムと最小像 Euclid の両方。"""
    NE, off = a["NE"], ~np.eye(a["NE"], dtype=bool)
    dpn = a["d"][:, off]                                        # (B, P)
    deu = mininum_image(a["positions"], a["lattice"], NE)[:, off]
    W = a["W"][:, :, :, off]                                    # (B, L, H, P)
    num_p = (W * dpn[:, None, None, :]).sum(-1)
    num_e = (W * deu[:, None, None, :]).sum(-1)
    den = W.sum(-1)
    ok = den > 1e-12
    rp = np.where(ok, num_p / np.maximum(den, 1e-12), np.nan)
    re = np.where(ok, num_e / np.maximum(den, 1e-12), np.nan)
    return np.nanmean(rp, 0), np.nanmean(re, 0)                 # (L,H) each


def r2_decomp(a, mask=None, nb=20):
    """(L,H) ごとに s を距離ビン / スピン / 両方で説明した R²。mask は使う対の bool。"""
    NE, off = a["NE"], ~np.eye(a["NE"], dtype=bool)
    d = a["d"][:, off]
    sp = np.asarray(a["spins"])
    par = (sp[:, np.where(off)[0]] * sp[:, np.where(off)[1]] > 0)
    NL, NH = a["NL"], a["NH"]
    out = np.zeros((NL, NH, 3)); frac = np.zeros((NL, NH))
    edges = np.linspace(d.min(), d.max() + 1e-9, nb + 1)
    db = np.clip(np.digitize(d, edges) - 1, 0, nb - 1)
    for l in range(NL):
        for h in range(NH):
            y = a["S"][:, l, h][:, off]
            m = np.ones_like(y, bool) if mask is None else mask[:, l, h]
            frac[l, h] = m.mean()
            yy, bb, pp = y[m], db[m] if db.shape == y.shape else db[m], par[m]
            if yy.size < 200 or yy.std() < 1e-9:
                out[l, h] = np.nan; continue
            v = yy.var()
            for k, code in enumerate([bb, pp.astype(int), bb * 2 + pp.astype(int)]):
                pred = np.zeros_like(yy)
                for c in np.unique(code):
                    s = code == c
                    pred[s] = yy[s].mean()
                out[l, h, k] = 1 - ((yy - pred) ** 2).mean() / v
    return out, frac


def show(tag, a, mask=None):
    r2, fr = r2_decomp(a, mask)
    m = np.nanmean(r2.reshape(-1, 3), 0)
    print("    %-34s R²_距離 = %.3f   R²_スピン = %.3f   R²_両方 = %.3f   残り = %.3f   使った対 %.0f%%"
          % (tag, m[0], m[1], m[2], 1 - m[2], 100 * fr.mean()))
    return m


print("\n=== (A) 実効レンジ r_eff — 名目の窓 w と、実際に重みが乗っている距離 ===")
print("    N=9 L4H4 種固定。r_eff は非対角の重み付き平均距離。括弧内は最小像 Euclid。\n")
print("      r_s   w̄(名目)  r_eff(周期ノルム)  r_eff(Euclid)   寄与重み付き r_eff")
rows = {}
for rs in [1, 5, 20, 30, 35, 40]:
    a = sprobe(SF % ("el5_4_rs%d_tri_det" % rs), nconf=64)
    rp, re = r_eff(a)
    C = np.load(os.path.join(SF % ("el5_4_rs%d_tri_det" % rs), "head_contrib.npy"))[0]
    ok = np.isfinite(rp)
    wc = float((C[ok] * rp[ok]).sum() / C[ok].sum())
    rows[rs] = (a["w"].mean(), np.nanmean(rp), np.nanmean(re), wc, rp, a["w"] / a["dref"])
    print("      %-5d %7.3f %13.3f %15.3f %17.3f"
          % (rs, a["w"].mean(), np.nanmean(rp), np.nanmean(re), wc), flush=True)
    if rs in (5, 40):
        globals()["A%d" % rs] = a
    jax.clear_caches()

print("\n=== (B0) Trim は実際に何かを切っているか ===\n")
for rs in [5, 40]:
    a = globals()["A%d" % rs]
    off = ~np.eye(a["NE"], dtype=bool); S = a["S"][:, :, :, off]
    thr = (1.0 - a["r"])[None, :, :, None]
    print("  r_s=%-3d  Trim 半径 r = %.3f ± %.3f  →  閾値 1−r = %.3f   s の範囲 [%.3f, %.3f]   切られた対 %.2f%%"
          % (rs, a["r"].mean(), a["r"].std(), (1 - a["r"]).mean(), S.min(), S.max(),
             100 * (S < thr).mean()))

print("\n=== (B) s の分散分解 — 全対 vs 実際に使われる対（α(s)>0）===\n")
for rs in [5, 40]:
    a = globals()["A%d" % rs]
    off = ~np.eye(a["NE"], dtype=bool)
    thr = (1.0 - a["r"])[None, :, :, None]                       # (1,L,H,1)
    mk = a["S"][:, :, :, off] > thr
    print("  r_s = %d  (N=9, 遮蔽)" % rs)
    show("全対", a)
    show("α(s)>0 の対だけ", a, mk)
    print()

print("=== (C) softmax との対照（N=49, r_s=5）===\n")
big_s = sprobe(GS % "el25_24_rs5_tri_l2h16", nconf=48); jax.clear_caches()
off = ~np.eye(big_s["NE"], dtype=bool)
mk = big_s["S"][:, :, :, off] > (1.0 - big_s["r"])[None, :, :, None]
print("  遮蔽 (Screenformer L2H16)")
show("全対", big_s); show("α(s)>0 の対だけ", big_s, mk)
big_p = pprobe(GP % "el25_24_rs5_tri_psi_l2h16", nconf=48); jax.clear_caches()
offp = ~np.eye(big_p["NE"], dtype=bool)
mkp = big_p["W"][:, :, :, offp] > (1.0 / big_p["NE"])
print("\n  softmax (Psiformer L2H16)")
show("全対", big_p); show("重み > 1/N の対だけ", big_p, mkp)

print("\n=== (D) ヘッド寄与の冗長度と分布の平坦さ ===\n")
for nm, fold in [("N=49 L2H16", GS % "el25_24_rs5_tri_l2h16"),
                 ("N=9  L4H4 rs5", SF % "el5_4_rs5_tri_det")]:
    C = np.load(os.path.join(fold, "head_contrib.npy"))[0]
    sh = 100 * C / C.sum()
    n = C.size
    print("  %-16s ヘッド総数 %2d  均等なら %.1f%%   最大 %.1f%% (=均等の %.2f 倍)  最小 %.1f%%"
          % (nm, n, 100 / n, sh.max(), sh.max() / (100 / n), sh.min()))
    print("  %-16s 上位 5 本の合計 %.1f%%   下位半分の合計 %.1f%%   ジニ係数 %.3f"
          % ("", np.sort(sh.ravel())[::-1][:5].sum(), np.sort(sh.ravel())[:n // 2].sum(),
             float(np.abs(np.subtract.outer(sh.ravel(), sh.ravel())).mean() / (2 * sh.mean()))))
