"""[lj3] 1 層目の q̄·k̄ の検査。lj2（入力 sin, cos, s）と lj3（12 成分）のチェックポイントのどちらにも使える。

    PYTHONPATH=<コードのフォルダ> python screening_analysis/lj3/input12_check.py <run フォルダ> [...] [--out x.json]

1. 対称性：全員の擬スピンを反転し、全員をセルの半分 (L_1 + L_2)/2 ずらしたときの max|Δ(q̄·k̄)|
   （チェックポイントのウォーカー 32 本、全ヘッド）。従来の入力では埋め込みにも q, k にもバイアスがないので、
   この操作で入力が −1 倍になり、学習した後でも 0 になる。12 成分の入力では 0 でなくなる。
2. 位置の周期：電子 i の位置を乱数で 8 点取り、相手 j の位置をセルの 96 x 96 の格子で動かして q̄_i·k̄_j を FFT する。
   一様な成分を除いた強さのうち、セル全体の波長の成分（|n_1|, |n_2| ≤ 1）と結晶の周期の成分（n_1, n_2 が 6 の倍数）の割合。
3. 12 成分のとき：埋め込みの重みの大きさ（sin・cos の 4 行、高調波の 6 行、one-hot の和 w_+ + w_-（定数項）と
   差の半分 (w_+ − w_-)/2（擬スピンの項））。初期値 'match' では高調波の行と w_+ + w_- が 0 から出発する。
"""
import argparse
import glob
import json
import os
import re
import types

import numpy as np
import jax
import jax.numpy as jnp

from periodicwave import CustomScreenformer as CS
from periodicwave.pbc import feature_layer as FL

NG = 96


def lattice_of(cfg):
    v = cfg["system"].get("pbc_lattice")
    if isinstance(v, list):
        return np.asarray(v, dtype=np.float64)
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", str(v))][:4]
    return np.asarray(nums, dtype=np.float64).reshape(2, 2)


def load(folder, step=None):
    cks = sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))
    if not cks:
        raise FileNotFoundError(f"チェックポイントがない: {folder}")
    ck = cks[-1]
    if step is not None:
        ok = [c for c in cks if int(re.findall(r"(\d+)\.npz$", c)[0]) <= step]
        ck = ok[-1] if ok else cks[0]
    z = np.load(ck, allow_pickle=True)
    params = jax.tree_util.tree_map(lambda x: np.asarray(x)[0], z["params"].item())
    data = z["data"].item()
    cfg = json.load(open(os.path.join(folder, "config.json")))
    return ck, params, data, cfg


def make_qk(params, cfg, L):
    cs = cfg["network"]["CustomScreenformer"]
    opts = types.SimpleNamespace(input_mode=cs.get("input_mode", "sincos_s"),
                                 input_harmonics=cs.get("input_harmonics"), pbc_lattice=L)
    nh, ad = int(cs["num_heads"]), int(cs["attn_dim"])
    P = params["layers"]
    at = P["attention"][0]
    inv_l = jnp.asarray(np.linalg.inv(L))

    def features(ae):
        """特徴層（原子 1 個、include_r_ae=False）と同じ (sin 2πu, cos 2πu)、u = A^{-1} r。"""
        u = ae[:, 0, :] @ inv_l.T
        return jnp.concatenate((jnp.sin(2 * jnp.pi * u), jnp.cos(2 * jnp.pi * u)), axis=-1)

    # 特徴層そのものと一致することを確かめておく（電子 5 個）
    _ae = jnp.asarray(np.random.default_rng(1).normal(size=(5, 1, 2)) * 3.0)
    _fl = FL.make_pbc_feature_layer(natoms=1, nspins=(5, 0), ndim=2, lattice=jnp.asarray(L), include_r_ae=False)
    _ref, _ = _fl.apply(ae=_ae, r_ae=None, ee=_ae[:, None, 0, :] - _ae[None, :, 0, :], r_ee=None)
    assert float(jnp.max(jnp.abs(_ref - features(_ae)))) < 1e-5

    @jax.jit
    def qk(ae, spins):
        """ae: [M, 1, 2]（原子からの位置）、spins: [M] -> 単位長の q̄, k̄ [M, H, D]。"""
        x = CS.one_electron_inputs(opts, features(ae), ae, spins) @ P["embed"]
        q = (x @ at["q_w"]).reshape(-1, nh, ad)
        k = (x @ at["k_w"]).reshape(-1, nh, ad)
        return CS._unit_length(q), CS._unit_length(k)

    return qk, opts, nh


def shares(field):
    """一様な成分を除いた |FFT|^2 のうち、結晶の周期の成分とセル全体の波長の成分の割合。"""
    n = np.fft.fftfreq(NG, 1.0 / NG).astype(int)
    n1, n2 = np.meshgrid(n, n, indexing="ij")
    zero = (n1 == 0) & (n2 == 0)
    cryst = (n1 % 6 == 0) & (n2 % 6 == 0) & ~zero
    first = (np.maximum(np.abs(n1), np.abs(n2)) <= 1) & ~zero
    p = np.abs(np.fft.fft2(field - field.mean())) ** 2
    tot = p.sum()
    if tot < 1e-20 * NG ** 4:
        return np.nan, np.nan
    return p[cryst].sum() / tot, p[first].sum() / tot


def check(folder, step=None):
    ck, params, data, cfg = load(folder, step)
    L = lattice_of(cfg)
    qk, opts, nh = make_qk(params, cfg, L)
    out = {"folder": folder, "ckpt": os.path.basename(ck), "input_mode": opts.input_mode}
    R = np.asarray(data["positions"])[0]
    nw = min(32, R.shape[0])
    R = R[:nw].reshape(nw, -1, 2)
    S = np.asarray(data["spins"])[0][:nw]
    atom = np.asarray(data["atoms"])[0][0][0]
    shift = 0.5 * (L[:, 0] + L[:, 1])
    dmax = 0.0
    for w in range(R.shape[0]):
        ae = jnp.asarray((R[w] - atom)[:, None, :])
        ae2 = jnp.asarray((R[w] + shift - atom)[:, None, :])
        qa, ka = qk(ae, jnp.asarray(S[w]))
        qb, kb = qk(ae2, jnp.asarray(-S[w]))
        sa = np.einsum("ihd,jhd->hij", qa, ka)
        sb = np.einsum("ihd,jhd->hij", qb, kb)
        dmax = max(dmax, float(np.max(np.abs(sa - sb))))
    out["flip_shift_max_dqk"] = dmax
    g = (np.arange(NG) + 0.5) / NG
    U = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    aeg = jnp.asarray((U @ L.T)[:, None, :])                     # r = L u（原子から）
    rng = np.random.default_rng(0)
    ui = rng.random((8, 2))
    aei = jnp.asarray((ui @ L.T)[:, None, :])
    cr, fi = [], []
    for si in (1.0, -1.0):
        qi, _ = qk(aei, jnp.full((8,), si))
        for sj in (1.0, -1.0):
            _, kj = qk(aeg, jnp.full((NG * NG,), sj))
            s = np.einsum("ihd,jhd->ihj", qi, kj)                 # [8, H, NG^2]
            for a in range(8):
                for h in range(nh):
                    c, f = shares(s[a, h].reshape(NG, NG))
                    if not np.isnan(c):
                        cr.append(c)
                        fi.append(f)
    out["qk_share_crystal_mean"] = float(np.mean(cr))
    out["qk_share_crystal_max"] = float(np.max(cr))
    out["qk_share_cell_mean"] = float(np.mean(fi))
    W = np.asarray(params["layers"]["embed"])
    if opts.input_mode == "lj12":
        nh2 = 2 * len(opts.input_harmonics)
        wp, wm = W[4 + nh2], W[5 + nh2]
        out["embed_norm_sincos"] = float(np.linalg.norm(W[:4]))
        out["embed_norm_harmonics"] = float(np.linalg.norm(W[4:4 + nh2]))
        out["embed_norm_onehot_sum"] = float(np.linalg.norm(wp + wm))
        out["embed_norm_onehot_halfdiff"] = float(np.linalg.norm(0.5 * (wp - wm)))
    else:
        out["embed_norm_sincos"] = float(np.linalg.norm(W[:4]))
        out["embed_norm_spin"] = float(np.linalg.norm(W[4]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--step", type=int, default=None, help="この step 以下で最後のチェックポイントを使う")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = []
    for f in a.folders:
        r = check(f, a.step)
        res.append(r)
        print(f"{os.path.basename(f.rstrip('/'))}  {r['ckpt']}  入力 {r['input_mode']}")
        print(f"   反転＋半分ずらし max|Δ(q̄·k̄)| = {r['flip_shift_max_dqk']:.2e}")
        print(f"   q̄·k̄ の位置の周期：結晶の周期 平均 {r['qk_share_crystal_mean']:.2e}（最大 {r['qk_share_crystal_max']:.2e}）、"
              f"セル全体の波長 平均 {r['qk_share_cell_mean']:.3f}")
        print("   埋め込みの大きさ：" + "、".join(f"{k[11:]} {v:.3f}" for k, v in r.items() if k.startswith("embed_norm")))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
