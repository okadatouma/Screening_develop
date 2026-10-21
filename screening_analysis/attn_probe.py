"""学習済みチェックポイントから注意の内部量を復元する。

    python screening_analysis/attn_probe.py results/2deg-Coulomb/CustomScreenformer/el5_4_rs5_tri_soft

やっていること: config.json の格子とチェックポイントの params を使い、repo 本体の
feature_layer / attention をそのまま再実行して、各層・各ヘッドで

    s      = q̄·k̄               （内容の類似度）
    m      = 遮蔽窓 m(d_ee)
    weight = α(s)·m(d)          （実際に v にかかる重み）

を記録する。出力は同フォルダの attn_probe.npz と、標準出力の要約表。

注意: d_ee は networks.make_periodic_construct_input_features の周期ノルム。
モジュール直下の Euclid 版 construct_input_features とは値が桁で違うので取り違えないこと。
"""
import glob
import json
import os
import re
import sys

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from periodicwave import networks, CustomScreenformer as CS      # noqa: E402
from periodicwave.pbc import feature_layer as FL                 # noqa: E402


def lattice_from_config(cfg):
    """config.json は numpy 配列を文字列で保存するので数値を拾い直す。列がベクトル。"""
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    v = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))]
    return np.array(v[:4]).reshape(2, 2)



def _centres_from(P, dref, NL):
    """checkpoint に mask_centre があれば (centres, sigmas)/d_ref を返す。

    make_screening_attention は centres が None かどうかで mask_centre を
    作る/作らないを決めるので、学習済みの構造を再現するために必要。値は
    どのみち P から上書きされるので、形さえ合っていればよい。
    """
    a = P["layers"]["attention"][0]
    if "mask_centre" not in a:
        return None
    c = np.asarray(a["mask_centre"], dtype=np.float64)
    sg = np.log1p(np.exp(np.asarray(a["mask_omega"], dtype=np.float64)))
    return (c, sg)


def probe(folder, checkpoint=None, trim_max=None, nconf=None):
    ck = checkpoint or sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))[-1]
    z = np.load(ck, allow_pickle=True)
    data = z["data"].item()
    P = jax.tree_util.tree_map(lambda a: np.asarray(a)[0], z["params"].item())
    cfg = json.load(open(os.path.join(folder, "config.json")))
    o = cfg["network"]["CustomScreenformer"]
    L = lattice_from_config(cfg)
    NH, AD, VD, MD, NL = o["num_heads"], o["attn_dim"], o["value_dim"], o["mlp_dim"], o["num_layers"]
    NE = sum(cfg["system"]["electrons"])
    dref = CS._max_periodic_distance(L)
    CS.TRIM_MAX = trim_max if trim_max is not None else getattr(CS, "TRIM_MAX", 2.0)

    fl = FL.make_pbc_feature_layer(natoms=1, nspins=tuple(cfg["system"]["electrons"]), ndim=2,
                                   lattice=jnp.asarray(L), include_r_ae=False)
    cif = networks.make_periodic_construct_input_features(jnp.asarray(L))
    _, attn = CS.make_screening_attention(
        NH, MD, AD, VD, NE, dref, use_softmask=o["use_softmask"], use_gate=o["use_gate"],
        use_tanh_norm=o["use_tanh_norm"], init_trim_r=min(o["init_trim_r"], CS.TRIM_MAX - 1e-3),
        init_softmask_w_factor=o["init_softmask_w_factor"],
        init_softmask_w_spread=o["init_softmask_w_spread"],
        init_softmask_centres=_centres_from(P, dref, NL),
        use_fixed_value=bool(o.get("use_fixed_value", False)), pair_dim=5,
        use_output_scale=o["use_output_scale"], init_output_scale=o["init_output_scale"])
    fixv = bool(o.get("use_fixed_value", False))
    _, ln = CS.make_layer_norm()
    _, mlp = CS.make_mlp(o["num_perceptrons_per_layer"], o["mlp_activation_fct"])

    def forward(pos, spins, atoms):
        ae, ee, r_ae, r_ee = cif(pos, atoms, ndim=2)
        f, eef = fl.apply(ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee)
        f = jnp.concatenate((f, spins[..., None]), axis=-1)
        x = jnp.dot(f, P["layers"]["embed"])
        d_ee = r_ee[..., 0]
        # 固定 value のとき運ぶ対の特徴 e(r_ij)（ネットワーク側と同じ作り方）
        pf = jnp.concatenate((eef[..., :1] / dref, eef[..., 1:]), axis=-1) if fixv else None
        S, W = [], []
        for l in range(NL):
            ap = P["layers"]["attention"][l]
            q = jnp.dot(x, ap["q_w"]).reshape(NE, NH, AD)
            k = jnp.dot(x, ap["k_w"]).reshape(NE, NH, AD)
            s = jnp.einsum("ihd,jhd->hij", CS._unit_length(q), CS._unit_length(k))
            r = (CS.TRIM_MAX * jax.nn.sigmoid(ap["trim_rho"]))[:, None, None]
            al = jnp.square(jax.nn.relu(s - (1.0 - r)) / r)
            if o["use_softmask"]:
                sg = dref * jax.nn.softplus(ap["mask_omega"])
                if "mask_rings" in ap:          # 全体窓 g の輪切り
                    bb = dref * ap["mask_rings"]; dl = sg[0]
                    # 階段は本体と同じ関数で作る（ガウス核の輪切りなら "mask_gauss" がある）
                    th = (CS._ring_steps(d_ee, bb, dl * ap["mask_gauss"], gauss=True) if "mask_gauss" in ap
                          else CS._ring_steps(d_ee, bb, dl))
                    xx = jnp.clip(d_ee / dref, 0.0, 1.0); ps = xx > 0.0
                    xp = jnp.where(ps, jnp.power(jnp.where(ps, xx, 1.0), ap["mask_envp"]), 0.0)
                    al = al * (th[:-1] - th[1:]) * (0.5 * (1 + jnp.cos(jnp.pi * xp)))[None]
                else:
                    if "mask_centre" in ap:
                        c = dref * ap["mask_centre"]
                        u = jnp.clip((d_ee[None] - c[:, None, None]) / sg[:, None, None], -1.0, 1.0)
                    else:
                        u = jnp.clip(d_ee[None] / sg[:, None, None], 0.0, 1.0)
                    al = al * 0.5 * (1 + jnp.cos(jnp.pi * u))
            S.append(s)
            W.append(al)
            x = x + attn(ap, x, d_ee, pf)
            x = ln(P["layers"]["ln"][l][0], x)
            x = x + mlp(P["layers"]["mlp"][l], x)
            x = ln(P["layers"]["ln"][l][1], x)
        return jnp.stack(S), jnp.stack(W), d_ee

    # nconf: 配置数の上限。N が大きいと (B,L,H,N,N) がメモリを食うので間引く。
    pos = np.asarray(data["positions"][0])
    sp_ = np.asarray(data["spins"][0])
    at_ = np.asarray(data["atoms"][0])
    if nconf is not None and pos.shape[0] > nconf:
        idx = np.linspace(0, pos.shape[0] - 1, nconf).astype(int)
        pos, sp_, at_ = pos[idx], sp_[idx], at_[idx]
    S, W, d = jax.vmap(forward)(jnp.asarray(pos), jnp.asarray(sp_), jnp.asarray(at_))
    _ap = [P["layers"]["attention"][l] for l in range(NL)]
    sg = np.asarray(dref * jax.nn.softplus(np.stack([a["mask_omega"] for a in _ap])))
    has_r = all("mask_rings" in a for a in _ap)
    has_c = all("mask_centre" in a for a in _ap) or has_r
    if has_r:        # 輪切り：代表長 = 輪の中点、外縁 = 輪の外側の境界
        _bb = np.asarray(dref * np.stack([a["mask_rings"] for a in _ap]))
        c = 0.5 * (_bb[:, :-1] + _bb[:, 1:]); sg = _bb[:, 1:] - c
    else:
        c = np.asarray(dref * np.stack([a["mask_centre"] for a in _ap])) if has_c else np.zeros_like(sg)
    # ヘッドを代表する長さ: 単調窓は w_h、帯域窓（殻）は中心 c_h。
    # 外縁 (殻なら c_h + sigma_h) は「窓が 0 になる距離」で d_ref との比較に使う。
    w_rep = c if has_c else sg
    r = np.asarray(CS.TRIM_MAX * jax.nn.sigmoid(
        np.stack([P["layers"]["attention"][l]["trim_rho"] for l in range(NL)])))
    return dict(S=np.asarray(S), W=np.asarray(W), d=np.asarray(d), w=w_rep, r=r, dref=dref,
                w_sigma=sg, w_centre=c, w_outer=(c + sg) if has_c else sg, shell=has_c,
                spins=sp_, positions=pos, lattice=np.asarray(L), NE=NE, NL=NL, NH=NH,
                checkpoint=os.path.basename(ck), folder=folder)


def summarize(a):
    NE, off = a["NE"], ~np.eye(a["NE"], dtype=bool)
    d = a["d"][:, off]
    si = a["spins"][:, np.where(off)[0]]
    sj = a["spins"][:, np.where(off)[1]]
    par = (si * sj > 0).ravel()
    db = np.clip(np.digitize(d.ravel(), np.linspace(d.min(), d.max(), 21)) - 1, 0, 19)

    def r2(y, g):
        tot = y.var()
        return sum(((g == u).mean() * (y[g == u].mean() - y.mean()) ** 2)
                   for u in np.unique(g) if (g == u).sum() > 10) / tot

    print("  checkpoint = %s   d_ref = %.3f   d = %.2f 〜 %.2f   電子 %d 個"
          % (a["checkpoint"], a["dref"], d.min(), d.max(), NE))
    print("  %3s%4s%8s%9s%12s%11s%9s%9s%9s"
          % ("層", "H", "w", "w/d_ref", "自己重み", "実効到達", "sd(s)", "R2距離", "R2spin"))
    for l in range(a["NL"]):
        for h in range(a["NH"]):
            W = a["W"][:, l, h]
            Wo, Ws = W[:, off], np.einsum("cii->ci", W)
            s = a["S"][:, l, h][:, off].ravel()
            reff = (Wo * d).sum(1).mean() / max(Wo.sum(1).mean(), 1e-12)
            print("  %3d%4d%8.2f%9.2f%11.1f%%%11.2f%9.3f%9.1f%9.1f"
                  % (l, h, a["w"][l, h], a["w"][l, h] / a["dref"],
                     100 * (Ws.sum(1) / (Ws.sum(1) + Wo.sum(1))).mean(), reff,
                     s.std(), 100 * r2(s, db), 100 * r2(s, par.astype(int))))
    print("  ※ 自己重み: 一様配分なら 1/%d = %.1f%%。R2 は s の分散の説明率[%%]。"
          % (NE, 100.0 / NE))
    print("  ※ 残差 100 - (R2距離 + R2spin) は、そのペア以外の %d 電子の配置に由来する成分。"
          % (NE - 2))


if __name__ == "__main__":
    folder = sys.argv[1]
    a = probe(folder)
    summarize(a)
    out = os.path.join(folder, "attn_probe.npz")
    np.savez_compressed(out, **{k: v for k, v in a.items() if isinstance(v, np.ndarray)})
    print("  saved:", out)
