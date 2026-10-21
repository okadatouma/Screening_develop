# -*- coding: utf-8 -*-
"""遮蔽アテンションの構造を、学習済みモデルで直接測る。

    python screening_analysis/structure_probe.py <run_folder> [--nconf 64]

測るもの
  (V)  value ベクトル v̄_j は電子ごとに違うか。違わなければ value は定数で、
       ヘッド出力は「窓の中の重みの和 × 定数」に退化している。
       さらに v̄_j を電子平均 / スピン平均で置き換えたときの log|ψ| の変化。
  (T)  内容チャネル s_ij = cos(q_i, k_j) は同スピン対と異スピン対で違うか。
       r を動かしたとき α(s) の形がどれだけ変わるか（出力スケールとの縮退）。
  (A)  反対称性：電子の交換で ψ がどう変わるかを数値で確かめる。
"""
import argparse, glob, json, os, sys
import numpy as np, jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from periodicwave import CustomScreenformer as CS, networks            # noqa
from periodicwave.pbc import feature_layer as FL                       # noqa
from attn_probe import lattice_from_config, _centres_from              # noqa


def internals(folder, nconf=64):
    """層 0 のアテンション内部量を返す（S, W, vbar, spins, x など）。"""
    ck = sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))[-1]
    z = np.load(ck, allow_pickle=True); data = z["data"].item()
    P = jax.tree_util.tree_map(lambda a: np.asarray(a)[0], z["params"].item())
    cfg = json.load(open(os.path.join(folder, "config.json")))
    o = cfg["network"]["CustomScreenformer"]; L = lattice_from_config(cfg)
    NH, AD, VD, MD = o["num_heads"], o["attn_dim"], o["value_dim"], o["mlp_dim"]
    NE = sum(cfg["system"]["electrons"]); dref = CS._max_periodic_distance(L)
    fl = FL.make_pbc_feature_layer(natoms=1, nspins=tuple(cfg["system"]["electrons"]), ndim=2,
                                   lattice=jnp.asarray(L), include_r_ae=False)
    cif = networks.make_periodic_construct_input_features(jnp.asarray(L))
    ap = P["layers"]["attention"][0]

    def fwd(pos, spins, atoms):
        ae, ee, r_ae, r_ee = cif(pos, atoms, ndim=2)
        f, _ = fl.apply(ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee)
        f = jnp.concatenate((f, spins[..., None]), axis=-1)
        x = jnp.dot(f, P["layers"]["embed"])
        d = r_ee[..., 0]
        q = CS._unit_length(jnp.dot(x, ap["q_w"]).reshape(NE, NH, AD))
        k = CS._unit_length(jnp.dot(x, ap["k_w"]).reshape(NE, NH, AD))
        v = CS._unit_length(jnp.dot(x, ap["v_w"]).reshape(NE, NH, VD))
        s = jnp.einsum("ihd,jhd->hij", q, k)
        r = (CS.TRIM_MAX * jax.nn.sigmoid(ap["trim_rho"]))[:, None, None]
        al = jnp.square(jax.nn.relu(s - (1.0 - r)) / r)
        sg = dref * jax.nn.softplus(ap["mask_omega"])
        if "mask_centre" in ap:
            c = dref * ap["mask_centre"]
            u = jnp.clip((d[None] - c[:, None, None]) / sg[:, None, None], -1.0, 1.0)
        else:
            u = jnp.clip(d[None] / sg[:, None, None], 0.0, 1.0)
        W = al * 0.5 * (1 + jnp.cos(jnp.pi * u))
        return s, W, v, x, d

    pos = np.asarray(data["positions"][0]); sp = np.asarray(data["spins"][0])
    at = np.asarray(data["atoms"][0])
    if pos.shape[0] > nconf:
        i = np.linspace(0, pos.shape[0] - 1, nconf).astype(int); pos, sp, at = pos[i], sp[i], at[i]
    S, W, V, X, D = jax.vmap(fwd)(jnp.asarray(pos), jnp.asarray(sp), jnp.asarray(at))
    r = np.asarray(CS.TRIM_MAX * jax.nn.sigmoid(ap["trim_rho"]))
    return dict(S=np.asarray(S), W=np.asarray(W), V=np.asarray(V), X=np.asarray(X),
                d=np.asarray(D), spins=sp, pos=pos, atoms=at, r=r, NE=NE, NH=NH,
                VD=VD, P=P, cfg=cfg, folder=folder, data=data)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("folder")
    ap.add_argument("--nconf", type=int, default=64); a = ap.parse_args()
    I = internals(a.folder, a.nconf)
    NE, NH = I["NE"], I["NH"]; sp = I["spins"]
    V = I["V"]                                   # (B, N, H, VD) 単位長
    off = ~np.eye(NE, dtype=bool)
    same = (sp[:, :, None] == sp[:, None, :])    # (B, N, N)

    print("=" * 72)
    print("(V) value ベクトルは電子ごとに違うか")
    # 電子間のコサイン類似度（同じ配置の中、j≠k）
    G = np.einsum("bjhd,bkhd->bhjk", V, V)       # (B,H,N,N)
    cos_all = G[:, :, off].mean()
    ss = np.broadcast_to(same[:, None], G.shape)
    cos_same = G[:, :, off][ss[:, :, off]].mean()
    cos_opp = G[:, :, off][~ss[:, :, off]].mean()
    # 電子方向の実効次元（共分散の参加率）: 1 なら全電子で同じ向き
    Vc = V - V.mean(axis=1, keepdims=True)
    cov = np.einsum("bjhd,bjhe->hde", Vc, Vc) / (V.shape[0] * NE)
    ev = np.linalg.eigvalsh(cov); pr = (ev.sum(-1) ** 2 / (ev ** 2).sum(-1))
    tot = np.einsum("bjhd,bjhd->h", V, V) / (V.shape[0] * NE)
    frac_var = ev.sum(-1) / tot                  # 定数からのずれの割合
    print("  電子間コサイン類似度  全体 %.3f   同スピン %.3f   異スピン %.3f"
          % (cos_all, cos_same, cos_opp))
    print("  v̄ の分散のうち電子ごとの変化が占める割合（ヘッド平均）: %.3f" % frac_var.mean())
    print("  電子方向の実効次元（ヘッド平均, 上限 %d）: %.2f" % (I["VD"], pr.mean()))
    # スピンだけで v̄ を説明できるか: v̄_j ≈ v̄_{σ_j}
    Vs = np.zeros_like(V)
    for b in range(V.shape[0]):
        for s_ in (-1, 1):
            m = sp[b] == s_
            Vs[b, m] = V[b, m].mean(axis=0, keepdims=True)
    res = ((V - Vs) ** 2).sum(-1).mean(); base = ((V - V.mean(1, keepdims=True)) ** 2).sum(-1).mean()
    print("  v̄ の電子依存のうちスピン（↑/↓ の 2 値）で説明できる割合: %.3f" % (1 - res / base))

    print("=" * 72)
    print("(T) 内容チャネルとスピン")
    S = I["S"]; Wt = I["W"]
    s_same = S[:, :, off][ss[:, :, off]]; s_opp = S[:, :, off][~ss[:, :, off]]
    print("  s_ij の平均   同スピン %.4f   異スピン %.4f   差 %.4f  （σ_s = %.3f）"
          % (s_same.mean(), s_opp.mean(), s_same.mean() - s_opp.mean(), S[:, :, off].std()))
    # 分離度: 同スピン対と異スピン対を s だけで見分けられるか（AUC）
    from itertools import product
    rng = np.random.default_rng(0)
    i1 = rng.choice(len(s_same), 20000); i2 = rng.choice(len(s_opp), 20000)
    auc = (s_same[i1] > s_opp[i2]).mean() + 0.5 * (s_same[i1] == s_opp[i2]).mean()
    print("  AUC（s で同/異スピンを見分ける確率, 0.5 = 見分けられない）: %.3f" % auc)
    w_same = Wt[:, :, off][ss[:, :, off]].mean(); w_opp = Wt[:, :, off][~ss[:, :, off]].mean()
    print("  W_ij の平均   同スピン %.4f   異スピン %.4f   比 %.3f" % (w_same, w_opp, w_same / w_opp))
    # r の縮退: r を 1.9 → 1.8 にしたとき α(s) は「形」が変わるか「大きさ」が変わるか
    s_obs = S[:, :, off].ravel()
    def alpha(s, r): return np.square(np.maximum(s - (1 - r), 0) / r)
    for r2 in (1.8, 1.5, 1.0):
        a1, a2 = alpha(s_obs, 1.9), alpha(s_obs, r2)
        m = a1 > 0
        ratio = a2[m] / a1[m]
        print("  r: 1.9 → %.1f   α の比の平均 %.3f, 変動係数 %.3f   （0 に落ちる対 %.1f%%）"
              % (r2, ratio.mean(), ratio.std() / ratio.mean(), 100 * (a2 == 0).mean()))

    print("=" * 72)
    print("(A) 反対称性の数値確認")
    import sampler as SM
    bnet, Pj, d0, cfg, L = SM.build(a.folder)
    net_full = jax.jit(lambda p, x, s, at_, c: bnet(p, x, s, at_, c))
    # sign も要るので 1 配置ぶんは直接呼ぶ
    from periodicwave import CustomScreenformer as CS2
    pos0 = np.asarray(d0.positions)[:4]; sp0 = np.asarray(d0.spins)[:4]
    at0 = np.asarray(d0.atoms)[:4]; ch0 = np.asarray(d0.charges)[:4]
    ntype = cfg["network"]["network_type"]
    # sampler.build と同じ net を作り直して sign/logabs 両方を取る
    import importlib
    fl = FL.make_pbc_feature_layer(natoms=ch0.shape[1], nspins=tuple(cfg["system"]["electrons"]),
                                   ndim=2, lattice=jnp.asarray(L), include_r_ae=False)
    from periodicwave import envelopes
    net = CS.make_fermi_net(tuple(cfg["system"]["electrons"]), jnp.asarray(ch0[0]), ndim=2,
                            determinants=cfg["network"]["determinants"],
                            envelope=envelopes.make_null_envelope(), feature_layer=fl,
                            jastrow=cfg["network"].get("jastrow", "NONE"),
                            jastrow_kwargs=cfg["network"]["jastrow_kwargs"],
                            bias_orbitals=cfg["network"]["bias_orbitals"],
                            complex_output=cfg["network"].get("complex", False),
                            pbc_lattice=jnp.asarray(L), **SM._net_kwargs(cfg, ntype, Pj))
    def psi(x, s):
        sg, la = net.apply(Pj, jnp.asarray(x), jnp.asarray(s), jnp.asarray(at0[0]), jnp.asarray(ch0[0]))
        return float(np.asarray(sg)), float(np.asarray(la))
    x = pos0[0].reshape(NE, 2).copy(); s = sp0[0].copy()
    up = np.where(s > 0)[0]; dn = np.where(s < 0)[0]
    sg0, la0 = psi(x.ravel(), s)
    def swapped(i, j, swap_spin):
        x2 = x.copy(); x2[[i, j]] = x2[[j, i]]; s2 = s.copy()
        if swap_spin: s2[[i, j]] = s2[[j, i]]
        return psi(x2.ravel(), s2)
    for lab, (i, j), ssw in [("同スピン ↑↑ の位置交換", (up[0], up[1]), False),
                             ("同スピン ↓↓ の位置交換", (dn[0], dn[1]), False),
                             ("異スピン ↑↓ の (位置, スピン) 同時交換", (up[0], dn[0]), True),
                             ("異スピン ↑↓ の位置だけ交換", (up[0], dn[0]), False)]:
        sg1, la1 = swapped(i, j, ssw)
        print("  %-34s  sign %+d → %+d   log|ψ| %.6f → %.6f   (差 %.2e)"
              % (lab, sg0, sg1, la0, la1, la1 - la0))
    # アテンション層の同変性: 入力の並べ替え → 出力の並べ替え
    I2 = internals(a.folder, 1)
    print("  （アテンションの重み W_ij は電子の並べ替えに対して同変。反対称性は行列式だけが担う。）")

    print("=" * 72)
    print("(C) カスプ: 異スピン対を近づけたときの log|ψ| の形")
    i, j = up[0], dn[0]
    dvec = x[j] - x[i]; dvec = dvec / np.linalg.norm(dvec)
    rs_ = np.array([-0.30, -0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.20, 0.30])
    vals = []
    for rr in rs_:
        x2 = x.copy(); x2[j] = x[i] + rr * dvec
        vals.append(psi(x2.ravel(), s)[1])
    vals = np.array(vals)
    print("  r        : " + " ".join("%7.2f" % v for v in rs_))
    print("  log|ψ|(r): " + " ".join("%7.3f" % v for v in vals))
    # 2 次関数と |r| のどちらでフィットがよいか
    A = np.stack([np.ones_like(rs_), rs_, rs_ ** 2], 1)
    c2, res2 = np.linalg.lstsq(A, vals, rcond=None)[:2]
    B = np.stack([np.ones_like(rs_), rs_, np.abs(rs_)], 1)
    c1, res1 = np.linalg.lstsq(B, vals, rcond=None)[:2]
    print("  2 次多項式でのフィット残差 %.2e   |r| を含むフィット残差 %.2e   |r| の係数 %.3f"
          % (float(res2[0]) if len(res2) else 0, float(res1[0]) if len(res1) else 0, c1[2]))
    print("  同スピン対で同じことをすると：")
    i, j = up[0], up[1]
    dvec = x[j] - x[i]; dvec = dvec / np.linalg.norm(dvec)
    vals = []
    for rr in rs_:
        x2 = x.copy(); x2[j] = x[i] + rr * dvec
        vals.append(psi(x2.ravel(), s)[1])
    print("  log|ψ|(r): " + " ".join("%7.3f" % v for v in vals))


if __name__ == "__main__":
    main()
