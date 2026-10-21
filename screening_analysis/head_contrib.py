"""各ヘッドが波動関数（残差ストリーム）に実際に足し込む量を測る。

    python screening_analysis/head_contrib.py <run_folder> [--nconf 64]

遮蔽注意でも通常の softmax 注意（Psiformer）でも、注意ブロックの出力は
ヘッド出力を [heads*value_dim] に並べて W_O を掛ける形なので、W_O の行が
ヘッドごとのブロックに分かれ、**出力はヘッドについて厳密に加法的**になる。

    遮蔽:    out_i = e^{s_O} Σ_h [ gate ⊙ TanhNorm(attn) ]_{i,h} · W_O^{(h)}
    softmax: out_i =           Σ_h [ softmax(qk/√d) v ]_{i,h}  · W_O^{(h)}

したがってヘッド h の寄与ベクトル c_h(i) = (h ブロック) · W_O^{(h)} をそのまま取り出せる。

出す量:
  ‖c_h‖        そのヘッドが足すベクトルの長さ（電子・配置平均）
  ノルム share ‖c_h‖ / Σ‖c‖            … 直感的だが加法の指標としては正しくない
  二乗 share   ‖c_h‖² / Σ‖c‖²          … 各ヘッドが足し込む量の二乗の割合。ヘッドどうしは打ち消し合うので
                                          ‖Σc‖² の内訳ではない（head_cancel.py：run 1929 の相手の分で
                                          ‖Σc‖²/Σ‖c‖² = 0.90 液体〜0.47 結晶、直交なら 1）
  Σ‖c_h‖/‖Σc_h‖                        … 1 なら完全整列、√H ならほぼ直交
"""
import argparse, glob, json, os, re, sys
import numpy as np, jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from periodicwave import networks                                     # noqa
from periodicwave.pbc import feature_layer as FL                      # noqa


def _lattice(cfg):
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    return np.array([float(x) for x in re.findall(
        r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))][:4]).reshape(2, 2)


def compute(folder, nconf=64, checkpoint=None, sel=None):
    """(C, Rn, w) を返す。C=(L,H) の ‖c_h‖、Rn=(L,) の ‖Σ_h c_h‖、w=(L,H) の w/d_ref。"""
    ck = checkpoint or sorted(glob.glob(os.path.join(folder, "qmcjax_ckpt_*.npz")))[-1]
    z = np.load(ck, allow_pickle=True); data = z["data"].item()
    P = jax.tree_util.tree_map(lambda x: np.asarray(x)[0], z["params"].item())
    cfg = json.load(open(os.path.join(folder, "config.json")))
    ntype = cfg["network"]["network_type"]
    softmax = ntype == "CustomPsiformer"
    o = cfg["network"][ntype]
    L = _lattice(cfg)
    NH, AD, VD, NL = o["num_heads"], o["attn_dim"], o["value_dim"], o["num_layers"]
    fixv = bool(o.get("use_fixed_value", False))       # [SCREEN] value を対の固定特徴に置換
    NE = sum(cfg["system"]["electrons"])
    from periodicwave import CustomScreenformer as CS
    dref = float(CS._max_periodic_distance(jnp.asarray(L)))

    fl = FL.make_pbc_feature_layer(natoms=1, nspins=tuple(cfg["system"]["electrons"]), ndim=2,
                                   lattice=jnp.asarray(L), include_r_ae=False)
    cif = networks.make_periodic_construct_input_features(jnp.asarray(L))

    if softmax:
        from periodicwave import CustomPsiformer as CP
        _, ln = CP.make_layer_norm()
        _, mlp = CP.make_mlp(o["num_perceptrons_per_layer"], o["mlp_activation_fct"])
        use_ln = o["use_layer_norm"]
    else:
        _, ln = CS.make_layer_norm()
        _, mlp = CS.make_mlp(o["num_perceptrons_per_layer"], o["mlp_activation_fct"])

    def _post(at, x, ap, sc):
        """注意の出力ベクトル at からヘッドごとの寄与ベクトルまで（共通の後段）。"""
        if (not softmax) and o["use_tanh_norm"]:
            n = jnp.sqrt(jnp.sum(at * at, -1, keepdims=True) + 1e-12)
            at = at * jnp.tanh(n) / n
        vd = at.shape[-1]
        flat = at.reshape(NE, NH * vd)
        if (not softmax) and o["use_gate"]:
            flat = flat * jnp.tanh(jax.nn.silu(jnp.dot(x, ap["g_w"])))
        WO = ap["attn_output"]["w"] if isinstance(ap["attn_output"], dict) else ap["attn_output"]
        return sc * jnp.einsum("ihd,hde->ihe", flat.reshape(NE, NH, vd),
                               WO.reshape(NH, vd, -1))

    def forward(pos, spins, atoms):
        ae, ee, r_ae, r_ee = cif(pos, atoms, ndim=2)
        f, eef = fl.apply(ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee)
        f = jnp.concatenate((f, spins[..., None]), axis=-1)
        x = jnp.dot(f, P["layers"]["embed"]); d_ee = r_ee[..., 0]
        # [SCREEN] 固定 value のとき運ぶ対の特徴 e(r_ij)（ネットワーク側と同じ作り方）
        pf = jnp.concatenate((eef[..., :1] / dref, eef[..., 1:]), axis=-1) if fixv else None
        eye = jnp.eye(NE)[None]
        C, Cd, Cp, X, R, A, Ap = [], [], [], [], [], [], []
        for l in range(NL):
            ap = P["layers"]["attention"][l]
            q = jnp.dot(x, ap["q_w"]).reshape(NE, NH, AD)
            k = jnp.dot(x, ap["k_w"]).reshape(NE, NH, AD)
            v = (None if fixv else jnp.dot(x, ap["v_w"]).reshape(NE, NH, VD))
            if softmax:
                s = jnp.einsum("ihd,jhd->hij", q, k) / jnp.sqrt(float(AD))
                wgt = jax.nn.softmax(s, axis=-1); vv = v; sc = 1.0
            else:
                s = jnp.einsum("ihd,jhd->hij", CS._unit_length(q), CS._unit_length(k))
                r = (CS.TRIM_MAX * jax.nn.sigmoid(ap["trim_rho"]))[:, None, None]
                wgt = jnp.square(jax.nn.relu(s - (1.0 - r)) / r)
                if o["use_softmask"]:
                    ww = dref * jax.nn.softplus(ap["mask_omega"])
                    if "mask_rings" in ap:       # 全体窓 g の輪切り
                        bb = dref * ap["mask_rings"]; dl = ww[0]
                        # 階段は本体と同じ関数で作る（ガウス核の輪切りなら "mask_gauss" がある）
                        th = (CS._ring_steps(d_ee, bb, dl * ap["mask_gauss"], gauss=True) if "mask_gauss" in ap
                              else CS._ring_steps(d_ee, bb, dl))
                        xx = jnp.clip(d_ee / dref, 0.0, 1.0); ps = xx > 0.0
                        xp = jnp.where(ps, jnp.power(jnp.where(ps, xx, 1.0), ap["mask_envp"]), 0.0)
                        wgt = wgt * (th[:-1] - th[1:]) * (0.5 * (1 + jnp.cos(jnp.pi * xp)))[None]
                    else:
                        if "mask_centre" in ap:      # 帯域窓（殻）
                            cc = dref * ap["mask_centre"]
                            u = jnp.clip((d_ee[None] - cc[:, None, None]) / ww[:, None, None], -1.0, 1.0)
                        else:                        # 単調減少窓
                            u = jnp.clip(d_ee[None] / ww[:, None, None], 0.0, 1.0)
                        wgt = wgt * 0.5 * (1 + jnp.cos(jnp.pi * u))
                vv = (None if fixv else CS._unit_length(v))
                sc = jnp.exp(ap["out_scale"]) if o["use_output_scale"] else 1.0
                if fixv:
                    wgt = wgt * (1.0 - eye)            # 固定 value では j = i を落とす
            if fixv:
                at_all = jnp.einsum("hij,ijd->ihd", wgt, pf)
                at_dia = jnp.zeros_like(at_all)
            else:
                at_all = jnp.einsum("hij,jhd->ihd", wgt, vv)
                at_dia = jnp.einsum("hij,jhd->ihd", wgt * eye, vv)
            # [PROBE] TanhNorm より前（重みの和そのもの）。読み出しの飽和を切り分けるため。
            A.append(jnp.linalg.norm(at_all, axis=-1))
            Ap.append(jnp.linalg.norm(at_all - at_dia, axis=-1))
            ch = _post(at_all, x, ap, sc)
            chd = _post(at_dia, x, ap, sc)
            C.append(jnp.linalg.norm(ch, axis=-1))            # [i,h] 全体
            Cd.append(jnp.linalg.norm(chd, axis=-1))          # [i,h] 自分だけ（1 体）
            Cp.append(jnp.linalg.norm(ch - chd, axis=-1))     # [i,h] 相手のぶん（2 体）
            X.append(jnp.linalg.norm(x, axis=-1))             # [i]
            R.append(jnp.linalg.norm(ch.sum(1), axis=-1))     # [i]  ‖Σ_h c_h‖
            out = ch.sum(1)
            if softmax:
                x = x + out
                if use_ln: x = ln(P["layers"]["ln"][l][0], x)
                x = x + mlp(P["layers"]["mlp"][l], x)
                if use_ln: x = ln(P["layers"]["ln"][l][1], x)
            else:
                x = ln(P["layers"]["ln"][l][0], x + out)
                x = ln(P["layers"]["ln"][l][1], x + mlp(P["layers"]["mlp"][l], x))
        return (jnp.stack(C), jnp.stack(Cd), jnp.stack(Cp), jnp.stack(X), jnp.stack(R),
                jnp.stack(A), jnp.stack(Ap))

    pos = np.asarray(data["positions"][0]); sp = np.asarray(data["spins"][0])
    at_ = np.asarray(data["atoms"][0])
    if sel is not None:                     # 呼び出し側が選んだウォーカー（誤差の見積もり用）
        sel = np.asarray(sel); pos, sp, at_ = pos[sel], sp[sel], at_[sel]
    elif pos.shape[0] > nconf:
        i = np.linspace(0, pos.shape[0] - 1, nconf).astype(int); pos, sp, at_ = pos[i], sp[i], at_[i]
    C, Cd, Cp, X, R, A, Ap = jax.vmap(forward)(jnp.asarray(pos), jnp.asarray(sp), jnp.asarray(at_))
    C = np.asarray(C).mean(axis=(0, 2)); Cd = np.asarray(Cd).mean(axis=(0, 2))
    Cp = np.asarray(Cp).mean(axis=(0, 2)); X = np.asarray(X).mean(axis=(0, 2))
    R = np.asarray(R).mean(axis=(0, 2))
    A = np.asarray(A).mean(axis=(0, 2)); Ap = np.asarray(Ap).mean(axis=(0, 2))
    if softmax:
        w = np.full((NL, NH), np.nan)
        meta = dict(shell=False, sigma=w, centre=w, outer=w)
    else:
        _ap = [P["layers"]["attention"][l] for l in range(NL)]
        _sg = np.asarray(dref * jax.nn.softplus(np.stack([a["mask_omega"] for a in _ap])))
        rings = all("mask_rings" in a for a in _ap)
        shell = all("mask_centre" in a for a in _ap)
        if rings:
            # 輪切りではヘッドを代表する長さは輪の中点。
            _bb = np.asarray(dref * np.stack([a["mask_rings"] for a in _ap]))
            _mid = 0.5 * (_bb[:, :-1] + _bb[:, 1:])
            w, meta = _mid / dref, dict(shell=True, sigma=np.diff(_bb, axis=1) / dref,
                                        centre=_mid / dref, outer=_bb[:, 1:] / dref)
        elif shell:
            # 殻ではヘッドを代表する長さは中心 c_h（そこの相関だけを見るので）。
            # 外縁 c_h + sigma_h は「窓が 0 になる距離」で、d_ref との比較に使う。
            _ct = np.asarray(dref * np.stack([a["mask_centre"] for a in _ap]))
            w, meta = _ct / dref, dict(shell=True, sigma=_sg / dref,
                                       centre=_ct / dref, outer=(_ct + _sg) / dref)
        else:
            w = _sg / dref
            meta = dict(shell=False, sigma=w, centre=np.zeros_like(w), outer=w)
    meta = dict(meta, A=A, Ap=Ap, fixv=fixv)
    return C, Cd, Cp, X, R, w, softmax, meta


def cached(folder, nconf=64):
    """<folder>/head_contrib.npz にキャッシュして dict を返す。

    C  … ヘッドが波動関数に足すベクトルの長さ（全体）
    Cd … そのうち自分自身だけを見たぶん（1 体的。窓が最短の対距離より狭いと
          ヘッドはこれだけになる）
    Cp … 相手を見たぶん（2 体。相関を運んでいるのはこちら）
    """
    p = os.path.join(folder, "head_contrib.npz")
    if os.path.exists(p):
        z = np.load(p)
        if "Cp" in z.files and "shell" in z.files and "Ap" in z.files:
            return {k: z[k] for k in z.files}
        os.remove(p)                                   # 旧形式は作り直す
    C, Cd, Cp, X, R, w, sm, mt = compute(folder, nconf)
    d = dict(C=C, Cd=Cd, Cp=Cp, X=X, R=R, w=w, softmax=np.asarray(sm),
             shell=np.asarray(mt["shell"]), w_sigma=mt["sigma"],
             w_centre=mt["centre"], w_outer=mt["outer"],
             A=mt["A"], Ap=mt["Ap"], fixv=np.asarray(mt["fixv"]))
    np.savez(p, **d)
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("folder")
    ap.add_argument("--nconf", type=int, default=64); a = ap.parse_args()
    z = cached(a.folder, a.nconf)
    C, Cd, Cp, X, R, w = z["C"], z["Cd"], z["Cp"], z["X"], z["R"], z["w"]
    softmax = bool(np.all(np.isnan(w)))
    NL, NH = C.shape
    sq = 100 * C ** 2 / (C ** 2).sum()
    sqp = 100 * Cp ** 2 / (Cp ** 2).sum()
    print("  %s   %d層×%dヘッド   %s"
          % (os.path.basename(a.folder), NL, NH, "softmax" if softmax else "遮蔽"))
    for l in range(NL):
        print("\n  第%d層   ‖x‖ = %.3f,  Σ‖c‖/‖Σc‖ = %.2f （直交なら √H = %.2f）"
              % (l + 1, X[l], C[l].sum() / R[l], np.sqrt(NH)))
        _hd = "中心" if bool(z.get("shell", False)) else "窓  "
        print("    %s    寄与率   うち 2 体   2 体の割合   2 体だけで見た寄与率" % _hd)
        for h in np.argsort(w[l] if not softmax else np.arange(NH)):
            print("    %6s %7.1f%% %8.4f %10.2f %14.1f%%"
                  % ("--" if softmax else "%.2f" % w[l, h], sq[l, h], Cp[l, h],
                     Cp[l, h] / max(C[l, h], 1e-9), sqp[l, h]))
    if not softmax:
        wt = (Cp ** 2) / (Cp ** 2).sum()
        print("\n  2 体の寄与で重み付けした平均%s = %.4f （周期ノルム単位）"
              % ("窓の山" if bool(z.get("shell", False)) else "窓幅",
                 float((wt * w).sum())))
        if "Ap" in z:
            # [PROBE] TanhNorm の前後で読み出しがどれだけ変わるか。
            # q_h は c_h（TanhNorm・ゲート・W_O の後）で定義しているが、
            # TanhNorm はヘッドごとに飽和するので、相手を多く見るヘッドほど潰れる。
            A_, Ap_ = z["A"], z["Ap"]
            # TanhNorm はベクトル全体に tanh|a|/|a| を掛ける。その分を割り戻すと
            # 残るのはゲートと W_O —— すなわち「学習した係数」で、読み出したいのはこれ。
            unsat = A_ / np.maximum(np.tanh(A_), 1e-12)
            q_a = (Ap_ ** 2) / (Ap_ ** 2).sum()                 # 窓が集めた量（ほぼ幾何）
            q_c = (Cp ** 2) / (Cp ** 2).sum()                   # いまの定義
            Cg = Cp * unsat
            q_g = (Cg ** 2) / (Cg ** 2).sum()                   # 飽和を割り戻した定義
            print("\n  読み出しの 3 通り   a = 窓が集めた量（幾何）/ c = いまの定義 / g = 飽和を割り戻す")
            print("    %-6s %9s %9s %9s %9s" % ("窓の山", "q_h(a)", "q_h(c)", "q_h(g)", "圧縮率"))
            for l in range(NL):
                for h in np.argsort(w[l]):
                    _cr = (float(np.tanh(A_[l, h]) / A_[l, h]) if A_[l, h] > 1e-12
                           else float("nan"))
                    print("    %6.3f %8.2f%% %8.2f%% %8.2f%% %9s"
                          % (w[l, h], 100 * q_a[l, h], 100 * q_c[l, h], 100 * q_g[l, h],
                             "—（相手なし）" if not np.isfinite(_cr) else "%.3f" % _cr))
            tv = lambda p, q: 0.5 * float(np.abs(p - q).sum())
            print("    平均の山:  a %.4f   c %.4f   g %.4f" %
                  (float((q_a * w).sum()), float((q_c * w).sum()), float((q_g * w).sum())))
            print("    全変動距離:  a-c %.4f   a-g %.4f   c-g %.4f" %
                  (tv(q_a, q_c), tv(q_a, q_g), tv(q_c, q_g)))


if __name__ == "__main__":
    main()
