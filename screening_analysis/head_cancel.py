"""ヘッドの寄与ベクトルが打ち消し合う度合い（スライド「距離の窓の継ぎ目と局所エネルギー」p.5）。

head_contrib.py と同じ c_h（TanhNorm・ゲート・W_O の後、ヘッドについて和をとる直前）を作り、
電子と配置について平均した

    rho   = < |sum_h c_h|^2 >   / < sum_h |c_h|^2 >
    rho_p = < |sum_h c^p_h|^2 > / < sum_h |c^p_h|^2 >     （c^p：相手 j != i から来た分）

を出す。ヘッドどうしが直交なら 1、打ち消し合うほど小さい。head_contrib.py の compute を写し、
R の行だけを [sum|c|^2, |sum c|^2, sum|c^p|^2, |sum c^p|^2] に替えたもの（キャッシュは書かない）。

    JAX_PLATFORMS=cpu python screening_analysis/head_cancel.py <run_folder> [<run_folder> ...] [--nconf 64]
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
            R.append(jnp.stack([jnp.sum(ch * ch, axis=(-1, -2)), jnp.sum(ch.sum(1) ** 2, -1),
                                jnp.sum((ch - chd) * (ch - chd), axis=(-1, -2)),
                                jnp.sum((ch - chd).sum(1) ** 2, -1)], -1))   # [i, 4]
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
    R = np.asarray(R).mean(axis=(0, 2))           # [L, 4]
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("folders", nargs="+")
    ap.add_argument("--nconf", type=int, default=64); a = ap.parse_args()
    for f in a.folders:
        C, Cd, Cp, X, R, w, sm, meta = compute(f, a.nconf)
        for l in range(R.shape[0]):
            S2, N2, S2p, N2p = R[l]
            print(f"{os.path.basename(f)}  層{l + 1}:  rho = {N2 / S2:.3f}   rho_p（相手の分） = {N2p / S2p:.3f}")


if __name__ == "__main__":
    main()
