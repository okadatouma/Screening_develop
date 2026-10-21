"""各ヘッドを 1 つずつ止めて、エネルギーがどれだけ悪くなるかを測る。

    python screening_analysis/head_ablation.py <folder> [--nconf 256]

寄与ベクトルの長さは窓が広いほど多くの対を足し込むぶん自動的に大きくなるが、
ヘッドを止めたときのエネルギーの代償にはその偏りが無い。窓の幅ごとに
「そのヘッドが無いとどれだけ損か」を並べれば、どの長さが効いているかが分かる。
"""
import argparse, json, os, sys
import numpy as np, jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import moire_analysis as MA                              # noqa
from energy_check import build_signed                    # noqa
from window_ablation import _measure                     # noqa


def _diag_head(params, l, h, NH):
    """そのヘッドだけ「自分自身しか見ない」ようにする（2 体成分だけを抜く）。"""
    p2 = jax.tree_util.tree_map(lambda x: x, params)
    for ll in range(len(p2["layers"]["attention"])):
        g = np.zeros(NH, dtype=np.float32)
        if ll == l: g[h] = 1.0
        p2["layers"]["attention"][ll]["_diag_only"] = jnp.asarray(g)
    return p2


def _zero_head(params, l, h, VD):
    p2 = jax.tree_util.tree_map(lambda x: x, params)
    ap = p2["layers"]["attention"][l]
    node = ap["attn_output"]
    W = node["w"] if isinstance(node, dict) else node
    W = W.at[h * VD:(h + 1) * VD, :].set(0.0)
    if isinstance(node, dict): node["w"] = W
    else: ap["attn_output"] = W
    return p2


def run(folder, nconf=256, nwalk=64, seed=0, mode="off"):
    net, params, le, data, cfg = build_signed(folder)
    ntype = cfg["network"]["network_type"]
    o = cfg["network"][ntype]
    NL, NH, VD = o["num_layers"], o["num_heads"], o["value_dim"]
    L = MA.load(folder)["L"]
    E0, e0, _ = _measure(net, params, le, data, nconf, nwalk, seed)
    print("  すべて有効        E = %9.3f ± %.3f" % (E0, e0))
    from head_contrib import cached as HC
    z = HC(folder, nconf=48)
    w_e = MA.w_to_euclid(np.asarray(z["w"]) * float(
        __import__("periodicwave").CustomScreenformer._max_periodic_distance(jnp.asarray(L))), L)
    out = []
    for l in range(NL):
        for h in range(NH):
            p2 = (_zero_head(params, l, h, VD) if mode == "off"
                  else _diag_head(params, l, h, NH))
            E, e, _ = _measure(net, p2, le, data, nconf, nwalk, seed)
            out.append(dict(layer=l, head=h, w=float(w_e[l, h]), E=E, Eerr=e, dE=E - E0))
            print("  第%d層 窓 %6s  E = %9.3f ± %.3f   ΔE = %+8.3f"
                  % (l + 1, ("%.2f" % w_e[l, h]) if np.isfinite(w_e[l, h]) else "制限なし",
                     E, e, E - E0))
    res = dict(folder=folder, mode=mode, E0=E0, E0err=e0, heads=out)
    json.dump(res, open(os.path.join(folder, "head_ablation_%s.json" % mode), "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("folder")
    ap.add_argument("--nconf", type=int, default=256)
    ap.add_argument("--nwalk", type=int, default=64)
    ap.add_argument("--mode", default="off", choices=("off", "pair"))
    a = ap.parse_args()
    run(a.folder, a.nconf, a.nwalk, mode=a.mode)
