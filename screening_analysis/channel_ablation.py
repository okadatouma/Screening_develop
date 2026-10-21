"""内容チャネル α(s) と距離チャネル m(d) を片方ずつ殺して、変分エネルギーがどれだけ悪化するかを測る。

    python screening_analysis/channel_ablation.py <folder> [--blocks 12]

再学習は一切しない。学習済みパラメータを書き換えるだけで 4 条件を作る:

  基準        W_ij = α(s_ij) · m(d_ij)                そのまま
  s を殺す    W_ij = ⟨α⟩_h  · m(d_ij)                内容チャネルを配置平均で置換
  m を殺す    W_ij = α(s_ij) · 1                      窓を無限大にする（距離依存を消す）
  両方殺す    W_ij = ⟨α⟩_h  · 1                      重みが定数（注意が何も選ばない）

パラメータだけで実現できる理由:
  ・k_w = 0 にすると kbar = 0（_unit_length は sqrt(・+eps²) なので NaN にならない）→ s ≡ 0
  ・そのとき α は定数 α(0) = ((r−1)/r)²。r = TRIM_MAX·sigmoid(trim_rho) を
    r* = 1/(1−√⟨α⟩) に置き直せば、α(0) がそのヘッドの ⟨α(s)⟩ にちょうど一致する
    （＝大きさを変えずに「対依存性だけ」を消せる）
  ・w → 大きくすると m(d) = ½[1+cos(π d/w)] → 1

注意: s を殺すと対角 α(s_ii) も ⟨α⟩ に潰れるので、自己/他者の区別も同時に消える。
これは「内容チャネルを消す」の一部として意図的。
"""
import argparse, json, os, sys
import numpy as np, jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from tv_split import build, _err                                   # noqa: E402
from w_response import scaled, energy                              # noqa: E402
from attn_probe import probe                                       # noqa: E402
from periodicwave import CustomScreenformer as CS                  # noqa: E402


def mean_alpha(folder, nconf=64):
    """学習済みモデルの ⟨α(s)⟩ をヘッドごとに測る（対角も含む＝重みに現れる通り）。"""
    a = probe(folder, nconf=nconf)
    r = a["r"][:, :, None, None]                                   # (L,H,1,1)
    al = np.square(np.maximum(a["S"] - (1.0 - r)[None], 0.0) / r[None])
    return al.mean(axis=(0, 3, 4)), a["r"]                         # (L,H), (L,H)


def kill_s(params, abar):
    """k_w = 0 にして s ≡ 0。trim_rho を r* = 1/(1−√⟨α⟩) に合わせ α(0) = ⟨α⟩ にする。"""
    p = jax.tree_util.tree_map(lambda x: x, params)
    at = p["layers"]["attention"]; new = []
    for l in range(len(at)):
        ab = jnp.clip(jnp.asarray(abar[l]), 1e-6, 0.999 ** 2)
        rstar = jnp.clip(1.0 / (1.0 - jnp.sqrt(ab)), 1e-3, CS.TRIM_MAX - 1e-3)
        rho = jnp.log(rstar / (CS.TRIM_MAX - rstar))               # logit(r/TRIM_MAX)
        new.append({**at[l], "k_w": jnp.zeros_like(at[l]["k_w"]), "trim_rho": rho})
    p["layers"] = {**p["layers"], "attention": new}
    return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("folder")
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--mcmc", type=int, default=20)
    ap.add_argument("--wide", type=float, default=8.0, help="m を殺すときの窓の倍率")
    a = ap.parse_args()

    b = build(a.folder)
    pos0 = jnp.asarray(b["data"]["positions"][0])
    abar, rr = mean_alpha(a.folder); jax.clear_caches()
    cfg = json.load(open(os.path.join(a.folder, "config.json")))
    NL, NH = cfg["network"]["CustomScreenformer"]["num_layers"], \
             cfg["network"]["CustomScreenformer"]["num_heads"]

    print("  %s   r_s=%g  N=%d  %d層×%dヘッド"
          % (os.path.basename(a.folder), b["r_s"], b["ne"], NL, NH))
    print("  ⟨α(s)⟩ ヘッド平均 = %.4f   Trim 半径 r = %.3f（TRIM_MAX=%.1f）\n"
          % (abar.mean(), rr.mean(), CS.TRIM_MAX))

    P = b["params"]
    conds = [("基準（そのまま）", P),
             ("s を殺す  W=⟨α⟩·m(d)", kill_s(P, abar)),
             ("m を殺す  W=α(s)·1", scaled(P, a.wide)),
             ("両方殺す  W=定数", scaled(kill_s(P, abar), a.wide))]
    out = []
    for nm, pp in conds:
        E = energy(b, pp, a.blocks, a.mcmc, 0, pos0)
        out.append((nm, E.mean(), _err(E)))
        print("    %-24s E = %10.3f ± %.3f" % (nm, E.mean(), _err(E)), flush=True)

    e0 = out[0][1]
    print("\n    条件                        ΔE = E − E(基準)      |E| に対する割合")
    for nm, e, s in out[1:]:
        print("    %-24s %+12.3f ± %.3f %14.1f%%" % (nm, e - e0, s, 100 * (e - e0) / abs(e0)))
    np.save(os.path.join(a.folder, "channel_ablation.npy"),
            np.array([[e, s] for _, e, s in out]))


if __name__ == "__main__":
    main()
