"""窓幅 w を人為的に動かしたときの変分エネルギー応答 ΔE(λ) を測る。

    python screening_analysis/w_response.py <folder> [--lam 0.25,0.4,...] [--perhead]

学習済みパラメータはそのままに、遮蔽窓だけを w → λ·w に置き換えて E = <T+V> を測り直す。
w は w = d_ref · softplus(ω) なので、λ 倍するには
    ω' = softplus^{-1}(λ·softplus(ω)) = log(exp(λ·softplus(ω)) − 1)
とすればよい（他のパラメータには一切触らない）。

  ・ΔE(λ) が λ=1 で最小 → w は本当に最適化されている（初期値の残りではない）
  ・最小の位置 λ* を r_s ごとに追えば、「w を決めている長さ」が
    Thomas–Fermi 遮蔽長（コード単位で ∝ 1/(2 r_s)、r_s とともに縮む）なのか
    相関長（コード単位でほぼ一定〜漸増）なのかを判別できる
  ・--perhead は各ヘッドを単独で ±30% 動かして dE/dln w_h（応答としての重要度）を出す
"""
import argparse, glob, json, os, sys
import numpy as np, jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from tv_split import build, _FND, _err                     # noqa: E402
from periodicwave import mcmc, CustomScreenformer as CS     # noqa: E402


def scaled(params, lam, head=None):
    """mask_omega を書き換えて w を λ 倍する。head を指定するとそのヘッドだけ。"""
    p = jax.tree_util.tree_map(lambda x: x, params)
    atts = p["layers"]["attention"]
    new = []
    for l in range(len(atts)):
        om = atts[l]["mask_omega"]
        sp = jax.nn.softplus(om)
        f = jnp.full_like(sp, lam) if head is None else \
            jnp.ones_like(sp).at[head].set(lam)
        om2 = jnp.log(jnp.expm1(jnp.clip(f * sp, 1e-6, None)))
        new.append({**atts[l], "mask_omega": om2})
    p["layers"] = {**p["layers"], "attention": new}
    return p


def energy(b, params, blocks, mcmc_steps, seed, pos0):
    d = b["data"]
    spins, atoms, chg = (jnp.asarray(d[k][0]) for k in ("spins", "atoms", "charges"))
    step = mcmc.make_mcmc_step(jax.vmap(b["logabs"], in_axes=(None, 0, 0, 0, 0)),
                               pos0.shape[0], steps=mcmc_steps, ndim=b["ndim"])
    bT = jax.jit(jax.vmap(b["kinetic"], in_axes=(None, 0, 0, 0, 0)))
    bV = jax.jit(jax.vmap(b["potential"], in_axes=(0, 0)))
    key = jax.random.PRNGKey(seed)
    pos = pos0
    for _ in range(3):                                     # 新しい w での再平衡
        key, s = jax.random.split(key)
        pos = step(params, _FND(pos, spins, atoms, chg), s, 0.2)[0].positions
    E = []
    for _ in range(blocks):
        key, s = jax.random.split(key)
        pos = step(params, _FND(pos, spins, atoms, chg), s, 0.2)[0].positions
        E.append(np.asarray(bT(params, pos, spins, atoms, chg)) +
                 np.asarray(bV(pos, atoms)))
    return np.concatenate(E)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("folder")
    ap.add_argument("--lam", default="0.3,0.5,0.7,0.85,1.0,1.2,1.5,2.0,3.0")
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--mcmc", type=int, default=20)
    ap.add_argument("--perhead", action="store_true")
    a = ap.parse_args()

    b = build(a.folder)
    pos0 = jnp.asarray(b["data"]["positions"][0])
    cfg = json.load(open(os.path.join(a.folder, "config.json")))
    o = cfg["network"]["CustomScreenformer"]
    NL, NH = o["num_layers"], o["num_heads"]
    dref = float(CS._max_periodic_distance(jnp.asarray(
        np.array([float(x) for x in __import__("re").findall(
            r"-?\d+\.?\d*(?:[eE][-+]?\d+)?",
            __import__("re").search(r"unserializable object: \[\[(.*?)\]\]",
                                    json.dumps(cfg)).group(1))][:4]).reshape(2, 2))))
    w0 = np.asarray(dref * jax.nn.softplus(np.stack(
        [b["params"]["layers"]["attention"][l]["mask_omega"] for l in range(NL)])))

    lams = [float(x) for x in a.lam.split(",")]
    print("  %s   r_s=%g  N=%d  %d層×%dヘッド  d_ref=%.3f"
          % (os.path.basename(a.folder), b["r_s"], b["ne"], NL, NH, dref))
    print("  学習後の w/d_ref =", np.round(w0.mean(0) / dref, 3), "\n")
    print("     λ      w̄/d_ref      E          ±       ΔE = E(λ)−E(1)")
    out = {}
    for lam in lams:
        E = energy(b, scaled(b["params"], lam), a.blocks, a.mcmc, 0, pos0)
        out[lam] = (E.mean(), _err(E))
        print("   %5.2f   %7.3f   %10.3f  %7.3f" % (lam, lam * w0.mean() / dref,
                                                    E.mean(), _err(E)), flush=True)
    e1 = out[1.0][0]
    print()
    for lam in lams:
        print("   λ=%-5.2f  ΔE = %+8.3f  (±%.3f)" % (lam, out[lam][0] - e1, out[lam][1]))
    np.save(os.path.join(a.folder, "w_response.npy"),
            np.array([[l, out[l][0], out[l][1]] for l in lams]))

    if a.perhead:
        print("\n  ヘッド単独 ±30%%（層は共通で動かす）  dE/dln w_h ≈ [E(1.3)−E(0.7)]/ln(1.3/0.7)")
        rows = []
        for h in range(NH):
            lo = energy(b, scaled(b["params"], 0.7, h), a.blocks, a.mcmc, 0, pos0)
            hi = energy(b, scaled(b["params"], 1.3, h), a.blocks, a.mcmc, 0, pos0)
            g = (hi.mean() - lo.mean()) / np.log(1.3 / 0.7)
            rows.append((h, w0.mean(0)[h] / dref, lo.mean() - e1, hi.mean() - e1, g))
            print("    H%-2d w/d=%.2f   ΔE(0.7)=%+7.3f  ΔE(1.3)=%+7.3f   dE/dlnw = %+7.3f"
                  % rows[-1], flush=True)
        np.save(os.path.join(a.folder, "w_perhead.npy"), np.array(rows))


if __name__ == "__main__":
    main()
