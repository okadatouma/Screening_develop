"""別の run のチェックポイントを種にして新しい run を始める。

    python screening_analysis/warm_start.py <src_folder> <dst_folder> [--w-lambda 1.4]

同じ形のネットワークなら、r_s やモアレの設定を変えても収束済みの波動関数から
始められる。--w-lambda を渡すと窓の梯子 [0.50λ, 2.50λ] だけを入れ替える。
窓は学習で動かないので、これは「同じ波動関数に別の窓を履かせる」操作になる。
"""
import argparse, glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)


def _dref(L):
    import jax.numpy as jnp
    from periodicwave import CustomScreenformer as CS
    return float(CS._max_periodic_distance(jnp.asarray(L)))


def _lat(cfg):
    m = re.search(r"unserializable object: \[\[(.*?)\]\]", json.dumps(cfg))
    return np.array([float(x) for x in re.findall(
        r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(1))][:4]).reshape(2, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--w-lambda", type=float, default=None)
    ap.add_argument("--lo", type=float, default=0.50)
    ap.add_argument("--hi", type=float, default=2.50)
    ap.add_argument("--t", type=int, default=0)
    ap.add_argument("--init", default=None, choices=("honeycomb",),
                    help="walker の初期配置を差し替える（波動関数はそのまま）")
    ap.add_argument("--jitter", type=float, default=0.12)
    ap.add_argument("--equil", type=int, default=0,
                    help="種を置いたあと walker だけ動かして落ち着かせる回数")
    a = ap.parse_args()

    ck = sorted(glob.glob(os.path.join(a.src, "qmcjax_ckpt_*.npz")))[-1]
    z = np.load(ck, allow_pickle=True)
    params = z["params"].tolist()
    msg = "params そのまま"
    if a.w_lambda is not None:
        cfg = json.load(open(os.path.join(a.src, "config.json")))
        dref = _dref(_lat(cfg))
        at = params["layers"]["attention"]
        H = np.asarray(at[0]["mask_omega"]).shape[-1]
        wf = np.geomspace(a.lo * a.w_lambda / dref, a.hi * a.w_lambda / dref, H)
        om = np.log(np.expm1(np.clip(wf, 1e-3, None))).astype(np.float32)
        for l in range(len(at)):
            old = np.asarray(at[l]["mask_omega"])
            at[l]["mask_omega"] = np.broadcast_to(om, old.shape).astype(old.dtype).copy()
        msg = ("窓 %.3f..%.3f -> %.3f..%.3f (λ=%g)"
               % (a.lo, a.hi, a.lo * a.w_lambda, a.hi * a.w_lambda, a.w_lambda))

    data = z["data"].item()
    if a.init == "honeycomb":
        import moire_analysis as MA
        d = MA.load(a.src)
        sites = MA.sites_in_cell(d)
        lab = np.rint(np.linalg.solve(d["MO"], sites.T).T).astype(int)
        sub = (lab[:, 0] + lab[:, 1]) % 3
        occ = sites[sub != 2]                       # 部分格子 2 を空ける
        ne = d["ne"]
        assert len(occ) == ne, (len(occ), ne)
        pos = np.asarray(data["positions"])         # (dev, walk, ne*2)
        rng = np.random.default_rng(0)
        new = np.empty_like(pos)
        for i in range(pos.shape[0]):
            for j in range(pos.shape[1]):
                r = occ[rng.permutation(ne)] + a.jitter * rng.normal(size=(ne, 2))
                new[i, j] = r.reshape(-1)
        data["positions"] = new
        msg += "、walker をハニカム配置から出発（ゆらぎ %.2f）" % a.jitter
        if a.equil:
            # 谷が深いとモンテカルロは谷を跳び移れないので、並びは保ったまま
            # 谷の中だけが落ち着く。これを済ませないと学習の出だしが荒れる。
            import jax, jax.numpy as jnp
            from periodicwave import networks, mcmc as MC
            import sampler as SM
            bnet, params2, d0, cfg2, Lm = SM.build(a.src)
            nw = new.shape[1]
            dat = networks.FermiNetData(
                positions=jnp.asarray(new[0]),
                spins=jnp.asarray(np.asarray(data["spins"])[0]),
                atoms=jnp.asarray(np.asarray(data["atoms"])[0]),
                charges=jnp.asarray(np.asarray(data["charges"])[0]))
            step = jax.jit(MC.make_mcmc_step(bnet, nw, steps=10, ndim=2))
            # 谷の割り当てが変わった walker は元に戻す（並びを保ったまま中だけ落ち着かせる）
            sites = MA.sites_in_cell(d)

            def pattern(pos):
                RR = np.asarray(pos).reshape(nw, ne, 2)
                dr = RR[:, :, None, :] - sites[None, None, :, :]
                fr = dr @ np.linalg.inv(d["L"]).T
                dd_ = np.linalg.norm((fr - np.round(fr)) @ d["L"].T, axis=-1)
                who = np.argmin(dd_, axis=-1)
                out = np.zeros((nw, len(sites)))
                for b in range(nw): np.add.at(out[b], who[b], 1.0)
                return out
            p0 = pattern(dat.positions)
            key = jax.random.PRNGKey(0); wdt = jnp.asarray(0.05); pm = 0.0
            for i in range(a.equil):
                key, sk = jax.random.split(key)
                prev = np.array(dat.positions)
                dat, pm = step(params2, dat, sk, wdt)
                cur = np.array(dat.positions)
                bad = (pattern(cur) != p0).any(axis=1)
                if bad.any():
                    cur[bad] = prev[bad]
                    dat = networks.FermiNetData(**(dict(dat) | {"positions": jnp.asarray(cur)}))
                if (i + 1) % 2 == 0:
                    wdt = wdt * (1.15 if pm > .55 else (1 / 1.15 if pm < .45 else 1.0))
            data["positions"] = np.asarray(dat.positions)[None]
            msg += "、%d 回ならし（受理率 %.2f、幅 %.3f）" % (a.equil, float(pm), float(wdt))

    os.makedirs(a.dst, exist_ok=True)
    out = os.path.join(a.dst, "qmcjax_ckpt_%06d.npz" % a.t)
    with open(out, "wb") as fh:
        np.savez(fh, t=a.t, data=data, params=params,
                 opt_state=np.asarray(z["opt_state"].tolist(), dtype=object),
                 mcmc_width=z["mcmc_width"])
    print("%s -> %s  (%s)" % (os.path.basename(ck), out, msg))


if __name__ == "__main__":
    main()
