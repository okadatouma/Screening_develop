"""解析ルーチンの自己検査。合成配置を入れて、期待どおりの値が出るか見る。

    python screening_analysis/test_moire.py <既存の run フォルダ>
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import moire_analysis as MA


def synth(d, kind, nb=256, jitter=0.10, seed=0):
    """kind='honeycomb' … 3 サイトに 2 つ埋める理想秩序、'liquid' … 一様乱数。"""
    rng = np.random.default_rng(seed)
    L, MO, ne = d["L"], d["MO"], d["ne"]
    if kind == "liquid":
        f = rng.random((nb, ne, 2))
        return (f @ L.T)
    sites, sub = [], []
    for m in range(-40, 41):
        for n in range(-40, 41):
            r = m * MO[:, 0] + n * MO[:, 1]
            f = np.mod(np.linalg.solve(L, r), 1.0)
            f = np.where((f < 1e-7) | (f > 1 - 1e-7), 0.0, f)
            k = (round(float(f[0]), 5), round(float(f[1]), 5))
            if k in [tuple(x) for x in sub]: continue
            sub.append(list(k)); sites.append((f @ L.T, (m + n) % 3))
    keep = np.array([p for p, s in sites if s in (0, 1)])
    assert len(keep) == ne, (len(keep), ne)
    R = np.repeat(keep[None], nb, 0) + jitter * rng.normal(size=(nb, ne, 2))
    return R


if __name__ == "__main__":
    d = MA.load(sys.argv[1])
    print("  セル %d サイト、電子 %d 個" % (
        round(abs(np.linalg.det(d["L"])) / abs(np.linalg.det(d["MO"]))), d["ne"]))
    for kind in ("honeycomb", "liquid"):
        dd = dict(d); dd["R"] = synth(d, kind)
        op = MA.order_parameter(dd); oc = MA.occupation_summary(dd)
        print("  %-10s S(K)/bg = %6.2f   S(G)/bg = %6.2f   占有平均 %.3f  "
              "コントラスト %.3f  占有分布 %s"
              % (kind, op["K_over_bg"], op["G_over_bg"], oc["mu"].mean(),
                 oc["contrast"], np.round(oc["hist"], 3)))
    # 完全秩序（ゆらぎなし）の上限
    dd = dict(d); dd["R"] = synth(d, "honeycomb", jitter=0.0)
    op = MA.order_parameter(dd)
    print("  完全秩序（ゆらぎ 0）: S(K)/bg = %.2f" % op["K_over_bg"])
    # 距離ルーチン
    L = d["L"]
    r = MA._scan(L, 200)
    de = MA.min_image_dist(r, L)
    print("  最小像距離の最大 %.4f （理論 a/√3 = %.4f）"
          % (de.max(), np.linalg.norm(L[:, 0]) / np.sqrt(3)))
    lad = np.geomspace(0.50, 2.50, 8)
    print("  窓 %s\n  → Euclid %s" % (np.round(lad, 3), np.round(MA.w_to_euclid(lad, L), 3)))
