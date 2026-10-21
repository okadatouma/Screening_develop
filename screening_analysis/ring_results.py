# -*- coding: utf-8 -*-
"""輪切り窓・V 固定の GPU run（ジョブ 2071–2073）を、これまでの run と並べてまとめる。

    python screening_analysis/ring_results.py <CustomScreenformer の結果フォルダ> [--out DIR]

出すもの
  1. 学習曲線：E と局所エネルギーの標準偏差 σ(E_L)（どちらも 25 電子の合計）
  2. 最後の 500 step の E（平均 ± ブロック平均の標準誤差）と σ(E_L)（中央値）
  3. 輪切りの run の読み出し q_h（a = 窓が集めた量、c = いまの定義）と利得 q_h(c)/q_h(a)
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))

# (記号, 説明, 窓, value, ヘッド数)
RUNS = [
    ("s", "入れ子窓", "学習", 16),
    ("t", "殻窓", "学習", 16),
    ("e", "輪切り τ=0.5", "固定", 8),
    ("v", "輪切り τ=1", "固定", 8),
    ("w", "輪切り τ=0.5", "学習", 8),
]
PHASES = [("l05", 5, "液体"), ("b30", 30, "境界"), ("c60", 60, "結晶")]
NE = 25


def folder(root, key, ph, rs):
    return os.path.join(root, f"el13_12_rs{rs}_tri_l2h16_{key}{ph}")


def load_stats(d):
    f = os.path.join(d, "train_stats.csv")
    if not os.path.exists(f):
        return None
    a = np.genfromtxt(f, delimiter=",", names=True)
    return a


def final(a, last=500, block=50, upto=6000):
    """step < upto の最後の last step：E の平均、block 平均から出した標準誤差、σ(E_L) の中央値。
    入れ子窓の run（ジョブ 1929）は 8000 step まで回っているので、6000 step で切って揃える。"""
    a = a[a["step"] < upto]
    e = a["energy"][-last:]; s = a["locstd"][-last:]
    nb = len(e) // block
    bm = e[:nb * block].reshape(nb, block).mean(1)
    return float(e.mean()), float(bm.std(ddof=1) / np.sqrt(nb)), float(np.median(s)), int(a["step"][-1])


def table(root):
    rows = []
    for key, win, val, nh in RUNS:
        for ph, rs, name in PHASES:
            a = load_stats(folder(root, key, ph, rs))
            if a is None:
                continue
            E, dE, S, st = final(a)
            rows.append(dict(key=key, win=win, val=val, nh=nh, ph=ph, rs=rs, name=name,
                             E=E, dE=dE, S=S, step=st, n=len(a)))
    return rows


def readouts(root, keys=("e", "v", "w")):
    import gain_readout as GR
    out = {}
    for key in keys:
        for ph, rs, name in PHASES:
            d = folder(root, key, ph, rs)
            if not glob.glob(os.path.join(d, "qmcjax_ckpt_*.npz")):
                continue
            a = load_stats(d)
            if a is None or a["step"][-1] < 5999:
                continue                     # 学習途中のものは読まない
            r = GR.readouts(d)
            out[(key, ph)] = {k: np.asarray(v).ravel() for k, v in r.items()}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root")
    ap.add_argument("--out", default=os.path.join(HERE, "out_rings"))
    ap.add_argument("--no-readout", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows = table(a.root)
    print("%-3s %-14s %-4s %3s %-4s %6s %12s %8s %9s" % ("", "窓", "V", "H", "相", "step", "E", "±", "σ(E_L)"))
    for r in rows:
        print("%-3s %-14s %-4s %3d %-4s %6d %12.2f %8.2f %9.1f"
              % (r["key"], r["win"], r["val"], r["nh"], r["name"], r["step"], r["E"], r["dE"], r["S"]))
    json.dump(rows, open(os.path.join(a.out, "final.json"), "w"), ensure_ascii=False, indent=1)
    if not a.no_readout:
        R = readouts(a.root)
        for (key, ph), r in sorted(R.items()):
            print(f"\n{key}{ph}:  ring mid d = " + " ".join("%5.2f" % (x * 3.2152) for x in r["w"]))
            for k in ("qa", "qc", "gain"):
                print("   %-4s " % k + " ".join("%5.3f" % x for x in r[k]))
        np.savez(os.path.join(a.out, "readouts.npz"),
                 **{f"{k}_{ph}_{q}": v for (k, ph), r in R.items() for q, v in r.items()})


if __name__ == "__main__":
    main()
