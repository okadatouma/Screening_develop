# -*- coding: utf-8 -*-
"""どの窓が液体と結晶をよく見分けるか：読み出し q_h の相の差を、ウォーカーを分けた誤差つきで比べる。

    python screening_analysis/phase_contrast.py <CustomScreenformer の結果フォルダ> [--groups 8]

各 run のチェックポイントに残っている 1024 本のウォーカーを K 組に分け、組ごとに
  q_h^(c) = |c_h|^2 の割合（相手から集めて波動関数に足した量）
  q_h^(a) = |a_h|^2 の割合（窓で集めた直後、ほぼ幾何）
を出す。液体と結晶の差は全変動距離 TV = (1/2) sum_h |q_h^L - q_h^C|。
同じ相の組どうしの TV を「雑音の床」とし、相の差がそれの何倍かを見る。
境界 r_s = 30 の位置 λ は、液体 → 結晶の直線への射影（0 = 液体、1 = 結晶）。
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import head_contrib as HC

RUNS = [("s", "入れ子窓・V 学習（16 ヘッド）"), ("t", "殻窓・V 学習（16 ヘッド）"),
        ("e", "輪切り τ=0.5・V 固定"), ("v", "輪切り τ=1・V 固定"), ("w", "輪切り τ=0.5・V 学習")]
PH = [("l05", 5), ("b30", 30), ("c60", 60)]


def folder(root, key, ph, rs):
    return os.path.join(root, f"el13_12_rs{rs}_tri_l2h16_{key}{ph}")


def groups(d, K):
    ck = sorted(glob.glob(os.path.join(d, "qmcjax_ckpt_*.npz")))[-1]
    nw = np.load(ck, allow_pickle=True)["data"].item()["positions"][0].shape[0]
    out_c, out_a = [], []
    for g in range(K):
        C, Cd, Cp, X, R, w, sm, mt = HC.compute(d, sel=np.arange(g, nw, K))
        pc = Cp[0] ** 2; pa = mt["Ap"][0] ** 2
        out_c.append(pc / pc.sum()); out_a.append(pa / pa.sum())
    return np.array(out_c), np.array(out_a), nw


def tv(p, q):
    return 0.5 * np.abs(p - q).sum(-1)


def summary(QL, QB, QC):
    K = len(QL)
    d = tv(QL, QC)                                    # 組ごとの液体–結晶の差
    fl = np.concatenate([tv(Q[: K // 2], Q[K // 2:]) for Q in (QL, QC)])   # 同じ相どうし
    mL, mB, mC = QL.mean(0), QB.mean(0), QC.mean(0)
    u = mC - mL
    lam = float(np.dot(mB - mL, u) / np.dot(u, u))
    lam_g = np.dot(QB - mL, u) / np.dot(u, u)
    return dict(tv=float(d.mean()), tv_err=float(d.std(ddof=1) / np.sqrt(K)),
                floor=float(fl.mean()), lam=lam, lam_err=float(lam_g.std(ddof=1) / np.sqrt(K)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root")
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "out_rings", "phase_contrast.json"))
    a = ap.parse_args()
    res = {}
    for key, name in RUNS:
        Q = {}
        for ph, rs in PH:
            Q[ph] = groups(folder(a.root, key, ph, rs), a.groups)
        sc = summary(Q["l05"][0], Q["b30"][0], Q["c60"][0])
        sa = summary(Q["l05"][1], Q["b30"][1], Q["c60"][1])
        res[key] = dict(name=name, c=sc, a=sa, walkers=int(Q["l05"][2]),
                        qc={ph: Q[ph][0].mean(0).tolist() for ph, _ in PH},
                        qa={ph: Q[ph][1].mean(0).tolist() for ph, _ in PH})
        print(f"{name:28s} TV(c) {sc['tv']:.3f}±{sc['tv_err']:.3f} (床 {sc['floor']:.3f}, {sc['tv']/sc['floor']:5.1f} 倍)"
              f" | TV(a) {sa['tv']:.3f}±{sa['tv_err']:.3f} (床 {sa['floor']:.3f})"
              f" | c/a {sc['tv']/sa['tv']:.2f} | λ(c) {sc['lam']:.2f}±{sc['lam_err']:.2f}  λ(a) {sa['lam']:.2f}")
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
