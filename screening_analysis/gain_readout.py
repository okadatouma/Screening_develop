# -*- coding: utf-8 -*-
"""読み出しを 3 通りで比べる：幾何 / いまの定義 / 飽和を割り戻した定義。

    python screening_analysis/gain_readout.py <run_folder> [<run_folder> ...]

q_h(a) = ||a_h^pair||^2 / sum   … 窓が集めた量。ほぼ幾何（どこに相手がいるか）
q_h(c) = ||c_h^pair||^2 / sum   … いまの定義（TanhNorm・ゲート・W_O のあと）
q_h(g) … c を tanh|a|/|a| で割り戻したもの。残るのはゲートと W_O = 学習した係数
利得 q_h(c)/q_h(a) … 幾何を超えて波動関数が要求した分。1 が並べば幾何だけ。
"""
import argparse, os, sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import head_contrib as HC


def readouts(folder, nconf=64):
    z = HC.cached(folder, nconf)
    Cp, A, Ap, w = z["Cp"], z["A"], z["Ap"], z["w"]
    un = A / np.maximum(np.tanh(A), 1e-12)
    qa = (Ap ** 2) / (Ap ** 2).sum()
    qc = (Cp ** 2) / (Cp ** 2).sum()
    Cg = Cp * un
    qg = (Cg ** 2) / (Cg ** 2).sum()
    return dict(w=w, qa=qa, qc=qc, qg=qg, gain=qc / np.maximum(qa, 1e-12))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("folders", nargs="+")
    ap.add_argument("--nconf", type=int, default=64); a = ap.parse_args()
    R = [(os.path.basename(f.rstrip("/")), readouts(f, a.nconf)) for f in a.folders]
    print("run                      | <w>_q  a / c / g        | 参加ヘッド数 (c) | 利得の log 標準偏差 | 利得 最大/最小")
    for nm, r in R:
        m = [float((r[k] * r["w"]).sum()) for k in ("qa", "qc", "qg")]
        part = 1 / float((r["qc"] ** 2).sum())
        lg = np.log(np.maximum(r["gain"], 1e-12))
        print("%-24s | %.4f / %.4f / %.4f | %14.2f | %17.3f | %12.1f"
              % (nm[:24], *m, part, lg.std(), r["gain"].max() / r["gain"].min()))
    if len(R) >= 2:
        tv = lambda p, q: 0.5 * float(np.abs(p - q).sum())
        print("\n最初と最後の run のあいだの全変動距離:  a %.4f   c %.4f   g %.4f"
              % tuple(tv(R[0][1][k], R[-1][1][k]) for k in ("qa", "qc", "qg")))


if __name__ == "__main__":
    main()
