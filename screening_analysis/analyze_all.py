"""終わった run をまとめて解析し、図を作る。

    python screening_analysis/analyze_all.py --pair eps10 eps5      # Fig.6 と q·k と到達距離
    python screening_analysis/analyze_all.py --scan rs2 eps10 rs5 eps5 rs9 rs14   # 主図
    python screening_analysis/analyze_all.py --nu  ns36 eps5 ns21 ns12 ns9        # a_M スキャン
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
S = "results/2deg-Coulomb/CustomScreenformer/el9_9_rs%s_tri_l2h16_%s"
P = "results/2deg-Coulomb/CustomPsiformer/el9_9_rs%s_tri_l2h16_%s"


def folder(tag):
    for pat in ("results/2deg-Coulomb/*/el9_9_rs*_tri_l2h16_%s" % tag,):
        g = sorted(glob.glob(os.path.join(ROOT, pat)))
        if g: return g[0]
    raise SystemExit("見つからない: " + tag)


def sh(*cmd):
    print("\n$", " ".join(cmd)); subprocess.run(cmd, cwd=ROOT, check=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, default=None)
    ap.add_argument("--scan", nargs="+", default=None)
    ap.add_argument("--nu", nargs="+", default=None)
    ap.add_argument("--lam", nargs="+", default=None)
    ap.add_argument("--nconf", type=int, default=6144)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    py = sys.executable
    F = lambda t: folder(t)                              # noqa: E731
    ext = ["--force"] if a.force else []
    if a.pair:
        f = [F(t) for t in a.pair]
        sh(py, "screening_analysis/fig_fig6.py", *f, "--nconf", str(a.nconf),
           "--labels", "ε=10（弱い相互作用）,ε=5（強い相互作用）",
           "--out", "screening_analysis/fig_fig6.png")
        sh(py, "screening_analysis/fig_qk.py", *f, "--labels", "ε=10,ε=5",
           "--out", "screening_analysis/fig_qk.png")
        sh(py, "screening_analysis/fig_kernel.py", *f, "--labels", "ε=10,ε=5",
           "--out", "screening_analysis/fig_kernel.png")
    if a.scan:
        f = [F(t) for t in a.scan]
        sh(py, "screening_analysis/moire_summary.py", *f, "--nconf", str(a.nconf), *ext)
        sh(py, "screening_analysis/fig_order.py", *f, "--x", "rs",
           "--nconf", str(a.nconf), "--out", "screening_analysis/fig_order.png")
    if a.nu:
        f = [F(t) for t in a.nu]
        sh(py, "screening_analysis/moire_summary.py", *f, "--nconf", str(a.nconf), *ext)
        sh(py, "screening_analysis/fig_order.py", *f, "--x", "aM",
           "--nconf", str(a.nconf), "--out", "screening_analysis/fig_nu.png")
    if a.lam:
        f = [F(t) for t in a.lam]
        sh(py, "screening_analysis/fig_lambda.py", *f, "--nconf", str(a.nconf),
           "--out", "screening_analysis/fig_lambda.png")
