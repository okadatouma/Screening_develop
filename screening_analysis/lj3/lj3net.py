"""lj2・lj3 の run（チェックポイント）から網を作り直す（フォルダで指定）。新しいコード（入力 12 成分に対応、従来と後方互換）を使う。
窓と Trim の形は環境変数で決まるので import の前に設定する。"""
import os, sys, glob, json
os.environ.setdefault("PW_WIN_SHAPE", "smooth7"); os.environ.setdefault("PW_TRIM_POW", "4")
sys.path.insert(0, '/home/claude/gh_repo_probe'); sys.path.insert(0, '/home/claude/gh_repo_probe/screening_analysis')
import numpy as np, jax, jax.numpy as jnp
from periodicwave import CustomScreenformer as CS, envelopes, networks
from periodicwave.pbc import feature_layer as FL
import sampler as SM
assert CS.WIN_SHAPE == "smooth7" and CS.TRIM_POW == 4
BASE = '/home/claude/lj3res/results/lambda-jellium/CustomScreenformer'
RUNS = [('0.33', 'WC', 'cw33', 'hw33'), ('0.33', 'AHC', 'ca33', 'ha33'), ('0.72', 'WC', 'cw72', 'hw72'), ('0.72', 'AHC', 'ca72', 'ha72')]


def folder(lam, tag):
    return f"{BASE}/N36_rs15_lam{lam}_{tag}"


def load(lam, tag, checkpoint=None):
    fd = folder(lam, tag)
    ck = checkpoint or sorted(glob.glob(os.path.join(fd, "qmcjax_ckpt_*.npz")))[-1]
    z = np.load(ck, allow_pickle=True)
    data = z["data"].item()
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(np.asarray(x)[0]), z["params"].item())
    cfg = json.load(open(os.path.join(fd, "config.json")))
    L = SM._mats(cfg)[0]
    kw = SM._net_kwargs(cfg, "CustomScreenformer", params)
    kw["lj_hf_path"] = "/home/claude/ljhf/" + os.path.basename(kw["lj_hf_path"])
    charges = jnp.asarray(np.asarray(data["charges"][0])[0])
    fl = FL.make_pbc_feature_layer(natoms=1, nspins=(36, 0), ndim=2, lattice=jnp.asarray(L), include_r_ae=False)
    net = CS.make_fermi_net((36, 0), charges, ndim=2, determinants=cfg["network"]["determinants"],
                            envelope=envelopes.make_null_envelope(), feature_layer=fl, jastrow="NONE",
                            jastrow_kwargs=cfg["network"]["jastrow_kwargs"], bias_orbitals=cfg["network"]["bias_orbitals"],
                            complex_output=True, pbc_lattice=jnp.asarray(L), **kw)
    R = np.asarray(data["positions"])[0].astype(np.float64)       # (W, 72)
    S = np.asarray(data["spins"])[0].astype(np.float64)           # (W, 36)
    return dict(net=net, params=params, cfg=cfg, L=np.asarray(L), R=R, S=S, kw=kw, ck=ck, tag=tag)
