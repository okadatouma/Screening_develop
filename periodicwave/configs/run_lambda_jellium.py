# -*- coding: utf-8 -*-
"""
[λ-jellium] 擬スピン付き 2 成分電子（Soejima et al., arXiv:2503.12704）を CustomScreenformer で解く。

    python run_lambda_jellium.py <N> <r_s> <lam> [<iterations> [<tag>]]

例（Valenti et al., arXiv:2512.07947 の相図で r_s = 15 は lam = 0 液体、0.33 WC、0.72 AHC）
    python run_lambda_jellium.py 36 15 0.72 6000 ah15n
    python run_lambda_jellium.py 36 15 0.33 6000 wc15n

系
  * 電子の実スピンは偏極、各電子が擬スピン s = ±1 を持つ（nspins = (N, 0) の行列式 1 枚、s は入力特徴）。
  * H = sum_i [-1/2 lap_i + Delta h(i)] + r_s sum 1/r（コード単位、pbc/lambda_jellium.py の冒頭を参照）。
    Delta = r_s * PW_DELTA_MULT（既定 1 = 論文の 2/r_s Ry）。
  * スーパーセルは三角格子の L x L（N = L^2、電子 1 個あたり面積 pi、ν = 1 の Wigner 結晶と整合）。
    C3 の固有値で Chern 数を読むなら L を 3 の倍数に（N = 9, 36）。
  * 結果：results/lambda-jellium/CustomScreenformer/N{N}_rs{r_s}_lam{lam}_{tag}

環境変数（run_2deg_moire.py と同じ意味）
  PW_BATCH PW_LR PW_LR_DELAY PW_LAYERS PW_HEADS PW_DET PW_FREEZE_W PW_SEED PW_SEED_FROM_TAG
  PW_W_LO / PW_W_HI   入れ子窓の幅 w_h の最小・最大（周期ノルム）。既定は a = 1.905（最近接）〜 d_ref（セルで決まる最大距離）
  PW_ENV=1            全体窓の輪切り（PW_ENV_P、PW_ENV_TAU、PW_ENV_EDGES）
  PW_ENV_KERNEL=gauss 輪切りの階段をガウス核でぼかす（標準偏差 = PW_ENV_SIGMA x 輪の幅、既定 0.5）
  PW_FIXV=1           value を対の固定特徴 e(r_ij) に
λ-jellium だけのもの
  PW_DELTA_MULT       Delta を 2/r_s Ry の何倍にするか（既定 1）
  PW_FLIPS            位置の一歩ごとの擬スピン反転の提案回数（既定 3）
  PW_FLIP_BATCH       反転項を何電子ずつ処理するか（0 = 全部まとめて。メモリが足りなければ 6 など）
  PW_EL_CHUNK         局所エネルギーを何ウォーカーずつ計算するか（既定 128、0 = 全部まとめて）
  PW_INIT_SPIN        'up'（既定、全員 +1）/ 'random'
  PW_LAPL             'folx'（既定）/ 'default'
  PW_BURN_IN          焼きなましの回数（既定 100）
  PW_COULOMB=0        Coulomb を切る（検算用。自由粒子の基底エネルギーは lam によらず sum_k k^2/2）
結晶から出発する版（2026-10-16、lj2）
  PW_LJ_HF=<表>       下のバンドに射影した HF の Bloch 軌道の表（pbc/lj_hf.py の orbitals_text）。軌道を
                      phi = phi^HF + eps [W_s h]、擬スピンの組ごとの Jastrow（カスプつき）にする（lj_orbitals.py）。
                      ウォーカーは HF の密度の山（格子点）に置いて始める。
  PW_LJ_NETSCALE      網の部分の初期の大きさ eps（既定 0.05）
  PW_LJ_JASTROW=0     擬スピンの組ごとの Jastrow を外す（検算用）
  PW_WIN_SHAPE=smooth7, PW_TRIM_POW=4   窓と Trim を 3 階微分まで連続に（CustomScreenformer.py）
  PW_SAVE_MIN         チェックポイントの間隔（分、既定 2）。チェックポイントは消さずに溜まるので、長い run では大きく
"""

# 実行用config：下でbase_configの既定値を上書きし、最後にtrainerを呼び出す。
# 単なる定数定義ではなく、読み込み時に保存先の準備・設定保存・学習が実行されるスクリプト。
# 物理条件はcfg.system、モデル構造はcfg.network、標本化はcfg.mcmc、更新則はcfg.optimを確認する。
import os
import sys
from sys import argv
from time import time

import numpy as np
import ml_collections
from absl import logging

from periodicwave import base_config
from periodicwave import train_screen as train
from periodicwave.utils import custom_logging
from periodicwave.utils import writers
from periodicwave.pbc import lattices

import jax
print("Jax Devices:", jax.devices())
jax.config.update("jax_default_matmul_precision", "float32")


def _flag(name, default="0"):
    return os.environ.get(name, default).lower() not in ("0", "", "false")


def format_for_folder_name(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


# ---------------- 引数 ----------------
N, r_s, lam, iterations, tag = 9, 15.0, 0.72, 200000, ""
if len(argv) > 1:
    if len(argv) < 4:
        raise ValueError("usage: run_lambda_jellium.py <N> <r_s> <lam> [<iterations> [<tag>]]")
    N, r_s, lam = int(argv[1]), float(argv[2]), float(argv[3])
    if len(argv) > 4:
        iterations = int(argv[4])
    if len(argv) > 5:
        tag = argv[5]
if iterations < 1:
    raise ValueError("iterations must be positive")
L = int(round(np.sqrt(N)))
if L * L != N:
    raise ValueError(f"N は L^2（三角格子の L x L セル）に限る。N = {N}")
if L % 3:
    print(f"[λJ] 注意：L = {L} は 3 の倍数でない。C3 固有値による Chern 数の判定には使えない。")

for _k in ("PW_PEAK", "PW_BANDS"):
    if _flag(_k):
        raise ValueError(f"{_k} は run_lambda_jellium.py では未対応（入れ子窓か PW_ENV の輪切りを使う）。")

delta_mult = float(os.environ.get("PW_DELTA_MULT", 1.0))
delta = r_s * delta_mult                      # コード単位の Delta（論文の 2/r_s Ry が r_s）
spin_flips = int(os.environ.get("PW_FLIPS", 3))
flip_batch = int(os.environ.get("PW_FLIP_BATCH", 0))
init_spin = os.environ.get("PW_INIT_SPIN", "up")
lapl = os.environ.get("PW_LAPL", "folx")
if init_spin not in ("up", "random"):
    raise ValueError("PW_INIT_SPIN は up か random")

# ---------------- 系 ----------------
cfg = base_config.default()
cfg.system.electrons = (N, 0)
cfg.system.ndim = 2
a_wc = np.sqrt(2 * np.pi / np.sqrt(3))        # 電子 1 個あたり面積 pi の三角格子（コード単位）
supercell_lattice, wc_lattice = lattices._triangular_lattice_vecs_periodic_potential(a_wc, N)
print(f"[λJ] N = {N} ({L}x{L}), r_s = {r_s}, lam = {lam}, Delta = {delta} (= {delta_mult} x 2/r_s Ry), "
      f"a = {a_wc:.4f}, flips/step = {spin_flips}, init spin = {init_spin}")

cfg.system.make_local_energy_fn = "periodicwave.pbc.lambda_jellium.local_energy"
cfg.system.make_local_energy_kwargs = {
    "lattice": supercell_lattice,
    "lam": lam,
    "delta": delta,
    "potential_type": "Coulomb" if _flag("PW_COULOMB", "1") else "none",   # 0 は検算用（自由粒子）
    "potential_kwargs": {"interaction_energy_scale": r_s},
    "kinetic_kwargs": {"laplacian_method": lapl},
    "flip_batch": flip_batch,
}
cfg.system.pbc_lattice = supercell_lattice
cfg.system.pseudospin = True
cfg.system.init_pseudospin = init_spin

cfg.network.make_feature_layer_fn = "periodicwave.pbc.feature_layer.make_pbc_feature_layer"
cfg.network.make_feature_layer_kwargs = {"lattice": supercell_lattice, "include_r_ae": False}
cfg.network.make_envelope_fn = "periodicwave.envelopes.make_null_envelope"

cfg.batch_size = int(os.environ.get("PW_BATCH", 64))
cfg.optim.optimizer = 'kfac'
cfg.optim.objective = 'vmc'
cfg.optim.iterations = iterations
cfg.optim.lr.rate = float(os.environ.get("PW_LR", 0.1))
cfg.optim.lr.delay = float(os.environ.get("PW_LR_DELAY", 20000))
cfg.optim.lr.decay = 1.0
# 局所エネルギーをウォーカー何本ずつ計算するか（0 = 全部まとめて）。N = 36 の前進ラプラシアンと
# 反転項は 1 本あたり ~80 MB（CPU で実測）なので、1024 本を一度にやると GPU 1 枚に収まらない。
cfg.optim.local_energy_chunk = int(os.environ.get("PW_EL_CHUNK", 128))
cfg.log.save_frequency = float(os.environ.get("PW_SAVE_MIN", 2.0))   # チェックポイントの間隔（分）。lj2 の長い run は 15

cfg.mcmc.burn_in = int(os.environ.get("PW_BURN_IN", 100))
cfg.mcmc.init_width = 1.0
cfg.mcmc.move_width = 0.2
cfg.mcmc.steps = 10
cfg.mcmc.move_width_updater = 'adaptive'
cfg.mcmc.spin_flips = spin_flips

cfg.network.complex = True        # 必須（下成分の lam q_+ は複素、AHC は時間反転を破る）
cfg.network.determinants = int(os.environ.get("PW_DET", 4))
cfg.network.jastrow = "NONE"
cfg.network.jastrow_kwargs = {'ndim': cfg.system.ndim}

# ---------------- CustomScreenformer（窓は run_2deg_moire.py と同じ定義） ----------------
import jax.numpy as _jnp
from periodicwave import CustomScreenformer as _CS
_dref = float(_CS._max_periodic_distance(_jnp.asarray(supercell_lattice)))
_H = int(os.environ.get("PW_HEADS", 8))
_lo = float(os.environ.get("PW_W_LO", a_wc))
_hi = float(os.environ.get("PW_W_HI", _dref))
w_factor = float(np.sqrt(_lo * _hi) / _dref)
w_spread = float(np.sqrt(_hi / _lo))
_rings = None
if _flag("PW_ENV"):
    _ep = float(os.environ.get("PW_ENV_P", 3.0))
    _et = float(os.environ.get("PW_ENV_TAU", 0.5))
    _ee = os.environ.get("PW_ENV_EDGES", "")
    _eb = (np.array([float(v) for v in _ee.split(";" if ";" in _ee else ",")])
           if _ee else np.linspace(0.0, _dref, _H + 1))
    if _eb.shape[0] != _H + 1:
        raise ValueError("PW_ENV_EDGES は PW_HEADS + 1 個必要")
    _ek = os.environ.get("PW_ENV_KERNEL", "sin").lower()
    if _ek not in ("sin", "gauss"):
        raise ValueError("PW_ENV_KERNEL は sin か gauss")
    if _ek == "gauss":
        _esg = float(os.environ.get("PW_ENV_SIGMA", 0.5)) * float(np.diff(_eb).min())
        _rings = (_eb / _dref, _esg / _dref, _ep, "gauss")
    else:
        _edl = _et * float(np.diff(_eb).min()) / 2
        _rings = (_eb / _dref, _edl / _dref, _ep)
_fixv = _flag("PW_FIXV")
_frz = "   [FROZEN]" if _flag("PW_FREEZE_W") else "   [learnable]"
print(f"[λJ] d_ref = {_dref:.4f}, value = " + ("固定 e(r_ij)" if _fixv else "学習 W_V"))
if _rings is not None:
    print("[λJ] 全体窓の輪切り: 境界 b_h = " + ", ".join(f"{v:.3f}" for v in _rings[0] * _dref)
          + (f"   ガウス核 sigma = {_rings[1] * _dref:.4f}" if len(_rings) > 3
             else f"   delta = {_rings[1] * _dref:.4f}") + f"   p = {_rings[2]:g}" + _frz)
else:
    print("[λJ] 入れ子窓 w_h = " + ", ".join(f"{v:.3f}" for v in np.geomspace(_lo, _hi, _H)) + _frz)

cfg.network.network_type = 'CustomScreenformer'
cfg.network.CustomScreenformer = ml_collections.ConfigDict({
    'num_layers': int(os.environ.get("PW_LAYERS", 1)),
    'num_heads': _H,
    'attn_dim': 16,
    'value_dim': 16,
    'mlp_dim': 64,
    'num_perceptrons_per_layer': 2,
    'use_layer_norm': True,
    'mlp_activation_fct': "GELU",
    'use_softmask': True,
    'use_gate': True,
    'use_tanh_norm': True,
    'use_output_scale': True,
    'init_trim_r': 1.9,
    'init_output_scale': 0.0,
    'init_softmask_w_factor': w_factor,
    'init_softmask_w_spread': w_spread,
    'init_softmask_centres': None,
    'init_softmask_rings': _rings,
    'use_fixed_value': _fixv,
})

# ---------------- [lj2] HF の結晶から出発 ----------------
_hf_path = os.environ.get("PW_LJ_HF", "")
if _hf_path:
    from periodicwave import lj_orbitals as _ljo
    _cusps = _ljo.cusp_values(r_s, lam, delta)
    _use_j = _flag("PW_LJ_JASTROW", "1")
    cfg.network.CustomScreenformer.lj_hf_path = _hf_path
    cfg.network.CustomScreenformer.lj_lam = float(lam)
    cfg.network.CustomScreenformer.lj_cusps = tuple(float(v) for v in _cusps)
    cfg.network.CustomScreenformer.lj_net_scale = float(os.environ.get("PW_LJ_NETSCALE", 0.05))
    cfg.network.CustomScreenformer.lj_jastrow = bool(_use_j)
    _k, _G, _c = _ljo.load_hf_table(_hf_path)
    # HF の密度 rho(r) = sum_j |phi_j(r, +)|^2 + |phi_j(r, -)|^2 の山を単位胞の格子で探し、格子点の列を作る
    _A = wc_lattice
    _f = np.linspace(0, 1, 49, endpoint=False)
    _pts = np.array([[_A[0, 0] * u + _A[0, 1] * v, _A[1, 0] * u + _A[1, 1] * v] for u in _f for v in _f])
    _q = _k[:, None, :] + _G[None, :, :]
    _nr = 1.0 / np.sqrt(1.0 + lam ** 2 * np.sum(_q ** 2, -1))
    _qp = _q[..., 0] + 1j * _q[..., 1]
    _ph = np.exp(1j * np.einsum('pd,kgd->pkg', _pts, _q))
    _rho = (np.abs(np.einsum('pkg,kg->pk', _ph, _c * _nr)) ** 2
            + np.abs(np.einsum('pkg,kg->pk', _ph, _c * lam * _qp * _nr)) ** 2).sum(1)
    _r0 = _pts[int(np.argmax(_rho))]
    _sites = np.array([_r0 + i * _A[:, 0] + j * _A[:, 1] for i in range(L) for j in range(L)])
    cfg.system.init_sites = [float(v) for v in _sites.reshape(-1)]
    cfg.system.init_sites_width = float(os.environ.get("PW_INIT_SITES_WIDTH", 0.3))
    print(f"[λJ] lj2：HF の表 {_hf_path}、カスプ (++, --, +-) = " + ", ".join(f"{v:.4f}" for v in _cusps)
          + f"、Jastrow {'あり' if _use_j else 'なし'}、網の初期の大きさ {cfg.network.CustomScreenformer.lj_net_scale}、"
          f"密度の山 r0 = ({_r0[0]:.3f}, {_r0[1]:.3f})、山/谷 = {_rho.max() / max(_rho.min(), 1e-12):.2f}")

# ---------------- 保存先・種 ----------------
folder_name = (f"results/lambda-jellium/CustomScreenformer/"
               f"N{N}_rs{format_for_folder_name(r_s)}_lam{format_for_folder_name(lam)}")
if tag:
    folder_name += f"_{tag}"
cfg.log.save_path = folder_name

writers.rename_file("device_info", folder_name, file_extension="log")
custom_logging.log_device_info(folder_name + "/device_info.log")
writers.rename_file("config", folder_name, file_extension="json")
custom_logging.save_config_dict_as_json(cfg, cfg.log.save_path + '/config.json')

cfg.debug.deterministic = True
if _flag("PW_SEED_FROM_TAG"):
    import zlib
    os.environ["PW_SEED"] = str(zlib.crc32(str(tag).encode()) % 2147483647)
print(f"[SEED] seed = {os.environ.get('PW_SEED', '42')}  (tag = {tag})")

t_init = time()
train.train(cfg)
logging.info("Training completed after t [s] = " + str(int(time() - t_init)))
