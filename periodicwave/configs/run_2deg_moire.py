# AUTO-GENERATED (cloud) batch=64 tag=ctl w=0.8/2.5
# Copyright (c) 2025 Max Geier, Massachusetts Institute of Technology, MA, USA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
[SCREEN] DUPLICATE of run_2deg.py that additionally supports the
'CustomScreenformer' network (screening / multiscreen attention).
The original run_2deg.py is left untouched. Differences:
  - imports train_screen instead of train,
  - 'CustomScreenformer' added to ALLOWED_NETWORK_TYPES (and made the default),
  - a cfg.network.CustomScreenformer config block (same sizes as
    CustomPsiformer, plus the screening knobs).

This code runs a calculation of a two-dimensional homogeneous electron gas with Coulomb interaction.

Usage:
    python run_2deg_screen.py [<nup>_<ndown> <r_s> <network_type> [<supercell_shape> [<iterations> [<tag>]]]]

Example:
    python run_2deg_screen.py 2_0 10 CustomScreenformer tri 600
    python run_2deg_screen.py 2_0 10 CustomScreenformer tri 600 v2

The optional <tag> is appended to the results folder name, e.g.
    results/2deg-Coulomb/CustomScreenformer/el2_0_rs10_tri_v2
so that several settings can be run side by side without overwriting each
other. IMPORTANT: give a new tag whenever the network parameters change
(e.g. after enabling use_output_scale), because a run restarts from any
checkpoint already present in its folder and old checkpoints do not contain
the new parameters.

Running the script without arguments uses the default parameters set below.

Note: the calculation restarts from the newest checkpoint if one is already
present in the results folder. To start a run from scratch, move that folder
aside first.
"""

import os
import sys
from sys import argv
import numpy as np
import ml_collections

from absl import logging
from periodicwave import base_config
from periodicwave import train_screen as train  # [SCREEN]

from periodicwave.utils import custom_logging
from periodicwave.utils import writers
from periodicwave.pbc import lattices

from typing import Tuple

from time import time

import jax
print("Jax Devices:", jax.devices())

# Always use float32 precision
print("Setting jax_default_matmul_precision to float32")
jax.config.update("jax_default_matmul_precision", "float32")
print(f"Matmul precision set to: {jax.default_matmul_precision}")

ALLOWED_NETWORK_TYPES = ("CustomPsiformer", "SlaterNet", "CustomScreenformer")  # [SCREEN]
ALLOWED_SUPERCELL_SHAPES = ("tri", "sq", "rect")


def format_for_folder_name(value: float) -> str:
    """Formats a number for use in a results folder name.

    Integer-valued parameters are written without a decimal point so that
    r_s = 10 and r_s = 10.0 map onto the same folder ("rs10" in both cases).
    """
    return str(int(value)) if float(value).is_integer() else str(value)


# Get parameters defining the physical system
print(f"Calculating 2DEG with parameters: {argv}")

# Default parameters, used when the script is called without command line arguments.
nspins = [2, 0]                     # number of spin up, spin down electrons
r_s = 10.0                          # ratio between interaction and kinetic energy scale
network_type = "CustomScreenformer" # "SlaterNet", "CustomPsiformer", "CustomScreenformer" [SCREEN]
supercell_shape = 'tri'             # "tri" (120 度菱形) / "rect" (三角格子と整合する長方形) / "sq"
iterations = 200000                 # number of optimization steps
tag = ""                            # [SCREEN] optional suffix for the results folder
n_sites = 27                        # [MOIRE] モアレの谷の数（スーパーセル内）
moire_U = 1.8374                    # [MOIRE] ポテンシャルの強さ（コード単位）
moire_phi = np.pi / 4               # [MOIRE] 位相
w_lambda = 1.0                      # [MOIRE] 窓の梯子全体に掛ける倍率

if len(argv) > 1:
    if len(argv) < 4:
        raise ValueError(
            "Expected at least three arguments: <nup>_<ndown> <r_s> <network_type> "
            "[<supercell_shape>]. Received: " + str(argv[1:]))
    program_name = argv[0]
    # number of spin up, spin down electrons
    nspins = [int(n) for n in argv[1].split('_')]
    if len(nspins) != 2:
        raise ValueError(
            "Expected the electron numbers as <nup>_<ndown>, e.g. 2_0. Received: " + str(argv[1]))
    # Ratio between interaction and kinetic energy scale
    r_s = float(argv[2])
    network_type = argv[3]
    supercell_shape = 'tri'
    if len(argv) > 4:
        n_sites = int(argv[4])
    if len(argv) > 5:
        moire_U = float(argv[5])
    if len(argv) > 6:
        moire_phi = float(argv[6]) * np.pi
    if len(argv) > 7:
        iterations = int(argv[7])
    if len(argv) > 8:
        tag = argv[8]
    if len(argv) > 9:
        w_lambda = float(argv[9])

if iterations < 1:
    raise ValueError("iterations must be a positive integer. Received: " + str(iterations))
if network_type not in ALLOWED_NETWORK_TYPES:
    raise ValueError(
        "network_type must be one of " + str(ALLOWED_NETWORK_TYPES) + ". Received: " + str(network_type))
if supercell_shape not in ALLOWED_SUPERCELL_SHAPES:
    raise ValueError(
        "supercell_shape must be one of " + str(ALLOWED_SUPERCELL_SHAPES) + ". Received: " + str(supercell_shape))

print(f"Running 2DEG with nspins = {tuple(nspins)}, r_s = {r_s}, "
      f"network_type = {network_type}, supercell_shape = {supercell_shape}, "
      f"iterations = {iterations}")

# --------------------------- Set up config file ---------------------------
cfg = base_config.default()

# Set number of electrons
cfg.system.electrons = (nspins[0], nspins[1])  # (spin up, spin down) electrons

# Create the 2D lattice
cfg.system.ndim = 2
num_electrons = sum(nspins)
if supercell_shape == "tri": # triangular supercell
    a_M = np.sqrt(2 * np.pi / np.sqrt(3) * num_electrons / n_sites)
    supercell_lattice, moire_lattice = \
        lattices._triangular_lattice_vecs_periodic_potential(a_M, n_sites)
    area = np.linalg.det(supercell_lattice)
    print(f"[MOIRE] n_sites = {n_sites}, nu = {num_electrons}/{n_sites} = "
          f"{num_electrons/n_sites:.4f}, a_M = {a_M:.4f}, "
          f"U = {moire_U}, phi/pi = {moire_phi/np.pi:.4f}, w_lambda = {w_lambda}")
elif supercell_shape == "rect":
    # 三角格子と整合する長方形スーパーセル。
    # 原始ベクトル a1 = (a, 0), a2 = (-a/2, a*sqrt(3)/2) に対して
    #   L1 = n1*a1                      = (n1*a, 0)
    #   L2 = n2*a1 + 2*n2*a2            = (0, n2*sqrt(3)*a)
    # すなわち M = [[n1, n2], [0, 2*n2]]、N = |det M| = 2*n1*n2。
    #
    # 120 度の菱形は周期ノルムにとって不利な形で、d_ref = 0.6430*sqrt(N) にとどまる。
    # 計量の非対角成分が -1/2 なので、最大値を与える深い穴 f=(2/3,1/3) で打ち消しが
    # 起きるため。長方形なら非対角成分が 0 になり、最大はセル中心 f=(1/2,1/2) の
    # sqrt(Lx^2+Ly^2)/pi となって d_ref = 0.7979*sqrt(N)、同じ N で 24% 広い。
    # 窓 w_h を d_ref より広く置いても無駄にならない範囲がそれだけ伸びる。
    # 代償として、スーパーセルの 6 回対称性は失われる。
    if num_electrons % 2:
        raise ValueError("rect スーパーセルは N が偶数のときだけ組める。N = %d" % num_electrons)
    _half = num_electrons // 2
    _best = None
    for _n2 in range(1, _half + 1):
        if _half % _n2: continue
        _n1 = _half // _n2
        # 縦横比 (n1*a) : (n2*sqrt(3)*a) が 1 に近いものを選ぶ
        _score = abs(np.log(_n1 / (_n2 * np.sqrt(3))))
        if _best is None or _score < _best[0]: _best = (_score, _n1, _n2)
    _, _n1, _n2 = _best
    a_M = np.sqrt(2 * np.pi / np.sqrt(3) * num_electrons / n_sites) if n_sites else 1.0
    _a = np.sqrt(2 * np.pi / np.sqrt(3))          # 電子 1 個あたり面積 pi となる格子定数
    moire_lattice = np.array([[_a, -_a / 2], [0.0, _a * np.sqrt(3) / 2]])
    supercell_lattice = np.zeros((2, 2))
    supercell_lattice[:, 0] = _n1 * moire_lattice[:, 0]
    supercell_lattice[:, 1] = _n2 * moire_lattice[:, 0] + 2 * _n2 * moire_lattice[:, 1]
    area = np.linalg.det(supercell_lattice)
    print(f"[CELL] 長方形スーパーセル {_n1}x{_n2}, N = {num_electrons}, "
          f"Lx = {supercell_lattice[0,0]:.4f}, Ly = {supercell_lattice[1,1]:.4f}, "
          f"縦横比 = {supercell_lattice[0,0]/supercell_lattice[1,1]:.3f}")
elif supercell_shape == "sq": # square supercell
    supercell_a = np.sqrt(np.pi * num_electrons)
    supercell_lattice = lattices._square_lattice_vecs(supercell_a)
    area = np.linalg.det(supercell_lattice)
    moire_lattice = supercell_lattice / max(int(round(np.sqrt(n_sites))), 1)
else:
    raise NotImplementedError(
        "Only supercell shapes " + str(ALLOWED_SUPERCELL_SHAPES) + " are implemented. Received: " + supercell_shape)

# Set up the Hamiltonian parameters
cfg.system.make_local_energy_fn = "periodicwave.pbc.hamiltonian.local_energy"
cfg.system.make_local_energy_kwargs = {"lattice": supercell_lattice,
    "potential_type": ('CoulombMoire' if moire_U != 0.0 else 'Coulomb'),
    "potential_kwargs": {'interaction_energy_scale': r_s,
                         'moire_lattice_vectors': moire_lattice,
                         'moire_potential_strength': moire_U,
                         'moire_potential_phi': moire_phi},
    "kinetic_kwargs": {'laplacian_method': 'folx'}, # use forward laplacian
    }
cfg.system.pbc_lattice = supercell_lattice

# feature layer for pbc
cfg.network.make_feature_layer_fn = ("periodicwave.pbc.feature_layer.make_pbc_feature_layer")
cfg.network.make_feature_layer_kwargs = { "lattice": supercell_lattice, "include_r_ae": False }

# envelopes
cfg.network.make_envelope_fn = ( "periodicwave.envelopes.make_null_envelope" ) # use no envelope with pbc

# Set batch size
cfg.batch_size = int(os.environ.get("PW_BATCH", 64))

# learning rate parameters
cfg.optim.optimizer = 'kfac'
cfg.optim.objective = 'vmc'
cfg.optim.iterations = iterations
cfg.optim.lr.rate  = float(os.environ.get("PW_LR", 0.1))     # 論文 Table II は 10
cfg.optim.lr.delay = float(os.environ.get("PW_LR_DELAY", 20000))  # 論文 Table II は 1e5
cfg.optim.lr.decay = 1.0

# logging parameters
cfg.log.save_frequency = 2.0 # minutes

# mcmc parameters
cfg.mcmc.burn_in = 100
cfg.mcmc.init_width = 1.0
cfg.mcmc.move_width = 0.2
cfg.mcmc.steps = 10
cfg.mcmc.move_width_updater = 'adaptive'

# general network parameters
cfg.network.complex = True # complex wavefunction work better for pbc systems
cfg.network.determinants = int(os.environ.get("PW_DET", 4))
cfg.network.jastrow = "NONE"
cfg.network.jastrow_kwargs = {'ndim':cfg.system.ndim}

# network architecture parameters
# Code to run Psiformer
if network_type == 'CustomPsiformer':
    cfg.network.network_type = 'CustomPsiformer'
    cfg.network.CustomPsiformer.num_layers = 2
    cfg.network.CustomPsiformer.num_heads  = 8
    cfg.network.CustomPsiformer.attn_dim   = 16
    cfg.network.CustomPsiformer.value_dim  = 16
    cfg.network.CustomPsiformer.mlp_dim    = 64
    cfg.network.CustomPsiformer.num_perceptrons_per_layer = 2
    cfg.network.CustomPsiformer.use_layer_norm = True
    cfg.network.CustomPsiformer.mlp_activation_fct = "GELU"

# [SCREEN] Code to run Screenformer (screening / multiscreen attention)
elif network_type == 'CustomScreenformer':
    cfg.network.network_type = 'CustomScreenformer'
    # [MOIRE] 窓の梯子を 0.50 * lambda 〜 2.50 * lambda（周期ノルム）に張る。
    # 下限は最短の電子間距離(~0.3)より上、上限は d_ref より下に取る。
    # そうしないと下端のヘッドは重みが恒等的に 0 になって二度と戻れず、
    # 上端のヘッドは窓が効かない（= 窓なし）ヘッドになって梯子が無駄になる。
    # d_ref はセルから決まるので、そこで割って factor / spread に直す。
    import jax.numpy as _jnp
    from periodicwave import CustomScreenformer as _CS
    _dref = float(_CS._max_periodic_distance(_jnp.asarray(supercell_lattice)))
    # 窓の梯子の下端・上端（周期ノルム）。PW_W_LO / PW_W_HI で上書きできる。
    _lo = float(os.environ.get("PW_W_LO", 0.50)) * w_lambda
    _hi = float(os.environ.get("PW_W_HI", 2.50)) * w_lambda
    w_factor = float(np.sqrt(_lo * _hi) / _dref)
    w_spread = float(np.sqrt(_hi / _lo))
    _H = int(os.environ.get("PW_HEADS", 8))
    # [SCREEN] PW_PEAK=1 で帯域窓（殻）に切り替える。中心を PW_C_LO..PW_C_HI に
    # 等間隔で置き、半値幅を中心の間隔に取る。raised cosine はこのとき 1 の分割に
    # なるので、sum_h m_h(d) = 1 となり q_h がそのまま動径分解になる。
    _centres = None
    if os.environ.get("PW_PEAK", "0").lower() not in ("0", "", "false"):
        _clo = float(os.environ.get("PW_C_LO", 0.9)) * w_lambda
        _chi = float(os.environ.get("PW_C_HI", 3.1)) * w_lambda
        _cc = np.linspace(_clo, _chi, _H)
        _sg = np.full(_H, (_chi - _clo) / max(_H - 1, 1)
                      * float(os.environ.get("PW_SIG_MULT", 1.0)))
        _centres = (_cc / _dref, _sg / _dref)
    # [SCREEN] PW_BANDS=1 で帯窓に切り替える。ヘッド h は輪 [a_h, a_{h+1}] を持ち、
    # その中で重みが 1 から 0 へ滑らかに落ちる（近い相手ほど重い）。
    #   m_h(d) = (1 + cos(pi * (d - a_h)/Delta)) / 2   (|d - a_h| <= Delta, 外は 0)
    # 中心 a_h = h * Delta を等間隔に置くと raised cosine は 1 の分割になり、
    # sum_h m_h(d) = 1 が [a_0, a_{H-1}] で厳密に成り立つ。
    # Delta = d_ref / (H - 1) と取るのが要点。こうすると a_{H-1} = d_ref ちょうどで、
    # いちばん外のヘッドは最大距離に山を持ち、届く範囲すべてで和が 1 になる。
    # Delta = d_ref / H だと外のヘッドが d_ref で 0 に落ちる。周期ノルムは弦なので
    # 対の分布は d_ref 近くに溜まり（中央値 0.88 d_ref、最外帯に対の 5 割）、
    # 重みを 28% 落としてしまう。
    # h = 0 は a_0 = 0 なので従来の単調窓そのもの。実装は殻窓（mask_centre）と同じ。
    if os.environ.get("PW_BANDS", "0").lower() not in ("0", "", "false"):
        _bhi = float(os.environ.get("PW_BAND_HI", _dref))   # いちばん外のヘッドの中心
        _D = _bhi / max(_H - 1, 1)
        _cc = _D * np.arange(_H, dtype=np.float64)
        _centres = (_cc / _dref, np.full(_H, _D) / _dref)
    # [SCREEN] PW_ENV=1 で「全体窓の輪切り」に切り替える。
    #   g(d) = (1 + cos(pi (d/d_ref)^p)) / 2     g(0) = 1, g(d_ref) = 0
    #   m_h(d) = g(d) [Theta(d - b_h) - Theta(d - b_{h+1})]
    # 境界 b_h は既定で [0, d_ref] を H 等分。PW_ENV_EDGES="0,1.2,..." で任意に置ける
    # （差の telescoping で sum_h m_h = g はどんな置き方でも厳密）。
    # 遷移の半幅 delta = PW_ENV_TAU * (いちばん狭い輪の幅) / 2。
    # PW_ENV_KERNEL=gauss で、階段を「輪の指示関数をガウス核でぼかしたもの」に替える
    # （標準偏差 sigma = PW_ENV_SIGMA * いちばん狭い輪の幅、既定 0.5）。窓はどの階数の微分も連続で、
    # 係数が輪ごとに 1 ずつ変わるときの余分な曲がりは (4 pi/Delta^2) exp(-2 pi^2 (sigma/Delta)^2)
    # （sigma = Delta/2 で 0.56。いまの tau = 1 は 30.5）。
    #   tau = 1   輪の芯なし（遷移が輪を覆う）、なめらか
    #   tau = 0.5 輪の半分が芯で、そこでは m_h = g そのもの（近い相手ほど重い）
    _rings = None
    if os.environ.get("PW_ENV", "0").lower() not in ("0", "", "false"):
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
    # [SCREEN] PW_FIXV=1 で value を学習させず、対の固定特徴 e(r_ij) を運ぶ。
    _fixv = os.environ.get("PW_FIXV", "0").lower() not in ("0", "", "false")
    _wl = np.geomspace(_lo, _hi, _H)
    print(f"[MOIRE] value = " + ("固定（対の特徴 e(r_ij)）" if _fixv else "学習（W_V）"))
    print(f"[MOIRE] d_ref = {_dref:.4f}, 窓 = {_lo:.3f}..{_hi:.3f} "
          f"-> factor {w_factor:.4f}, spread {w_spread:.4f}")
    _frz = ("   [FROZEN]" if os.environ.get("PW_FREEZE_W","0").lower()
            not in ("0","","false") else "   [learnable]")
    if _rings is not None:
        print("[MOIRE] 全体窓の輪切り: 境界 b_h = "
              + ", ".join(f"{v:.3f}" for v in _rings[0] * _dref)
              + (f"   ガウス核 sigma = {_rings[1]*_dref:.4f}" if len(_rings) > 3
                 else f"   delta = {_rings[1]*_dref:.4f}") + f"   p = {_rings[2]:g}" + _frz)
    elif _centres is None:
        print("[MOIRE] 入れ子窓 w_h = " + ", ".join(f"{v:.3f}" for v in _wl) + _frz)
    else:
        _cv = _centres[0] * _dref
        print("[MOIRE] 帯窓: 輪の内側 a_h = "
              + ", ".join(f"{v:.3f}" for v in _cv)
              + f"   帯幅 Delta = {_centres[1][0]*_dref:.4f}"
              + f"   いちばん外の山 {_cv[-1]:.3f} (d_ref = {_dref:.3f})" + _frz)
    # base_config.py has no CustomScreenformer block, so create it here
    # (keeps base_config.py untouched).
    cfg.network.CustomScreenformer = ml_collections.ConfigDict({
        # Same architecture sizes as the CustomPsiformer reference run,
        # so that any difference comes from the attention mechanism itself.
        # 論文 arXiv:2502.05383 Table II は 3 層 x 6 ヘッド。環境変数で切替。
        'num_layers': int(os.environ.get("PW_LAYERS", 2)),
        'num_heads':  int(os.environ.get("PW_HEADS", 8)),
        'attn_dim': 16,
        'value_dim': 16,
        'mlp_dim': 64,
        'num_perceptrons_per_layer': 2,
        'use_layer_norm': True,
        'mlp_activation_fct': "GELU",
        # Screening knobs:
        'use_softmask': True,   # raised-cosine window on the physical distance r_ee
        'use_gate': True,       # gate tanh(SiLU(W_G x)) on the attention output
        'use_tanh_norm': True,  # bounded aggregation without a softmax denominator
        'use_output_scale': True,       # learnable output scale e^{s_O} on the attention branch
        # Trim acceptance width r in (0, 2). Start SOFT (close to 2): at init the
        # query/key directions are random, so cos similarities sit near 0 and a
        # sharp trim (r = 1) would zero out almost every weight.
        'init_trim_r': 1.9,
        # Initial e^{s_O}. 0 = auto-calibrate to the residual-stream magnitude.
        'init_output_scale': 0.0,
        # Initial screening lengths: geometric centre x/: spread, per head.
        # centre 0.8 puts the window edge inside the cell (where the raised
        # cosine actually has slope), spread 2.5 gives the four heads
        # w/d_ref = 0.32, 0.56, 1.14, 2.0 -> genuine multiscreen from step 0.
        'init_softmask_w_factor': w_factor,
        'init_softmask_w_spread': w_spread,
        # 帯域窓（殻）を使うときの (中心, 半値幅) / d_ref。None なら従来の単調窓。
        'init_softmask_centres': _centres,
        'init_softmask_rings': _rings,
        # 帯窓（bin）を使うときの (境界, なめらかさ) / d_ref。None なら上に従う。
        # True なら value を学習させず、対の固定特徴 e(r_ij) を運ぶ。
        'use_fixed_value': _fixv,
    })

# Code to run SlaterNet
elif network_type == 'SlaterNet':
    cfg.network.network_type = 'SlaterNet'
    cfg.network.SlaterNet.num_layers = 4
    cfg.network.SlaterNet.mlp_dim    = 64
    cfg.network.SlaterNet.num_perceptrons_per_layer = 2
    cfg.network.SlaterNet.use_layer_norm = True
    cfg.network.SlaterNet.mlp_activation_fct = "GELU"
    cfg.network.determinants = 1 # when using a single determinant, SlaterNet is equivalent to Hartree-Fock

# Get folder name to save the results
# 末尾の l2h16 は昔からの固定の札で、実際の層数・ヘッド数とは無関係。
# 途中で変えると走行中の run がフォルダを見失うので、この campaign では触らない。
folder_name = f"results/2deg-Coulomb/{network_type}/el{nspins[0]}_{nspins[1]}_rs{format_for_folder_name(r_s)}_{supercell_shape}_l2h16"
if tag:
    folder_name += f"_{tag}"

# save path
cfg.log.save_path = folder_name

# log device info.
# If same folder contains previous runs, rename old log file to keep information.
# The renaming follows the same convention as renaming train_stats.csv
writers.rename_file("device_info", folder_name, file_extension="log")
# [PROBE] PW_PIN=honeycomb を立てると、モンテカルロが「どの谷が空いているか」を
# 変えないように拘束される。ハニカム秩序を保ったまま最適化して、そのエネルギーを
# 乱れた状態と比べるための解析用スイッチ（既定では何も起きない）。
if os.environ.get("PW_PIN", "") == "honeycomb":
    from periodicwave import mcmc as _MC
    _MOi = np.linalg.inv(moire_lattice); _Li = np.linalg.inv(supercell_lattice)
    _seen, _sites, _sub = set(), [], []
    for _m in range(-40, 41):
        for _n in range(-40, 41):
            _r = _m * moire_lattice[:, 0] + _n * moire_lattice[:, 1]
            _f = np.mod(_Li @ _r, 1.0)
            _f = np.where((_f < 1e-7) | (_f > 1 - 1e-7), 0.0, _f)
            _k = (round(float(_f[0]), 5), round(float(_f[1]), 5))
            if _k in _seen: continue
            _seen.add(_k); _sites.append(supercell_lattice @ _f); _sub.append((_m + _n) % 3)
    _sites = np.array(_sites); _sub = np.array(_sub)
    _tgt = (_sub != 2).astype(np.float32)          # 部分格子 2 を空ける = ハニカム
    assert int(_tgt.sum()) == num_electrons, (int(_tgt.sum()), num_electrons)
    _MC.PIN = dict(sites=_sites, lattice=supercell_lattice, target=_tgt)
    print(f"[PROBE] モンテカルロを谷の割り当てに拘束: {len(_sites)} 谷、"
          f"うち {int(_tgt.sum())} 谷に 1 個ずつ")

custom_logging.log_device_info(folder_name + "/device_info.log")
writers.rename_file("config", folder_name, file_extension="json")
custom_logging.save_config_dict_as_json(cfg, cfg.log.save_path + '/config.json')

cfg.debug.deterministic = True   # [SCAN] 全 r_s で同じ初期化にする
# 注意：config.json はこの行より前に保存されるので、そこには deterministic = False と残る。
# 実際の種は train_screen.py で PW_SEED（既定 42）。つまり既定ではどの run も同じ初期化。
# [CONTROL] PW_SEED_FROM_TAG=1 なら種をタグから決める。1 つのジョブの 3 本は環境変数を
# 共有するので、PW_SEED を直接渡すと 3 本とも同じ種になってしまうため。
if os.environ.get("PW_SEED_FROM_TAG", "0").lower() not in ("0", "", "false"):
    import zlib
    os.environ["PW_SEED"] = str(zlib.crc32(str(tag).encode()) % 2147483647)
print(f"[SEED] seed = {os.environ.get('PW_SEED', '42')}  (tag = {tag})")
t_init = time()
# --------------------------- train ---------------------------
train.train(cfg)
logging.info("Training completed after t [s] = " + str(int(time() - t_init)))
