# Copyright 2022 DeepMind Technologies Limited.
# Modifications Copyright (c) 2025 Max Geier, Khachatur Nazaryan, Massachusetts Institute of Technology, MA, USA
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

# NOTICE: This file has been modified from the original DeepMind version.
# Changes:
# - Streamlined for materials calculations

""" Default base configuration for VMC calculations for periodic systems. """

import ml_collections
from ml_collections import config_dict

# 通常の学習用設定の土台。configs/run_*.pyでdefault()を呼び、系やAnsatzに合わせて上書きする。
# この関数は設定を作るだけで、学習やパラメータ初期化は実行しない。
# system.electronsなどの未設定値は、実行用configで必ず指定する。
# window_experimentsの独立入口は、この設定ではなく専用CLIの引数を使う。

def default() -> ml_collections.ConfigDict:
  """Create set of default parameters for running qmc.py.

  Note: placeholder (cfg.system.electrons must be replaced with appropriate values.

  Returns:
    ml_collections.ConfigDict containing default settings.
  """
  # ConfigDictはcfg.system.ndimのように、属性形式で設定を読み書きできる。
  cfg = ml_collections.ConfigDict({
      'batch_size': 1024,  # 同時に保持するMCMC標本（walker）の総数。独立標本数とは限らない。
      'optim': {
          # 最適化：|ψ|²で標本化し、エネルギー期待値〈H〉を下げるVMC。
          'objective': 'vmc',
          'iterations': 1000000,  # パラメータ更新の総ステップ数
          'optimizer': 'kfac',  # adam / kfac / lamb / none（noneは更新なしの評価）
          # 微分方式の設定項目。実際のHamiltonianへの受け渡しは実行用configも確認する。
          'laplacian': 'default',  # default：通常の自動微分、folx：前進ラプラシアン
          'lr': {
              # 学習率：rate * (1 + step / delay)^(-decay)
              'rate':  0.05,  # 初期学習率
              'decay': 1.0,  # 減衰の指数
              'delay': 10000.0,  # 学習率が変化するステップ数の目安
          },
          # 勾配推定に使う局所エネルギーの外れ値を制限する幅。0なら無効。
          # 幅の単位は平均絶対偏差。報告用エネルギー自体を置き換える設定ではない。
          'clip_local_energy': 5.0,
          # True：clippingの中心を中央値にする。False：平均値を使う。
          'clip_median': False,
          # True：勾配内のELの差を、clipping後の平均を基準にして中心化する。
          'center_at_clip': True,
          # 更新後にNaNが生じたら、パラメータとoptimizer状態を前ステップへ戻す。
          'reset_if_nan': False,
          # 明示有効時のみ追加。PH / Laplacianスペクトル / 重み平滑化は別の目的。
          'topology_regularization': {
              'mode': 'off', 'gamma': 0., 'perturb_sigma': .03,
              'max_samples': 8, 'ph_floor': 1e-4,
              'landscape_levels': 3, 'landscape_grid': 21,
              'sampling_score': True,
          },
          # KFAC：勾配の曲率情報を近似して更新する。Adamとは別の設定群。
          'kfac': {
              'invert_every': 1,  # 曲率近似の逆行列を更新する間隔
              'cov_update_every': 1,  # 共分散の更新間隔
              'damping': 0.001,  # 逆行列計算を安定化する正則化
              'cov_ema_decay': 0.95,  # 共分散の指数移動平均の減衰率
              'momentum': 0.0,  # 過去の更新方向を残す割合
              'momentum_type': 'regular',
              # 現在のtrainerではdampingの自動調整は使わない。
              'min_damping': 1.0e-6,
              'norm_constraint': 0.001,
              'mean_center': True,
              'l2_reg': 0.0,
              'register_only_generic': False,
          },
          # Adam：勾配の一次・二次モーメントの移動平均で更新幅を調整する。
          'adam': {
              'b1': 0.9,  # 一次モーメントの減衰率
              'b2': 0.999,  # 二次モーメントの減衰率
              'eps': 1.0e-8,  # ゼロ除算防止：平方根の外側に加える
              'eps_root': 0.0,  # 平方根の内側に加える安定化項
          },
      },
      'log': {
          # 結果の記録とcheckpoint。頻度の単位（step / 分）に注意。
          'stats_frequency': 1,  # 統計量を記録する間隔［step］
          'save_frequency': 10.0,  # checkpointを保存する間隔［分］
          # 保存先。空なら作業ディレクトリに時刻付きのフォルダを作る。
          'save_path': '',
          # 再開元。save_pathにcheckpointがあれば、そちらの復元を優先する。
          'restore_path': '',
      },
      'system': {
          # 物理系：(上向き電子数, 下向き電子数)。空tupleは未設定。
          # λ-jelliumの動的擬スピンは専用configのpseudospin設定も必要。
          'electrons': tuple(),
          # 実空間の次元。2DEG・モアレ・λ-jelliumでは2。
          'ndim': 2,
          # 局所エネルギー EL=(Hψ)/ψ を作る関数を「module.function」で指定する。
          # 通常PBCとλ-jelliumでは別Hamiltonianを選ぶ。trainerはこの文字列から関数を読む。
          'make_local_energy_fn': '',
          # 上の関数に渡す物理条件（相互作用・λ・Delta・微分方式など）。
          'make_local_energy_kwargs': {},
          # 周期セルの格子行列。格子ベクトルを列に並べる。長さの単位は各モデルに従う。
          'pbc_lattice': None,
      },
      'mcmc': {
          # MCMC：現在の波動関数の|ψ|²に従う標本を位置移動で生成する。
          # 学習開始前の平衡化回数。0の場合の初期化／復元の扱いはtrainer側も確認する。
          'burn_in': 300,
          'steps': 30,  # NNパラメータを1回更新する間のMCMCステップ数
          # 初期電子配置を生成するGaussianの幅（提案移動幅とは別）。
          'init_width': 1.0,
          # Metropolis法で位置を動かすGaussian提案の幅。
          'move_width': 0.1,
          # const：移動幅を固定。adaptive：受理率に応じて移動幅を調整。
          'move_width_updater': 'adaptive',
          # adaptiveで移動幅を見直す間隔［学習step］。
          'adapt_frequency': 100,
          'blocks': 1,  # 電子を何組に分けて提案するか。1は全電子をまとめて移動。
      },
      'network': {
          # 波動関数Ansatzの選択。Screenformer固有の設定は各実行用configで追加する。
          'network_type': 'CustomPsiformer',
          # True：複素波動関数（位相とlog振幅）を扱う。
          'complex': False,
          # CustomPsiformerを選んだときのTransformer設定。
          'CustomPsiformer': {
              # Customized Psiformer architecture: Geier et al,. arXiv:2502.05383
              'num_layers': 4, # Transformer層数
              'num_heads': 4,  # 各層のAttention Head数
              'attn_dim': 32,  # HeadごとのQuery/Key次元（内積を取る空間）
              'value_dim': 32, # HeadごとのValue次元（集約する情報の幅）
              'mlp_dim': 128,  # 電子ごとの内部特徴Xの幅。MLPの入出力もこの次元。
              'num_perceptrons_per_layer': 2, # Attention後のMLPの線形層数
              'use_layer_norm': True, # Attention/MLPの残差接続後にLayerNormを使う
              'mlp_activation_fct': "GELU", # MLPの活性化関数
          },
          # SlaterNetの電子ごとのMLP設定（Attentionは使わない）。
          # determinants=1は単一Slater行列式。外部HF係数表を読む指定ではない。
          'SlaterNet': {
              'num_layers': 4, # 残差接続付きMLPブロック数
              'mlp_dim': 128,  # 電子ごとの内部特徴Xの幅
              'num_perceptrons_per_layer': 2, # 各ブロック内の線形層数
              'use_layer_norm': True, # 各ブロックでLayerNormを使う
              'mlp_activation_fct': "GELU", # MLPの活性化関数
          },
          # 全Ansatzに共通する軌道・行列式・相関因子の設定。
          'determinants': 4,  # 足し合わせるSlater行列式の数
          'bias_orbitals': False,  # 軌道を作る最終線形射影にbiasを付けるか
          # Jastrowは電子対の短距離相関を表す。none / default / simple_eeを選ぶ。
          # simple_eeのCoulombカスプ条件には次元と相互作用強度が必要。
          # λ-jelliumのHF用擬スピン対Jastrowとは別の設定経路。
          'jastrow': 'none',
          # 通常Jastrowへ渡す引数。相互作用強度はHamiltonianの単位系と揃える。
          'jastrow_kwargs': {'ndim': 2, "interaction_strength": 1.0}, 
          # 入力特徴量を作る関数を「module.function」で指定する。
          # PBCのsin/cos特徴量などをここで選択し、その後に学習可能Embeddingへ渡す。
          # 空の場合はnetworks.make_ferminet_featuresを使う。
          'make_feature_layer_fn': '',
          # 特徴量生成に渡す引数（格子、電子-原子距離を含めるか、など）。
          'make_feature_layer_kwargs': {},
          # 軌道へ掛けるEnvelopeの生成関数と引数。特徴量生成とは別の役割。
          'make_envelope_fn': '',
          'make_envelope_kwargs': {},
      },
      'debug': {
          # optimizer状態・パラメータ・lossにNaNがあれば例外を出す。
          'check_nan': False,
          'deterministic': False,  # Trueなら固定seedを使う（乱数の生成方法はtrainer側）。
      },
  })

  return cfg

