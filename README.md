# Screening_develop

遮蔽アテンション（screening attention）を使ったニューラルネットワーク変分モンテカルロ（VMC）のコード。
[PeriodicWave](https://github.com/mg607/PeriodicWave)（M. Geier, K. Nazaryan, T. Zaklama, L. Fu, Phys. Rev. B **112**, 045119 (2025)）をもとに、次のものを足しています。

- 遮蔽アテンションの波動関数 `periodicwave/CustomScreenformer.py`、その学習 `periodicwave/train_screen.py`、2DEG の設定 `periodicwave/configs/run_2deg_screen.py`
- λ-jellium（擬スピン 2 成分の 2 次元電子系）：局所エネルギー `periodicwave/pbc/lambda_jellium.py`、下のバンドに射影した Hartree–Fock `periodicwave/pbc/lj_hf.py`、
  HF の軌道を使う軌道 `periodicwave/lj_orbitals.py`、設定 `periodicwave/configs/run_lambda_jellium.py`
- 学習済みのネットワークの解析 `screening_analysis/`：いまの λ-jellium（WC と AHC）は `lj_hf_branch.py`（VMC の出発点の HF の表）、`lj2/`、`lj3/`。
  ほかは 2DEG・モアレの時期に作り、別の解析が import している共有の部品（`screening_analysis/README.md` の末尾）
- 検算 `test_lambda_jellium.py`、`test_screenformer.py`、`test_gauss_rings.py`
- どのファイルがどの工程で呼ばれるかの図：`doc/code_map/code_map.pdf`
- λ-jellium の多点学習（HF なし）：1 本のネットワークで多くの (r_s, λ) の基底状態を同時に学習する。点ごとに変えるのは (r_s, λ) だけで、相は渡さない。
  `periodicwave/lj_multi.py`、点の表 `periodicwave/configs/lj_multi_points_32.txt`、検算 `test_lj_multi.py`、
  GPU の計算機・スパコン（Slurm、TSUBAME4.0）で走らせるジョブの雛形 `hpc/`（`hpc/README.md`）
- いまの λ-jellium（lj3）の計算を、処理の段ごとに 1 ファイルずつ書き直した参照用コード `lj_reference/`。読むためのもので学習には使わない。本体と同じ値を出すことを `lj_reference/check.py` で確かめた（`lj_reference/README.md`）
- 2DEG・モアレ・窓の設計（2026-08〜09）と λ-jellium の最初のスキャン（2026-10 上旬）だけで使ったスクリプト・設定・説明は `archive/` に移した（`archive/README.md`）

元の PeriodicWave から変えたファイルは `.gitignore`、`evaluate-density.py`、`evaluate-energies.py`、`periodicwave/configs/run_2deg.py`、`periodicwave/loss.py`、`periodicwave/mcmc.py` です。
元の README（インストールと実行の方法）は `README_PeriodicWave.md` にあります。

## 開発の流れ

各自のブランチ（`wada_dev`、`okada_dev`）で開発して push → `develop` にプルリク → `main` に統合。
計算結果（`results/`）、チェックポイント、解析の出力、研究室の計算サーバ用の設定（`ws_pw_kit/`）はリポジトリに入れません（どの計算機でも使えるジョブの雛形だけ `hpc/` に置く）。

## ライセンス

Apache License 2.0（`LICENSE`、元の PeriodicWave と同じ）。
