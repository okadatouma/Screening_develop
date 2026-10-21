# Screening_develop

遮蔽アテンション（screening attention）を使ったニューラルネットワーク変分モンテカルロ（VMC）のコード。
[PeriodicWave](https://github.com/mg607/PeriodicWave)（M. Geier, K. Nazaryan, T. Zaklama, L. Fu, Phys. Rev. B **112**, 045119 (2025)）をもとに、次のものを足しています。

- 遮蔽アテンションの波動関数 `periodicwave/CustomScreenformer.py`、その学習 `periodicwave/train_screen.py` と設定 `periodicwave/configs/run_2deg_screen*.py`
  （説明は `README_SCREEN.md` と `CustomScreenformer_解説.md`）
- λ-jellium（擬スピン 2 成分の 2 次元電子系）：局所エネルギー `periodicwave/pbc/lambda_jellium.py`、下のバンドに射影した Hartree–Fock `periodicwave/pbc/lj_hf.py`、
  HF の軌道を使う軌道 `periodicwave/lj_orbitals.py`、設定 `periodicwave/configs/run_lambda_jellium.py`
- 学習済みの網の解析と図のスクリプト `screening_analysis/`（`lj2/` は λ-jellium の結晶の解析）
- 評価と比較 `evaluate-*-screen.py`、`compare_screen.py`、検算 `test_screenformer.py`、`test_gauss_rings.py`、`test_lambda_jellium.py`

元の PeriodicWave から変えたファイルは `.gitignore`、`evaluate-density.py`、`evaluate-energies.py`、`periodicwave/configs/run_2deg.py`、`periodicwave/loss.py`、`periodicwave/mcmc.py` です。
元の README（インストールと実行の方法）は `README_PeriodicWave.md` にあります。

## 開発の流れ

各自のブランチ（`wada_dev`、`okada_dev`）で開発して push → `develop` にプルリク → `main` に統合。
計算結果（`results/`）、チェックポイント、解析の出力、計算サーバ用の設定はリポジトリに入れません。

## ライセンス

Apache License 2.0（`LICENSE`、元の PeriodicWave と同じ）。
