# Screening_develop

新しい窓研究の実行入口：**[入力E0〜E3・Window W0〜W3・WC/AHC HF比較・Slurm手順](periodicwave/window_experiments/README.md)**。
λ=0.64を優先。HF表と作成条件メタデータのパスを実行時に指定する。GPU本計算は未実行、CPU小規模検証済み。

遮蔽アテンション（screening attention）を使ったニューラルネットワーク変分モンテカルロ（VMC）のコード。
[PeriodicWave](https://github.com/mg607/PeriodicWave)（M. Geier, K. Nazaryan, T. Zaklama, L. Fu, Phys. Rev. B **112**, 045119 (2025)）をもとに、次のものを足しています。

- 遮蔽アテンションの波動関数 `periodicwave/CustomScreenformer.py`、その学習 `periodicwave/train_screen.py` と設定 `periodicwave/configs/run_2deg_screen*.py`
  （説明は `README_SCREEN.md` と `CustomScreenformer_解説.md`）
- λ-jellium（擬スピン 2 成分の 2 次元電子系）：局所エネルギー `periodicwave/pbc/lambda_jellium.py`、下のバンドに射影した Hartree–Fock `periodicwave/pbc/lj_hf.py`、
  HF の軌道を使う軌道 `periodicwave/lj_orbitals.py`、設定 `periodicwave/configs/run_lambda_jellium.py`
- 学習済みの網の解析と図のスクリプト `screening_analysis/`（`lj2/` は λ-jellium の結晶の解析）
- 評価と比較 `evaluate-*-screen.py`、`compare_screen.py`、検算 `test_screenformer.py`、`test_gauss_rings.py`、`test_lambda_jellium.py`

原版からの変更には、上記に加えてAttention・Ansatz・学習・評価・設定・解析の実装が含まれます。現在の差分は`git diff`で確認してください。
元の README（インストールと実行の方法）は `README_PeriodicWave.md` にあります。

## 現在の整理状況と研究インターフェース

Attention、Embedding、Wavefunction Construction、Hamiltonianを主な研究上の変更境界とします。
MCMC/VMC/Optimizer/Checkpointは原則共通基盤として維持します。
Screening Attentionと主要3解析入口のDiagnostic移行はE2bまで完了。F1で学習可能Embeddingを共通化しました。
F2a/bでは周期ノルムの独立実装を維持し、幾何・距離・特徴量の責務を調査しました。
F3では通常の軌道構成を`wavefunction.py`へ移し、`networks.make_orbitals`は同じ関数を再exportします。
HF構成・Jastrow/Envelope・行列式reducerは独立に維持しています。大規模な構造整理はF3で停止しました。
F4は研究で必要な新Ansatzが決まるまで保留します。G1では独立検証に基づいてloss.pyのVMC主勾配だけを修正しました。

- [現行の数式・shape・単位・対称性・互換性](tests/CONTRACTS.md)
- [検証方法・Diagnostic API](tests/README.md)
- [最新の実行結果と既知FAIL/ERROR](tests/RESULTS.md)
- [未commit資産を含む引継ぎ](HANDOFF.md)

ProductionとDiagnosticは独立JITです。診断値を学習へ戻さず、固定snapshotと研究データを上書きしません。
G1の主勾配は `2 Re mean((E_L−center) conj(d logψ))`。clipping方式は維持しています。
波動関数の旧数値基準は厳密保持しますが、VMC勾配・更新・学習軌跡は意図的に変わります。
G2bで通常PBCの複素folx局所エネルギーの虚部を保持する1行修正を行いました。
補助JVP契約・folx定数出力・Psiformer微分・倍精度KFACなどは未修正です。
G3aで自由電子系の短期VMCを独立Fourier/FFT基準、MCMC blocking、checkpointで検証しました。本体は変更していません。
実数SlaterNet/float32 Screenformerは検証範囲で適合しましたが、complex通常trainerの初期INFOログ不具合とdouble NNの厳しい微分比較差が残ります。
既存33件とG2b固定snapshotの結果は維持しています。全経路が正常とは扱わず、詳細はtests/RESULTS.md先頭を参照してください。
G3bの相互作用系には進んでいません。
G3a追補では、Screenformer64の微小勾配差を検証側のdtype昇格へ、Psiformerの微分差を使用中JAXのsoftmax混合精度へ局在化しました。
本体は未修正です。追加検証済みの実数SlaterNet・2電子・float64・AdamをG3b初回候補とし、同じセル/背景定数を含む独立変分参照と実行計画を用意しました。
独立seed検証は凍結測定に限定され、複数の全学習乱数軌道は未検証です。
G2cで両trainerの報告境界を修正し、複素局所エネルギーを保持したままCSV・EW統計・ログへ実数の平均エネルギーを渡します。
locstdは複素局所エネルギーのRMS揺らぎであり、平均値の標準誤差とは区別します。旧checkpointのoptimizer状態は旧勾配の履歴を含むため、再開方針を明示してください。
専用CPU環境は`.venv-pw`。導入根拠と固定依存はtests側を参照し、既存の研究環境を更新しないでください。

## 開発の流れ

各自のブランチ（`wada_dev`、`okada_dev`）で開発して push → `develop` にプルリク → `main` に統合。
計算結果（`results/`）、チェックポイント、解析の出力、計算サーバ用の設定はリポジトリに入れません。

## ライセンス

Apache License 2.0（`LICENSE`、元の PeriodicWave と同じ）。
