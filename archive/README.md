# archive — いまの計算では使っていないもの

2026-11-02 に、いまの λ-jellium（WC と AHC）の計算と解析で使っていないファイルをここに移した。中身は書き換えていない（git の履歴では移動として残る）。
元の場所からの並びをそのまま残している（`2deg_moire/screening_analysis/x.py` は元の `screening_analysis/x.py`）。

| フォルダ | 中身 |
|---|---|
| `2deg_moire/` | 2DEG・モアレ・窓の設計の時期（2026-08〜09）。評価と比較（`compare_screen.py`、`evaluate-*-screen.py`）、ノートブックを書き出した `screening_electrons.py`、説明（`README_SCREEN.md`、`CustomScreenformer_解説.md`。どちらも 2026-08 時点のネットワーク）、設定の派生（`periodicwave/configs/run_2deg_screen_*.py`、`run_2deg_moire.py`）、解析（`screening_analysis/` の 75 本） |
| `lj_scan_v1/` | λ-jellium の最初のスキャン（液体から出発、2026-10-08〜09）の解析 |

2DEG の基本の設定 `periodicwave/configs/run_2deg_screen.py`、検算 `test_*.py`、`screening_analysis/` の共有の部品（`attn_probe.py` など）は、
ほかのコードやテストが使うので元の場所に残した。

## 走らせるとき

ここのスクリプトは `periodicwave` と `screening_analysis` の共有の部品（`style`、`sampler`、`attn_probe` など）を import する。リポジトリの根で

```
PYTHONPATH=.:screening_analysis:archive/2deg_moire/screening_analysis python archive/2deg_moire/screening_analysis/<スクリプト>.py ...
```

のように走らせる（λ-jellium の最初のスキャンは `archive/lj_scan_v1/screening_analysis` も足す）。中に書いてあるデータのパス（`results/2deg-Coulomb/...` など）は当時のもの。
