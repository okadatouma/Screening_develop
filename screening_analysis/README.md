# screening_analysis — 遮蔽注意の設定生成と内部解析

2026-08-14 のクラウド実験で使ったものを、GPU で再実行できる形に置いたもの。
`periodicwave/` のオリジナルは書き換えていない（`CustomScreenformer.py` の `TRIM_MAX` 追加のみ、既定値は従来と同じ）。

## 何が分かっているか（要約）

N=9 (↑5 ↓4)、batch 64、400 step、E/N は step 300–400 の平均。

| 構成 | r_s=5（液体） | r_s=40（結晶） |
|---|---|---|
| CustomPsiformer (softmax) | −3.372 ± 0.025 | −38.289 ± 0.107 |
| Screenformer 減衰なし | −3.358 ± 0.025 | — |
| 硬い窓 ＋ Trim（原論文どおり） | −3.372 ± 0.016 | −37.673 ± 0.194 |
| 硬い窓のみ | −3.675 ± 0.011 | −38.677 ± 0.024 |
| **緩い減衰（w > d_ref）** | **−3.710 ± 0.005** | **−39.032 ± 0.044** |

- **効くのは滑らかな動径減衰のみ。** 鋭い打ち切りも内容ベースの Trim も逆効果
- ゲート / TanhNorm / 出力倍率は ±0.07 の範囲で中立
- 遮蔽長 w も Trim 幅 r も学習でほとんど動かない（±3%、+0.1%）。事実上ハイパーパラメータ
- ヘッドは 自分 → 近距離 → 中距離 → 平均場 の粗視化階段になる（自己重み 58/30/23/14%、一様なら 11%）
- q·k の分散は距離とスピンで最大 45%（層0のヘッド1）しか説明できず、残りはそのペア以外の
  7 電子の配置に由来する = 3 体以上の情報

原論文 Ken M. Nakanishi, *Screening Is Enough*, arXiv:2604.01178 との差は Trim の変域のみ
（論文 r = σ(s_r) ∈ (0,1)、実装 r = 2σ(ρ) ∈ (0,2) で初期値 1.9 = 実質無効）。
`CustomScreenformer.TRIM_MAX` で切り替えられるようにした。既定 2.0 = 従来どおり。

## 使い方

### 1. 設定を作る

```
python screening_analysis/make_config.py --tag soft  --batch 1024 --w-factor 1.4 --w-spread 1.4
python screening_analysis/make_config.py --tag paper --batch 1024 --w-factor 0.75 --w-spread 1.31 --trim 0.9 --trim-max 1.0
python screening_analysis/make_config.py --tag nomask --batch 1024 --softmask off
python screening_analysis/make_config.py --tag psi   --batch 1024 --psiformer
```

`periodicwave/configs/run_2deg_screen_<tag>.py` ができる。結果は `..._<tag>` フォルダに出るので
既存の結果を上書きしない。置換が 1 箇所に当たらなければ停止する。

### 2. 走らせる

ローカル CPU:

```
.venv-pw\Scripts\activate
set PYTHONPATH=%CD%
python -X utf8 -u periodicwave\configs\run_2deg_screen_soft.py 5_4 5 CustomScreenformer tri 400
```

ws の GPU（`ws_pw_kit` 経由）: `jobs/pw_2deg_screen.sh` の `SRC` を生成した config に向けるか、
`ws_pw_kit\ws_screen_all.ps1` の `run_2deg_screen.py` の箇所を差し替える。
引数は `<nup>_<ndown> <r_s> <network> <shape> <iterations> [<tag>]`。

推奨: **`--w-factor 1.4 --w-spread 1.4`（全ヘッド w/d_ref = 1.0〜1.96）**。
`w < d_ref` にすると硬い打ち切りになって悪化する。

### 3. 内部を見る

```
python screening_analysis/attn_probe.py results/2deg-Coulomb/CustomScreenformer/el5_4_rs5_tri_soft
python screening_analysis/attn_plots.py results/2deg-Coulomb/CustomScreenformer/el5_4_rs5_tri_soft
```

`attn_probe.npz`（s, 重み, 距離）と `attn_qk.png` / `attn_heads.png` が結果フォルダに出る。
要約表には自己重み・実効到達距離・sd(s)・分散の説明率（距離 / スピン）が並ぶ。

## 落とし穴

- `d_ee` は **周期ノルム**（`networks.make_periodic_construct_input_features`）。モジュール直下の
  Euclid 版 `construct_input_features` と取り違えると距離が桁で変わる
- 分数座標への変換は `f = dr @ inv(L).T`。格子ベクトルは行列の**列**なので転置を落とすと壊れる
- `jnp.clip(d/w, 0, 1)` は `d ≥ w` で勾配ゼロ。窓が距離分布の下に落ちたヘッドは飢餓状態になる
- チェックポイントの `opt_state` は kfac_jax が無いと unpickle できない。`data` と `params` だけ読むこと
- 結果フォルダに `qmcjax_ckpt_*.npz` があると自動で再開する。ゼロから走らせたいときは tag を変える

## 次にやること

1. **q, k を外して α ≡ 1 に固定した参照系**。落ちなければ「効いているのは動径フィルタと
   値ベクトルの多体性で、注意ではない」と確定する
2. **ΔE(w) 曲線**。w を固定して 1.0〜3.0 で掃引すると「相関を距離 w で重み付けたときの変分コスト」
   が得られる。ゲージの曖昧さのない物理量で、人手の 2 体波動関数との比較の物差しになる
3. 減衰を raised cosine から `exp(−d/w)` に変える（打ち切りが消え、勾配も死なない）
