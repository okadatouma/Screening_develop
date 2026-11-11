# CustomScreenformer — スクリーニング注意（Multiscreen）版 Transformer 波動関数

Liang Fu グループの Transformer 波動関数コード（PeriodicWave）に、
教科書『Transformerを理解する』Part 3 で扱ったスクリーニング注意
（Trim・Softmask・分母なし集約・TanhNorm・ゲート）を移植したものです。

**元のファイルは一切変更していません。** すべて複製・新規ファイルとして追加してあり、
従来の CustomPsiformer（softmax 注意）と同一条件で並走比較できます。

## 追加ファイル（6 個）

| ファイル | 内容 |
|---|---|
| `periodicwave/CustomScreenformer.py` | 本体。`CustomPsiformer.py` の複製で、softmax 注意をスクリーニング注意に置換（変更箇所は `[SCREEN]` 印付き） |
| `periodicwave/train_screen.py` | `train.py` の複製。変更は import 1 行と `'CustomScreenformer'` の分岐 1 個のみ |
| `periodicwave/configs/run_2deg_screen.py` | `run_2deg.py` の複製。`CustomScreenformer` を選べるようにし、設定ブロックを追加 |
| `compare_screen.py` | 2 つの結果フォルダのエネルギー曲線を重ね描き（PNG 出力）し、学習された受容幅 r と遮蔽窓 w を表で表示 |
| `test_screenformer.py` | 正しさの検査（反対称性・格子周期性・勾配とラプラシアンの有限性） |
| `run_2deg_screen.bat` | ダブルクリックで検証run（2 電子, r_s = 10, 600 ステップ）を実行 |
| `evaluate-density-screen.py` | `evaluate-density.py` の複製 + [SCREEN] 拡張：密度・ペア分離・g(r) に加えて、学習された遮蔽窓のエッジ（d = w の等高線）をペア分離図に重ね描き。r_h・w_h の表も出力 |
| `run_evaluate_density_screen.bat` | ダブルクリックで上記を CustomScreenformer の結果フォルダに実行 |

## 設計の要点：トークン＝電子

言語モデルではトークンは文中の位置をもつため、遮蔽窓 w は「トークン間の位置差」に
かけました。この コードではトークンは電子そのものであり、電子に並び順はありません
（波動関数は置換反対称）。そこで、

- **Softmask の「距離」には物理的な電子間距離 r_ee（周期ノルム）を使います。**
  遮蔽窓 w_h はヘッド h ごとに学習される長さで、そのまま**物理的な遮蔽長**として
  読めます（2DEG 設定では長さの単位は r_s·a_B）。
- ヘッドごとに異なる遮蔽長 w_h が並ぶことが、そのまま **Multiscreen**
  （複数の遮蔽スケールの並列）になります。
- **MiPE（位置符号化の回転）は導入しません。** 並び順に基づく符号化は置換同変性を
  壊すためです。位置の情報は既存の周期特徴量（sin/cos）から入ります。

## タイル 1 個の処理（`make_screening_attention`）

入力は各電子のストリーム x_i。ヘッド h ごとに：

1. 射影：q, k, v (, g) = x·W_Q, x·W_K, x·W_V (, x·W_G)
2. 単位長正規化：q̄ = q/|q|, k̄ = k/|k|, v̄ = v/|v|
3. cos 類似度：s_ij = q̄_i · k̄_j （必ず −1 ≤ s_ij ≤ 1。1/√d の割り算は不要）
4. Trim（受容幅 r_h, 学習可能）：α_ij = ( ReLU(s_ij − (1 − r_h)) / r_h )²
   — s_ij ≤ 1 − r_h の項は厳密に 0
5. Softmask（遮蔽窓 w_h, 学習可能）：m_ij = ( 1 + cos(π·d_ij / w_h) ) / 2（d_ij < w_h）、
   d_ij ≥ w_h では厳密に 0 — d_ij は電子 i, j の物理的距離（周期ノルム）
6. 分母なし集約：h_i = Σ_j α_ij·m_ij·v̄_j （softmax の分母を使わない）
7. TanhNorm：o_i = tanh(|h_i|)·h_i/|h_i| — 出力の大きさを 1 未満に抑える
8. ゲート：o_i ← o_i ⊙ tanh(SiLU(g_i))、最後に出力射影 W_O

学習パラメータの持ち方：受容幅は r_h = 2·sigmoid(ρ_h)（0 < r < 2）、
遮蔽窓は w_h = d_ref·softplus(ω_h)（d_ref はセル内の最大周期距離）。
初期値は r = 1.0、w = 3·d_ref（ほぼ全開の窓から出発し、学習で狭める）。

これ以外（残差接続・LayerNorm・MLP・行列式・envelope・Jastrow）は
CustomPsiformer と完全に同一なので、**差はすべて注意機構の違い**に帰着します。

## VMC 特有の確認事項（局所エネルギーの滑らかさ）

局所エネルギーは log|ψ| の電子座標に関するラプラシアンを使うため、
ネットワークは座標について 2 回微分可能である必要があります。

- raised cosine の Softmask は境界 d = w で C¹ 級（1 階微分が連続）で、
  2 階微分は有界。デルタ関数的な特異性は出ません。
- d → 0 で dm/dd → 0 なので、電子どうしが近づく点に**新たなカスプを作りません**
  （カスプ条件は従来どおり特徴量・Jastrow の担当のまま）。
- Trim の ReLU² も C¹ 級・2 階微分有界で同様に安全です。

## 実行方法

検証run（Windows）：`run_2deg_screen.bat` をダブルクリック
（2↑ 電子, r_s = 10, 三角スーパーセル, 600 ステップ。ログは `run_2deg_screen.log`）。

コマンドラインでは（`.venv-pw` を有効化して）：

```
python -X utf8 -u periodicwave\configs\run_2deg_screen.py 2_0 10 CustomScreenformer tri 600
```

引数の形式は run_2deg.py と同じです：`<n↑>_<n↓> <r_s> <ネットワーク> <セル形状> <ステップ数>`。
結果は `results\2deg-Coulomb\CustomScreenformer\el2_0_rs10_tri\` に入ります。

## 比較方法

1. 従来版を同条件で実行：
   `python -X utf8 -u periodicwave\configs\run_2deg.py 2_0 10 CustomPsiformer tri 600`
2. 比較スクリプトを実行：
   `python compare_screen.py`
   - `compare_screen.png`：両者のエネルギー vs 最適化ステップ
   - 学習された r_h・w_h の表（w/d_ref < 1 なら「遠くの電子を遮蔽する」ことを学習した証拠）

## 電子密度・ペア相関の見方

チェックポイント（`qmcjax_ckpt_*.npz`）には学習済み |ψ|² からサンプルされた
MCMC ウォーカー配置（1024 個/チェックポイント）が保存されているので、
**再計算なしで**密度が見られます。

- `run_evaluate_density_screen.bat` をダブルクリック（または
  `python evaluate-density-screen.py <結果フォルダ>`）
- 左パネル: セルに折りたたんだ電子位置（一様電子ガスなら特徴なしのはず）
- 右パネル: 同一配置内の電子間分離 r₂ − r₁。原点まわりの空白が
  **交換相関ホール**。破線は学習された遮蔽窓のエッジ（周期ノルムで d = w となる線。
  最小と最大の w_h の 2 本）。w がセル内最大距離より大きい間は
  「窓がセル全域をカバー＝まだ空間的遮蔽なし」なので破線は描かれません
- テキスト出力: 動径平均した g(r) と、全層・全ヘッドの r_h・w_h の表

従来版の結果フォルダには元の `evaluate-density.py` がそのまま使えます
（チェックポイント形式は同一。遮蔽パラメータの表と破線が付かないだけです）。

## 切り替えつまみ（アブレーション）

`periodicwave/configs/run_2deg_screen.py` の `CustomScreenformer` ブロックで変更できます：

| 設定 | 既定値 | 意味 |
|---|---|---|
| `use_softmask` | True | False にすると Trim のみ（距離による遮蔽なし） |
| `use_gate` | True | ゲート tanh(SiLU(W_G x)) の有無 |
| `use_tanh_norm` | True | TanhNorm の有無 |
| `init_trim_r` | 1.0 | 受容幅 r の初期値（0 < r < 2） |
| `init_softmask_w_factor` | 3.0 | 遮蔽窓の初期値 w = この値 × d_ref |

## 検証済みの項目（クラウド環境, jax 0.8.0, CPU）

- `test_screenformer.py`：両ネットワークで
  同スピン電子の交換で位相が π 反転・|ψ| 不変（反対称性）、
  全電子の格子ベクトル並進で ψ 不変（周期性）、
  勾配と folx 前進ラプラシアンが有限、をすべて確認。
- 30 ステップの学習run：CustomScreenformer・CustomPsiformer とも
  エネルギーが単調に低下し、NaN なし、MCMC 採択率 ~0.7。
- 学習パラメータ r_h・ω_h に勾配が届いていること（チェックポイントで確認）。
