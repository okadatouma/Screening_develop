# λ-jellium：入力・距離窓の比較実験

大学の **Slurm・単一GPU** 用の独立した実験入口。通常のconfig/trainer・既存checkpointを置き換えない。
物理Hamiltonian、HF軌道＋NN補正、擬スピン対Jastrow、位置／擬スピンMCMC、G1のVMC勾配は既存関数を使用する。
最初は **λ=0.64、WC型HF／AHC型HF、3 seed**。その後0.33、0.40、0.48、0.56、0.72へ展開する。
WC/AHCは**初期HF表のラベル**であり、学習後の相・Chern数を判定する機能ではない。

## 大学で最初に実行すること

リポジトリrootで、既存のJAX GPU研究環境を有効化する。依存の自動更新・インストールは行わない。
必要なのは通常の研究環境のJAX/jaxlib、folx、Optax、KFAC-JAX（既存trainer import用）、NumPy等。
依存バージョン・ソースハッシュ・デバイス・dtypeは各manifestに保存する。GPU実機での性能／収束は未検証。

1. WC/AHCのHF表と、作成時の条件を確認したメタデータを別々に指定する。
2. 下の設定検査を行う。
3. 入力比較を投入する。結果確認後に窓比較、対依存比較へ進む。

```sh
export PW_PYTHON="$(command -v python)"
export WC_HF=/大学の実パス/WC_lam064.txt
export AHC_HF=/大学の実パス/AHC_lam064.txt
export WC_META=/大学の実パス/WC_lam064.json
export AHC_META=/大学の実パス/AHC_lam064.json

# GPU割当て中に実行。初期化まで検査し、学習も出力作成もしない。
"$PW_PYTHON" -m periodicwave.window_experiments.run --validate-only --require-gpu \
  --out /新規出力先/check --hf "$WC_HF" --hf-meta "$WC_META" --label WC --lam .64

# E0/E1/E2/E3 × WC/AHC × 3seed = 24 task。窓を固定し、Value/Gate等を揃える。
# 旧梯子0.45〜3.0の比較には --w-first .45 --w-max 3 を明示する。
"$PW_PYTHON" -m periodicwave.window_experiments.suite \
  --plan /新規出力先/input-plan.json --out-root /新規出力先/input-runs \
  --wc-hf "$WC_HF" --ahc-hf "$AHC_HF" --wc-meta "$WC_META" --ahc-meta "$AHC_META" \
  --lam .64 --phase inputs --python "$PW_PYTHON" --w-first .45 --w-max 3

# suiteが表示した --array と --export を使用。account/partition/memoryは大学の値を追加。
sbatch --array=0-23%1 --export=ALL,PW_PLAN=/新規出力先/input-plan.json \
  periodicwave/window_experiments/submit.slurm
```

`sbatch`は**リポジトリrootから**実行する。plan内のPythonは生成時の絶対パス（別環境なら`--python`）。
taskはGPU未検出時に停止する。GPU1枚を超える並列学習・multi-hostには対応しない。
`%1`は同時実行数。空きGPU数に応じて増やせる。既存plan・runは上書きしない。
本計算の既定値はN36、rs15、Delta=rs、三角6×6セル、Coulomb/Ewaldあり、
1層・16head・幅64・QK/Value次元16・行列式4、LayerNormあり、Adam、batch128、20000更新。
局所エネルギーchunk4、反転計算batch4。GPUメモリに応じて`--batch`/`--chunk`/`--flip-batch`を全比較で揃えて変更する。
MCMCは全電子位置移動＋擬スピン反転。move幅は固定0.2（既存trainerのadaptive幅とは異なる）。
座標は一様、初期擬スピンは全up、burn100。HF初期配置の山への配置は行わない。
旧KFAC runとの直接比較ではoptimizer等も異なる。新しい比較群の条件を揃えるためのAdam入口である。

## HF作成条件の照合

6列 `kx ky Gx Gy Re(c) Im(c)` から**作成時のλは復元できない**。
各表に作成記録から確認したJSONを用意する。解析からλ・格子を推測して補わない。
以下はschema例であり、実際の表の作成条件を確認して記入すること。

```json
{
  "lam": 0.64,
  "rs": 15.0,
  "delta": 15.0,
  "n": 36,
  "label": "WC",
  "lattice": [[11.42775368236749, -5.713876841183745], [0.0, 9.89672499711541]],
  "sha256": "実際のHF表のsha256"
}
```

**例の格子数値をコピーせず**、元のHF計算の列ベクトル格子を十分な桁数で保存する。
この入口の格子は `lattices._triangular_lattice_vecs_periodic_potential(sqrt(2*pi/sqrt(3)), N)`。
ハッシュは `shasum -a 256 "$WC_HF"` 等で確認できる。
検査するのはメタデータと実行条件の一致、表のハッシュ・6列・有限値・軌道数・G列の整合性。
メタデータ自身の物理的な由来はHF作成記録で保証する。ハッシュ照合だけで表の正しさを証明するものではない。
HF表／JSON欠落、λ・rs・Delta・N・label・格子不一致では停止する。表の生成・上書きはしない。

## 実験の切り替え

|入口|内容|
|---|---|
|`--phase inputs`|E0〜E3、W0固定。Value/Gate/HF/学習条件を揃える|
|`--phase windows --chosen-input E2`|選んだ入力でW0〜W3、warmup後に窓を学習|
|`--phase pairs`|E3、W1/W2それぞれ共有窓と対依存窓を比較。他の条件は同じ|
|`--phase recommended`|推奨A/B/C/Dを実行。B/DはValue/Gateも変わるので窓だけの比較ではない|
|`--phase all`|上記全条件を列挙。入力の自動選抜はしない。計算量に注意|

どのphaseもWC/AHCを同じ条件で実行する。`--seeds 0,1,2`で独立初期化・MCMC。
rs、層数、head数、dtype、学習率等の共通引数はsuiteからrunへ渡す。
窓比較では`--w-max`省略時にセルのd_refを使う。W2/W3の最外境界はd_refに固定する（異なる明示値はエラー）。
W3の既定基底境界は `[0,a,sqrt(3)*a,2*a,d_ref]`。小セルや別の上限では`--basis-edges 0,...,dmax`を明示する。
`--cases /path/cases.json`なら複数λを一括列挙できる（以下の形式。0.64を先頭にする）。

```json
[
  {"lam":0.64,"wc_hf":"/path/WC064.txt","wc_meta":"/path/WC064.json","ahc_hf":"/path/AHC064.txt","ahc_meta":"/path/AHC064.json"},
  {"lam":0.33,"wc_hf":"/path/WC033.txt","wc_meta":"/path/WC033.json","ahc_hf":"/path/AHC033.txt","ahc_meta":"/path/AHC033.json"}
]
```

### 入力：Q/Kだけに追加する

`windows.input_features`がraw特徴 `[N,F]` を定義する。
θ=2πL⁻¹(r−atom)、p=(sinθ1,sinθ2,cosθ1,cosθ2)、
χ=(cos hθ1,sin hθ1,cos hθ2,sin hθ2,cos h(θ1−θ2),sin h(θ1−θ2))。
h=6は**スーパーセル**に対する調和数で、N36の6×6結晶周期用。別のNなら`--harmonic`を物理条件に合わせて明示する。

|入力|raw特徴|次元|
|---|---|---|
|E0|p,s|5|
|E1|p,s,χ|11|
|E2|p,χ,n+,n−|12|
|E3|p,n+,n−,n+χ,n−χ|18|

Value・残差stream Xは従来の`embed(p,s)`から作り、QK用には `X_QK=X+extra @ qk_extra` を使用する。
E2/E3のone-hotは(s,1)の線形基底として表現し、元のs射影を再利用する。
追加射影はゼロ初期化し、初期のQK関数をE0と揃える。学習後のQKには特徴が効く。
**入力全体を全streamへ置換する方式ではない**。2層目以降も追加特徴をQKへ渡す。
`--gate position`はpとχのみの位置依存Gate、`--gate learned`は従来のX依存Gate、`--gate off`は無効。
固定Valueは既存の電子対特徴e(r_ij)で、擬スピン特徴を新しくValueへ足してはいない。

### Window

実装は`windows.py`。長さはすべてr_s a_B。

* W0：独立softplus幅の旧型累積cos窓。学習後の順序保証なし。
* W1：δ=softplus(η)、c=cumsum(δ)/sum(δ)、w=wmin+(wmax−wmin)c。
  最外幅固定。初期値はw_first〜wmaxの幾何梯子に一致させる。
* W2：同じ正増分でb_h=dmax c_h。内部累積C(d;b)=sigmoid((b²−d²)/(2bτ))、C0=0、CH=1、m_h=C_h−C_{h−1}。
  これは研究用に選択した具体的な滑らかさの定義。旧sin/Gaussian輪切りとは別物。
  非負・合計1、τ=`--transition`、原点の距離一次微分0。自己項は保持し、固定Valueの集約時のみ既存どおり除く。
* W3：同じ滑らかな固定ring基底Bに対してm_h=Σ_r softmax(β)_hr B_r。
  保存するのは幅でなく混合係数π。Head間の順序は保証しない。

`--min-gap-fraction 0`が添付の厳密な正増分式（既定）。順序だけでは密集を防げない。
必要なら例`.02`で正規化増分へ2%の一様成分を混ぜ、共有境界の最小間隔を設ける。これは明示的な追加仮定。
W1/W2の対依存版は `log w_ss=log w+δ_ss`、3チャネル(++/--/+-)、混合チャネルは対称。
補正をtanhと近傍log-gapの45%未満へ制限し、順序と外端を保つ。
W2では保存量は**ring外側境界**であり、累積窓幅と同義ではない。
最終端の対補正は0。`--pair-max-log-shift .15`を全比較で揃える。

窓Adamはwarmup中、パラメータだけでなく内部状態も凍結する。
`--warmup 1000 --window-ratio .05 --window-ramp 100`：1000更新後に解放し、100更新で窓LRをnetworkの5%まで増やす。
`--freeze-window`は全期間固定。入力／Window以外にエネルギー正則化や行和正規化は追加しない。

## 保存と再開

runごとに`manifest.json`、実数報告値の`stats.csv`、既存形式のcheckpoint、独立RNG sidecarを新規保存する。
エネルギーは**全電子のコード単位**と**一電子あたりHa=E/(N rs²)**を別列にする。
`variance_abs_complex=mean|EL−mean EL|²`と`variance_real=var(Re EL)`は標準誤差ではない。
虚部はVMC勾配まで保持する。診断は別JITで、通常forward内部のビット列との一致は主張しない。

再開は同一引数・環境で`--resume --iterations <新しい総更新数>`を追加する。
パラメータ、Adam状態、データ、RNGを復元し、旧ログを編集せず別のresume CSVへ書く。
依存バージョン・ソース・実効設定が変わっていれば停止する。
この新しい実験用PyTreeは**旧checkpointと互換ではない**。旧研究の再開にこの入口を使わない。
checkpointはpickleを含む既存形式なので、信頼できる自分のファイルだけを読む。

## 診断とΔC_core

```sh
python -m periodicwave.window_experiments.analyze /path/run \
  --burn 100 --measure 256 --walkers 8 \
  --core-centers '[[0.0,0.0],[1.905,0.0]]' --core-radius .3 \
  --out /新規出力先/WC_window_analysis_v1.npz
```

上の中心座標は**コマンド形式の例**。実際の研究で選んだ中心を明示する。中心・半径が片方だけならエラー、両方なければcore統計を計算しない。
coreは指定した中心の和集合。Query iのみcore内に制限し、Key jはcore外でも含め、i=jは必ず除く。
core距離は `min_{center,n∈Z²}|r_i−center−L n|` の**最小像Euclidean距離**。
斜交格子でfractional座標を成分ごとにroundするだけの近似は使わない。
Windowの距離は従来の**周期ノルム**であり、core距離とは別である。

M_ss=1[i∈core]1[s_i=s_j=s]1[i≠j] として、標本bを含む全分子・全分母を合算する：

```
C_core,h,ss = sum_bij M_ss m_hij C_hij / sum_bij M_ss m_hij
ΔC_core,h = C_core,h,-- − C_core,h,++
C_hij = Qbar_ih · Kbar_jh  （本体のepsilon付き長さ正規化）
```

Head出力寄与はこの式に含めない。分母0ならNaNと有効Window質量／対数を記録する。
保存NPZの`layer0_`等のprefixで層を区別する：

|キー|意味・shape|
|---|---|
|`C`, `alpha`|cosine類似度、Trim後係数、それぞれ[S,H,N,N]|
|`window`, `W_window`, `W_aggregation`|m、αm、固定Value対角除去後の実集約W、[S,H,N,N]|
|`core_C_pp/mm`, `core_delta_C`|Window加重C、ΔC、[H]|
|`core_alpha_pp/mm`, `core_delta_alpha`|同じcore／Window重みで平均したα。Cとは別|
|`core_W_aggregation_pp/mm`, `core_delta_W_aggregation`|同じWindow重みで平均したW（mを更に掛けた平均であることに注意）|
|`head_contribution`|c_h=exp(scale) z_h W_h、[S,N,H,E]。共有biasは別|
|`q_head_mean_squared_norm`|mean||c_h||² / sum_h mean||c_h||²。[H]。旧mean-norm²指標と混同しない|
|`dbar`, `dbar_offdiag`|実集約Wによる平均周期距離。自己項あり／なし。[H]|
|`scales_pair`|W0/W1の幅、W2の外側境界、[S,3,H]。チャネル++,--,+-|
|`mixture`, `basis_edges`|W3のπ[S,H,B]と固定基底境界|
|`delta_w_mm_minus_pp`|W1の幅差／W2の外側境界差。[H]|
|`local_energy`|独立MCMC測定の複素EL、[time,batch]|

WC/AHCの比較には**同じλ、格子、core定義、入力、窓の定義／設定**を使う。
AHC解析時に `--compare /path/WC_window_analysis_v1.npz` を追加すれば不一致をエラーにし、
`core_delta_C_this_minus_other`を保存する。学習された幅自体は異なり得るため両方のscales/πを確認する。
core定義のJSON sidecarを図・報告書と一緒に保持する。

### R*：必要な場合だけ重い再標本化を実行

上の解析コマンドへ `--radii .8,1.2,1.8,2.5 --cut-burn 200 --cut-measure 256` を追加する。
必要なら`--cut-layer 0 --cut-head 3`で特定層／Headのみ切る。既定は全層・全Head。
窓にC3のsmooth7切断襟（内端(1−beta)R、外端R、距離²で接続）を掛ける。
ψが変わるので各半径で位置・擬スピンを**再平衡化**し、元のELの単なる再評価では代用しない。
全エネルギー差を保存する。これはHeadのエネルギー加法分解ではない。
blockingのplateauが未確認／悪化が不確かな場合はRstar=null。
暫定の離散R*規約は「有意な正の最大悪化の10%以内に、3SEM込みで戻った最小の測定R」。
閾値規約・全曲線・SEMを保存し、単調性を強制せず、物理的遮蔽長と断言しない。
WC/AHCで同じcut半径、beta、対象層／head、測定条件を使用する。

## CPUで配線だけを確認

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m unittest tests.test_window_experiments -v
JAX_PLATFORMS=cpu PW_TEST_X64=0 .venv-pw/bin/python -m unittest tests.test_window_experiments -v
JAX_PLATFORMS=cpu .venv-pw/bin/python -m periodicwave.window_experiments.run --smoke --out /tmp/NEW_WINDOW_SMOKE
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_window_experiment_run /tmp/NEW_WINDOW_INTEGRATION
```

smokeはN4、Coulombなし、少数walker／更新。HFなしも許すが、大学のWC/AHC検証の代わりにはならない。
実行済み範囲・既知問題は`tests/RESULTS.md`の窓研究欄を参照。
float64 KFAC、定数folx、Psiformer点微分、K10、既存Envelope不整合の修正はこの実験に含めない。
新入口はAdamのみ。旧ProductionのKFAC snapshotを保護したことと、新モデルをKFACで検証したことは別である。
