# 解析境界の安全性：削除防止・失敗伝播・Head辞書対応（2026-10-24・最新）

今回の変更はscreening_analysisの指定3入口と小規模テストだけ。大規模整理や他の解析移行は行っていない。
Production、Hamiltonian/loss、MCMC/Optimizer、Diagnostic計算、head_contrib.py、研究データ・旧結果は不変。

- **moire_summary.py**：force時のos.removeを撤去。通常の既存summary.json読込は維持。
  forceはsample cacheを使わず再標本化、Headは正式analyzeで再評価し、summary_recomputed.jsonへ新規保存。
  下位cacheや旧summaryは削除/上書きしない。再保存は単一folderの--outputで新規名を指定する。
  保存はexclusive create。研究固有の集計式は変更していない。
- **analyze_all.py**：子工程はcheck=True。失敗で例外終了し後続を実行しない。
  --outの既存図は子工程実行前に拒否。正常な--pair等の実行順/引数は維持。
  --forceと--scan/--nuの併用は、別名summaryを未移行の後続図が読まないため、実行前に明確に拒否。
  後続図を改造したり、古いsummaryで成功を装ったりしない。単独のmoire_summaryで安全に再計算できる。
- **fig_channels.py**：tuple展開をやめ、正式Head analyzeの辞書C/wを使用。
  share=100*C**2/sum(C**2)を維持。--settings/--latticeは正式入口へそのまま渡し、欠損は例外で通知。
  cachedを使わず旧cacheを読まない。methodがhead-contrib-diagnostic-v1以外なら拒否。
  新規fig_channels_diagnostic_v1.pngに保存。既存出力を拒否し、PNGに入力由来metadataを付与するコードを追加。
  CLI引数/出力保護のためmainへまとめたが、描画本体の式・固定数値・注記はASTで変更前と一致を確認。
  固定の旧研究注記や③④の既報数値を最新の科学的結論として再認定したわけではない。

検証：`JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_analysis_safety` → **10 PASS /0 FAIL /0 ERROR /0 SKIP**。
一時ディレクトリと既存の小さなN=2 fixtureを使用し、大量の合成データは作成していない。
実際のDiagnosticでTrim^4のC/shareを照合、設定欠損/文字列格子の明示エラー、旧cache不使用・非変更を確認。
削除/上書き防止、force時の再計算、通常cache経路と同じ入力での同じ集計値、子工程の正常順序と途中停止を確認。
Python子プロセスの非ゼロ終了伝播も実際に実行。fig_channels --helpも成功。

**未実行**：Matplotlibが現環境にないため、PNGの描画・埋込metadataの保存/再読込・見た目は未検証。
インストールや研究環境変更は行っていない。実際のN9/N49全runでの作図、一括解析全体、moire_summaryの実MCMC再計算は未実行。
テストで科学計算をmockした箇所は境界制御の検証であり、その物理量を新たにPASS認定していない。
既存33件/全snapshotは今回は再実行せず、本体・旧数値資産の非変更をハッシュで確認する。

残存：structure_probe/lj2.readoutのAttention再実装、複数のcheckpoint/config復元、他の図の旧API/cache対応等は未変更。
moire_summaryの下位解析にも旧設定復元の制約が残る。欠けた設定を推測する修正は行っていない。
今回は指定3点で停止する。

---

# G3a残存問題の分類・追加収束検証・G3b準備（2026-10-24・最新）

**調査と追加検証のみで停止。Productionの修正、G3bの相互作用VMC学習は実施していない。**
G3aの13完走条件を「全経路成功」へ読み替えない。元の4適合/9制約の記録は保持し、今回の原因特定を別記する。
新規コードは `check_g3a_followup.py` 1ファイル。旧 `check_g3a_vmc.py` 等も変更していない。
新しい2つの96step自由系学習、独立seed測定、診断用反実仮想、参照計算は `g3a_followup/` に新設。
機械可読な分類は [classification-v1.json](g3a_followup/classification-v1.json)。

## 13条件の分類と使用範囲

ここで「適合」はG3aの全検査に対する判定であり、全粒子数/全Hamiltonianで使用できるという意味ではない。
5条件のSlaterNet complexは物理検証に適合し、制約は周辺の初期ログにあった。

| G3a保存条件 | 元の判定 | 制約の種類 | 今回の調査結果 | 研究使用の判断 |
|---|---|---|---|---|
| slater-real64 | 適合 | なし | 検証した小規模条件で全検査適合 | 検証範囲内で使用可 |
| slater-real32-v2 | 適合 | なし | 検証した小規模条件で全検査適合 | 検証範囲内で使用可 |
| slater-complex64-v2 | 制約 | 初期INFO整形のみ | 検査した物理量は適合、ログは未修正 | 計算経路は候補、無監視の新規complex学習は回避 |
| slater-complex32-v2 | 制約 | 初期INFO整形のみ | 検査した物理量は適合、ログは未修正 | 計算経路は候補、無監視の新規complex学習は回避 |
| slater-two64 | 制約 | 初期INFO整形のみ | 検査した物理量は適合、ログは未修正 | 計算経路は候補、無監視の新規complex学習は回避 |
| slater-kfac64 | 制約 | 初期INFO整形のみ | 検査した物理量は適合、ログは未修正 | 計算経路は候補、無監視の新規complex学習は回避 |
| slater-kfac32 | 制約 | 初期INFO整形のみ | 検査した物理量は適合、ログは未修正 | 計算経路は候補、無監視の新規complex学習は回避 |
| screen-complex64 | 制約 | 別graph方向微分差 | 検証側のdtype昇格を特定、native JVP適合 | 当該N1 Adam条件は候補、G3bでは非選択 |
| screen-default64 | 制約 | 別graph方向微分差 | 検証側のdtype昇格を特定、native JVP適合 | 当該N1 Adam条件は候補、G3bでは非選択 |
| screen-complex32 | 適合 | なし | 検証した小規模条件で全検査適合 | 検証範囲内で使用可 |
| screen-kfac32 | 適合 | なし | 検証した小規模条件で全検査適合 | 検証範囲内で使用可 |
| psiformer-complex64 | 制約 | 点微分の混合精度＋初期INFO | 倍精度softmax/AD経路を未認定 | 精密な局所energy・微分研究には回避 |
| psiformer-default64 | 制約 | 点微分の混合精度＋初期INFO | 倍精度softmax/AD経路を未認定 | 精密な局所energy・微分研究には回避 |

これとは別に、完走13条件に含まれない **Screenformer64 KFACはERRORで使用回避**。
通常λ/HFの同じ混合dtype KFACも今回新たに正常認定しない。
Psiformerは平均energyが統計的に合ったことだけをもって点微分を正しいとは扱わない。
float32 Psiformerの本格的な物理最適化、相互作用系、多device/GPU等も今回未検証である。

## 2電子系：同じ固定参照に対する追加測定

元の参照 **1.2347060624819985** を変更せず、元の24step checkpointのparamsを固定した。
既存checkpointを上書きせず、測定だけ独立seed=2303/2304/2305/2306へ変更。
各seedは初期walkerと遷移乱数を独立に生成、32walker、burn128、MCMC幅1、blocks1。
同じseedの1024/4096/16384sweepは長い系列のprefixであり、3個の独立試行とは数えない。
元のseed2303/1024sweepの平均とSEMも再現した。独立Fourier係数のNumPy再評価との差は約3e−16で、参照を調整したものではない。

| 24stepの同じψを測定 | 標本総数 | Re E平均 | blocked SEM | 固定参照との差 | z |
|---|---:|---:|---:|---:|---:|
| 元の1seed×1024sweep | 32,768 | 1.2353392281 | .0002783228 | .0006331656 | 2.275 |
| 4seed×1024sweep | 131,072 | 1.2350144182 | .0001388645 | .0003083558 | 2.221 |
| 4seed×4096sweep | 524,288 | 1.2347927768 | .0000693053 | .0000867143 | 1.251 |
| 4seed×16384sweep | 2,097,152 | 1.2347429463 | .0000347476 | .0000368839 | 1.061 |

SEMはseedごとの保守的blocked SEMを独立系列として二乗和合成した値。
最後の4seed平均間から推定したSEM=.0000136353も別記録し、都合のよい方へ切り替えない。
長い系列の各seedで|z|<=.98、blocking末尾3点のplateauを確認。
相関時間τ_int=(SEM_block/SEM_IID)²/2は約1.67〜1.77sweep（½から始まる定義）。
元のIID SEM=.000149265に対してblocked SEM=.000278323であり、自己相関を無視すると約1.86倍過小評価する。
209万標本でも独立標本数は生の総数より少ない。blocking plateauは有限系列の診断であり独立性の厳密証明ではない。

同じ保存系列の先頭をさらに捨て、burnを128→512→1024相当とすると、平均は
1.2347429463→1.2347359092→1.2347262424、参照から1.061→.848→.564SEM。
前半/後半の平均は1.2347709287±.0000492949 /1.2347149640±.0000491779。
顕著な初期化偏りや長時間ドリフトを検出する結果ではなかった。
これらは標本を共有する感度分析であり、独立試行を増やしたと数えない（stability-v1/summary.json）。

**解釈**：当初の2.275SEMのずれは統計的にあり得る。標本数/seedを増やして参照へ近づき、
今回の範囲ではサンプリング揺らぎと整合する。少数標本でのずれを系統バイアスやHamiltonian不一致と断定する証拠はない。
一方、「完全にバイアスがない」ことを証明したとも言わない。差が常に単調に減ることも要求していない。
配置ごとのProduction局所energyと独立Slater公式の差はランダム256点/seedで最大1.67e−14程度（診断記録）。
既存の固定配置・固定snapshotの厳密条件は別に保持する。

### 学習不足と測定誤差を分離

complex SlaterNetの学習を同じseed42のまま96stepまで新規ディレクトリで実行。
**最初の24stepはparams/data/optimizer_state等、全checkpointフィールドが旧24stepと厳密一致**（prefix-exact.json）。
従って、学習延長のために初期条件や元の参照状態を差し替えていない。

| 完了step | 同じψの厳密変分E | 基底E₀=1.23370055013617との差 |
|---|---:|---:|
| 8 | 1.2721353677 | .0384348176 |
| 24 | 1.2347060625 | .0010055123 |
| 48 | 1.2339193720 | .0002188218 |
| 96 | 1.2337033123 | .0000027622 |

これは**最適化不足による変分残差**で、同じψのMC平均とE_refの差とは別問題。
96step後の4seed×4096sweepでは **1.2337052601 ± .0000035894**、同状態の参照との差.543SEM。
基底からの残差も大きく減ったが、まだゼロとは認定しない。

G3b候補として実数SlaterNet・N2・float64・Adam96stepも既存trainerで実行した。
固定配置のψ/局所energy/交換/PBC、勾配、更新、checkpoint、初期ログを含む既存検証CLIは全項目適合。
最終参照は **1.23372305785**、独立4seed×4096sweepでは **1.23374498339 ± .00001433773**（1.529SEM）。
最適化残差は2.2508e−5。個々の測定seedの最大|z|=2.701、plateauも確認した。

追加2条件の学習初期seedは42固定であり、**異なる初期値/SGD乱数seedでの学習収束は今回未検証**。
独立化したのは凍結パラメータでの測定seedである。相互に独立な学習試行まで検証したと報告しない。

## 残存4問題の原因と最小修正候補（今回未実装）

### 1. complex通常trainerの初期INFO

`periodicwave/train.py:640` がcomplex initial_energyを `%03.4f` へ渡す。
INFOが有効な新規学習/burn-in経路でLogRecordの整形がTypeErrorになる。handlerによってはlogging errorとして表示され学習は継続する。
G3aのobserverはこれを捕捉したが、正常なログとしては扱っていない。
train_screen.py:686は既にrealを使用。G2cのstep後のreported_energy境界は別箇所である。
最小候補はこのINFO引数だけ `jnp.real(initial_energy[0])`。raw E_L、loss/aux、G1勾配、CSV定義を変更する必要はない。
別段階でreal/complex・INFO有効・optimizer有無・初期/復元分岐を確認してから修正する。

### 2. Psiformerの点微分差

`CustomPsiformer.py:107`付近の `jax.nn.dot_product_attention` を調査。
**使用環境のJAX0.7.2の実装** `jax/_src/nn/functions.py:966` は、入力がfloat64でも
`padded_logits.astype(np.float32)`を経てsoftmaxを計算する。64bitパラメータだけでは全計算が64bitになるわけではない。

テスト専用の反実仮想として同じQK/√d → softmax → V式をfloat64で計算し、既存N2テストを比較した。
Productionへこの代替式を導入したり、学習に使用したりしていない。

| N2・既存固定条件 | 元のJAX Attention | 検証専用float64 Attention |
|---|---:|---:|
| phaseのfolx−Hessian trace | 1.782359425e−7 | −8.3267e−16 |
| log振幅のfolx−Hessian trace | 4.811755616e−8 | 4.4409e−16 |
| phaseのforward値 | −.162316763023 | −.162316771045 |
| log振幅のforward値 | 1.375909433889 | 1.375909424843 |

N1でもsoftmax=1なのでAttentionをValueの恒等写像に置いたテストでは、forward軌道は厳密に同じなのに、
raw軌道Hessianとlog/phase Hessian由来の局所energy差が **6.21e−17** へ減る。
元のraw軌道ADからの局所E=.3228033963261491+.07254664273777983iに対して、
恒等写像での値は **.3228033938484307+.0725466390818130i**。
後者は元のfolx値とほぼ一致し、5点有限差分もh=.004/.002でこちらへ近づく。
**旧raw軌道Hessian側も絶対的な正解とは扱えない**。folxだけを原因と断定せず、混合精度Attentionを通る微分経路全体の問題と分類する。

最小候補はPsiformerのAttention評価でdtypeを保持するsoftmaxを明示すること。
ただしN2ではforwardも約1e−8変わるため、単なる構造整理ではなく数値動作修正として別承認・version付き基準が必要。
現行snapshotを更新して隠さず、解析的な小attention・finite difference・folx/各AD・JIT・gradient・optimizerを再検証する。
今回、ライブラリ更新やProductionのkernel/コンパイル設定変更は一切行っていない。

### 3. Screenformer64 KFAC

既知ERRORをもう一度再現し、元の `kfac_jax.utils.weighted_sum_of_objects` に入るdtypeを観測した。
係数はfloat64、第一vectorを係数倍したaccumulatorはmask_omega/out_scaleまでfloat64になるが、
第二vector（velocity）の同じ葉はfloat32のまま。
KFAC0.0.7 utils/math.py:144–147のabstract比較でValueErrorになる（shapeは一致）。
本体の明示float32初期化はCustomScreenformer.py:365、377付近。optionalなmask centre/rings等にも同じ設計がある。
**エネルギーやVMC係数の不一致ではなく、更新を構成するdtype契約のERROR**。

最小候補は(a)Screening学習パラメータのdtypeを基幹weightへ揃える、または(b)更新の係数倍を葉のdtype契約に合わせる。
(a)は初期値/旧checkpointのdtype・optimizer state移行を伴い、(b)はKFAC境界の数値仕様へ影響する。
今回はどちらも導入せず、旧stateを無条件castして再開可能とも主張しない。float32 ScreenformerとSlaterNet64は別の動作経路。

### 4. Screenformer64の5.8341e−12の差

**Productionの不具合ではなく、G3a検証の異なるdtype条件を比較したことが原因**と特定した。
G3aはスカラーt（float64）で `p+t*v` を作ったため、t=0でも本来float32の
mask_omega/out_scaleをfloat64へ昇格させていた。同じ数値のパラメータでも、別dtypeの計算の微分である。
`jax.jvp(lambda params: loss(params,...)[0].real, (p,), (v,))` のように
元の混合dtype PyTreeをそのまま渡すと、独立centered-score縮約との差は以下になる。

| local_energy_chunk | 旧スカラー摂動との差 | 元dtype保持のPyTree JVPとの差 |
|---|---:|---:|
| 0 | 5.83410e−12 | −2.77556e−17 |
| 32 | 5.83410e−12 | −3.46945e−17 |
| 128 | 5.83410e−12 | −3.46945e−17 |

元の32eps基準のまま適合。EL配列差~2.6e−15、score差~1.4e−16では旧5.8e−12を説明できなかった点も確認。
最小対応は検証入口でdtypeを保つことだけで、Production修正は不要。
旧G3aの検証コード/不適合記録は保存し、今回の新規検証に正しい入口を実装した。
全条件の学習を再認定したわけではないが、当該2条件をProductionの勾配不具合と解釈する理由はなくなった。

## G3bの最小構成と独立参照（実行計画のみ）

**採用経路：実数SlaterNet・num_layers=0・N2完全偏極・float64・Adam。**
G3aの正常な実数N1に加え、今回実数N2で96step/4測定seedを確認した経路を選ぶ。
Screenformer/Psiformer、多層化、KFAC、新Ansatzは初回G3bへ持ち込まない。
null Envelope、Jastrowなし、1行列式、biasあり、include_r_ae=False、blocks1を維持する。
この小Ansatzはcuspや高運動量の相関を十分に表さないが、配管と変分参照の独立検証に適する。

| 物理仕様 | 指定 |
|---|---|
| 粒子/スピン | N=2、nspins=(2,0)、同種フェルミオン |
| 格子/PBC | L=4 a₀の正方格子、A=16 a₀²、twist=(0,0) |
| 密度 | n=2/16=.125 a₀⁻²、r_s=1/√(πn)=1.5957691216 a₀ |
| 運動項 | −½ΣΔ、ℏ=m=e²=1のHartree原子単位 |
| 相互作用 | 平面内の3D Coulomb 1/r、interaction_energy_scale=1、対数2D Coulombではない |
| Ewald | 通常PBC factory、convergence_radius=5、α=2.8/√A=.7 a₀⁻¹ |
| 背景/自己項 | 一様正電荷背景、G=0成分を除き、Nξ/2の自己像＋背景定数を含む |
| 報告 | まず全エネルギー。1電子当たりは明示的に2で割る。モアレ/λ/外場なし |
| optimizer | Adam、初回clip=0、lr=.01。通常の公開API/パラメータ構造を維持 |

周期ノルムや最小像の単一1/rをEwaldの代わりに使わない。
コードの定数は `ξ=Madelung_const−phi_S_q0` と書き直すと、
**V=Σᵢ<ⱼ Φ(rᵢⱼ)+Nξ/2**、ΦのFourier係数はq≠0で2π/(A|q|)、q=0はゼロになる。
これは説明上の式で、Productionは書き換えていない。

独立なPython/math.erfc和でα=.375/.7/1.0を比較し、選んだ3変位でαに依らず~4e−16以内に一致。
Production Ewaldとの点差はcutoff2で最大1.005e−6、3で1.288e−10、**5/7で2.22e−16**。
G3aはscale0なのでこのcutoff誤差は当初の自由系energy差の原因ではない。
G3bでcutoff2を漫然と引き継がず5を選ぶ。セル全域/節近傍の追加確認は実行段階の条件にする。

このセルでξ=−.9750662300004891 a.u.。
校正用のSlater軌道(1, exp(iqx))、または実数(1, cos(qx))の期待値は独立に
**E=q²/2−2π/(Aq)+ξ=.008634320135680662 Hartree（全体）**、
**.004317160067840331 Hartree/電子**。
これは**その波動関数の厳密期待値であり、相互作用系の厳密基底値ではない**。
今回の実数96step checkpointをそのまま固定して相互作用Hamiltonianへ評価する場合の独立予測は
T=1.2337230578498934、対のHartree−exchange=−.2499980641485502、自己/背景=ξ、合計 **.008658763700854166**。
今回このHamiltonianでProduction学習は開始していない。

さらに任意の現行0層Slater paramsについて、有限Fourier係数Cから正確な変分参照を作れる。
S=C†C、Rq=C†Uq C、Uqは平面波の運動量をqだけ移す行列として、

`E_int = (1/2A) Σ[q≠0] (2π/|q|) {Tr(S⁻¹Rq)Tr(S⁻¹R−q) − Tr(S⁻¹Rq S⁻¹R−q)} + Nξ/2`

を用いる。有限軌道基底なのでこの期待値のq和は有限であり、Productionの局所energyを参照に流用しない。
今回plane/standingで同じ解析値を再現することを確認し、一般係数の参照関数もtests内に用意した。
真の相互作用基底との比較が必要な後続段階では、同じN・spin・セル・twist・G0・ξを用いた
平面波CI/EDをcutoffと全運動量sectorについて収束させる。それが済むまで文献の異なる単位/境界/粒子数の値を基底参照として引用しない。

### 実行順序と合格条件（未実行）

1. 校正状態と保存済実数N2 paramsで、相互作用の点ごとのdefault/folx、Ewald cutoff5/7、独立Fourier期待値を確認。
2. パラメータ固定の4独立測定seedで開始。32walker×16384sweep/seedを初期計画とし、burn-inとblock幅を変えてplateauを確認。
   energy差は**4.5 blocked SEM＋独立参照/Ewaldの評価誤差**で判定、全energyのSEM目標は1e−4 Hartree以下。
   達しなければ未収束として記録し、閾値を広げない。ReのSEMと複素揺らぎ/EWを分離する。
3. 新規出力先でAdam24/48/96/192step。各checkpointのparamsに対する独立E_refと方向微分を計算し、
   凍結標本の平均/勾配と比較。energyが減っただけでは合格にしない。初回はreal・clip=0。
4. 収束した候補を4測定seedで再評価し、checkpoint/Adam状態/CSVの保存復元を確認。
   保存先は将来の `tests/g3b_validation/<version>/<condition>`。存在する出力先には書かない。
5. Production通常trainerはdeterministic時seed42固定であり、**今回独立化したのは測定seedだけ**。
   別初期値/SGD乱数まで独立な学習再現性は未検証として残す。
   seed別params/初期walkerを別checkpointから投入する案も、実装・復元時の乱数契約を確認してから採用する。
   完全に独立した学習RNGを要求する場合、既存の時刻seedへ依存せず、別途明示seedの試験入口を検討する必要がある。

G3bの最初の目的は**相互作用を含むVMCの配管と厳密変分期待値の一致**。
有限SlaterによるAnsatz誤差をHamiltonianや推定器の不具合と混同しない。
真の相関基底状態の高精度認定、独立seedの全学習再現性、complex系の初期ログ修正等は別の未完了事項である。

## 回帰・保存・終了

両精度のG2b固定snapshotを新しいプロセスで再比較し、**全項目厳密一致**。
float64は902配列/21651要素、float32は1319配列/31142要素（Adam/KFAC経路を含む）。
ログはg3a_followup/frozen64.txt、frozen32.txt。
追加のcomplex96学習CLIは既知の初期ログ検査でexit2、real96はexit0。
KFAC診断CLIのexit0は原因観測の完了を示すだけで、内部の学習は明示的にERRORのまま記録する。
既存33件は今回は全体再実行していない。直前G3aの **30 PASS/0 XFAIL/1 FAIL/2 ERROR** は履歴であり、
Psiformer/KFAC等の既知問題を今回修正してPASSへ変更したとは報告しない。
本体・既存解析・旧テスト/数値資産はSHA256で保護し、追加前後を照合する。
定数folx、K10、isotropic Envelope、block MCMCは未修正。選んだG3b入口は既知の該当不整合を踏まない構成に限定する。
今回の学習は追加の自由系2条件のみ。相互作用系の実行や修正へ進まず停止する。

---

# G3a：自由電子系の独立参照とProduction短期VMC（2026-10-24・最新）

**G3aの検証を実施し停止。本体・既存解析・既存snapshotは変更していない。全面的な正常判定ではない。**
実数SlaterNetの両精度、float32 ScreenformerのAdam/KFACは今回の検証範囲で正常。
複素通常trainerの初期ログ、倍精度ニューラル経路の厳しい微分比較、倍精度Screenformer KFACには下記の問題が残る。
G3bの相互作用系には進んでいない。過去のG1/G2b以前の記録は上書きしていない。

## 追加した検証コードと保存先

- `check_g3a_vmc.py`：既存trainerを呼ぶ短期学習、独立Fourier/FFT参照、凍結MCMC、方向勾配、checkpoint照合。
- `check_g3a_exact.py`：既存HF Ansatzによる厳密固有状態、λ=0、周期性・反対称性・MCMC分布。
- `check_g3a_controls.py`：独立な確率ヒストグラム、同一パラメータの精度比較、Adam状態の再計算、HF入力制約の再現。

総括機械可読記録は [audit-v1.json](g3a_validation/audit-v1.json)。13完走条件のうち全検査適合4、部分不適合9、別に既知KFAC ERROR1。
新規資産は `g3a_validation/`。各条件の `summary.json`、`environment.json`、`initial_params.npz`、
`effective-config.json`、`run/train_stats.csv`、全stepのcheckpoint、`frozen_samples.npz`を保存。
traceには報告energy、複素分散、勾配ノルム、独立参照energy前後、受理率、Adam更新誤差を記録する。
保存名/設定の実ファイル一覧とSHA256は `g3a_validation/sha256-v1.json` を参照。既存checkpointへの書込みはない。
出力先が既に存在する場合、検証CLIは停止する。
既存trainerはconfig自体を保存しないため、今回のeffective-config.jsonは未変更のconfig生成関数と保存済summaryから事後再構成した記録であり、trainerが自動保存した設定とは呼ばない。新しいCLI実行では生成直後に記録する。

環境：CPU・1 device、Python3.13.14、JAX/jaxlib0.7.2、folx0.2.19、KFAC-JAX0.0.7、Optax0.2.6、NumPy2.3.3。
既存 `.venv-pw` を利用し、依存/システム/JAX-XLA設定は変更していない。実効情報は各environment.json。
GPU、多deviceの短期学習、長期収束、大粒子数、相互作用系は今回未実行。

## 模型・Hamiltonianと独立基準

単位は既存Hamiltonianの原子単位、正方格子L=4、q=2π/L、H=−½ΣᵢΔᵢ。
通常PBC factoryの既存設定 `interaction_energy_scale=0` を使用し、Coulomb/背景の寄与をゼロにする。
相互作用を無効化するための本体修正や新Ansatzは追加していない。
1電子の厳密基底energyは0、同スピン2電子は k=0 と最低非零運動量の占有で **q²/2=1.23370055013617**。
短期学習の参照値は、その時点の波動関数の変分energyであり、基底energyと混同しない。

SlaterNetの既存 `num_layers=0`、null Envelope、Jastrowなし、軌道biasありを用いると、
軌道は既存1体特徴 sin(qx),sin(qy),cos(qx),cos(qy),spin の線形結合。
Fourier係数Cを学習パラメータから独立に構成し、S=C†C、T=C†diag(|k|²/2)C、
**E_ref=Re Tr(S⁻¹T)** を全占有軌道について求める。
点ごとの独立波動関数は det(M)、運動energyは −½Σᵢⱼ(M⁻¹)ⱼᵢ ΔᵢMᵢⱼ。
log微分やProductionの局所energy関数をこの参照に流用しない。

Screenformer/Psiformerは1 Transformer層・1電子・1head・幅4・MLP8・GELU、LayerNormなし。
独立参照は raw軌道のFFTによるDirichlet/Parseval形式
**E_ref=Σₖ |k|²|cₖ|²/(2Σₖ|cₖ|²)**。32/64/128/256格子で収束を確認し、
最後の差の2倍を参照離散化誤差予算とする（最終budget<1e−7を要求）。
Screenformer64ではE128=.0544630010602、E256=.0544630009957、budget=1.29e−10。
Psiformer64では全格子.108516411758332付近（差~3e−17）。
この検証は1tokenのAttention経路を通すが、電子間Attentionの多粒子物理を認定するものではない。

## Production短期学習：13完走条件と既知1 ERROR

通常はAdam24step/B128、非線形NNは16step/B64、KFACは8step/B64。
初期化seed42、MCMC各step8 proposal、burn-in16回、幅1固定、blocks=1。
初期化→Production MCMC→局所energy→G1勾配→optimizer→CSV→checkpointを既存trainerで実行した。
数学演算は変更せず、INFOのフォーマットエラーを記録するテスト内observerだけを使用。
**observerが捕捉したエラーは正常終了とは判定しない。**

| 条件（保存ディレクトリ） | step | 独立変分E 初期 → 最終 | 凍結MCMCのRe E ± blocked SEM | 受理率 | 判定・制限 |
|---|---:|---:|---:|---:|---|
| slater-real64 | 24 | 0.78788967 → 0.01175440 | 0.01008968 ± 0.00105447 | 0.908 | 検証範囲PASS |
| slater-real32-v2 | 24 | 0.43034515 → 0.00388203 | 0.00422297 ± 0.00057016 | 0.945 | 検証範囲PASS |
| slater-complex64-v2 | 24 | 0.84798858 → 0.00454387 | 0.00477928 ± 0.00052340 | 0.951 | 初期logエラー |
| slater-complex32-v2 | 24 | 0.66068739 → 0.00589954 | 0.00610327 ± 0.00041744 | 0.959 | 初期logエラー |
| slater-two64 | 24 | 1.42733529 → 1.23470606 | 1.23533923 ± 0.00027832 | 0.483 | 初期logエラー |
| slater-kfac64 | 8 | 0.84798858 → 0.34973838 | 0.35110480 ± 0.00262369 | 0.722 | 初期logエラー |
| slater-kfac32 | 8 | 0.66068739 → 0.22396214 | 0.22292704 ± 0.00240793 | 0.760 | 初期logエラー |
| screen-complex64 | 16 | 0.81767754 → 0.05446300 | 0.05735472 ± 0.00211836 | 0.838 | 縮約差 |
| screen-default64 | 16 | 0.81767754 → 0.05446300 | 0.05735472 ± 0.00211836 | 0.838 | 縮約差 |
| screen-complex32 | 16 | 0.56484866 → 0.05620886 | 0.05577442 ± 0.00263584 | 0.838 | 検証範囲PASS |
| screen-kfac32 | 8 | 0.56484866 → 0.06820709 | 0.07082029 ± 0.00498709 | 0.860 | 検証範囲PASS |
| psiformer-complex64 | 16 | 1.21322580 → 0.10851641 | 0.11225010 ± 0.00331968 | 0.758 | 局所微分差, 初期logエラー |
| psiformer-default64 | 16 | 1.21322580 → 0.10851641 | 0.11225010 ± 0.00331968 | 0.758 | 局所微分差, 初期logエラー |

加えて `screen-kfac64-known` は既知K08のKFAC abstract structure不整合でERROR、学習完走として数えない。
KFACの有限更新・参照energy・勾配・全状態保存を検証したが、complex Fisherの設計を新たに認定したわけではない。
実数波動関数の最適化はSlaterNetのみ、Psiformer float32・実数NN各種・多電子Transformer短期学習は未実行。

### 波動関数・勾配・更新

SlaterNetではsin/cosによる独立軌道行列から振幅と位相（位相は2π同値）を照合。
N2のψ誤差1.60e−15、局所energy誤差1.12e−15、電子＋spin交換の反対称性誤差2.78e−16。
格子周期の座標シフトも検証する。基底energy以上という変分不等式を全stepで確認した。

参照energyのパラメータADと、正規化したランダム方向(seed71)の中心有限差分(h=1e−5)を比較。
両者の差はdouble計算~3.4e−13～4.1e−11（float32由来の方向/パラメータでは~3.4e−9まで）。
有限差分の打切り・丸め誤差とVMCの統計誤差は別記録とし、ビット一致とは呼ばない。
凍結標本で **2 Re[(E_L−E_ref) conj(O_dir)]** と厳密方向微分を比較し、全13条件で4.5 blocked SEM以内。
実装本体の標本平均によるcenterと、厳密E_refによるcenterは異なるため、別途
Production lossのchunk128方向JVPと **2 Re mean[(E_L−mean E_L) conj(O_dir)]** の一致も検証する。
後者のfloat64 Screenformerの微小不一致は下記の通りFAILのまま残す。

Adam10条件の全stepについて、実際の更新前paramsと更新時のMCMC標本からOptaxを独立に再評価。
パラメータ差はfloat64最大2.22e−16、float32最大1.19e−7、所定の32eps基準以内。
moments・countを含むoptimizer状態も別途照合し全条件適合。Adamの係数/実装は変更していない。
精度ごとのJAX乱数初期値は異なるため、float32/64の学習曲線差を純粋な精度差とは解釈しない。
`controls-v2.json`では同じdoubleパラメータをfloat32へcastし、実数/複素のdefault/folxを比較して適合を確認する。

## 厳密固有状態とMCMC分布

既存HF経路で、λ=0、delta=2、potential_type=none、NN scale=0、Jastrowなしを指定。
`exact64-v2` / `exact32-v2` の両精度で次の3状態を検証した。

| 状態 | ψ（規格化定数を除く） | E_L | 検証する分布 |
|---|---|---|---|
| 平面波 | exp(iqx) | q²/2+i0 | 一様 |
| 定在波 | cos(qx) | q²/2+i0（節以外） | (1+cos(2qx))/L |
| 同スピン2電子 | exp(iqx₂)−exp(iqx₁) | q²/2+i0（節以外） | 相対xの密度(1−cos(qr))/L |

default/folxの実部と虚部、周期性、交換反対称性、NN無効時のゼロ勾配を独立解析式と比較した。
NNを無効にしたこの厳密校正だけを「NNが正しく最適化された」根拠にはしない。
各状態64walker×2048sweep、burn128。cos/sin期待値、定在波cos(2qx)=1/2、2電子cos(qr)=−1/2を検証。
さらに12bin×6状態=72binを解析的積分確率と比較し全項目4.5 blocked SEM以内（最大|z|=2.787）。
HF入力は小さなテスト専用tableを新設し、内容とSHA256を各summaryに記録する。
単一行tableで既存loaderがIndexErrorになる制約が見つかったため、平面波は2行目をゼロ係数とする有効な既存形式を使用。
1行版の失敗ログも `exact32.txt` / `exact64.txt` に残す。loaderを修正したり、この失敗をPASS扱いしたりしない。

## 統計誤差と物理的揺らぎ

学習後はparams固定、32walker×1024sweep、burn128の新しいProduction MCMC系列を使用。
周期系の分布は **座標mod L** で検証する。既存MCMCの非ラップ座標そのものに正規化分布を要求しない。
各walkerの時間方向をblock=1,4,16,32,64,128で平均し、block>=16の最大SEMを保守的に採用。
最後の3blockのSEM比<1.5をplateauの診断とし、全条件で確認した。これは有限系列の診断であって厳密な独立性の証明ではない。
エネルギー・虚部・方向勾配・cos特徴についてそれぞれblockingを行い、**4.5SEM +32eps +測定した参照格子budget** を事前基準とした。
これは測定後に広げた許容誤差ではない。統計的適合は個々の配置の微分の正しさとは別条件。

例：N2 SlaterNet64の局所Re E標準偏差=.02701946、IIDを仮定したSEM=.00014927、
blocked SEM=.00027832、標本平均=1.23533923、同じψの厳密変分値=1.23470606（差2.275SEM）。
Im平均=.00036990±.00022581でゼロと適合。
方向勾配は厳密.00144390に対し.00097135±.00024920（−1.896SEM）。
短期学習の最終状態は真の基底1.23370055からまだ.00100551上にあり、MC誤差と最適化残差は区別する。
locstd=sqrt(mean|E_L−mu|²)、EW分散、Re E平均のblocked SEMを同一視しない。

## 既知問題・新たに検出した制約

1. **初期ログの残存不具合（新規検出、未修正）**：`periodicwave/train.py:640` の
   `Initial energy: %03.4f`へ複素initial_energyが渡る。`train_screen.py:686`はreal済み。
   G2cはstep後の報告を修正したが、通常trainerの新規complex学習のINFO初期報告は未修正だった。
   Slater/Psiformer complex条件でobserverがTypeErrorを記録。物理更新/CSV/checkpointは生成できたが、報告全体の正常認定はしない。
2. **Psiformerの点ごとの微分差（既知K09と関連する経路、原因同一とは未確定）**：
   学習後1電子でraw orbital Hessianからの局所energyとの差はfolx4.416e−9、default7.381e−9。
   doubleの32eps基準に不適合。defaultへの切替で直るとは言えない。平均energy/勾配は統計範囲内でも点微分は未認定。
   既存N2 Psiformer微分テストのFAIL（差1.78235943e−7）も再現した。
3. **Screenformer64の別graph縮約差（新規測定、未解決）**：fullbatch/chunk128のProduction方向JVPと
   別評価したcentered scoreに5.8341e−12の差（default/folx双方）。32eps基準に不適合のまま残した。
   勾配SEM約5.84e−4に比べ小さいが、原因を完全に特定せず「無害な丸め」と断定しない。
4. **K08**：Screenformer64 KFACは既知型不整合でERROR。SlaterNet64/32、Screenformer32は実行できた。
5. **定数folx**：文字通り定数のlogabsを返す最小関数は `.laplacian` AttributeErrorを再現。
   既存HF表現の固有状態は実行できたが、定数出力一般の問題を解消したわけではない。
6. **K10**：補助JVP契約は未修正。主勾配を検証するだけで、auxの接線/二次ADの正しさを認定しない。
7. **isotropic Envelope / block MCMC**：既存の失敗再現テストを実行。今回null Envelope・blocks1なので該当不整合を踏まない。
8. **HF単一行loader**：`lj_orbitals.py:35`付近の配列shape制約。2行入力を使う既存表現は正常。入力形式の別問題として記録。

旧pilot（slater-complex64）と、JSONにnp.float32を直接書いて失敗した初回slater-*-32、単一行HF入力の初回は保持する。
それらを完走13条件へ重複計上しない。直したのは新規テスト側のJSON型変換だけで、許容誤差/本体は変えていない。

## 既存回帰・保護対象

G1独立検証は **7 PASS**。既存のblock MCMC/Envelope失敗再現は **2 PASS（不具合の再現に成功）**。
既存33件はG2cと同じ **30 PASS /0 XFAIL /1 FAIL /2 ERROR**。
全体runnerでは29 PASS /1 SKIP /1 FAIL /4 ERROR記録（KFACの3 subtest ERRORを含む）で、
frozen1件を別プロセスでPASS確認したため、メソッド単位では上記の33件となる。
FAILはPsiformerの二階微分、ERRORは定数folxと倍精度KFAC。新規の既存テスト失敗はない。
両精度G2b固定snapshot（5経路・11設定）は**全項目厳密一致**：float64 902配列/21651要素、float32 1319配列/31142要素。
後者はAdam/KFAC経路を含む。`frozen64.txt` / `frozen32.txt` / `numerical33.json`を参照。
この既存snapshotの成功は新しい点微分検査の合格を意味しない。
CLIにexit2がある条件は上記の厳しい比較やログ検査で不適合であり、全条件成功とは報告しない。
G3a開始時の `before-sha256.json` は既存818ファイルを記録し、検証追加後にも変更ゼロを確認した。
以後の変更は文書と新規testsのみ。Production、既存snapshot、既存研究データは保持する。

## checkpointとG3b判断

全13完走条件で保存・復元後のt/data/params/optimizer_state/mcmc_widthを再保存し、全leafの厳密一致を確認。
Adamはさらに独立再計算した状態を検証した。既存形式にPRNG状態はなく、再開乱数軌道の完全一致は要求も認定もしない。
検証MCMCのPRNGはテスト専用npzに別保存し、Production形式には追加しない。
G1/G2b以前のcheckpointは読めても、再開後の勾配・energy虚部・optimizer軌跡は修正後の定義になる。

**G3bは条件付きで着手可能**：今回認定したSlaterNet/float32 Screenformer、blocks1、null Envelopeなどに範囲を明示する必要がある。
複素通常trainerの初期ログとPsiformer点微分を未解決のまま「全Production正常」とは扱わない。
まず別段階で初期ログ境界を修正/検証し、微分差と倍精度KFACは独立課題として解明するのが安全。
今回相互作用系の検証や本体修正には進まず、G3aで停止する。

---

# G2c：複素物理量を保持した実数の報告・統計（2026-10-24・最新）

**R08を解消。変更は両trainerの報告境界とstatistics.pyのdocstringのみ。**
loss.py、G2bのHamiltonian、λ専用、Attention/Embedding/波動関数、MCMC/Optimizer数式、checkpoint形式を変更していない。
旧研究データ・checkpoint・固定snapshotは保持する。G2cで停止し、G3は開始しない。
数式/契約は[CONTRACTS.md](CONTRACTS.md)末尾、実行方法は[README.md](README.md)先頭。

## 問題の場所と変更

optimizerありはG1のJVPの実主lossを報告するが、optimizerなしのmake_loss_stepはrawな複素平均を返す。
この値がtrain.pyのtrain / train_screen.pyのtrainでEW・CSV・%fログへ渡り、R08を起こしていた。
両trainのstep評価後に、報告用 `reported_energy = jnp.real(loss)` を作った。
その値だけをEWのobservation、CSVのenergy、実数ログへ渡す。

| ファイル・関数 | 変更 |
|---|---|
| train.py:686–691、train | loss[0]は保持し、688行で報告用実部を分離。CSV/logも同じ実値を使用 |
| train_screen.py:736–741、train | 同様に738行で分離。eimagはcomplex auxから従来どおり算出 |
| utils/statistics.py:36以降 | EWのdocstringだけ修正。alphaの説明と式が旧文書では逆だったためコードに合わせた |

元のloss/auxは残し、NaN検査もraw lossを参照する。虚部を含む局所エネルギー配列にrealをかけていない。
make_loss_step、make_opt_update_step、JITされたMCMC/学習step等の非train関数はASTで不変を確認。
EWアルゴリズムの数値コードもASTで不変を確認した。K10のraw/JVP dtype差や補助接線の問題を同時修正していない。

## エネルギー・分散・標準誤差の定義

- 複素平均：mu=pmean(mean(E_L))。raw loss/aux.energyに保持する。
- 報告energy：epsilon=Re(mu)。EW・CSV・%fログへ渡す実数値。
- 複素局所エネルギーの分散：V_C=mean(abs(E_L−mu)^2)=V_R+V_I。従来のaux.varianceを保持。
- locstd：sqrt(V_C)。複素局所エネルギーのRMS揺らぎであり、実平均の標準誤差ではない。
- V_R=mean((Re E_L−Re mu)^2)、V_I=mean((Im E_L−Im mu)^2)。虚部診断は別量として扱う。

実平均のSEMは、独立標本で母分散が既知ならsqrt(sigma_R^2/B)。標本分散の推定ではB/B−1も区別する。
MCMCでは自己相関・有効標本数/ブロック平均等の評価が必要であり、sqrt(V_C/B)をSEMとは呼ばない。
今回新しいSEM列や推定器は導入していない。既存real CSVの形式を維持する。
既存analysis_tools.estimate_statsのblocking stderr（pyblock使用）も調査したが、trainerのEW/locstdとは別API。
今回の2 stepの系列では有効な長期SEMを認定せず、blocking精度の実測検証は行っていない。
検証の二状態Hermitian例はV_R=1、V_I=4、V_C=5。4個のIID標本なら実平均SEM=.5、locstd=sqrt(5)と明確に異なる。
同じ例でG1の勾配(振幅2、位相−4、全体位相0)も確認し、虚部を捨てていない。

EWは各stepの実数バッチ平均epsilon_tの時系列統計。
既存alpha=.1で `m_t=.9 m_{t−1}+.1 epsilon_t`、
`v_t=.9[v_{t−1}+.1(epsilon_t−m_{t−1})^2]`、初回m=epsilon、v=0。
これはバッチ平均の重み付き中心二次モーメントで、walker分散でもm_tのSEMでもない。
2 stepの値を重み(.9,.1)の直接平均・中心二次モーメントから独立に検算した。

## optimizerなしの実評価・CSV・ログ・checkpoint

check_g2c_reporting.pyでは、保存したG2bの旧train関数と新train関数を同じ初期checkpointから2 stepずつ実行した。
Productionに旧版wrapper/クラスを追加せず、テスト内で旧ソースのtrain関数だけを読み込む。
旧版で複素CSV/ログエラーになる対照も残し、新版には正常な実数出力を要求する。

- 両trainer：Psiformer/train、Screenformer/train_screen。
- 実数/複素 × float32/64 × default/folx：計16条件。
- λ専用の複素評価とeimag：両精度2条件。
- 保存された旧/新checkpointのt/data/params/state/mcmc_widthが**全フィールド厳密一致**。
- real波動関数のCSVは全項目が旧版と同じ。complex波動関数もenergyの実部、locstd、pmoveを厳密維持。
- 新版のCSV energy/ewmean/ewvar/locstdはfloatで再読込でき、%fログは全条件で正常に整形できた。
- λのeimag列は旧値と厳密一致。通常系も複素auxを保持し、検証JSONに虚部/分散の対照を記録する。新しいCSV列は追加しない。

G2bで再現したfloat32 Screenformer/folxの例を同じ物理状態で比較：

| 量 | 旧報告 | G2c |
|---|---|---|
| step1 energy | (1.574007+1.1401422j) | 1.574007 |
| step2 energy | (−.18113184−1.9403098j) | −.18113184 |
| step2 ewmean | (1.3984932+.832097j) | 1.3984932 |
| step2 ewvar | (−.5767805+.9731918j) | **.27724612** |
| step2 locstd | 3.3867812 | 3.3867812 |
| 実数用ログ | TypeError | 正常な実数文字列 |

ewvarは「旧複素分散の実部」へ置き換えたのではなく、実数時系列に対する正しい分散になった。
この差は報告値の定義の修正で、波動関数/局所エネルギー/MCMC/勾配の変更ではない。
旧CSVは上書き・自動変換しない。旧複素EW分散から単にrealを取って新EW分散へ流用するのも不適切。

check_g2b_downstreamも更新し、保存済み検証checkpointからの実推論に、ログエラーなし・実数CSV・非負EW分散を要求した。
raw make_loss_stepが返す複素値を直接%fへ渡せないこと自体は変えていない。実際の報告境界で実部を選択する設計。
このraw/JVP契約はK10と区別して記録し、R08が残っているという判定にはしない。
NPZの複素局所エネルギー/平均の保存復元は引き続き成功した。

## G1・G2bの回帰と全固定snapshot

| 実行 | 結果 |
|---|---|
| G1独立物理・clip・主JVP・Adam | 7 PASS、228 records。K10保持再現を含む |
| G2b局所エネルギー/G1接続/clip/λ対照 | 7 PASS、180 records |
| G2b 2 CPU device回帰 | 1 PASS、6条件。隔離プロセスのXLA_FLAGSのみ使用 |
| G2c旧/新報告比較・分散定義 | 3 PASS、20 records。16条件+λ2条件+分散定義2精度 |
| downstream/Adam/KFAC/CSV/保存復元 | 4 PASS、31 records。R08の再現要求を正常性要求へ変更 |

G1/G2bの比較許容値は変更していない。両精度の解析式・AD/FD・default/folx・clipping無効/非活性/活性は引き続きPASS。
VMCへIm E_Lが渡り、θ=.4の位相勾配.2を保持する。活性clipの推定量を元の厳密勾配と同一視しない。

今回は物理計算を変えていないため、**G2b snapshotの全値**に除外なしで厳密一致を要求した。

| 精度 | 比較対象 | 結果 |
|---|---|---|
| float64 | 902配列 / 21651要素、5経路・11設定、Adam更新・保存stateを含む | 全項目厳密一致 |
| float32 | 1319配列 / 31142要素、5経路・11設定、全経路Adam/KFAC更新・保存stateを含む | 全項目厳密一致 |

全2221配列、shape/dtype/PyTree/実効設定/乱数条件を維持。既存snapshotを新しいコードで再生成していない。
物理値の新versionを作る必要はないため、G2cの新記録は報告値と旧/新比較JSONへ分けた。
float64 KFACは既存33件で実行可能なPsiformer/SlaterNetを確認し、他3経路の既知ERRORを残す。
依存更新や通常ProductionのJAX/XLA設定変更はない。専用CPU環境.venv-pwを継続利用した。

## 既存33件・変更ファイル・保存場所

既存33件はG2bから変わらず **30 PASS / 0 XFAIL / 1 FAIL / 2 ERROR**。新しい失敗はない。
Psiformer微分FAILの差は同じ1.78235943e−7、ERRORは定数log振幅folxとfloat64 KFAC screen/lambda/HF。
KFAC3 subtestを1メソッドとして数える。raw runnerのsnapshot SKIPは残し、
同一メソッドをG2b float64基準指定で別実行してPASS（93.191秒）。final-results.jsonで証拠付き集計。
全数値テストが成功したという報告ではない。

- Production：train.py / train_screen.pyのtrain内報告処理、statistics.pyのdocstringのみ。
- tests：新規check_g2c_reporting.py、check_g2b_downstream.pyの現行期待値、test_wavefunction.pyの全G2b比較指定。
- 文書：README/HANDOFF/tests README/CONTRACTS/RESULTSを現行仕様へ更新。
- 記録：g2c_validation/reporting-v1.json、downstream-v1.json、g1-regression.json、g2b-regression.json、two-devices.json。
- snapshot-float32.txt / snapshot-float64.txt：保存済みG2bとの全数値厳密比較の実行ログ。
- numerical-results.json / final-results.json：既存33件のraw/集計結果。

旧trainerと旧downstream検証コードはg2c_validation/*-before.txtに保存する。

## checkpoint互換性とG3へ進む判断

既存checkpoint形式/パラメータ/stateは不変。G2bの旧checkpointによる実評価も、実数CSV/logとして正常終了した。
G2c適用によってoptimizer状態を初期化する必要はない。G1/G2b以前の物理修正に伴う履歴や、
checkpointがPRNG keyを保存しない既存の再現性制約は別問題として残る。
研究データ・過去checkpoint・旧CSV・数値snapshotは変更していない。

残存：K10、定数出力folx、Psiformer微分、倍精度KFAC、PBC/isotropic Envelope、block MCMC等。
GPU/multi-host、実研究HFデータ/研究checkpoint、長期学習・自己相関を考慮したSEM推定は今回未検証。
**G3の総合物理検証へ進める状態**だが、既知問題を対象別に識別して検証する必要がある。今回はG2cで停止しG3は実施していない。

以下はG2b以前の履歴。

---

# G2b：通常PBC複素folxの虚部保持（2026-10-24・G2c前の記録）

**Productionの変更は `periodicwave/pbc/hamiltonian.py:234` の1行のみ。**
`return jnp.real(result)`を`return result`へ変更し、独立解析式に一致する複素局所エネルギーを返す。
G1 loss、root/λ専用Hamiltonian、Attention/Embedding/波動関数、MCMC、Diagnostic/解析、checkpoint形式は変更していない。
G2bで停止する。**optimizerなしの評価では、複素平均が報告処理に渡る既存問題R08が通常PBCにも露出した。これは未修正で、全報告経路が正常という完了報告ではない。**
通常Adam/KFACの学習、複素E_Lと主勾配の物理検証は完了した。R08の修正は別段階が必要。

## 物理式・返却型・数値比較

ψ=exp(a+iφ)、通常T=−∇²/2では
`T_L=−½[Δa+|∇a|²−|∇φ|²+i(Δφ+2∇a·∇φ)]`。
実部化前の式・順序・分岐・folx設定はそのままで、最後の虚部破棄だけを取り除いた。
complex_output=TrueのE_Lはcomplex64/complex128になる。振幅のみで虚部0でも複素dtypeを保持する。
complex_output=Falseの分岐は従来の実数dtypeのまま。Ewaldや実外場のreal処理は変えない。

G2と同じ2π周期関数、振幅のみ・位相のみ・混合、1/2電子、両精度、通常/JIT、default/folxを比較した。
位相/振幅の座標微分・Laplacian・共役なしの二乗と、λ専用の重み付き運動項/擬スピン反転を独立NumPy式で検算する。
1電子R=(.31,−.27)の例：

| ケース | 修正前PBC folx | 修正後（解析式/defaultにも一致） |
|---|---|---|
| 振幅のみ | .0448278684403 | .0448278684403 + 0i |
| 位相のみ | .0735831930436 | .0735831930436 − .0209087989024i |
| 混合 | .118411061484 | .118411061484 + .00112313131067i |

| PBCと解析式の差（各精度24条件） | float32 | float64 |
|---|---:|---:|
| 最大実部差 | 2.98e−8 | 1.39e−17 |
| 最大虚部差 | 1.49e−8 | 2.78e−17 |
| 複素誤差RMS | 1.11e−8 | 1.42e−17 |

比較はG2と同じ実部/虚部別32 machine epsilon。既存テストの許容誤差は緩めていない。
詳細はg2b_validation/metrics.json。G2の約.15025の虚部欠落は解消した。

## G1の主勾配、clipping、batch/device

実際のlocal_energyを変更していないG1 lossへ渡した。
α=0、θ=.4、ψ=exp(αcos x+iθsin x)の周期例ではE=θ²/4、∂θE=.2。
**通常PBC folxの勾配0→.2**となり、default/独立DirichletエネルギーのAD・有限差分・解析勾配に一致する。
両精度の主勾配最大差は1.49e−8 / 2.78e−17。
λ専用はλ=0なら.2、Δ=2,λ=.4なら振幅/位相勾配(−.32,.264)をそのまま維持する。

非零の実振幅勾配も検証した。実際のモアレ外場U=.3、均等周期格子、
a=αcos(q·r)、φ=θsin(q·r)、α=0では∂αE=−2U=−.6、∂θE=θ|q|²/2。
実波動関数・複素振幅・位相・混合方向について、default/folx、両精度、通常/JIT、独立FDと一致した。

- clipping無効：厳密な変分エネルギー勾配と比較。
- clip=5（この例では非活性）：default/folxとも位相勾配.2。
- clip=.25（活性）：両者とも独立clipスコア約.0406772。元エネルギーの厳密微分.2とは同一視しない。
- batch複製1/2、chunk=0/3、JIT、2 CPU deviceのglobal平均/pmeanでも正しい位相勾配を保持。

clipping関数・中心化・JVP・KFAC登録は変更していない。K10の補助JVP契約不整合は残る。

## 後段処理とR08の具体的な制約

通常Adam/KFACはG1のJVPを通り、主lossは実数で、auxのE_Lは複素のまま保持される。
aux分散はmean(|E_L−mean(E_L)|²)で、位相由来の虚部の分散も含む。実部だけの分散へ置き換えていない。
両trainerの実Adam/KFAC 1 stepのCSVでenergy/ewmean/ewvar/locstdをfloatとして再読込し、有限値を確認した。
複素E_L/平均はNPZで厳密保存・復元できる。CSVもcomplexを明示すれば読み込める。

**R08：optimizer='none'の通常PBC評価に新たに露出する報告互換性の問題。**
make_loss_step（train.py:210、train_screen.py:216）は未微分のraw lossを返すので、複素平均がそのまま報告へ渡る。
現行lossのraw primalとJVP primalのdtype差に関連し、λ専用/default complex等には以前からある問題。

既存のLayerNorm前の検証checkpoint（float32 Screenformer、t=0、state=None）を読み、別の一時出力先で2 step実推論した。
checkpointのhashは不変。推論計算・新checkpoint保存自体は完了するが、次の問題を確認した：

| 対象 | 実測・問題 |
|---|---|
| inference_stats.csvのenergy | 例 `(1.574007+1.1401422j)`。floatとして読む既存consumerはValueError |
| EW統計 | 複素平均が実数用のdiff*incrへ入り、例 `(-.5767805+.9731918j)` のEW分散になる |
| %fのログ書式 | 複素JAX scalarでTypeError。logging errorとして出力される |

このためoptimizerなしの報告経路を正常とは認定しない。正常なAdam/KFAC学習のCSVと区別する。
報告する平均エネルギーを実数にすることと、VMCに必要な各E_Lの虚部を消すことは異なる。
loss/trainers/statistics/既存consumerは今回変更せず、R08の解消も別段階に残す。

## 既存数値基準と新しいversion

比較先は既存G1の両精度snapshot。全項目・PyTree・shape・実効設定を照合した。
PBC3経路のenergy/VMC勾配/更新/学習後checkpoint/統計は意図的な変更として分ける。
許すdtype変化はE_Lとenergy auxの対応する実数→複素昇格だけ。パラメータ/stateの複素化は許さない。
lambda/HFはエネルギー・勾配・更新を含む**全数値**、他経路は波動関数・軌道・初期params・微分/Hessian・MCMC・11 Attention設定を厳密比較した。

| 精度 | G1と厳密一致する配列 / 要素 | energy/gradient/update区分 | 実際の値変更 | 許可したdtype昇格 |
|---|---:|---:|---:|---:|
| float64 | 586 / 13196 | 316配列 | 265配列 | 21 |
| float32 | 792 / 17850 | 527配列（KFAC含む） | 328配列 | 21 |

旧CPU/G1/G2のsnapshot・研究データは更新していない。
新記録：

- physics-v1.json：独立式・実部/虚部・G1勾配・clip・λ対照の180 records。
- two-devices-v1.json：2 CPU deviceの6条件。
- downstream-v3.json：実/複素振幅・位相・分散・Adam・保存再読込・実trainer・R08の31 records。
- snapshot-float32-g2b-v1/：全5経路・11設定、Adam/KFAC 1 step・保存復元。
- snapshot-float64-g2b-v1/：全5経路・11設定、Adam 1 step・保存復元。

新snapshotには設定・依存版・入力hash・初期params・source/harness hashも保存する。旧基準を自動更新しない。
将来G2bの全値と比べるときは通常の全項目厳密比較を用い、今回の変更区分を永久除外にしない。

## テスト結果と残る問題

既存33件：**29 PASS / 1 XFAIL / 1 FAIL / 2 ERROR → 30 PASS / 0 XFAIL / 1 FAIL / 2 ERROR**。
K03の期待失敗だけがPASSへ改善。旧実部化を要求する試験は複素値保持へ改名・更新し、旧ソースを保存した。
許容誤差を変更したり、不具合経路を除外したりしていない。
raw runnerのsnapshot未指定SKIPは残し、同じメソッドを別プロセスで基準指定してPASS（74.930秒）。
final-results.jsonは証拠を明記して33メソッドへ集計。KFACの3 subtest ERRORは1メソッドとして数える。

| 実行 | 結果 |
|---|---|
| check_g2_local_energy | 7 PASS / 180 records、27.815秒 |
| 同2 CPU device | 1 PASS / 6条件 |
| check_g2b_downstream | 4 PASS / 31 records、21.217秒。R08保持再現は報告の正常性のPASSではない |
| F0 12 + Wavefunction 3 + FormalDiagnostic 7 | 22 PASS、198.986秒。36診断条件・既存検証checkpoint評価・微分/JIT等を含む |
| f32全5経路の実学習 | Adam/KFACとも1 step・有限性・保存復元を確認 |
| f64実学習 | Adam全5経路、KFACは実行可能なPsiformer/SlaterNetを確認 |

既存33件に新規失敗はないが、別途R08の新露出を明示している。
残るFAILはPsiformer folx/Hessian差（同じ1.78235943e−7）。ERRORは定数log振幅folxとf64 KFACのscreen/lambda/HF。
定数位相のfolx ERROR、K10、PBC＋isotropic Envelope、block MCMC等も未修正。
CPU専用.venv-pw（Python3.13.14/JAX0.7.2/folx0.2.19）で実行し、依存や通常XLA設定を変更していない。
GPU/multi-host、実研究checkpoint/HFデータ、長時間再学習・観測量への定量影響は未検証。

downstream-final.txt/jsonの初回追加推論テストには1 FAILも保存した。
state=None/t=0のcheckpointはt=1へ復元されるため、iterations=2では1行だったというharnessの区間設定ミス。
iterations=3として既存のt+1契約に従う2 stepを実行した。checkpointや本体・許容誤差を変更して通したのではない。

## checkpointと学習再開

既存検証checkpointのパラメータからの波動関数評価と、旧checkpointを読む実推論を確認した。
新しいAdam/KFACのcheckpointではstate/data/paramsを復元し、形式・PyTree・dtypeは維持した。
研究用の過去checkpoint全般の互換性や、optimizer状態を継承する長期学習は未検証。
旧stateには虚部欠落時の勾配履歴があり、再開後の軌跡は変わる。勝手にstateをリセットせず、
継承/初期化の研究方針とG2b適用を記録して別の出力先へ保存する。
λ専用/HFの対象外経路をこの通常PBC修正の影響と混同しない。

以下はG2調査と過去段階の記録。

---

# G2：複素局所エネルギーの虚部欠落を原因特定（2026-10-24・修正前の記録）

**調査のみ。本体はG1終了時から一切変更していない。loss.py・Hamiltonian・checkpoint・研究データ・既存snapshotは保持。**
新規コードは `tests/check_g2_local_energy.py` の1ファイル、測定値は `g2_validation/`。
数式・関数の対応と将来の最小修正案は[CONTRACTS.md](CONTRACTS.md)末尾、実行コマンドは[README.md](README.md)先頭。
G2で停止し、Hamiltonian/K10/folx定数出力問題の修正には進んでいない。

## 原因と正しい式

ψ=exp(a+iφ)、通常のT=−∇²/2について、

`T_L = −½(∇²a + |∇a|² − |∇φ|²) − i/2(∇²φ + 2∇a·∇φ)`。

logψの勾配の二乗は共役しない。VMCスコア共分散のconj(∂logψ)とは別の演算。
Hermitian Hで平均エネルギーが実数でも、点ごとのE_Lの虚部や位相スコアとの相関は一般に非零。
実部だけが一致しても正しい複素局所エネルギーとは認定しない。

**直接の原因は `periodicwave/pbc/hamiltonian.py:234` の `return jnp.real(result)`。**
228–233行はすでに位相Laplacian・位相勾配二乗・振幅/位相交差項を正しく計算している。
folxの中間微分と、この実部化より前の値を解析式で照合し、虚部は234行で消えると特定した。

| 経路 | 結果 |
|---|---|
| root hamiltonian.local_kinetic_energy default/folx（124–155行） | 有効な微分出力では複素解析値と一致、虚部を保持 |
| 通常PBC default（215–219行） | root defaultへ委譲し複素値を保持 |
| 通常PBC folx/complex（224–234行） | 最後に実部だけを返す。K03は未修正 |
| CoulombMoire（298–300行） | 実外場を足しても失われた虚部は戻らない |
| λ専用 weighted lap（59–62行）、local_energy（188–200行） | 重み付き運動項と擬スピン反転項を複素のまま返す |

Ewaldの実ポテンシャルのreal処理（173行）は別の責務。全real処理を一括削除する案ではない。

## 独立した解析式と実測

2π周期のa=Σ(α cos x+β cos y+u s)、φ=Σ(θ sin x+η cos y+v s)を試験入力とする。
振幅のみ・位相のみ・混合、1/2電子、両擬スピン、λ=0/.4、Δ=2、両精度、通常/JITを含む。
NumPyの明示的なsin/cos微分から1階微分・Hessian対角・Laplacianを求め、AD/folxと比較した。
λの反転比はexp[−2s(u+iv)]で独立に検算。2電子積関数は演算子の検証入力であり、新しい研究用Ansatzではない。
Coulombは強度0の実際のPBC入口を使って運動項を分離する。Ewald全般の新規物理検証を行ったとは主張しない。
モアレは別に非零の実外場を独立NumPy式から加えて検証した。

1電子R=(.31,−.27)、float64の例：

| ケース | 複素解析値 | 通常PBC folx |
|---|---|---|
| 振幅のみ | .0448278684403 + 0i | .0448278684403 |
| 位相のみ | .0735831930436 − .0209087989024i | .0735831930436 |
| 混合 | .118411061484 + .00112313131067i | .118411061484 |

微分保護のない特異な原点や節ではなく、滑らかで有限なψでも再現する。
純平面波はIm T_L=0でK03を検出できないため、位相曲率・振幅/位相交差項を含む例が必要。

| 比較対象 | float32 最大実部差 / 最大虚部差 | float64 最大実部差 / 最大虚部差 |
|---|---|---|
| folx微分から実部化前に合成したT_Lと解析式 | 1.49e−8 / 1.49e−8 | 2.78e−17 / 1.73e−17 |
| 通常PBC defaultと解析式 | 1.49e−8 / 1.49e−8 | 1.39e−17 / 2.78e−17 |
| 通常PBC folxと複素解析式 | 2.98e−8 / **.15024653** | 1.39e−17 / **.15024653** |
| λ専用全局所エネルギーと解析式 | 4.02e−8 / 5.84e−8 | 8.33e−17 / 5.55e−17 |

通常PBC folxの複素誤差RMSは約.0764369。倍精度でも消えず、単なる丸め誤差ではない。
λ専用の複素誤差RMSはfloat32 3.31e−8、float64 3.44e−17。
条件数・RMS・参照スケール・相対L2誤差も `g2_validation/metrics.json` に保存した。
比較は実部/虚部別の32 machine epsilon。既存テストの許容誤差は一切変更していない。

## folx側の挙動と別の呼出し不整合

folx 0.2.19の実座標入力・実出力tuple(φ,a)の微分は、上記解析式と一致した。
したがってK03は「folxでは複素波動関数を扱えない」という制約ではない。
installed interpreter.pyのcomplex-input警告は複素**入力**の微分に関するもので、今回の実座標とは別。
ソースの場所/hash・依存バージョン・実行デバイスをJSONに保存した。library更新はしていない。

別問題：座標に依存しないliteral constant出力は、folxから通常Arrayとして返りlaplacian属性を持たない。
その属性を必ず参照するroot/PBC/λの呼出し側ではAttributeErrorになる。
定数log振幅だけでなく、定数位相も両精度で再現した（計12エラー条件）。数学的には零微分であり物理的特異点ではない。
零係数×座標依存式の表現では微分属性が残り、同じ解析波動関数の計算を検証できるが、この回避をProductionに入れてはいない。
この呼出し契約と、正しく算出した複素値を最後に実部化するK03を分離する。
今回のPASSは期待エラーの**再現成功**も含み、その経路の物理的正しさを認定しない。

## G1修正後のVMC勾配への影響

G1は変更せず、実際のHamiltonian出力をloss.make_lossへ渡した。
α=0、θ=.4、ψ_s=exp(α cos x+iθ sin x)、1電子・両擬スピン均等標本の周期例を用いた。
通常のDirichletエネルギーはE=θ²/4、∂θE=.2。
λ専用はCONTRACTSの周期エネルギー式から、Δ=2、λ=.4で(∂αE,∂θE)=(−.32,.264)。
正規化Dirichlet積分のADと有限差分（h=1e−5）、独立解析勾配を比較した。

| Hamiltonian | 実装の振幅勾配 / 位相勾配 | 厳密値 |
|---|---|---|
| 通常PBC default | ≈0 / .2 | 0 / .2 |
| 通常PBC folx | ≈0 / **0** | 0 / .2 |
| λ専用 λ=0 default/folx | ≈0 / .2 | 0 / .2 |
| λ専用 λ=.4 default/folx | −.32 / .264 | −.32 / .264 |

通常PBC defaultの勾配最大絶対差はfloat32 1.49e−8 / float64 2.78e−17。
λ専用は7.75e−8 / 5.55e−17。通常PBC folxの物理的誤差.2は両精度・batch/deviceで再現した。
これはG1の係数不具合の再発ではなく、入力されたE_Lから位相情報が消えているため。
非clip時、欠落項は `2 mean((Im E_L−mean(Im E_L)) ∂φ)`。
純位相方向の寄与が失われ、振幅も変えるパラメータでは振幅由来の成分だけが残る。

clip=5（非活性）でもdefault .2 / folx 0。clip=.25（活性）では独立clipスコアが
約.0406772 / 0となり、その値を元エネルギーの厳密勾配.2と同一視しない。
G1の正しい係数・共役も、clippingも、捨てた虚部を復元できない。
16点周期格子×両擬スピン=32標本、batch複製1/2、chunk=0/3、JITを検証。
隔離2 CPU deviceでは各device 16標本でglobal中心化/pmeanを使い、6条件で同じ結論になった。
通常ProductionのJAX/XLA設定は変更しない。K10も未修正。

## 実行結果と保護範囲

- `physics-v1.txt/json`: **7 PASS / 180 records**、27.389秒。独立物理検証とK03/定数出力の現状再現を区別する。
  内訳：微分24、通常PBC/root比較48、λ専用48、モアレ4、G1接続36、clipping8、定数出力エラー12。
- `two-cpu-devices-final.txt/json`: **1 PASS / 6条件**、1.910秒。
- `g1-regression.txt/json`: **7 PASS**。G1の完全和/AD/FD・clipping・主JVP・Adamの基準は維持。K10保持再現を含む。
- `existing-targeted.txt`: **16件＝13 PASS / 1 XFAIL / 1 FAIL / 1 ERROR**。23.584秒。
  XFAILはK03、FAILはPsiformer微分（同じ最大差1.78235943e−7）、ERRORは定数log振幅の平面波folx。
  既存VMC 7件はすべてPASS。既存の失敗や期待失敗を隠していない。

CPU専用.venv-pwを使用。Python3.13.14、JAX/jaxlib0.7.2、folx0.2.19、NumPy2.3.3。
既存33件のG1全面実行基準は29 PASS / 1 XFAIL / 1 FAIL / 2 ERROR。
G2ではその関連16件を再実行したもので、全33件・全固定snapshot・KFAC学習の再計算を行ったとは報告しない。
残るfloat64 KFACの既知ERRORは今回再実行せず、未解決のまま。
Production/主要解析/旧snapshot・checkpoint/旧検証データは変更前hashと比較して保護する。
GPU、multi-host、別バージョン、研究用HF/checkpoint、大規模NNの全微分、長時間学習・観測量は未検証。

初回 `single-cpu.txt/json` は新規harnessの2 FAILも保存した。
1件は2電子の虚部が部分相殺して.001123となる配置に、根拠なく>.01を要求した検出条件の誤り。
もう1件はfloat32で表されたθ=.40000000596によるFDを、厳密な.4から求めた閉形式期待値と比較した入力不一致。
解析入力を同じ表現値に揃え、非零信号の判定を丸めスケールへ合わせた。数値誤差の許容値は緩めていない。

## 将来の最小修正とsnapshot管理（今回は未実施）

候補は通常PBC複素folxの234行だけを `return result` にすること。
先行する数式・演算順・folx設定を変えず、root/LJ/lossは変更しない。
修正後にK03をPASSへ変更し、旧実部化の再現は旧版として保存する。
定数出力、K10、Psiformer/KFACの問題を同時修正しない。

通常PBCのE_Lのdtype・虚部・複素分散・VMC勾配・optimizer更新は意図的に変わる。
旧snapshot/G1基準は上書きせず、新しい独立物理基準と全経路更新を別versionへ保存する。
波動関数・Attention・Embedding・軌道・微分・初期params・checkpoint形式は厳密一致を維持し、
専用LJ/HFの対象外経路はG1のエネルギー/更新基準も比較する。
clip・JIT・device、Adam/KFAC・保存復元、CSV/解析の複素dtype取扱いも確認する。今回は実際の修正はしていない。

## 過去のλ-jellium研究への判断に必要な情報

現行run_lambda_jellium.py:112はlambda_jellium.local_energyを選択し、通常PBCのreal投影を通らない。
AnsatzがScreenformer/HFという名前だけで影響を決めない。Psiformer/Screenformer/SlaterNet/HFのいずれも、
通常PBC＋complex_output=True＋folxを選んだ場合が直接対象になる。
誤って通常PBCを使ったλ runなら、虚部だけでなくλ固有項の欠落も別に確認する。
専用経路でもG1以前のloss問題や既存folx/NN微分問題まで解決済みとは言わない。

過去結果の定量判断には、次が必要：

1. 実行時のHamiltonian/loss/Ansatzソース（commitと未commit差分）、JAX/folx版、dtype・device・実効設定。
2. `make_local_energy_fn`、laplacian_method、λ/Δ/r_s、格子/単位、complex、clipping/median/center、batch/chunk/device。
3. checkpointのparams・walker座標/擬スピン、HF外部表/hash、network設定、spin更新規約。
4. 同じ固定標本で再評価した複素E_Lと位相スコアの共分散。eimag平均≈0だけでは勾配の正しさを判定できない。
5. 学習再現には初期seed/RNG経路とoptimizer状態も必要。既存checkpointはPRNG keyを保存しないため完全再現には制約がある。

研究データを今回は変更・再評価しておらず、過去のλ研究結果が正しい/誤りと一括断定しない。

以下はG1および過去段階の記録。

---

# G1：VMC主勾配の最小修正（2026-10-24）

**本体の変更は `periodicwave/loss.py` の `total_energy_jvp` だけ。**
既存のclipping、KFAC登録、primal/auxの戻り方は維持し、主勾配の収縮式を修正した。
Hamiltonian・複素folx・Attention・Embedding・波動関数・MCMC・optimizer・Diagnostic/解析は変更していない。
大規模な構造整理はF3で停止し、F4は新しい研究対象が必要になるまで保留する。
数式と適用範囲は[CONTRACTS.md](CONTRACTS.md)末尾、再実行方法は[README.md](README.md)先頭。

変更した検証コードは `check_g1_vmc.py`（新しい独立物理検証）、`check_g1_snapshots.py`（新規保存と旧forward比較）、
`baseline.py`（G1の明示的な比較区分）、`test_vmc_mcmc.py`（厳密期待値への修正）、
`test_wavefunction.py`（G1比較区分の指定）、`check_vmc_gradients.py`（旧挙動を保存ソースへ固定）。
README/HANDOFF/CONTRACTS/RESULTSも現在の仕様へ更新した。変更前lossと該当旧テストはg1_validation/*-before.txtへ保存。

## 修正した式と独立した物理基準

実パラメータ、Hermitian H、ψ=exp(a+iφ)について
`∂E = 2 Re〈(E_L−E) conj(∂logψ)〉`。
`diff`の作成は変更せず、両分岐を `2 * real(vdot(psi_tangent, diff)) / B` にする（loss.py:294–297）。
第1引数を共役するvdotにより虚部/位相項はplus、実振幅にも係数2が付く。

旧複素clip=0の式は、u=Re E_L、v=Im E_L、μ=平均、A=∂a、P=∂φとして
`2 mean((u−μ_R) A + (2v−μ_I) P)`。
修正後は `2 mean((u−μ_R) A + (v−μ_I) P)`。
旧実数式 `mean(A diff)` は `2 mean(A diff)` になる。

| 二状態の完全和、clipping無効 | 旧実装 | G1 | 厳密値 |
|---|---:|---:|---:|
| H=diag(−1,+1)、ψ=exp(αx)、α=0の振幅微分 | 1 | 2 | 2（E=tanh(2α)） |
| H=[[0,i],[−i,0]]、ψ=exp(iθx)、θ=0の位相微分 | −4 | −2 | −2（E=−sin(2θ)） |
| 混合H、p=(.2,.8)の振幅微分 | −.5962500704 | −.5962500704 | −.5962500704 |
| 同じ混合Hの位相微分 | −2.8641653964 | −1.4320826982 | −1.4320826982 |

正規化完全和、NumPy複素共分散、JAX AD、中心差分h=1e−5を独立に比較した。
非等確率は状態数1:4/4:1で実現し、固定標本のmean(E_L)だけを有限差分する誤検証を避ける。
実数/複素振幅・純位相・混合・全体位相、JIT有無の40条件（各精度20条件）を実行した。

| 主勾配の差 | float32 | float64 |
|---|---:|---:|
| 独立共分散との最大絶対差 | 1.9322e−7 | 5.5511e−16 |
| 同RMS差 | 4.8187e−8 | 1.7723e−16 |
| 完全和ADとの最大絶対差 | 1.1921e−7 | 4.9960e−16 |
| NumPy完全和の中心差分と共分散の最大絶対差 | 2.7861e−10 | 2.7861e−10 |

差分はfloat64で評価し、float32のパラメータはその表現値を使用する。Production差分をfloat64へ変更したわけではない。
主勾配/ADの比較は従来の独立検証と同じ32 machine epsilon、差分法は既存の独立FDと同じ2e−9。
既存テスト・snapshot比較の許容誤差は緩めていない。

## clipping：方式を維持し、推定量を区別

- **無効**：diff=E_L−global_mean(E_L)。完全和で厳密エネルギー微分に一致する。
- **有効・非活性**：閾値100で値が切り詰められない場合、非clip推定量と一致する。純位相例はoff/onとも−2。
- **有効・活性**：既存の実部/虚部別clipと指定中心Cに対する `2 Re mean((E_clipped−C) conj(O))`。
  元エネルギーの厳密微分とは一般に異なる。返すエネルギーは従来通り元の平均。

窓は平均絶対偏差×閾値。median指定は実部median/虚部0、通常は元の複素平均を窓中心にする。
center=TrueはC=clip平均、FalseはC=元平均。これらの処理・集約・dtypeは変更していない。
両精度128条件で、実数/複素、clip=0/.25/1/100、median/centerの全組合せを独立NumPy式と照合した。
clip推定量の最大絶対差はfloat32 2.0129e−7、float64 3.0531e−16。

混合例の閾値.25・mean窓・center=Trueでは、位相微分はG1 −.2291332317、非clip厳密値−1.4320826982。
旧式の−.3580206745へ合わせない。center=Falseでは同例の全体位相スコアも−.2148124047になり得る。
これはclip平均と元平均の相違を残す既存規則で、G1では別のclip方式へ変更しない。
標本数3:1の純位相例はcentered推定−1.5/全体位相0であり、完全和の−2と混同しない。

## custom JVP・batch・device

主スカラーの零方向、斉次性、加法性、独立方向微分、パラメータ勾配を検証した。
batch複製1/2/4、chunk=0/1/2、JITは18条件、clip側も零方向・斉次性・独立スコア収縮を確認した。
隔離プロセスで2 CPU deviceを作り、実数/複素・等分布/偏った標本・clip=0/.25/100の24条件を実行。
既存のglobal中心化→各deviceでB除算→trainer pmeanが、全batchの独立共分散と一致した。
通常のProduction JAX/XLA設定は変更していない。

**K10は未修正**。aux tangentがauxそのものなので、零方向でもvariance tangent=1、方向を倍にしても1となる。
complex primalの通常呼出し/JVPのdtype差も残る。clip有効時にaux.clipped_energyが中心スカラーへ変わる既存の意味・shapeも維持した。
これを完全なcustom JVP契約のPASSとは扱わない。
一方、主出力だけのgradとhas_aux=Trueのgradは同じ正しい一階勾配であり、主tangentの独立検証を通過する。
aux微分・lossの一般的な高階微分の正しさは保証しない。K10は別段階で扱う。

## 既存33件とProduction固定snapshot

既存33件：**27 PASS / 2 XFAIL / 2 FAIL / 2 ERROR → 29 PASS / 1 XFAIL / 1 FAIL / 2 ERROR**。
改善はK04のXFAIL→PASSと、位相VMCのFAIL→PASSの2件だけ。新しい失敗はない。
実数の旧値1の保持試験とAdamの参照勾配は、厳密式E'(0)=2へ変更した。旧テストソースも保存している。

raw runnerではsnapshot未指定の1件がSKIP。別プロセスで同じメソッドを基準指定し実行してPASS（75.872秒）。
`final-results.json`はその証拠を併記して33メソッドで集計する。raw JSONのSKIPは消さない。
KFACの3 subtest ERRORを1メソッドと数えるため、raw ERROR4はメソッドERROR2になる。

残る問題：

| 状態 | 対象 | G1後も同じ症状 |
|---|---|---|
| XFAIL | K03、通常Hamiltonianの複素folx経路 | 虚部が落ちる。独立複素エネルギーとの差 .04264 |
| FAIL | Psiformer folx/通常微分の比較 | 最大絶対差1.78235943e−7 |
| ERROR | 定数log振幅の平面波folx | Arrayにlaplacian属性がない |
| ERROR（3 subtest） | float64 KFAC screen/lambda/HF | weighted_sum_of_objectsのabstract構造不一致 |

旧固定snapshotの全項目・tree・shape・dtype・実効設定を照合し、保持対象の値を厳密比較した。

| 精度 | 厳密一致の保持対象 | 旧勾配/更新の別扱い | 実際に値が変化した配列 |
|---|---:|---:|---:|
| float64 | 282配列 / 5158要素 | 620配列 | 253 |
| float32 | 282配列 / 5158要素 | 1037配列（KFAC含む） | 351 |

保持対象は5経路の初期params、入力、格子、位相/log振幅、ψ、軌道、行列式、座標勾配/Hessian、
既存局所エネルギー、MCMC、HF表、11種類のAttention内部量。
値の旧一致から外すのは明示したloss_gradient/Adam/学習更新後checkpoint/統計のグループだけ。
これらも項目・PyTree・shape・dtypeは照合し、変化した全項目名をcomparison.jsonへ記録した。
旧の全VMC/optimizer値へ一致した、という報告ではない。

修正後の独立物理基準は `g1_validation/physics-final.json`（228 records）、device基準はphysics-two-devices.json（24 records）。
新しい全経路の記録はsnapshot-float32-g1-v1/、snapshot-float64-g1-v1/。旧snapshot/旧研究checkpointは変更しない。
新基準は明示的な新規保存先へ生成し、自動更新・上書きしない。

## optimizer・checkpointと研究への影響

float32の全5経路でAdam/KFACの実trainer 1ステップ・保存・復元・有限性を確認。
float64ではAdam全5経路と、実行可能なPsiformer/SlaterNetのKFACを確認した。
二状態では正しい物理勾配を先に検証し、両trainerのAdam更新/stateを独立Optax更新と比較した。
KFACについては実行可能性の確認であり、位相を含む曲率近似の物理的最適性や長期収束を認定しない。

float32の全体位相勾配には最大8.9407e−8の丸め残差があり、Adamの初回更新では
厳密な零勾配を使う場合とパラメータが最大.00899398異なる例がある（float64では1.6653e−10）。
勾配の係数誤りとは別の零近傍感度であり、optimizerやepsilonを変更して隠さない。
テストは厳密スコアと実測スコアを別記録し、実測スコアからのoptimizer更新を検算する。

checkpointの形式・パラメータ構造は維持する。新規checkpointのroundtripと既存検証checkpointの評価を区別する。
旧optimizer stateには旧勾配のAdamモーメント/KFAC履歴がある。継承か初期化かを研究上の方針として明示し、
勝手にリセットしない。再開後の学習軌跡は従来と変わる。新runはG1の適用と実効clip設定を記録し別出力先に保存する。
研究用の過去checkpointからの長期再開、GPU/multi-host、収束・観測量への定量影響は未検証。

lambda/HFは同じlossを使う。Im(E_L)と位相スコアの相関がある場合や、clip中心の虚部が非零の場合に勾配が変わる。
clip=0の完全和で位相寄与が二重だった場合に修正が効くが、既定clip=5を含む全runが一律半分になるわけではない。
通常Hamiltonianのfolx虚部欠落は別問題で、loss修正だけでは復元できない。Hamiltonianの修正には進まない。

## 実行記録と中断

- `physics-final.txt/json`: G1と独立物理7件PASS。K10保持再現は問題の解消を意味しない。
- `physics-two-devices.txt/json`: 隔離2 CPU deviceの1件/24条件PASS。
- `legacy-reproduction.txt/json`: 旧lossの保存ソースに固定した8件PASS。旧不具合の数値も保持。
- `numerical-output.txt` / `numerical-results.json` / `final-results.json`: 上述の33件。
- `snapshot-float32.txt` / `snapshot-float64.txt`: 新規snapshot生成と旧保持対象の厳密比較。
- `analysis-static-final.txt`: 27件を再実行し、26 PASS / 1 XFAIL（既知K02）。400.535秒。
- `additional-tests-final.txt`: F0 12件、Embedding 3件、波動関数 3件、Window 6件、LayerNorm 4件、
  Screening 4件、B2内部量 5件、正式Diagnostic 7件の**44 PASS**。406.194秒。
  座標/パラメータ微分・Hessian・JIT、実行可能なfloat64 KFAC、保存済み検証checkpoint、
  独立JIT Diagnosticの36条件を含む。診断とProductionの差をゼロに変更した検証ではない。

最初の追加harnessは2 FAIL（physics-single.*）も保存している。
1件は新規方向微分のFD誤差2.74e−11へ誤ってmachine epsilonを要求していたため、
既存の独立FDと同じh=1e−5/2e−9基準へ合わせた。もう1件はAdamが零勾配の丸め誤差を増幅する上記現象。
物理勾配の検算と、実測勾配からのoptimizer配線検算を分け、残差と更新差を隠さず記録する。
既存テストの許容誤差を変更したものではない。
中断前のadditional-tests.txt/analysis-static.txtは完了として数えず、finalログを使用する。

以下は修正前の調査と過去段階の履歴。

---

# VMC勾配の独立検証：原因特定、Production未修正（2026-10-24・最新）

**大規模リファクタリングはF3で停止。F4は研究で必要なAnsatzが決まるまで保留。今回の作業は検証コードと文書のみ。**
追加は`tests/check_vmc_gradients.py`、数値記録はvmc_gradient_validation/。
loss/Hamiltonian/optimizer/既存テスト/固定snapshot/研究データは変更していない。
導出・コードとの対応・修正後の計画は[CONTRACTS.md](CONTRACTS.md)の末尾へ集約した。

## 正しい式と最初の不一致

実パラメータ、固定Hermitian H、ψ=exp(a+iφ)、O=∂logψについて
`dE=2 Re〈(E_L-E)conj(O)〉=2〈(Re E_L-E)∂a+Im E_L∂φ〉`。
共役と虚部の符号はplus、実数のlog振幅にも係数2が必要。
完全和の正規化期待値のAD/中心差分/独立NumPy共分散式が一致することを先に確認した。

| 経路 | 発生箇所と原因 |
|---|---|
| 複素、clip=0 | loss.py:251でclipped_energyはE_L配列。loss.py:288のdiff+clipped_energyが**2E_L−平均E_L**になる。term2はRe(O)だけを引くため、振幅項は正しく戻るが位相項の余分な分を引けない |
| 実数 | loss.py:301のmean(O*diff)に**係数2がない**。trainerはlog確率でなくlog振幅を渡している |
| 複素、clip>0 | clip_local_valuesの返す中心Cを同じclipped_energy欄へ格納するので、clipped_elはE_clippedになる。clip=0とは別式。中心化したclip scoreとの差は2 Im(C) mean(∂φ) |

clipping無効時、中心化した正しい標本規則との差は`2 mean(Im E_L * ∂φ)`。
完全和で平均虚部が0なら、振幅は一致・位相寄与だけ2倍になる。
NumPyの現行式の展開がProduction JVPを再現するため、JAXの複素微分規約やdevice係数が原因ではない。
今回の二状態は節がなく、E_L=Hψ/ψを直接計算し、folxも使わない。

## 数値結果

以下は全状態和を表す確定batchであり、MCの標本誤差ではない。表はfloat64、float32でも係数不一致を再現。

| 条件 | 厳密勾配 | 現行勾配 |
|---|---|---|
| H=diag(-1,1)、振幅alpha=0、複素分岐 | 2 | 2 |
| 同じ模型、実数分岐 | 2 | 1 |
| H=[[0,i],[-i,0]]、純位相theta=0、clip=0 | -2 | -4 |
| 同じ位相模型、clip=100（値は一切clipされない） | -2 | -2 |
| 同じ位相模型、テストで局所エネルギーの虚部を捨てる | -2 | 0 |
| 混合H、alpha=log(2)/2, theta=.31：振幅 | -0.5962500704077354 | -0.5962500704077351 |
| 同じ混合模型：位相 | -1.4320826981978145 | -2.8641653963956286 |

混合H=[[.4,.7+.6i],[.7-.6i,-.2]]、ψ_x=exp(alpha*x+i(theta*x+gamma))、x=(-1,+1)。
global phase gammaもパラメータに含める。全模型・全入力・O・E_L・各termと結果は
[single-cpu-final.json](vmc_gradient_validation/single-cpu-final.json)に保存（93 record）。

独立基準の最大絶対差：

| 比較 | float32 | float64 |
|---|---|---|
| 正規化完全和のJAX AD vs NumPy共分散 | 1.3851e-7 | 4.4409e-16 |
| NumPy中心差分 vs 共分散（差分h=1e-5） | 2.7861e-10 | 2.7861e-10 |
| テスト内の修正候補scoreのAD vs 厳密式 | 5.9605e-8 | 0 |
| 現行JVP vs 独立NumPyで展開した現行式 | 3.8152e-7 | 3.8858e-16 |

[metrics.json](vmc_gradient_validation/metrics.json)に実測値を保存。float32/64・差分打切り誤差と、
-4/-2や1/2の有限の係数差を区別する。新テストの丸め比較は32 machine eps、FDは2e-9。
既存テストの許容誤差を変更していない。純位相-4と複素振幅2は両dtypeで厳密な値としても確認した。

## clipping、平均、複素局所値の切分け

- 正の閾値の式は`2[mean((u_c-C_R)A)+mean(v_c P)]`、同じCによる中心化clip scoreとの差は`2 C_I mean(P)`。
  C_I=0なら一致する。clippingは一般に厳密エネルギー勾配とは異なる更新規則なので、clip結果を全て厳密勾配のFAILとは扱わない。
- 混合例clip=.25、mean窓、center_at_clip=TrueではC_I=-.10740620236。
  位相の現行値-.35802067455、中心化clip scoreは-.22913323171。
  global phase方向は現行-.21481240473、中心化規則は丸め範囲で0。
- 有限batchが状態数[3,1]（真のpは[.5,.5]）の位相例では、正しい中心化標本推定-1.5、現行clip=0は-3.5、厳密完全和は-2。
  標本推定と完全和の差を、そのまま実装バグとは判定しない。global phaseも非零虚部平均によるノイズと中心化規則を分ける。
- base_configの既定はclip=5、median=False、center_at_clip=True。**過去の全複素runが位相2倍だったとは言わない。**
  実効clip設定、虚部、位相依存パラメータを各runで確認する必要がある。
- 普通のpbc.hamiltonianのfolx複素枝は最後にreal(result)へ落とす（K03）。lambda独自経路は複素運動/反転項を保持する。
  このHamiltonian差とlossの係数問題は独立。虚部を捨てた対照の0は、情報欠落のablationであり別の正しいHの検算ではない。

## batch/deviceと補助JVP

複素混合模型のbatch複製1/2/4倍、chunk 0/1/2、JIT、混合方向JVPを比較した。
隔離した別プロセスだけで`XLA_FLAGS=--xla_force_host_platform_device_count=2`を指定し、
CPUのlogical device 2個で実際のpmap/pmeanを実行（両精度・実数/複素・均一/不均一状態数・clip 0/.25/100の24条件）。
trainerと同じlocal score→gradient pmeanは、全batch単一device値と**最大差0**で一致した。
2deviceの均一batchでも位相-4・実数振幅1が残るため、device数による2倍ではない。
Productionや親プロセスのコンパイル設定は変えていない。GPU/multi-host/KFAC内部collectiveは今回未検証。

別の契約問題K10も記録：零tangentのaux.variance tangentが1、方向を2倍にしても同じ1。
主loss tangentは零方向で0・方向2倍で2倍なので、K07の原因とは分ける。
通常complex primal lossはcomplex128、JVP内のprimalはfloat64という出力型の差もある。
補助値や高階微分の一般的正しさを主勾配の検証から保証しない。

## 実行結果と保護範囲

既存 .venv-pw CPU、Python3.13.14、JAX/jaxlib0.7.2、Optax0.2.6、KFAC0.0.7、NumPy2.3.3。
依存更新なし。新しいProduction Ansatz、Hamiltonian、optimizerは追加していない。

| 実行 | 結果と意味 |
|---|---|
| 新しい単一device試験 | **8 PASS / FAIL0 / ERROR0 / SKIP0**（7.235秒）。独立物理2件＋現行挙動/不具合再現6件 |
| 隔離した2 CPU device試験 | **1 PASS / FAIL0 / ERROR0 / SKIP0**（2.558秒）、24条件 |
| 既存VMCEstimator 7件 | **5 PASS / 1 XFAIL / 1 FAIL / ERROR0 / SKIP0**（0.728秒）。K04は既存XFAIL、K07は既存FAIL |

新しいCurrentBehaviorのPASSは不具合再現の成功で、Productionが物理的に正しくなったことを意味しない。
初回7件のsingle-cpu.jsonも履歴として保持し、追加のscore検証を含む最新版はsingle-cpu-final.jsonとする。
ログは[単一device](vmc_gradient_validation/single-cpu-final-output.txt)、[2device](vmc_gradient_validation/two-cpu-devices-output.txt)、
[既存VMC](vmc_gradient_validation/existing-vmc-output.txt)。同じ環境情報/実行時XLA_FLAGS/パラメータとHをJSONに記録した。

今回は既存33件全体、Production固定snapshot、Diagnostic、学習/optimizer更新、checkpoint復元は再実行していない。
既存VMCのone_adam_update_both_trainersは7件の中で実行したが、全モデルの学習検証ではない。
最新の全面検証はF3の27 PASS / 2 XFAIL / 2 FAIL / 2 ERRORであり、本調査で再実行した数字ではない。
既存ソース/テスト/数値基準の不変はscope-check.jsonへ記録する。研究結果の再解釈・過去runの定量評価は未実施。

## lambdaへの影響と最小修正案（未実施）

lambdaはcomplex wavefunctionを要求するため、実数枝K04は直接は適用されない。
trainableな位相スコアとIm E_Lに相関があり、clip=0ならK07の余分な位相項が作用する。
clip>0ではC_Iとcenterの規則、有限標本、実際のclippingを別に評価する。
NNの同じ実パラメータに振幅/位相が混ざるので、単一の学習率倍率やAdam/KFACでの相殺は仮定できない。
固定HFのみ/補正scale=0など位相の動かない方向や厳密固有状態ではこの差が見えない場合がある。
lambda=0を安全条件とはしない。実研究checkpoint/run設定を本調査では評価していないため、過去結果への定量影響は未判定。

最小案はloss.pyの主tangentの収縮を、既存diffから
`2 * real(vdot(psi_tangent, diff)) / device_batch_size`
とすること。実数でも係数2を使う。テスト内の凍結残差scoreのADで、この収縮が厳密勾配に一致することを検証した。
**本体は修正していない。** aux tangent/出力dtype/clip中心の意味の整理、K03の虚部保持は別契約として小段階で検討する。

修正後は完全和/FD/共役符号、振幅/位相/混合/global phase、clip境界/各center、JVP線形性、
JIT/chunk/batch、2deviceと実機GPU、Adam/KFAC/復元、lambda小系の方向微分・短期学習を確認する。
forwardの固定基準は厳密に保持し、意図的に変わる勾配/optimizerの基準は旧版を残して別versionで物理的に検算する。
旧基準を自動更新せず、既存のFAIL/XFAILや現行再現試験を黙って削除/緩和しない。
**原因調査までで停止。Production修正・F4実装へは進まない。**

---

# 段階F3：通常Wavefunction Constructionをwavefunction.pyへ移動（2026-10-24・過去の段階記録）

**通常のmake_orbitalsだけを移動。HF/Attention/Embedding/Hamiltonian/学習ループ/解析は変更しない。F4は実施しない。**
通常の幾何→trunk→軌道射影→Envelope/Jastrow→行列式入力という責務が1関数にまとまっていたため、
専用のwavefunction.pyへ置くことで研究者が変更場所を見つけやすくなると判断した。
新たなクラス、factory、設定機構、数値処理wrapperは作らない。

## 実際の変更と依存関係

| ファイル | 変更 |
|---|---|
| periodicwave/wavefunction.py（新規） | 通常make_orbitalsの定義を移動。著作権・由来を保持。既存の数値本体はそのまま |
| periodicwave/networks.py | 元の定義を除き、同じ関数オブジェクトを直接再export |
| tests/check_wavefunction_construction.py（新規） | 移動前factory、API接続、初期化/軌道/微分とfloat64 KFACの厳密比較3メソッド |
| tests/CONTRACTS.md、tests/README.md | 実装場所・数式と関数の対応・実行方法 |
| README.md、HANDOFF.md、tests/RESULTS.md | 到達点と実測・未検証範囲の更新 |

3 Ansatzは引き続きnetworks.make_orbitalsを呼ぶ。nets側にcall-through wrapperは置かない。
geometry/Network型はnetworksへ残し、wavefunctionは型をTYPE_CHECKINGで参照、幾何のimportをfactory呼出し時に行う。
モジュール初期化中の逆importを避け、networks先・wavefunction先の両順序を独立プロセスで検証した。
引数名・順序・既定値・戻り値は維持する。型注釈は新モジュールで遅延評価する。
型定義や周期幾何を別モジュールへ移す依存再設計は行わない。

## 波動関数を作る流れと独立維持

1. geometryでae/ee/r_ae/r_eeを構成し、従来のtrunkへ渡してh[N,E]を得る。
2. PRE_ORBITAL Envelopeが指定された場合はhへ適用する。
3. hの線形射影から軌道列を作り、complexなら偶数/奇数列をRe+i*Imへ合成する。
4. PRE_DETERMINANT Envelopeを適用し、reshape(N,-1,N)→transpose(1,0,2)で全電子N×N行列を作る。
5. 通常JastrowのJをexp(J/N)として全軌道要素へ掛け、[matrix]を返す。spin別ブロックには戻さない。
6. 既存3 Ansatzのnetwork_applyがnetwork_blocks.logdet_matmulを呼ぶ。
   slogdetと安定化した行列式の和から、realはsign、complexはangle、共にlog振幅を返す。

行列式reducerは既存network_blocksに残す。複数Headやlayer、Attention/MLP/Embeddingは変更しない。
通常Jastrowはjastrowsのmerged (N,0)の式。HFはlj_orbitalsの固定Bloch軌道＋spin別NN補正と
動的spin専用Jastrowであり、標準Envelopeを通さない。両factoryを無理に統合しない。
Envelope各種類、周期ノルム、Ewald、lambda Hamiltonianも独立に維持する。

## 数値検証

既存CPU .venv-pwを使用。Python3.13.14、JAX/jaxlib0.7.2、folx0.2.19、KFAC0.0.7、
Optax0.2.6、NumPy2.3.3。matmul precision=float32、CPU1 device。
依存更新・ProductionのJAX/XLA設定変更・snapshot更新・許容誤差緩和は行わない。
環境は[f3_validation/environment.json](f3_validation/environment.json)に記録した。

| 検証 | 結果 |
|---|---|
| F3の直接比較2＋F0の12 | **14 PASS**（50.823秒） |
| F3追加のfloat64 KFAC旧/new比較 | **1 PASS**（16.824秒）。Psiformer/SlaterNet各1ステップでparams/state/data/CSV厳密一致 |
| Production float64、5経路/11設定 | **902配列/21651要素、厳密一致**。指定snapshotテスト1 PASS（73.777秒） |
| Production float32、5経路/11設定 | **1319配列/31142要素、厳密一致**。全5経路のAdam/KFAC更新を含む |
| 既存33メソッド | **27 PASS / 2 XFAIL / 2 FAIL / 2 ERROR**。F1からstatus変更0 |
| Window/LayerNorm/B1/B2/正式Diagnostic | **26 PASS**（311.914秒）。独立Diagnosticの36条件・保存済み内部量と整合 |
| F1 Embedding＋主要3解析入口＋静的/再現性 | **29 PASS / 1 XFAIL**（378.298秒）。Embedding3・解析16・静的10 PASS、既存静的XFAIL1 |

追加3メソッドは以下を確認する（2件＋F0、KFAC1件を別プロセスで実行）。

- 公開再exportが同じ関数であること、呼出しsignature、両import順、数値本体のAST。
  新しい局所geometry import以外、元factoryの本体文は同一。
- 変更前factoryと同じseed=17で全PyTree/全初期化値を比較。
  両精度×非周期isotropic/PBC null×real/complex×Jastrowなし/ありの16条件。
  軌道、各det、位相/log振幅、eager/JIT、座標/パラメータ勾配、座標Hessianが**厳密一致**。
  Jastrowを使う条件では、同じパラメータの適用前の軌道も旧/newで厳密比較した。
  F0の独立した軌道packing/Jastrow/行列式/反対称性/節/HFの物理式検算も全件再実行した。

- float64固定snapshotはKFACを含まないため、実行可能なPsiformer/SlaterNetのKFACを旧factory差替え/新factoryで実学習する追加試験を行った。
  初期化を含む1ステップ全体のcheckpoint（params/optimizer state/data/width/step）とCSVが厳密一致。
  float64のscreen/lambda/HFの既知ERRORをこの対照試験で隠さない。
  [float64-kfac-old-new-output.txt](f3_validation/float64-kfac-old-new-output.txt)を参照。

[construction-interfaces-output.txt](f3_validation/construction-interfaces-output.txt)、
[frozen-output.txt](f3_validation/frozen-output.txt)、[float32-output.txt](f3_validation/float32-output.txt)を参照。
33件のrawは[numerical-results.json](f3_validation/numerical-results.json)。指定なしsnapshotのSKIPを消さず、
別プロセスでPASSしたsnapshotを反映したメソッド集計を[final-results.json](f3_validation/final-results.json)に保存した。
KFAC3 subtestのERRORは1メソッドに集約。runnerのexit 2は既知FAIL/ERRORによるもので全件成功ではない。
移動前ソースはnetworks-before.py.txt、差分はsource-diff.patchへ保存した。固定数値基準は更新しない。

## checkpoint、既知問題、未検証項目

checkpointは従来のPyTree/NPZ形式のまま。保存済みLayerNorm移動前の5経路×2精度の10個で、
hash、復元したparams、波動関数評価を確認した。Screen/lambda/HFのDiagnostic復元も別に検査した。
新規checkpointの保存/復元は既存33件で再実行した。実研究用過去checkpointとの互換性、
長期再開・関数オブジェクトを含む一般のpickle互換性を確認したとは主張しない。
HF入力は合成表。GPU/複数device/実HF表/長期統計比較/opt-inの長い旧検算3件は未実行。

I01は既存F0の実make_orbitalsでERRORを再現し、F2bの8 shape probeも同じ値/例外/誤broadcastを維持した。
[known-envelope-probes.json](f3_validation/known-envelope-probes.json)は**正常なモデル動作のPASSではない**。
この局所probeはCPU float64であり、全Envelope・全寸法を網羅する検証ではない。
電子一致の微分NaN、節の未定義、位相VMC(-4 vs -2)、Psiformer folx/Hessian差、
定数log振幅folx ERROR、float64 screen/lambda/HF KFAC ERROR、複素局所エネルギー虚部欠落、実VMC係数は全て未修正。
通常/Jastrow/HFの物理的に違う式を一律のSlater-Jastrowへ書き直していない。

## 次の研究変更の入口と停止位置

通常の軌道・Jastrow適用位置はwavefunction.make_orbitals、Slater行列式の安定化/和/位相はnetwork_blocks.logdet_matmul、
HF入力/NN補正/専用Jastrowはlj_orbitals。それぞれの役割を既存CONTRACTSへ記録した。
新Ansatzは同じNetwork契約へ接続し、形が違う物理モデルを通常factoryの多数の条件分岐へ詰め込まない。
F3は新規失敗なく完了し、F4は研究上必要なAnsatz接続が指定された時に検討できる。**今回F3のみで停止し、F4は未実装。**

実行コマンド（ログ/JSONの保存先は新しいものを指定）：

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m unittest tests.check_wavefunction_construction tests.check_interfaces -v
JAX_PLATFORMS=cpu PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.run_numerical NEW_RESULTS.json
JAX_PLATFORMS=cpu PW_BASELINE_DIR=tests/cpu_validation/snapshot-v1 .venv-pw/bin/python -m unittest tests.test_wavefunction.WavefunctionChecks.test_frozen_numerical_baseline -v
JAX_PLATFORMS=cpu PW_TEST_X64=0 PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.baseline compare tests/cpu_validation/snapshot-float32-v1 --exact
JAX_PLATFORMS=cpu .venv-pw/bin/python -m unittest tests.check_window tests.check_layernorm tests.check_screening tests.check_attention_aux tests.check_formal_diagnostic -v
JAX_PLATFORMS=cpu .venv-pw/bin/python -m unittest tests.check_embedding tests.check_attn_probe tests.check_head_contrib tests.check_head_cancel tests.test_reproducibility -v
```

---

# 段階F2b：周期幾何・距離・特徴量の責務境界調査（2026-10-24・過去の段階記録）

**F2bの設計調査だけ完了。本体/Production/Diagnostic/解析コードはF1から変更なし。F3は開始せず停止。**
関数別の利用表、入力shape、A〜Eの変更範囲、独立維持/共有の判断を
[CONTRACTS.md](CONTRACTS.md)のF2b欄へ追記した。新規geometryモジュール、クラス、設定機構、解析コードは追加していない。

## 主な結論

- 生変位ae=r_i-R_a、ee=r_j-r_iは周期ノルムのベクトルではない。geometryは座標全体をwrapせず、
  特徴はsin/cos、距離は用途別periodic_norm、Ewaldは内部の中心セル写像と各像のEuclid距離を使う。
  最小像距離は別の量で、対象10ファイルの距離へ代用しない。
- **現在の幾何r_eeはScreeningと通常/HF Jastrowに共有されている。** geometryのノルム変更はAttentionだけの変更にならない。
- **固定ValueはFeatureLayerのpairを使う。** 特徴のノルムやpair定義の変更は、Embedding以外のValue経路へも影響し得る。
  既定include_r_ae=Falseのoneはsin/cosなので、特徴用periodic_normを変えるだけではoneの式は変わらない。
- 標準軌道/標準JastrowとHF軌道/動的spin Jastrowは別実装のまま。lambda Hamiltonianを選ぶことと
  HF専用Jastrowを使うことは同義ではない。NN-only lambdaで通常Jastrowを有効にすればmerged spin仕様である。
- **ファイル移動は不要**。既存のEwald、Embedding、LayerNorm、Attention/Diagnosticの同目的の計算共有を維持する。
  CS固定ValueのFeatureLayer.apply 2回は将来の最小整理候補だが、XLAが既に重複除去している可能性があり性能改善は未測定。今回は触らない。

## 研究ごとの変更入口

| テーマ | 変更すべき入口と注意 |
|---|---|
| A. 周期Embedding特徴のみ | pbc.feature_layer.make_pbc_feature_layer.applyのone側、または既存make_feature_layer_fnで別FeatureLayerを選択。pairは保持し、固定Valueを巻き込まない。幅変更はembed PyTree/checkpointへ影響 |
| B. Screeningの距離依存のみ | 窓の式はattention.screening_window。距離定義はCS.make_screenformer_layersのd_eeの境界で局所設計する。networksのノルムを変えるとJastrowも変わる。d_ref/clip/固定Value距離との関係を別々に指定 |
| C. Jastrowのみ | 通常はjastrows.make_simple_ee_jastrow、HFはlj_orbitalsの専用Jastrow。距離定義を変えるならJastrow入力の境界も必要。通常の距離だけのAPIとHF動的spin APIを混同しない |
| D. 新Ansatz | 既存trunkを使い通常軌道構成/HF構成を参考に、Networkのinit/apply/orbitals/options契約へ接続。新network_type選択はtrainerの生成分岐の小さな接続が必要。計算ループの再実装は不要 |
| E. 新相互作用 | pbc.hamiltonianまたはlambda側local_energyのポテンシャル選択。既存make_local_energy_fnで別factoryも選べる。Ewald共有先、cusp/Jastrow/HF入力/単位は物理的な整合確認が必要 |

共通geometryオブジェクトに全てを入れたり、全利用者の距離を一斉に変更したりする設計は採用しない。
HFのBloch位相とEnvelopeの局在距離も別の責務。diagonal/full Envelopeはraw aeからEuclid距離を作るので、
shapeが通ることだけでPBC周期性を保証しない。

## 既知問題の確認と測定

1. **別電子の一致は原点cuspという特異条件**。F2aで保存した両精度/両格子のNaNを再確認し、
   発生源→距離入り特徴・Screening・Jastrow→波動関数微分→運動エネルギー/反転比の依存を整理した。
   全てのAnsatzに必ずNaNが出ると一般化しない。距離不使用の枝や、異なるspin状態で節でない場合を区別する。
   同一内部状態の一致はフェルミオンの節であり、Coulombの非対角一致も物理的に特異。
2. **I01は非特異配置でも起こる実装上のshape/軸の不整合**。PBC r_ae[N,A]に対して
   isotropic Envelopeは[N,A,1]を想定する。例外の有無だけでは判定できないことを8条件で追加確認した。

| N,A,O（Oは複素packing後の軌道列数） | 低レベルの実測 |
|---|---|
| 3,1,6 / 2,1,4 | 軌道への乗算でbroadcast ERROR |
| 3,2,6 | Envelope内の乗算でERROR |
| 2,1,2 / 1,1,2 / 2,2,2 | shapeは通るが期待する軌道別Envelopeと異なる値 |
| 3,1,1 | 軌道shapeが[N,1]から[N,N]へ誤拡張。通常N*K軌道の設定ではなく低レベルprobe |
| 1,1,1 | この特殊条件では最終値が対照と一致 |

測定はCPU float64、非一様sigma/pi、有限の距離を用いた。rank3の非周期shape対照は全8条件で期待式と厳密一致。
[envelope-shape-cases.json](f2b_validation/envelope-shape-cases.json)に入力生成式・shape・差・例外を記録した。
これはEnvelopeと軌道乗算の局所測定であり、全Ansatzの組合せを網羅して実行したという意味ではない。
標準make_orbitalsのN=3/K=2/complexの実経路は既存F0テストで別に再現した。
CP/CS/SNは同じ標準make_orbitalsを使うため構造上対象。HFは標準Envelopeを適用せず、このI01の経路を通らない。
主要PBCプリセットはnull EnvelopeとPBC FeatureLayerを明示している。Feature未指定時の非周期fallbackとPBCのrank不一致はさらに別の設定上の注意として記録した。
**修正は一切していない。**

## 実行結果と保存範囲

既存 .venv-pw CPU、Python3.13.14、JAX/jaxlib0.7.2。依存更新・システム変更なし。

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m unittest tests.check_interfaces -v
```

**12 PASS（15.360秒）**：[log](f2b_validation/interface-checks-output.txt)。Current 4 / Independent 7 / Known再現 1。
Known再現のPASSはPBC isotropic Envelopeの正常動作認定ではない。テストと許容誤差は変更していない。
ソースと保存済み検証資産はF2aのmanifestに対し**331ファイルでhash変更0**：
[scope-check.json](f2b_validation/scope-check.json)。本体/解析/固定snapshot/検証checkpointを含む。

**今回、既存33件・全Production snapshot・Diagnostic36条件・Adam/KFAC・checkpoint復元は再実行していない。**
本体は不変。最新の全面実行はF1の両精度5経路11設定の厳密一致、Diagnostic36条件、保存済み10 checkpoint、
既存33件 **27 PASS / 2 XFAIL / 2 FAIL / 2 ERROR**であり、F2bで再実行した結果と混同しない。
GPU・研究用過去checkpoint・実HF表は未検証。位相VMC係数を最優先とする既知FAIL/ERRORは未修正。

## F3への判断

**通常Wavefunction Constructionの小単位の整理へ進むための責務調査は整った。**
通常make_orbitalsに限定して幾何→trunk→射影/複素packing→Envelope→J/Nの境界を保持する計画を立てる。
HFとの統合、I01修正、距離変更、位相VMC/folx修正、学習ループ統合は含めない。
実装する際は初期化/params/checkpointと既存厳密比較を再実行する。**今回はF2bで停止し、F3は未実装。**

---

# 段階F2a：周期距離の比較と分離維持（2026-10-24・過去の段階記録）

**F2a調査完了。2つのperiodic_normは共通化せず、本体コードを変更しないで停止した。**
数式が同じでも、研究で独立に変えたい距離と特徴量を結合しない方針を優先する。
詳細な式・shape・微分/JITの契約は[CONTRACTS.mdのF2a欄](CONTRACTS.md#f2a周期ノルムは用途別の独立実装を維持2026-10-24)。

## 比較と判断

| 比較 | networks.pyのlocal periodic_norm | pbc/feature_layer.pyの公開periodic_norm |
|---|---|---|
| 数式・入力 | G=L.T@L、分率変位uからsqrt(aᵀGa+bᵀGb)/(2π) | 同一。a=1-cos(2πu)、b=sin(2πu) |
| kernel出力 | u[...,d] -> D[...]、長さ | 同一。kernelの実行文ASTも同一 |
| 座標変換 | (2π*inv(L))/2/π | inv(L) |
| 主用途 | 幾何入力、Screening Window距離、通常/HF Jastrow、Envelopeへの距離 | 1体・電子対特徴、学習可能Embedding入力、固定Valueの特徴 |
| 距離のshape | r_ae[N,A]、r_ee[N,N,1] | 内部r_ae[N,A]、r_ee[N,N]を特徴へ連結 |
| 対角保護 | 分率対角に全成分+1→ノルム→0 mask | 同じ手順。1体・非対角一致は保護しない |
| dtype/JIT | 格子からclosure構築、既存dtype/promotion。JIT設定変更なし | 同じ構築方式。入力r_ae/r_eeを再利用せず距離を計算 |

**kernelだけを共有することは技術上可能**であり、異なる数式だから共有できないという結論ではない。
しかし将来、Attention/Jastrow用距離と特徴量の表現を独立に変える際、共有kernelの変更は両側に及ぶ。
今回は研究上の変更境界を守る利益が重複削減を上回るため、現在の独立実装を維持する。
呼出し側まで統合したりinv(L)へ一本化したりする案も採用しない。演算順の違いを消さない。

周期ノルム、Ewaldの像への実空間距離、最小像距離は別物である。
Ewaldの中心セルへの剰余写像は斜交格子の最小像選択と同義ではなく、像の総和も維持する。
Screenformerのd_ref走査、Q/K正規化のepsilonなども今回のperiodic_normと混同しない。

## 今回実行した測定

既存CPU環境 .venv-pw、Python3.13.14、JAX/jaxlib0.7.2、matmul precision=float32。
依存・Productionコンパイル設定を変更していない。

| 対象 | 実測結果 |
|---|---|
| F0の幾何/周期ノルム解析微分/特徴・共変性・PBC | **3 PASS**、6.516秒。[log](f2a_validation/geometry-tests-output.txt) |
| kernel直接比較 | 両精度×正方/斜交格子×通常/原点近傍/原点の12条件。値・1階・2階・JITの両実装が一致。ただし原点の微分NaNは正しさのPASSではない |
| 公開経路の比較 | 両精度×2格子×通常/境界跨ぎ/別電子一致の12条件。通常と境界跨ぎは値・座標微分・Hessianが両経路で厳密一致、eager同士/JIT同士とも最大差0 |
| 原点/一致条件 | 値0でも微分が未定義。両経路で同じNaNを再現。新規回帰ではなく既存の特異条件として記録 |
| ソース・既存検証資産のhash | **331ファイル変更0**。本体、解析、Production/Diagnostic固定基準、保存済み検証checkpointを含む |

測定の各shape、dtype、有限成分の最大差/RMS差/相対差/スケール、非有限成分数、入力は
[comparison.json](f2a_validation/comparison.json)に保存。
原点を除くkernel比較も全て一致。原点ではgradient 2成分/Hessian 4成分が非有限で、正則化していない。
別電子が一致するN=3配置では、両精度・両格子とも距離の座標gradientは12/54成分、
行列Hessian（jacfwd(jacrev)）は216/324成分が非有限。そのうち対角出力に対応するHessianも72/108成分が非有限。
通常配置で確認した対角保護が、他の電子対のcuspによるADのNaN伝播まで防ぐとは主張しない。
特異点でのNaNをequal_nanで一致と記録することと、物理的に微分可能であることを明確に分ける。

同じ経路のeagerとJITの間には、既存の微分丸め差もあった（通常/境界配置の最大絶対差）。

| 精度 | 距離 | 座標gradient | 座標Hessian |
|---|---|---|---|
| float32 | 0 | 1.1920928955078125e-7 | 3.5762786865234375e-7 |
| float64 | 0 | 2.220446049250313e-16 | 8.881784197001252e-16 |

両距離経路で同じ測定値だった。これは変更前から存在するeager/JITの違いであり、
新実装との比較失敗やProduction固定基準の許容誤差変更として扱わない。

逆格子変換の演算順について、seed=41の128格子の**一括評価**で
(2π*inv(L))/2/πとinv(L)の差を別途測定した。
float32は58格子で最大2.9802322387695312e-8、float64は57格子で最大1.1102230246251565e-16。
各精度の最初の候補格子を使った公開距離経路の比較では、値/勾配/Hessianに差は観測しなかった。
これは今回の公開経路が不一致だったという記録ではなく、代数的な約分を一般にビット同値と扱えない証拠である。
入力格子と全測定値は[reciprocal-roundtrip.json](f2a_validation/reciprocal-roundtrip.json)。

測定用コマンドは`JAX_PLATFORMS=cpu .venv-pw/bin/python < tests/f2a_validation/measurement-source.txt`。
既存JSONを上書きしないようopen('x')を使う。再測定はoutを新規保存先へ変更して実行する。
元ソースのfactory内kernelのみを抽出して比較し、本体へのhookや新APIを追加しない。

## 実行していないものと次段階

**本体を変更しなかったため、全Production固定snapshot、Diagnostic36条件、Adam/KFAC、checkpoint復元、
既存33件はF2aでは再実行していない。** hash不変と、F1の最新実測を区別する。
F1での両精度・5経路・11設定の厳密一致、Diagnostic36条件、10検証checkpoint、
既存33件の27 PASS / 2 XFAIL / 2 FAIL / 2 ERRORをF2aで再実行した結果とは報告しない。
GPU・実研究HF/過去checkpointも未検証。既知FAIL/ERRORの修正・許容誤差変更はない。

今後、周期特徴はpbc/feature_layer.py、幾何として使う距離はnetworks.pyのfactoryを確認する。
後者の変更はAttentionだけでなくJastrow/Envelopeにも影響するので、独立変更には別途境界の検討が必要。
Ewaldはpbc/hamiltonian.pyで別に扱い、今回触らない。
**F2bは責務境界の調査へ進めるが、幾何の共通化を既定とする計画は見直す。今回F2bは開始しない。**
変更は既存文書とf2a_validation/の測定記録だけ。数値snapshot・研究データ・解析結果は保持した。

---

# 段階F1：学習可能Embeddingだけの共通化（2026-10-24・過去の段階記録）

**本体変更は4ファイルのEmbedding初期化/評価だけ。F2a以降には進まない。**
既存の未commit差分は保持し、F1着手直前との差分を
[f1_validation/source-diff.patch](f1_validation/source-diff.patch)へ記録した。
新しい本体ファイル、factory、クラス、設定機構は追加していない。

## 共通化した計算と残した責務

| ファイル/関数 | F1の変更 |
|---|---|
| network_blocks.init_input_embedding(key,F,E) | 既存init_linear_layerのbiasなしWだけを返す。W[F+1,E]、normal(key1)/sqrt(F+1) |
| network_blocks.apply_input_embedding(W,f,spins) | f[N,F]とspins[N]をconcatし、単一のjnp.dotでX[N,E]を返す |
| CustomPsiformer.make_psiformer_layers | init/applyの学習可能部分だけを上の2関数へ委譲 |
| CustomScreenformer.make_screenformer_layers | 同上。通常/Diagnosticが使う同じEmbedding。Diagnostic入口は変更なし |
| SlaterNet.make_SlaterNet_layers | 同上。後段のMLPは変更なし |

以前は初期化とconcat→dotを3箇所で記述していた。共通の式を各1箇所にし、各Ansatzは呼出しだけにした。
`params['layers']['embed']`は**裸の配列**、`['input']`は従来のdictのまま。
feature用split（未使用subkeyを含む）→embed用split→block用splitは全て各Ansatzに残す。
blockへ最後のkey側を渡す仕様も維持する。初期化の内部split、dtype、スケール、biasなしを変えない。
著作権・由来表記を保持し、移動元と共通関数の関係を記載した。

**共通化しなかったもの**：周期幾何、periodic_norm、固定1体/対特徴、FeatureLayerのinit/apply、
固定Valueの2回目の特徴評価とd_ref正規化、距離の取得、Attention/Transformer、軌道/HF/Jastrow/Hamiltonian。
これらは学習可能な射影と別の物理的責務であり、F1で式や演算順を変えない。
PBCとisotropic Envelopeのshape不整合も維持する。

## CPUで実行した検証

環境は既存`.venv-pw`のまま：Python3.13.14、JAX/jaxlib0.7.2、folx0.2.19、KFAC0.0.7、
Optax0.2.6、NumPy2.3.3。CPU1 device、matmul precision=float32。各テストがfloat32/float64を切り替える。
[environment.json](f1_validation/environment.json)にバージョン、実効環境変数、ソースhashを記録。
ライブラリやProductionのJAX/XLA設定は変更しない。

| 検証 | 結果 |
|---|---|
| F1追加3件＋F0の12件 | **15 PASS**（34.349秒）。[log](f1_validation/embedding-interfaces-output.txt) |
| 元の33メソッド | **27 PASS / 2 XFAIL / 2 FAIL / 2 ERROR**。F0からstatus変更0 |
| float64 Production：5経路・11設定 | **902配列/21651要素、厳密一致**。独立snapshotテスト1 PASS（73.256秒） |
| float32 Production：5経路・11設定 | **1319配列/31142要素、厳密一致**。全5経路Adam/KFACを含む |
| Window6＋LayerNorm4＋B1の4＋B2の5＋正式Diagnostic7 | **26 PASS**（295.236秒）。Diagnostic全36条件・各層内部量も保存基準と厳密一致 |
| 主要3解析入口16＋静的/再現性11 | **26 PASS / 1 XFAIL**（334.047秒）。E1/E2a/E2bすべてPASS、静的既知XFAILを維持 |

元33件の[raw JSON](f1_validation/numerical-results.json)は指定なしsnapshotのSKIPをそのまま保持する。
別プロセスで指定実行してPASSしたsnapshotを反映したメソッド集計は
[final-results.json](f1_validation/final-results.json)。複数KFAC subtestのERRORは1メソッドに集約する。
runnerのexit 2は既知FAIL/ERRORによるもので、全件成功とは報告しない。

新しい`check_embedding.py`の3件：

1. 初期化：両dtype×3種類のF/E、既存linear初期化との厳密比較。余分なdictを作らない。
2. 評価：両dtype×電子数2/3/4、異なるF/E。eager/JIT出力、特徴微分/Hessian、weight勾配を厳密比較。
   単体のJAXPRとlowered HLOテキストも一致（同一entry名）。
3. 3 Ansatz：変更前ソースのfactoryだけを読み、init後の**全trunk PyTree**を異なるseed=23で厳密比較。
   全blockの乱数割当も比較対象。実際の周期幾何を通したtrunk出力、JIT座標勾配/Hessian、パラメータ勾配が厳密一致。

変更前factory参照はimportやfolx登録を再実行しない。FeatureLayer/Attentionなど未変更の計算は実装を使う。
全面的な旧ネットワークの複製ではなく、F1抽出の直接比較であり、独立した固定snapshot比較も併用する。
F0の周期性・電子交換共変性・独立した特徴式のテストも全て再実行した。
XLA融合による新しい差は、検査したProduction出力・微分・更新では検出していない。
性能の再ベンチマークはしていない。Embeddingに新しいJIT境界や保持配列は追加していない。

## checkpoint、既知問題、停止位置

保存済み検証checkpointは、LayerNorm共通化前の5経路×2精度の10個について、
元のhash・PyTree・波動関数評価を検証する。Screen/lambda/HFの診断入口での復元も別に検査する。
新規保存/復元の形式とstep、Adam/KFACは既存テストと固定基準で確認する。
**研究用の過去checkpoint互換性、実HF研究データは未検証**。HF経路は合成表である。
GPU・複数device・長期学習/統計比較、opt-inの長い旧検算3スクリプトは今回実行していない。

既存の問題を修正していない：位相VMC係数(-4 vs -2) FAIL、Psiformer folx/Hessian差
(max abs 1.78235943e-7) FAIL、定数log振幅のfolx.laplacian ERROR、
float64のscreen/lambda/HF KFAC混合dtype ERROR。
複素局所エネルギーの虚部欠落・実数VMC係数は既存XFAIL、I01のPBC Envelope shapeは再現PASSであり正常経路の認定ではない。
旧D候補のON/OFFビット一致テストは撤回仕様なので採用せず、正式な独立Diagnosticを検証する。
ProductionとDiagnosticを同じJITへまとめず、診断結果を学習へ戻さない。

新しい学習可能Embeddingを試すときは上記network_blocksの2関数を見る。
Feature/PBCを変える場合は別の変更境界で扱う。F/Eやspin表現を変更すればshape/対称性/checkpointの新たな検証が必要。
同じ形の式変更でも物理モデルの変更であり、F1の単純移動とは区別する。
F1は新規失敗なく完了。F2aの小さな幾何抽出に着手可能と判断する。**今回F2aは実施しない。**

再実行コマンド（記録用の出力先は新規のものを指定する）：

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m unittest tests.check_embedding tests.check_interfaces -v
JAX_PLATFORMS=cpu PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.run_numerical NEW_RESULTS.json
JAX_PLATFORMS=cpu PW_BASELINE_DIR=tests/cpu_validation/snapshot-v1 .venv-pw/bin/python -m unittest tests.test_wavefunction.WavefunctionChecks.test_frozen_numerical_baseline -v
JAX_PLATFORMS=cpu PW_TEST_X64=0 PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.baseline compare tests/cpu_validation/snapshot-float32-v1 --exact
JAX_PLATFORMS=cpu .venv-pw/bin/python -m unittest tests.check_window tests.check_layernorm tests.check_screening tests.check_attention_aux tests.check_formal_diagnostic -v
JAX_PLATFORMS=cpu .venv-pw/bin/python -m unittest tests.check_attn_probe tests.check_head_contrib tests.check_head_cancel tests.test_reproducibility -v
```

---

# 段階F0：4領域の契約明文化・不足テスト・文書整合（2026-10-24・過去の段階記録）

**F0のみ完了。F1以降は実装していない。本体・Diagnostic・解析3入口を変更していない。**
新しい物理文書は増やさず、`tests/CONTRACTS.md`のF0欄へ式・入出力・単位・初期化/微分仕様を追記した。
README/HANDOFF/既知問題台帳を現在の到達点に合わせ、古い研究結果と段階記録は改変しない。

## 明文化したインターフェース

| 領域 | 数式と現行の接続 |
|---|---|
| 幾何/特徴 | 列格子L、ee_ij=r_j-r_i、分率座標、sin/cos、周期ノルム、自己項の微分保護。PBC/非PBC r_aeのshape差を明記 |
| Embedding | 1体特徴＋spinの連結、biasなし線形射影、layers.input/embed、未使用splitを含むRNG順。固定Valueの電子対特徴とd_ref正規化は別 |
| Wavefunction | 全電子N×N行列、複素packing、行列式の和、位相/sign・log振幅、Envelope/Jastrowの適用位置、HF固定軌道＋NN補正、擬スピン交換 |
| Hamiltonian | 通常2DEG/モアレ/lambdaの入口、複素運動項、Ewaldの背景/自己項、反転後波動関数比・微分、プリセット別単位 |

普通のJastrowはnspins_merged=(N,0)で全対parallel。実際の擬スピン対を使うHF専用Jastrowとは統合しない。
周期ノルムをEwaldの実空間変位へ代入しない。trunkの共変性、全波動関数の反対称性、格子周期性を区別する。
Hamiltonian変更に伴うHF/Jastrow cusp、内部状態変更に伴うMCMCの到達可能状態は物理的な依存として残す。

## 追加テストと区分

`tests/check_interfaces.py`を新規追加。既存のtest_*.pyと33件runnerは一切変更せず、別に明示実行する。

| 区分/メソッド | 検証 |
|---|---|
| Current: geometry_shapes_orientation_and_masked_derivatives | 実shape、eeの向き、r_ee対角の値/1階/2階微分0 |
| Current: embedding_rng_tree_and_evaluation_all_three_ansatz | 3 Ansatz×両dtype、RNG手順からweightを厳密再現、tree、実embedding入口、交換共変性 |
| Current: orbital_packing_merged_jastrow_and_envelope_position | 制御trunkの3電子/2det/複素軌道、exp(J/N)→exp(J)、ee_anti不使用、非周期isotropic Envelopeの対照 |
| Current: node_and_near_node_current_behavior | t→0の符号/位相/log振幅、1/tと-1/t²、全特異と非特異det相殺の節 |
| Independent: periodic_norm_analytic_first_second_derivatives | 直交格子の独立解析値/勾配/Hessian、JIT、幾何constructorとの比較、両dtype |
| Independent: features_values_covariance_and_pbc | NumPy参照、include_r_ae/rescale、1体/対特徴、3電子置換、個別格子移動、両dtype |
| Independent: determinant_sum_real_sign_and_complex_phase | NumPy行列式和、real sign/complex phaseとlog振幅 |
| Independent: hf_nn_correction_dynamic_spins_and_exchange | G=0合成HF＋非零NN補正、++/--/+-状態、専用Jastrow、位置と擬スピンの同時交換 |
| Independent: lambda_spin_flip_and_weighted_complex_kinetic | 2電子の解析的複素関数、重み付きラプラシアンと反転比/微分、default/folx、flip_batch=0/1、両spin配置 |
| Independent: ewald_independent_background_self_and_length_scaling | 1/2/3電子、math.erfc/NumPyの有限Ewald参照、Madelung・背景・自己項、長さ拡大で1/L |
| Independent: moire_value_derivatives_and_local_energy | 余弦和の値/勾配/Hessian、格子周期、平面波default運動項＋外場の全局所値 |
| Known: current_periodic_isotropic_envelope_shape_incompatibility | 既存PBC/非null Envelopeのbroadcastエラー再現。**正常な物理経路の認定ではない** |

直接両精度で新たに検査したのは幾何解析微分・特徴・Embedding。残るF0独立式はfloat64。
Production全5経路/11設定は既存両精度固定snapshotでも別に厳密比較した。
Embedding観察用にテスト内だけで後段blockを恒等に差し替えた。本体にhook/新APIを加えたものではない。

## 実行結果

CPU専用`.venv-pw`、Python3.13.14、JAX/jaxlib0.7.2、folx0.2.19、KFAC0.0.7、Optax0.2.6、NumPy2.3.3。
CPU1 device、matmul precision=float32。研究環境・ライブラリ・Productionのコンパイル設定は変更しない。

| 検証 | 実測結果 |
|---|---|
| F0新規12メソッド | **12 PASS**。Current 4 / Independent 7 / Known再現 1を別区分 |
| 既存Window/LayerNorm/B1/B2 | **19 PASS** |
| 正式Diagnosticの既存参照と独立API全36条件 | **2 PASS** |
| 静的/再現性11件 | **10 PASS / 1 XFAIL**、既存基準どおり |
| Production float64・5経路/11設定 | **902配列/21651要素で厳密一致** |
| Production float32・5経路/11設定 | **1319配列/31142要素で厳密一致**、全5経路Adam/KFACを含む |
| 既存33件 | **27 PASS / 2 XFAIL / 2 FAIL / 2 ERROR / 0 SKIP**、E2bからメソッドstatus変更なし |

raw runnerにはsnapshot指定なしの1 SKIPがあり、同じsnapshotメソッドを別プロセスで実行してPASSを確認した。
KFAC screen/lambda/HFの3 subtest ERRORは1メソッドERROR。raw ERROR4とメソッドERROR2を混同しない。
既存FAILは位相VMC -4/厳密-2とPsiformer folx/Hessian差。既存ERRORは定数logabsのfolxとfloat64 KFAC dtype。
XFAILは複素局所エネルギー経路差と実数VMC係数。どれも本体修正・閾値緩和・除外で消していない。
K01のTrim4/smooth7はE1以後の既存PASSであり、今回直したものではない。

主なログ：`f0_validation/interfaces-complete-output.txt`、`additional-output.txt`、`static-output.txt`、
`frozen-output.txt`、`float32-output.txt`、`numerical-output.txt`。
`final-results.json`に既存33件のメソッド別結果と差分、`interfaces-results.json`に追加12件の区分別結果を保存。
初回`interfaces-output.txt`はテスト実装側のtuple引数順/chex keyword constructor誤用でERROR。
テストだけを直し、`interfaces-final-output.txt`、その後の区分/共変性追加を含む`interfaces-complete-output.txt`は各12件完走。

## 新たに確認した既存不整合と未検証範囲

**I01：PBC r_ae [N,A]とisotropic PRE_DETERMINANT Envelopeの組合せ。**
N=3/K=2/複素の例では`ValueError: Incompatible shapes for broadcasting: shapes=[(3, 6), (3,)]`を再現。
非周期r_ae [N,A,1]では同Envelopeが独立式に一致する。主要なPBC null-envelope経路とは別である。
Known再現テストのPASSは、この問題の解消や当該組合せの物理的正しさを意味しない。
詳細は`inconsistency.json`。F0で既存問題を見つけたもので、今回のコード変更による新規回帰ではない。

節での非有限値、通常Jastrowのmerged分類、folxの虚部欠落等も現行仕様/未解決問題として明記した。
GPU/複数device/研究用過去checkpoint/実HF表は未検証。HFの新検算は合成G=0の2軌道で、実際の非零G多モードHF表や自己無撞着解を認定しない。
Ewaldは同じ有限打切りの独立式と長さスケールを検証し、無限和への全条件収束を保証したわけではない。
追加Envelope検証はnullと非周期isotropicの対照/PBCエラー再現まで。diagonal/fullやPRE_ORBITAL全組合せは未検証。
モアレ全局所値の新試験はdefault微分。既知の平面波folx ERRORは既存33件に残している。
主要3解析入口のコードと既存データは不変。F0ではE1/E2a/E2bの全解析行列を再実行しておらず、過去の実測と今回のDiagnostic再検証を区別する。

## 文書整合とVMC調査計画

- ROOT README・README_SCREEN・screening_analysis/README：原版が未変更という古い記述や導入時の一覧を履歴として明示し、現行契約へリンク。
- HANDOFF：B2止まり/古い25 PASS・4 XFAILを現在の到達点へ更新。D候補の撤回と独立JIT方式、E1/E2a/E2b、F0、未commit資産と停止条件を記載。
- tests/README：F0の明示実行、区分と精度範囲、既存snapshot優先、captureが新モデル用の別手順であることを明記。
- CONTRACTS：式/shape/単位/対称性/params/RNG/微分順、K01の到達状況、I01、最優先K07の具体的調査計画を追記。
- RESULTS：この記録を先頭に追加。過去の実測・固定snapshotは保持。

位相VMCは最優先の未解決問題。非clippingの目標式は
`2 Re <(E_L-<E_L>)*conj(partial log Psi)>`。
二状態完全和→複素E_L/Oの独立共分散→custom JVPの各項→clipping中心/閾値→虚部保持/実部化→chunk/batch/device/精度→短い独立学習→過去runの順で切り分ける。
複素振幅の対照2/2と純位相-4/-2を分け、学習率の一括調整で済ませない。
clippingありを非clippingの厳密期待値微分と混同せず、global phase勾配・中心化・共役・虚部を独立に確認する。
F0ではこの追加調査行列を実行・修正したとは主張しない。既存K07再現FAILはそのまま維持する。

SHA-256で本体48 Pythonファイル・解析113 Pythonファイル・既存テスト29 Pythonファイルの不変を確認。
既存テスト/測定/固定基準454ファイルも保持し、変更は7つの既存文書と新規テスト/記録だけ（`scope-check.json`）。

## F1への判断と停止

**主要な検証済み経路で、学習可能Embeddingだけを共通化するF1へ進める準備はできた。今回は実装せず停止。**
params['layers']['input'/'embed']、初期化順、空feature params、concat/dotの演算順を固定した直接試験が加わり、
将来の関数移動が何を保持すべきか明確になった。PBC距離やWavefunction構成まで一緒に動かさない。
I01や位相VMC等は別課題であり、全モデルの物理的正しさを認定した状態ではない。
4領域の変更場所・物理的依存・未対応条件を既存文書から追えるようにし、F1以降の境界判断を研究者が行える状態にした。

---

# 段階E2b：head_cancelを正式Diagnostic APIへ移行（2026-10-24・以前の記録）

**E2bのみ完了。アプリケーションコードの変更は `screening_analysis/head_cancel.py` だけ。**
head_contrib・attn_probe・Attention/Transformer/Diagnostic本体・Embedding・Wavefunction・Hamiltonian・学習系は変更しない。
新規テストは `tests/check_head_cancel.py`、記録は `tests/e2b_validation/`。従来の数値基準と研究データを更新していない。

## 旧・新の数式と物理的な意味

`c_h(i)=exp(out_scale)*z_ih W_h`。zはTanhNorm/Gate後のHead出力、W_hはW_Oの行ブロック。
倍率なしは1。共有射影biasは別項であり、Head間の相殺指標に含めない。
実数演算で `Attention_out_i = Σ_h c_h(i) + scaled_bias`。
実際のモデルの射影はbiasなしの配列。一般的なbiasの分離はE2aの射影検算と今回の相殺検算で確認するが、
bias付きcheckpointを本体で新たにサポートしたという意味ではない。

以下の**平均二乗の比**を、旧head_cancelから維持する。平均は配置と電子について取る。

- `S2=<Σ_h ||c_h||²>`、`N2=<||Σ_h c_h||²>`、`rho=N2/S2`。
- `c_h^p=c_h-c_h(self)`、`S2p=<Σ_h ||c_h^p||²>`、`N2p=<||Σ_h c_h^p||²>`、`rho_p=N2p/S2p`。
- 旧戻り値R `[L,4]` の列順は `[S2,N2,S2p,N2p]`。C/Cd/Cp/X等も従来の平均ノルムを維持。
- E2aのR=`<||Σc||>`とは異なる。この値は新結果の `mean_head_sum_norm` に別保存する。

`rho = 1 + 2*Σ_{h<k}<c_h・c_k>/S2`。S2>0なら理論上 `0<=rho<=H`。
rho<1は平均交差項が負（相殺）、rho>1は正（強め合い）。rho=1でも個々のHeadの直交を証明しない。
これは中心化した相関係数ではなく、平均ベクトルを含む射影出力の指標。
比の配置平均、平均ノルムの二乗の比、E2aのalignment_ratioとは別である。

selfは**同じ層入力Xを固定**して自己項だけを集約し、TanhNorm/Gate/射影を通した応答。
`F(a)-F(a_self)` と `F(a-a_self)` は非線形性により異なる。
したがって旧表示の「相手の分」は厳密な非対角項の独立応答・純粋な二体項ではない。この解釈上の問題は別に明記し、
c^pの数値定義を黙って後者へ変更していない。
Headそのものの除去、残差Xの追加、LayerNorm・MLP・後続層の応答、波動関数・局所エネルギーの変化は測らない。
相殺比をエネルギーへのHead寄与と解釈しない。

## 移行内容と共通計算

`head_cancel.analyze(...) -> head_contrib.analyze(...) -> 独立JIT(make_screenformer_diagnostic(...))`。
E2aが取得するHeadベクトル・自己応答ベクトル・実集約用aggregation_weightsをそのまま利用する。
自己応答のためのB2 kernel再評価もE2aと共有。解析側にQKV/正規化/Trim/Window/Value集約/Transformerの式を持たない。
Head分解の和を層更新に戻さず、各層は正式Diagnosticの出力を次の層へ渡す。
Diagnosticは同じソース・入力・paramsでの独立再評価であり、Production内部のビット列とは主張しない。

- `cancellation_statistics(c,c_self)`：実数 `[B,L,N,H,E]` から上記R・rho・rho_pと有限/正分母フラグを求めるだけ。
- `analyze(...)`：E2aの結果に相殺統計・定義・このファイルのソースSHA-256を追加。
- `compute(...)`：旧8要素tupleとR `[L,4]` を維持。settings/latticeはkeyword-only。
- `save(...)`：E2aの排他的NPZ保存を共有し、旧head_cancel/head_contrib名への保存を拒否。
- `main()`：複数folder対応を維持。import時に計算・保存・CLI処理を実行しない。循環importはない。

旧 `_post` 12行・`forward` 71行を含むAttention/Transformerのコピーを削除。
格子の曖昧なregex読出し、feature/Embedding/LayerNorm/MLP再構築、checkpoint読出しの重複も除去した。
ファイル全体は192行から125行。新しいクラス・hook・設定フレームワークはない。

## 未定義条件と不安定な比

分母0/非有限値ではrho/rho_pをNaN、`rho_defined`/`rho_p_defined`をFalseとする。epsilonで分母を正則化しない。
これらのdefinedフラグは**浮動小数点値として有限な分子と正の分母が得られたこと**で、物理的な安定性の保証ではない。
全ての非対角aggregation_weightsが0なら数学的にはc^p=0、rho_pは未定義。
`rho_p_self_only_warning [L]` を返し、CLIでも警告する。この十分条件以外の全ての悪条件を検出するものではない。
別JITの丸め残差が正の微小分母を作っても、生の比を丸め直したり、全体の許容誤差を拡大したりしない。

実測した全Head自己項のみの両精度ケースでは、c-c(self)=0、S2p=N2p=0、rho_p=NaN、warning=True。
今回この条件で非ゼロの丸め残差を観測したとは主張しない。
通常条件でも微小な自己応答差の比には注意し、保存した分子/分母とWeightを合わせて判断する。
単独Headや狭い窓ではrho_p≈1に退化する条件があるため、非自明な相殺の検証をこれだけで済ませていない。

## 旧解析との数値差と独立検算

CPU、両精度、N=2、2walker、seed7。**42条件+広い窓4条件=46条件**。
L=1/2/3、H=1/2/3、Trim²/⁴、cos/smooth7、窓なし、通常/固定Value、LNあり/なし、band/sin/Gaussian、FREEZE_W、
Gateなし・倍率なし・TanhNormなし・全てなし、diag-only・Value介入・CLIP・複合介入を含む。
Gate等のflag ablationは構造を整合させた合成checkpointを使い、未変更の本体で評価する。

各条件のHeadベクトル・共有bias・Weight・層入出力・Attention出力は正式APIへの直接呼出しと**厳密一致**。
旧解析は[legacy-head-cancel.txt](e2b_validation/legacy-head-cancel.txt)に固定し、同じprecision・params・入力・設定で比較した。
JSON中のnullは未定義を示す。異なるprecisionは初期化乱数配列も異なり、下表をprecision間の誤差比較には使わない。

L2/H2における旧→新rhoの最大絶対差：

| 条件 | float32 | float64 | 分類 |
|---|---:|---:|---|
| cos/Trim² | 1.19209e-7 | 2.22045e-16 | 旧eager/JAX縮約・Head和による次層更新と独立JIT/NumPy縮約の丸め・演算順 |
| Trim⁴ | 0.251350 | 0.0254349 | 旧解析は常にTrim² |
| smooth7 | 0.0101383 | 0.000699762 | 旧解析は常に余弦窓 |
| 固定Value | 0 | 2.22045e-16 | 旧head_cancelも対角除去済み。微小な縮約差のみ |
| Gateなし | 2.38419e-7 | 1.11022e-16 | 旧方式もこのflagは対応。丸め差 |
| 倍率なし | 1.19209e-7 | 2.22045e-16 | 旧方式もこのflagは対応。丸め差 |
| 複合Value/diag/CLIP介入 | 0.727634 | 0.205585 | 旧解析は介入を無視 |
| 全Head diag-only | 0.406964 | 0.00800242 | 同上。新rho_pは数学的に未定義 |
| LNなし / 窓なし | 旧IndexError / KeyError | 同左 | 旧の無条件LN / 表示用mask_omega参照。新方式は評価成功 |

rho_pの比較では双方で定義される層だけを比較し、未定義をPASSの差0に置き換えない。
R4列・rho/rho_p・Head和の最大絶対差、RMS差、相対RMS差、参照スケール、ゼロ近傍の扱いは
[legacy-comparison.json](e2b_validation/legacy-comparison.json)、[wide-window-comparison.json](e2b_validation/wide-window-comparison.json)に保存。
一覧：[legacy-comparison.md](e2b_validation/legacy-comparison.md)。

広い窓では全Headの非対角寄与を有効にし、例えばfloat32/L3/H3のrho_pは0.810534、0.406855、1.32604。
float64の対応する構成でも0.735590、0.432010、0.943496で、相殺と強め合いを独立に検算した。
固定Valueでも複数Headによる非自明なrho_p=rhoを確認した。

実APIのAttention出力と `sum(head_vectors)+bias` の差（46条件の各統計の最大値）：

| dtype | 最大絶対差 | RMS差 | 相対RMS差 |
|---|---:|---:|---:|
| float32 | 5.96046e-8 | 1.97371e-8 | 1.37369e-7 |
| float64 | 1.11022e-16 | 4.34377e-17 | 1.63172e-16 |

Head別縮約は本体の単一行列積と加算順が異なるため、この再構成はビット一致とはしない。
既存Production/Diagnosticの基準は緩めていない。
独立スカラー積とHeadペアの内積で4平均二乗を検算。直交・完全相殺・完全整列でrho=1/0/H、
比の平均との違い、共有biasを含めると別指標になること、分母0/非有限値もテストした。

## 保存・互換性・使い方

```python
from screening_analysis.head_cancel import analyze, save
r = analyze(run_folder, nconf=16, settings=known_settings, lattice=known_lattice)
print(r['rho'], r['rho_p'], r['R'])  # R [L,4]を必ず参照
save(r, 'NEW_head_cancel_diagnostic_v1.npz')
```

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python screening_analysis/head_cancel.py RUN_FOLDER \
  --settings KNOWN_SETTINGS.json --lattice '[[3.2,-1.6],[0.0,2.7712812921102037]]' --nconf 16
```

格子は例。そのrunで使用した元精度の格子を指定する。複数runには各runに合う設定を使用する。
`--checkpoint/--output`は単一folder時に指定できる。デフォルトは各folderの **head_cancel_diagnostic_v1.npz**。
以前はstdoutのみだったが、新CLIはこの新規ファイルにも保存する。既存ファイル・旧解析名への保存は拒否し、キャッシュは再利用しない。
再実行時は別outputを指定する。既存の図・cache・checkpointは読み書きで更新しない。

E2aの厳格な設定/格子/feature/NN構成とparamsの整合チェックをそのまま使う。
未保存のTRIM/Window/凍結/CLIP/JAX精度設定や元精度の格子を推測しない。
全deviceのparams replica一致、device-major walker選択、dtype/設定復元と例外時の復元も共有する。
softmax/PsiformerはこのScreenformer診断入口では未対応として明示エラー。

NPZはallow_pickle=False。符号付きHead/自己応答ベクトル、実Attention出力、共有bias、実集約Weight、
層入力/出力、R・比・defined/warning、入力座標/spin/atoms/charges/格子を保存する。
metadata_jsonに `method=head-cancel-diagnostic-v1`、親手法、4列の順序・量の定義・解釈上の制限、
checkpoint/config/settings/実行ソースのSHA-256、実効設定と指定元、walker index、dtype・backend等を残す。
importのみでは計算・ファイル保存・CLI出力が起きないことをsubprocessで確認した。

## Production・Diagnostic・既存テスト

独立CPU環境 `.venv-pw`、Python3.13.14、JAX/jaxlib0.7.2、folx0.2.19、KFAC0.0.7、Optax0.2.6、NumPy2.3.3。
CPU1 device、matmul_precision=float32、両dtypeを別々に実行。依存ライブラリと環境は変更しない。

| 検証 | 実測結果 |
|---|---|
| E2b追加5メソッド | **5 PASS**。旧新46条件、独立数式、未定義条件、保存/CLI/importを含む |
| E2a既存5メソッド | **5 PASS** |
| Window/LayerNorm/B1/B2 | **19 PASS** |
| 正式Diagnostic既存参照と実API全36条件 | **2 PASS**。既存D'内部配列との比較を含む |
| Production float64・5経路/11設定 | **厳密一致、902配列/21651要素** |
| Production float32・5経路/11設定 | **厳密一致、1319配列/31142要素**。全5経路Adam/KFACを含む |
| 既存33メソッド | **27 PASS / 2 XFAIL / 2 FAIL / 2 ERROR / 0 SKIP**、E2aからstatus変更なし |

snapshotには波動関数/位相・座標微分/Hessian・パラメータ微分・MCMC・Adam・checkpoint等を含む。
raw runnerのsnapshot 1 SKIPは別プロセスで同じsnapshotメソッドを指定実行してPASSを確認した。
KFAC screen/lambda/HFの3 subtest ERRORは1メソッドERROR。rawのERROR4とメソッドERROR2を区別する。
既存phase VMC勾配FAIL、Psiformer folx/Hessian FAIL、定数log振幅folx ERROR、float64 KFAC mixed dtype ERROR、
複素局所エネルギー/実VMC係数の2 XFAIL、block MCMC問題等は修正していない。
GPUと研究用過去checkpoint実データは今回未検証。解析比較は小規模合成checkpointで、実研究runを再解析した報告ではない。

記録：`head-output.txt`（3メソッド）、`extended-output.txt`（広い窓・警告・IOの3メソッド、IOは重複実行）、
`additional-output.txt`（E2aを含む26メソッド）、`frozen-output.txt`、`float32-output.txt`、`numerical-output.txt`。
`final-results.json`は既存33件のメソッド別結果、`self-only-diagnostics-final.json`は自己項のみの実測。
最初の`self-only-output.txt`はテスト2件PASS後に報告スクリプトのnested suite参照でStopIterationとなった記録。
報告スクリプトを修正し、`self-only-final-output.txt`と`extended-output.txt`は正常完走した。本体/数式の変更で対処していない。

SHA-256で本体48 Pythonファイル、他の解析116ファイル、既存テスト/基準/記録435ファイルの不変を確認。
既存ファイルの変更はhead_cancel.pyとRESULTS.mdだけ。新規ファイルも許可範囲内（`scope-check.json`）。

## 残るAttention解析の再実装（今回は変更なし）

| ファイル | 残っている計算・区別 |
|---|---|
| structure_probe.py | 第1層Embedding/QKV/Similarity/Trim²/cos Window |
| seed_spin.py | 初期化研究用のEmbedding/QK/Similarity |
| lj_maps.py | NumPyによるEmbedding/QK/Similarity。attn_dim=16の固定前提 |
| lj2/readout.py | 第1層QKV/Trim⁴/smooth7/集約/TanhNorm/Gate/Head射影。NE=36等の前提 |
| lj2/harmonics.py | Embedding/QK長さ正規化/Trim⁴。分母の保護も本体と別実装 |
| channel_ablation.py | probeのSからTrim²を再計算して介入係数を作る |
| lj2/pstype.py | 取得したSimilarityからTrim⁴を再計算 |
| psi_probe.py | Psiformerの手動QK/softmaxと層ループ。今回のScreenformer移行対象外 |
| CustomScreenformer_更新版.py | Attention/Transformerを含む別コピー。現用/保存用の扱いは別途確認が必要 |

Windowの曲線・幾何平均等を再計算する表示/検算コードも残る：
`band_check.py, fig_window_derivs.py, fig_ladder.py, fig_interp.py, fig_bins.py, fig_gauss_rings.py, fig_handover.py,`
`check_window_derivs.py, slide_framework.py, fig_window_types.py, fig_sharp.py, fig_dref.py, deck_intro.py,`
`check_ring_readout.py, fig_metric.py, fig_radial.py, deck2.py, density.py, fig_rankcap.py, fig_shell.py`。
独立な参照式・説明図・kernel研究は、学習済みforwardの再実装と同一視して機械的に削除すべきではない。
`check_ring_kernels.py`の別kernel、`qk_decomp.py`の係数行列分解もこの区別が必要。
`neighbour_map.py/bigN_maps.py`の解析上の行和規格化も、Productionのaggregation_weightsとは別量。

## 区切りの判断

**主要3入口（attn_probe / head_contrib / head_cancel）のScreenformer診断移行として、ここで区切れる。**
全Attention解析から再実装が消えたという意味ではない。残る各解析の目的・量の定義とデータを確認して別作業で移行判断する。
Embedding/Wavefunction/Hamiltonianの変更境界の整理、他の解析移行には着手しない。
E2bで停止し、既存結果を再生成・上書きしない。

---

# 段階E2a：head_contribを正式Diagnostic APIへ移行（2026-10-24・以前の記録）

**E2aのみ完了。アプリケーションコードの変更は `screening_analysis/head_contrib.py` だけ。**
本体48 Pythonファイル、Diagnostic API、attn_probe、head_cancel、他の解析、固定snapshot、
既存の図・キャッシュ・checkpointは変更していない。E2b・Embedding・Wavefunction・Hamiltonianの整理には進んでいない。

## Head寄与の定義と、評価しないもの

層番号l、電子i、Head h、Value次元D、残差ストリーム次元Eとする。
正式APIの `attention.gate_output` を `[N,H,D]` に戻したものをzとし、
射影行列W_O `[H*D,E]` をHeadの行ブロックW_h `[D,E]` に分ける。

`c_h(i) = s * z_ih @ W_h`、`s = exp(out_scale)`（倍率なしでは1）。

通常ValueではD=value_dim、固定ValueではD=pair_featuresの末尾次元。値を固定5等にしない。
**実数演算で Attention_out = Σ_h c_h + s*b。** biasは共有項で、Headに恣意的に分配しない。
現行本体のattn_output/checkpointはbiasなしの配列。`projected_heads`の独立検算ではbias付きの一般的な線形射影も確認したが、
本体がbias付きcheckpointを受け付けるよう拡張したという意味ではない。

| 旧キー（維持） | 数式・意味 |
|---|---|
| C [L,H] | `<||c_h||>`。walker・電子平均。波動関数そのものへの加法的寄与ではなく残差ストリームへの寄与 |
| Cd [L,H] | `<||c_h(self)||>`。**同じ層入力Xを固定**し、自己項だけを集約してTanhNorm/Gate/射影を通す反実仮想 |
| Cp [L,H] | `<||c_h - c_h(self)||>`。非線形な自己応答との差。純粋な二体項ではない |
| A / Ap [L,H] | `<||a_h||>` / `<||a_h-a_h(self)||>`。aはTanhNorm前のValue集約 |
| X [L] | `<||層入力X_i||>` |
| R [L] | `<||Σ_h c_h||>`。biasを含まない旧定義を維持 |
| q_C / q_Cp / q_Ap [L,H] | C² / Cp² / Ap²を**全層・全Head**で規格化。平均ノルムの二乗であり、平均二乗ではない |
| q_g_heuristic [L,H] | normalize(`[Cp*A/max(tanh(A),1e-12)]²`)。旧CLIの平均後の飽和補正を保持。正確な逆変換ではない |
| alignment_ratio [L] | `Σ_h C_h/R`。平均の比。各配置での比の平均ではない。直交時に√Hとなるのは等ノルムの場合 |

全量0等で規格化不能ならshare/比はNaNを明示する。
`compression_heuristic=tanh(A)/A`も平均後の旧指標。零近傍は未定義としてNaN。
旧CLIのCpの「2体」という表示、一般に直交なら√Hという説明、飽和を正確に割り戻すという解釈は訂正し、数値定義は上記として明示した。

旧解析はAttention式とTransformer更新を再実装し、`Σ_h c_h`を**次層更新にも使用**していた。
新解析ではHead和を層評価に一切戻さず、正式Diagnosticの各層入力・出力を使う。
TanhNormは非線形なので、`F(a)-F(a_self)`と`F(a-a_self)`は一般に異なる。
Headの局所的な線形射影分解、自己応答差、Head除去後に層列全体を再評価するablationは別物である。
このファイルはHead除去実験、波動関数差、局所エネルギー差を計算しない。c_hをエネルギー寄与と解釈しない。
相殺については従来のノルム比を維持し、head_cancelの解析へは進んでいない。

## 実装とDiagnostic APIの対応

- `analyze(...)`：厳格なcheckpoint/config復元後、`CS.make_screenformer_diagnostic(nspins, options)`を**独立JIT**し、各保存walkerを評価。
  全内部量を返す正式APIをそのままコンパイルし、その後にPython側で層のrecordを読む。解析用にTransformerループをコピーしていない。
- `projected_heads(...)`：APIのGate後の連結値から、射影行列のHeadブロックによる寄与だけを計算。
- Cd/Cpのための自己応答：同じ診断層入力・params・周期幾何を固定し、本体B2 kernelの既存 `_diag_only=1` を**別JITで評価**。
  QKV・Trim・Window・集約・TanhNorm・Gateの式を解析側に再実装しない。この反実仮想には追加のkernel実行コストがある。
  自己応答も独立した診断再評価で、元の内部ビット列とは主張しない。層の状態を更新しない。
  _diag_onlyの変更はparamsの一時的なコピーに限定し、ProductionのPyTree/checkpointには書き戻さない。
- `compute(...)`：外部consumer向けの旧8要素tupleの順序・shapeを維持。追加settings/latticeはkeyword-only。
- `save(...)` / `cached(...)`：新方式だけを新規保存。古いcacheを削除/再利用せず、新cacheも暗黙に再利用・上書きしない。

Attention Weightは正式APIの **aggregation_weights** `[B,L,H,N,N]`。
`weights_window`（Trim×Window）、`weights_intervened`（介入後・固定Value対角除去前）とも区別して保存する。
行和正規化は加えない。固定Valueではaggregation_weightsの対角が0となり、Cd=0、Cp=Cとなる。
通常ValueではAPIのv_effectiveに施された介入も尊重する。

削除した重複は、Q/K/V生成・長さ正規化・Similarity・固定Trim²・Window全分岐・本集約・TanhNorm/Gateの手動実装、
Embedding評価、LayerNorm/MLP/残差によるTransformer評価ループ。
解析側に残す数式はHead射影の分解、ノルム・平均・旧share指標、表示用の学習済み窓パラメータ読出しだけ。
E1とのcheckpoint/config復元I/Oの一部重複は残る。今回のファイル制限内で無理な共通基盤を新設しない。
他の解析に残るAttention/Transformerの再実装は今回の対象外。

## 実効設定・保存形式・使用例

E1の設定検証/格子読出し関数を再利用し、未保存グローバル8項目を現在の環境から推測しない。
必要な値：`TRIM_MAX,TRIM_POW,WIN_SHAPE,FREEZE_W,CLIP_R,CLIP_BETA,jax_enable_x64,jax_default_matmul_precision`。
configのscreening_attention_settingsまたはsettings dict/JSONが必要。矛盾はエラー。
文字列化格子は元精度のlattice指定が必要。feature factory/kwargs、NN寸法、LayerNorm、
Window/Gate/TanhNorm/固定Value/倍率フラグも明示されたconfigを要求し、params構造と照合する。
成功・例外時ともCSグローバルとJAX設定を復元する。同一プロセスでProductionのtraceと並行せず別解析プロセスを推奨。

全保存deviceのparams replicaの一致を確認し、walkerはdevice-major順に扱う。
これは最初のdeviceしか見なかった旧解析とのサンプル母集団の違いにもなる。
旧新数値比較では単一deviceの同一walkerを使って、この違いと式の差を混同しない。
`sel`は全保存walkerに対する整数index列。nconfはlinspaceによる決定的選択。

```python
from screening_analysis.head_contrib import analyze, save
r = analyze(run_folder, nconf=16, settings=known_settings, lattice=known_lattice)
# [B,L,N,H,E]：各Headの符号付きベクトル、Productionに戻さない
c = r['head_vectors']
actual = r['attention_output']       # [B,L,N,E]、APIの単一射影+倍率の結果
save(r, 'NEW_head_contrib_diagnostic_v1.npz')
```

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python screening_analysis/head_contrib.py RUN_FOLDER \
  --settings KNOWN_SETTINGS.json --lattice '[[3.2,-1.6],[0.0,2.7712812921102037]]' --nconf 16
```

格子の数値は例であり、そのrunで使用した元の値を指定する。
デフォルトは `head_contrib_diagnostic_v1.npz`。`--checkpoint` / `--output` も指定できる。
既存ファイルは排他的openで上書き禁止。旧head_contrib.npz/npyの名前には新結果を保存しない。
`cached`という旧関数名は残すが、条件不明なcacheの自動再利用は行わない。再実行は別outputを指定する。

NPZはallow_pickle=Falseで読める。旧C/Cd/Cp/X/R/w等に加え、符号付きHead/selfベクトル、
共有bias `[B,L,E]`、実際のattention_outputと解析上のdecomposition_sum、Value集約、TanhNorm/Gate出力、
層入力・出力 `[B,L,N,E]`、layer_index `[L]`、各種Weight、位置/spin/atoms/charges/格子を保存する。
wはd_ref単位。輪切りw_sigmaは旧head_contribと同じ**区間全幅**（E1 probeの半幅/遷移幅と混同しない）。
窓なしはw=NaNだがsoftmax=False。NaN窓幅をsoftmaxの判定に使わない。
metadata_jsonにmethod=`head-contrib-diagnostic-v1`、量の定義、入力checkpoint/config/設定ファイル/実行ソースのSHA-256、
実効設定と指定元、選択index、保存device数、dtype、JAX/jaxlib/NumPy、CPU/device/XLA情報を記録する。

旧softmax/Psiformer解析は正式Screenformer APIの対象外として明示エラーにする。
他の図/解析consumerは今回移行していない。未保存設定を明示できない呼出しもエラーとなり、既存図やcacheを自動更新しない。

## 旧解析との差：28条件・両精度

N=2、L=1/2/3、H=1/2/3、Q次元3、通常V次元2/固定pair次元5、2walker、seed7。
両精度×14条件：cosの3構成、Trim4、smooth7、窓なし、固定Value、LNなし2構成、band、sin輪切り、Gaussian輪切り、FREEZE_W、介入。
比較対象は保存した変更前ソース `e2a_validation/legacy-head-contrib.txt`。
全条件で新データの各層入力/出力・Gate/TanhNorm・Value集約・Weight・Attention出力が正式APIに**厳密一致**。
cos/L2の両精度では、既存D'固定params・入力に対応するWeightにも厳密一致した。

以下はL2/H2のCの最大絶対差（旧→新）。同じprecision内で同じparamsを使用。
precision間はJAX初期化dtypeで乱数配列も異なるため、この表を精度誤差の大小の比較には使わない。

| 条件 | float32 | float64 | 差の分類 |
|---|---:|---:|---|
| cos/Trim2 | 2.98023e-8 | 5.55112e-17 | 旧eager vmap/Head和での層更新と独立JIT等の丸め・縮約順 |
| Trim4 | 0.257517 | 0.114815 | 旧解析はTrim²、本体はTrim⁴ |
| smooth7 | 0.0329428 | 0.00136267 | 旧解析はcos、本体はsmooth7 |
| 固定Value | 0 | 3.46945e-18 | 旧head_contribも対角除去済み。旧attn_probeとの違いと混同しない |
| `_v_mode`/diag/CLIP介入 | 0.309274 | 0.157391 | 旧解析はこれらの介入を無視 |
| Windowなし | 旧KeyError | 旧KeyError | 旧表示用mask_omegaアクセス。新解析は評価成功 |
| LayerNormなし | 旧IndexError | 旧IndexError | 旧解析は無条件LN。新解析は評価成功 |

band/sin/Gaussian/FREEZE_Wも比較済み。全28条件のC/Cd/Cp/X/Rおよび分解和について、
最大絶対差、RMS差、相対RMS差、参照スケール、ゼロ近傍を除いた相対差を
[e2a_validation/legacy-comparison-complete.json](e2a_validation/legacy-comparison-complete.json)へ保存。
一覧は[legacy-comparison.md](e2a_validation/legacy-comparison.md)。
旧式の不一致へ新結果を合わせたり、Production/既存Diagnosticテストの許容誤差を変更していない。

## Head分解和の独立検算

Headの各成分の射影をNumPy float64のスカラー積で独立に計算し、dtype epsilonと縮約長から定めた前進誤差上界で比較。
bias・倍率の合成例、固定Valueの自己応答0、実際のWeight対角×v_effectiveによる自己項、
自己項のTanhNorm/Gate/射影を独立参照式で検算した。
TanhNormの非加法性も反例で確認。ON/OFFフラグ、全device選択、設定不足/矛盾、キャッシュ保護、CLIも検証した。

`Σ_h c_h+bias`と実API出力の差（14条件それぞれの統計の最大値）：

| dtype | 最大絶対差 | RMS差 | 相対RMS差 |
|---|---:|---:|---:|
| float32 | 5.96046e-8 | 1.87575e-8 | 1.03153e-7 |
| float64 | 1.11022e-16 | 3.11270e-17 | 1.63172e-16 |

異なる縮約順でありビット一致は要求しない。一方、APIから受け取る内部量そのものとProduction固定snapshotは厳密比較のまま。
比の分母が小さい成分での相対誤差のみを根拠に判定しない。局所エネルギー精度の保証とは別である。

## 実行結果と、保持した既知問題

独立CPU環境 `.venv-pw`、Python3.13.14、JAX/jaxlib0.7.2、folx0.2.19、KFAC0.0.7、Optax0.2.6、NumPy2.3.3。
CPU1 device、matmul_precision=float32、float32/float64をそれぞれ明示。既存環境・依存は変更していない。

| 検証 | 実測結果 |
|---|---|
| E2a追加5メソッド | **5 PASS**。旧新28条件と独立数式・保存保護を含む |
| 最終の設定不足/保存/互換入口の再確認 | 3 PASS（上記と重複するメソッドの再実行） |
| Window/LayerNorm/B1/B2 | **19 PASS** |
| 正式Diagnostic既存参照・実API全36条件 | **2 PASS**、4596内部配列の既存基準検査を含む |
| Production float64・5経路/11設定 | **厳密一致、902配列/21651要素** |
| Production float32・5経路/11設定 | **厳密一致、1319配列/31142要素**、全5経路Adam/KFACを含む |
| 既存33件・個別snapshot実行を合算 | **27 PASS / 2 XFAIL / 2 FAIL / 2 ERROR / 0 SKIP**。E1からstatus変更なし |

Productionの座標微分・Hessian・パラメータ勾配・MCMC・Adam・checkpoint/PyTree基準も維持。
既存数値runner単独ではsnapshot指定なしの1 SKIPを記録し、別プロセスで同じsnapshotメソッドを実行してPASSを確認した。
KFACはscreen/lambda/HFの3 subtest ERRORを**1メソッドERROR**にまとめる。raw runnerのERROR4とメソッドERROR2を混同しない。

保持した問題：phase VMC勾配-4対期待-2（FAIL）、Psiformer folx/Hessian差（FAIL）、
定数log振幅のfolx `.laplacian`欠落（ERROR）、float64 KFAC mixed dtype（ERRORメソッド）、
複素局所エネルギーの虚部欠落/実VMC係数（2 XFAIL）。block MCMC引数等も修正していない。
GPU、実研究runの未保存実効設定、研究用過去checkpointの実データ互換性は今回未検証。
workspaceには研究用runのnpzがなく、解析比較は小規模合成checkpointで行った。
有限なHead/Attention内部量だけで波動関数の節・局所エネルギーの安定性を保証しない。

ログ：`e2a_validation/head-complete-output.txt`、`guard-final-output.txt`、`additional-output.txt`、
`frozen-output.txt`、`float32-output.txt`、`numerical-output.txt`。
`final-results.json`はメソッド別結果とE1との差分、`verification-summary.json`は検証集約。
開発途中の`head-output.txt`は一時的な設定変数NameErrorの記録であり修正済み。最終判定には上記完走ログを使用する。
本体や物理の既知問題を修正したという意味ではない。
SHA-256で本体48 Pythonファイル、他の解析116ファイル、既存テスト/数値基準/記録417ファイルの不変を確認。
変更前から存在したファイルの変更はhead_contrib.pyとRESULTS.mdだけ（scope-check.json）。

## E2bへの判断

**Head寄与分解の共有基盤としてE2bへ進める。ただし今回は停止。**
head_cancel移行では、この符号付きc_hを使う相殺指標と、Head除去で層列を再評価するablationを分離して定義する必要がある。
旧cacheとの比較・明示設定・独立JIT・新しい保存名・Production厳密一致を維持する。
実研究データを解析する前には当時の設定/元精度格子を確定し、明示的に新出力を生成する。

---

# 段階E1：attn_probeを正式Diagnostic APIへ移行（2026-10-23・以前の記録）

**E1のみ完了。アプリケーションコードの変更は `screening_analysis/attn_probe.py` だけ。**
本体48 Pythonファイル、head_contribを含む他の解析、既存データ/図/キャッシュ、固定snapshot、D/D'記録は変更していない。
Embedding・Wavefunction Construction・Hamiltonian・学習の整理には進んでいない。

## 変更内容と共有境界

旧probeのfeature/Attention/Transformer手動評価ブロック56行を置き換えた。
Q/K生成・正規化・Similarity・Trim・Window・Value集約や層更新を解析側で再実装せず、
`CS.make_screenformer_diagnostic(nspins, options)` を独立JITし、保存walkerごとに評価する。
正式APIの全内部量を返した後、Python側で必要な配列を取り出すため、取り出し用JITの出力枝刈りで融合を変えない。
Transformerループのコピーはない。距離表示dも既存の周期幾何関数を使い、Windowには再計算した値を渡さない。

trunk用の既存ScreenformerOptionsを構築し、Embedding/特徴評価も正式APIへ任せる。
解析のためにWavefunction/Envelope/HF軌道/Hamiltonianを構築・ロードしない。
表示用のw/r等はcheckpointパラメータの読出しであり、Weightの計算には使用しない。
輪切りのw_sigmaは従来どおり区間の半幅で、Gaussian遷移幅の意味ではない。
既存のstructure_probeがimportする `_centres_from` は互換性のため保持した。

出力の定義：

| キー | shape・意味 |
|---|---|
| S | [B,L,H,N,N]、APIのsimilarity |
| Alpha | 同上、実際のTrim係数alpha |
| Window | 同上、実際のWindow。Windowなしではキー自体なし |
| W_window | 同上、alpha×Window。Windowなしではalpha |
| W_intervened | 同上、APIのweights（diag/clip介入後、固定Valueの対角除去前） |
| W | 同上、**aggregation_weights**。固定Valueでは対角0 |
| d | [B,N,N]、本体関数による周期ノルム |
| feature_rank_deficient / finite_features | [B]、独立した状態チェック。波動関数・局所エネルギーは未評価 |

Wは行和正規化せず、自己項も本体の仕様のまま。Window積を実集約用Weightと混同しない。
電子の位置・spinを同時に置換するとS/Wの両電子軸が共変する。QとKは異なる射影なので、S/Wの対称行列性は要求しない。
これらは同じソースからの診断再評価であり、Production内部のビット列ではない。

## 実効設定の復元と、復元できない情報

- 名前付き `system.pbc_lattice` を読み、JSON中で最初に見つけた任意の行列を使わない。
- 保存されたfeature factoryとkwargsを使用する。natoms=1/include_r_ae=False/pair_dim=5等を強制しない。
- 学習済みmask_centre/rings等はcheckpointから読み、初期化用設定の文字列化に対応する。paramsのinitは呼ばない。
- configの層数・LN構造・Value/Gate/Window/出力倍率・Q/K/head/MLP次元をcheckpointと照合し、矛盾したconfigで黙って別モデルを評価しない。
- 全保存deviceのwalkerをdevice-major順で使用する。params replicaが異なれば停止し、最初のreplicaへ無条件に落とさない。
- dtypeを保存し、float64 checkpointをx64無効で切り捨てない。nconfの選択indexも記録する。

**現行checkpoint/configは、当時のTRIM_POW/WIN_SHAPE等のグローバルやJAX精度を保存していない。これらは配列から一意に復元できない。**
現在の環境やfolder名から推測せず、次のflat dict/JSONを `settings=` / `--settings` で明示する。

`TRIM_MAX, TRIM_POW, WIN_SHAPE, FREEZE_W, CLIP_R, CLIP_BETA, jax_enable_x64, jax_default_matmul_precision`

config内に `screening_attention_settings` の記録があれば使用し、明示値と矛盾したら停止する。
これは新しい学習設定システムではなく、未保存だった実効値を解析呼出しに渡す入口。
当時の未保存値は利用者の確認・申告であり、checkpointから自動復元したとは主張しない。
`--trim-max` の明示変更はanalysis_overrideとして記録する。

同様に旧writerの `unserializable object: [[...]]` は印字丸めがあり、元の格子のビット列を保証できない。
その場合は既知の元精度の共通格子を `lattice=` / `--lattice` で指定する。省略記号付きの配列は復元しない。
独立したfeature格子を持つ旧文字列configは、共通格子指定で無理に置き換えず、元精度の構造化configを必要とする。
記録済み構造化配列はその値を使用する。

CSグローバルとJAX精度の一時設定は成功時・例外時に復元する。
同一プロセスでProductionのtraceと並行実行する方式は未対応。別プロセスでの解析を推奨する。
現在の研究用実データ/ログはこのworkspaceで見つからず、実研究runの未保存設定を確定・再解析したという報告ではない。

## 保存と使い方

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python screening_analysis/attn_probe.py RUN_FOLDER \
  --settings KNOWN_RUNTIME_SETTINGS.json \
  --lattice '[[3.2,-1.6],[0.0,2.7712812921102037]]' --nconf 16
```

上の格子は例。そのrunの元の格子・設定を確認して使用する。
ライブラリは `probe(folder, settings=known_settings, lattice=known_lattice)`。

デフォルト出力は **attn_probe_diagnostic_v1.npz**。旧名attn_probe.npzは使用せず、
既存ファイルを排他的openで上書き禁止とする。別出力は `--output` で明示する。
metadata_jsonに手法/Weight定義、設定と指定元、checkpoint/config/設定ファイル/実行ソースのSHA-256、
入力・parameter dtype、JAX/jaxlib/NumPy/device/XLA情報、保存t・保存device数・選択walker indexを含める。
positions/spins/atoms/charges/latticeも保存し、NPZはallow_pickle=Falseで読める。
summaryにも新Weight定義と特徴ランク警告を表示する。

他の解析呼出しや図の移行は未実施。旧probeを直接呼ぶスクリプトに未保存設定があれば明示エラーとなり、
暗黙に新しい定義や現在の環境で再解析しない。旧キャッシュ/図を自動更新しない。

## 数値差：旧式と新式を分ける

実際の保存形式で作った小規模合成checkpoint/configについて、両精度×12条件=24比較を実施。
新probeの各walker・全層のS/W/W_window/W_intervened/Alpha/Windowは正式APIに**厳密一致**。
cosの両精度ではD'保存入力・paramsも照合し、最初のwalkerの既存Weight基準へ直接厳密一致した。
異なるWeight定義の旧解析へ無理に一致させたり、比較許容誤差を広げたりしていない。

旧W→新Wの最大絶対差：

| 条件 | float32 | float64 | 主な理由 |
|---|---:|---:|---|
| cos/Trim2 | 1.19209e-7 | 3.33067e-16 | 主に旧eager vmapと独立JITの丸め・演算順 |
| Trim4 | 0.249732 | 0.249124 | 旧解析はu²、本体はu⁴ |
| smooth7 | 0.0344222 | 0.0135595 | 旧解析は余弦窓、本体は1-S7 |
| 固定Value | 0.871662 | 0.875855 | 旧Wは対角を含む。新Wの対角は厳密に0 |
| band | 1.78814e-7 | 5.55112e-17 | 意味は一致、丸め差 |
| sin / Gaussian輪切り | 3.72529e-9 / 1.19209e-7 | 2.63678e-16 / 1.11022e-16 | 意味は一致、演算順・丸め差 |
| diag/clip介入 | 0.437934 | 0.161335 | 旧Wが介入を反映していない |
| LNなし | 旧IndexError → 新PASS | 同左 | 旧解析は無条件にLNを参照 |
| Windowなし | 旧KeyError → 新PASS | 同左 | 旧表示処理がmask_omegaを要求 |
| include_r_ae=True | 旧shape例外 → 新PASS | 同左 | 旧解析はFalseを強制 |

float32の旧d_refはNumPy float64計算で、Nativeとの差が約1.63e-8あった。
新probeはNativeと同じdtype・周期幾何を使用する。Freeze_Wの値はcosケースと同じで、設定も保存・復元を確認した。
RMS差・相対差・参照スケール・S/d/d_refの差は [`e1_validation/legacy-comparison.md`](e1_validation/legacy-comparison.md) とJSONに保存した。
旧解析の6例外（3条件×両精度）は旧方式の再現結果で、新probeやProductionの失敗ではない。
4電子・幅4・初期LNのケースも特徴ランク警告を返し、Attentionは有限。波動関数や局所エネルギーの安定性はこのprobeでは認定しない。

## 検証結果

同じ独立CPU環境を使用。Python 3.13.14、JAX/jaxlib 0.7.2、folx 0.2.19、KFAC-JAX 0.0.7、
Optax 0.2.6、NumPy 2.3.3、Apple M5 CPU 1 device。依存・Productionのコンパイル設定は変更していない。

| 対象 | 最終結果 |
|---|---|
| Production float64、5経路・11設定 | 902配列/21651要素、厳密一致PASS |
| Production float32、5経路・11設定、Adam/KFAC/復元 | 1319配列/31142要素、厳密一致PASS |
| 既存33メソッド | **27 PASS / 2 XFAIL / 2 FAIL / 2 ERROR / 0 SKIP** |
| 既存Window/LN/checkpoint/B1/B2追加19件 | 19 PASS |
| 正式診断のD'全配列照合・単独API36条件再実行 | 2メソッドPASS（各36条件を照合） |
| E1追加 | **6メソッドPASS**。API/旧式24比較、設定欠落/矛盾、全device/間引き、上書き防止、CLI、特異ケース |

**33件の変更は解析用の2 XFAIL→PASSだけ**：Trim4とsmooth7は今回本体APIへ移行したので旧別式の不一致が解消した。
`test_known_attention.py` のfixtureを実config/保存形式・明示実効値へ合わせ、この2件のexpectedFailureを外した。
独立参照式と既存atol=1e-9は変更していない。旧不一致の再現は保存済み旧sourceを使うE1比較で保持した。
残る2 XFAIL（複素folx・VMC係数）、2 FAIL（位相VMC・Psiformer微分）、2 ERROR（plane-wave folx・倍精度KFAC）は従来どおり。
raw unittestのKFAC3 subtest ERRORは1メソッドに集約する。frozen1件の初回SKIPは別プロセスで実行しPASS確認済み。
集約と前回との差分は `e1_validation/final-results.json`。

初回検証では旧簡略fixtureの必須config/設定欠落でERRORになったが、上記fixture対応後に33件を全再実行して解消。
E1の設定矛盾ガードが先に発火したため一つのテストの期待エラー文も修正した。途中の失敗ログを最終PASSへ混ぜていない。
既存の物理的・数値的不具合の修正や既存基準の緩和は行っていない。

## ファイルと停止範囲

- アプリケーション：`screening_analysis/attn_probe.py` のみ。
- テスト：`tests/check_attn_probe.py` を追加、`tests/test_known_attention.py` を移行仕様へ対応。
- 文書/記録：`tests/README.md`, `tests/RESULTS.md`, `tests/e1_validation/`。

ファイル全体は191→325行。重複計算56行を除去した一方、設定欠落/矛盾の検出・精度保護・provenance・安全な保存のIO処理を追加したため、総行数は増えた。
数式は本体へ一本化し、解析の設定復元・保存と分離した。新しいクラス階層、registry、学習設定機構は導入していない。
`scope-check.json` で本体・他解析・既存データ・固定基準・D/D'記録のhash不変を確認した。
**E1で停止。head_contrib等の移行やEmbedding/Wavefunction/Hamiltonianの再設計は未実施。**

---

# 段階D：独立した多層Screenformer診断APIを正式実装（2026-10-23・履歴）

**段階Dまで完了。Productionは両精度の固定snapshotに厳密一致し、診断はD'の36条件に一致した。Eの解析移行には進んでいない。**
通常Network.apply、Network構造、初期化、PyTree、checkpoint、学習/VMC/MCMC/Hamiltonian、Psiformer/SlaterNetは今回変更していない。
本体で変更したのは `CustomScreenformer.py` だけで、attention.pyのB2計算はそのまま使用する。

## 正式APIと使用例

```python
import jax
from periodicwave import CustomScreenformer as CS

production = jax.jit(net.apply)
diagnostic = jax.jit(CS.make_screenformer_diagnostic(nspins, net.options))
args = (params, positions, spins, atoms, charges)  # 同じcheckpoint PyTree・入力
phase, logabs = production(*args)               # Productionだけで波動関数を評価
final_x, layers = diagnostic(*args)            # 別のコンパイル済み関数として呼ぶ
weights = layers[1]['attention']['aggregation_weights']
state = CS.screening_diagnostic_status(final_x) # 別評価。波動関数は未検査(None)
```

公開入口は**trunkの診断**で、戻り値は `(final_X, layers_tuple)`。波動関数やHF軌道を別実装で再構築するAPIではない。
引数は通常Network.applyと同じ5引数、paramsは全PyTreeをそのまま渡す。
`layers[i]` のindexが層番号で、下の配列を取得する。

| キー | 意味（境界shapeは[N,E]） |
|---|---|
| `input` | 層への入力X |
| `attention` | B2辞書：raw/正規化QKV、Similarity、alpha、Window、積Weight、実集約Weight、集約、TanhNorm/Gate、射影 |
| `attention_output` | 出力倍率まで適用し残差へ加える値 |
| `attention_residual` / `ln1_output` | 最初の残差加算後、最初のLN前/後 |
| `ln1_output` / `mlp_output` | MLPの入力/出力 |
| `mlp_residual` / `ln2_output` | MLP残差加算後、2回目のLN前/後 |
| `output` | 次層へ渡すX。ln2_outputと同じ値 |

LNなしではLN前後は恒等。固定Valueではraw V等はNone、pair_featuresを返す。
Window積 `weights_window` と介入後 `weights`、固定Valueの対角除去後 `aggregation_weights` を区別する。
詳細なshape・意味・batch利用の注意は [README.mdの正式API節](README.md#正式な多層screenformer診断api段階d) を参照。
通常と診断を同じJIT関数内で同時に評価しない。設定変更時は両側の構築・初回traceの設定を揃える。
グローバル設定の読取りは従来のtrace時点を維持し、設定システムを新設していない。

## 共有実装と変更ファイル

- `CustomScreenformer.make_screening_attention` のapply：B2の静的return_auxを通す。
- `make_self_attention_block` のapply：**一つの既存ループ**に静的な参照記録分岐を追加。
- `make_screenformer_layers` のapply：同じ埋め込み・固定Value特徴・層評価へreturn_auxを通す。
- 新規 `make_screenformer_diagnostic`：通常と同じ層factory・周期幾何を使用する小さい独立入口。
- 新規 `screening_diagnostic_status`：forwardとは別にSVDによる特徴/軌道ランク・有限性を調べる。

通常のNetwork.applyの関数本体、make_fermi_net、make_mlp、初期化、乱数splitは無変更。
取得OFFのASTは静的取得分岐を除くと変更前に一致（`formal_d_validation/production-ast-check.json`）。
通常時は辞書や内部量保持を作らず、同じ演算を実行する。head別射影の再合成も行わない。
正式APIにexec・パッチ適用・hook・registry・クラス階層はない。

検証側の変更：

| ファイル | 内容 |
|---|---|
| `check_formal_diagnostic.py` | 新規7メソッド：通常の旧ループとの厳密比較、異なるQKV/head、状態検査、checkpoint、36条件のwave比較と単独API比較 |
| `measure_diagnostic_layers.py` | `--formal-api`を追加。既存の数値指標・波動関数比較手順を再利用 |
| `diagnostic_layers.py` | D'の候補生成は保存済みB2ソースへ固定して過去試作を再現。正式API検証では公開入口と既存軌道処理を接続するだけ |
| `check_diagnostic_layers.py` | 過去試作のsource hash確認を保存済みソースへ対応。数値の許容誤差は無変更 |
| `check_layers.py` | 撤回済み旧D候補用3件は従来どおりSKIP。正式な分離APIを旧ON/OFF厳密一致仕様で判定しない |
| `README.md`, `RESULTS.md`, `formal_d_validation/` | API、再現方法、測定・回帰・スコープ記録 |

旧Dの3件をPASSへ変更したのではない。旧仕様の再現用に残し、新仕様は新しい7件で検証した。
旧D'の36条件・snapshot・checkpoint・研究データは更新していない。

## Productionの厳密検証と既存結果

同じ独立CPU環境：Python 3.13.14、JAX/jaxlib 0.7.2、folx 0.2.19、KFAC-JAX 0.0.7、
Optax 0.2.6、NumPy 2.3.3、Apple M5 CPU 1 device。ライブラリ・JAX/XLAコンパイル設定の変更なし。

| 対象 | 結果 |
|---|---|
| float64固定snapshot：5経路・11設定 | 902配列/21651要素、厳密一致PASS |
| float32固定snapshot：5経路・11設定、Adam/KFAC/状態/復元 | 1319配列/31142要素、厳密一致PASS |
| 既存33メソッド | **25 PASS / 4 XFAIL / 2 FAIL / 2 ERROR / 0 SKIP**、前回との差分なし |
| Window6・LayerNorm/JIT/checkpoint4・B1比較4・B2比較5 | 19 PASS |
| D'検証 | 4 PASS |
| 静的再現性 | 10 PASS / 1 XFAIL |
| 正式APIの新規検証 | **7メソッドPASS**（各ログの重複実行分は二重に数えない） |
| 旧D候補の廃止仕様 | 3 SKIP、成功扱いにしない |

全体実行でSKIPしたfrozen1件は別プロセスで再実行してPASS。raw unittestのKFAC ERROR3 subtestは1メソッドに集約する。
集約結果は `formal_d_validation/final-results.json`。既存のVMC位相勾配FAIL、Psiformer微分FAIL、
plane-wave folx ERROR、float64 KFACのmixed dtype ERRORを修正していない。
float32 Adam/KFACは既存全5経路で固定基準に一致する。

保存済み検証checkpointはProductionで従来の10件を再検証し、正式診断入口もscreen/lambda/HF・両精度の6件で復元評価した。
paramsは初期の検証PyTreeと厳密一致し、診断実行前後のProduction、aux破棄時の通常trunkとの一致も確認した。
実際の過去研究checkpoint・長期学習再開の互換性は、実データ未入手のため引き続き未検証。

## Diagnosticの数値判定

1/2/3層、LNあり/なし、N=2/4、通常/固定Value、Windowなし、cos/smooth7/Trim4/sin/Gaussianを含むD'36条件を再実行した。
さらにhead数1/3、Q次元2/5、V次元5/2の3層モデルも検証した。

**条件×精度×量ごとのD'保存配列を判定基準とし、このCPU環境では数値許容誤差を増やす必要がなかった。**

- 正式入口を使う診断wave評価：D'の全**4596配列が厳密一致**。入力・全初期params・各内部量・位相/log振幅・座標微分・Hessian・パラメータ微分を含む。
- 公開trunk APIを**単独でJIT**した場合も、36条件の全層内部配列がD'へ厳密一致。
- Production↔Diagnosticの最大絶対差・RMS差・相対差・参照スケールの指標も、条件ごとにD'と完全一致。
- 残差/LN/MLPの関係、次層入力と前層出力、対角除去、集約の独立式も確認。

波動関数の診断比較では、公開入口で得たXを既存make_orbitalsの依存注入点へ接続する検証用adapterを使った。
軌道/行列式/Transformerを複製せず、Production Network.applyは変更していない。
このadapterは通常Screenformer用の検証コードであり、HF波動関数全体を新規に再構築する正式APIではない。

非特異30条件のProduction↔Diagnosticの最大絶対差（D'と同じ）：

| 対象 | float32 | float64 |
|---|---:|---:|
| 位相 / log振幅 | 1.19209e-6 / 8.64267e-7 | 1.66533e-15 / 6.66134e-16 |
| 座標勾配の両成分最大 | 1.71661e-5 | 3.10862e-15 |
| Hessianの両成分最大 | 5.79834e-4 | 4.26326e-14 |
| パラメータ勾配の両成分最大 | 1.09673e-5 | 3.55271e-15 |
| 局所エネルギー実部 / 虚部 | 2.84643e-4 / 7.49432e-5 | 1.06581e-14 / 2.13163e-14 |

RMS・相対差・スケールは [`formal_d_validation/ERRORS.md`](formal_d_validation/ERRORS.md) と各JSONへ記録。
層途中のProduction内部値を直接観測したとは主張しない。prefix/Weight単独JIT参照の意味も維持している。
局所エネルギーへの影響は、同じ複素運動エネルギー恒等式を測定した位相/log振幅の微分へ適用して分離評価したもの。
Hessian項によるΔTと勾配項の寄与もD'から不変。既存folxの虚部欠落問題やVMC係数の問題とは別である。

## 特異条件は明示的に不適合とする

4電子・幅4・初期最終LN・null envelope・軌道biasなしではrank(X)≤3なので4×4行列式は数学的にゼロ。
該当6条件を削除せず、定量的なwave/log微分/局所エネルギーの比較対象としては不適合と明記する。
浮動小数点で再現した値がD'と一致しても、物理的な正しさ・安定性のPASSにはしない。

`screening_diagnostic_status` は dtypeのeps×行列サイズを閾値にSVDのrcondを調べる。
特徴のみなら `feature_rank_deficient` と `finite_features` を返し、波動関数はNone（未検査）。
`orbitals=...` を渡すと各軌道行列のrcond、singular_determinants、all_determinants_rank_deficientを返す。
`phase_logabs=...` の非有限値も調べ、wavefunction_unstable_or_undefinedとlocal_energy_unstable_or_undefinedへ警告を返す。
この2フラグのTrueは不安定/未定義のリスク、Falseは精度保証ではない。軌道入力のない有限性チェックだけでは軌道ランクを証明しない。

検証では36条件の特徴ランクが該当6条件だけを識別し、実際の既存軌道射影でも両精度・幅4/8の特異/非特異状態を識別した。
特異条件のAttention内部量は有限でD'と一致し、Value集約を内積の前進誤差上界でも独立に検算した。
全行列式間の打消しなどをこの検査ですべて検出できるとは保証しない。値の修正、epsilon追加、detachは行っていない。

## CPUコスト

他の検証ジョブ終了後に公開trunk APIを単独JITして測定した。warmup3回、9組×10回、各回同期。
4電子・幅8・LNあり・2headの例：

| 条件 | 時間中央値ms | 出力byte | 一時領域byte |
|---|---:|---:|---:|
| float32 1層 | 0.0352 | 3024 | 16 |
| float32 2層 | 0.0431 | 5912 | 16 |
| float64 1層 | 0.0356 | 5840 | 32 |
| float64 2層 | 0.0426 | 11416 | 32 |

生記録は `standalone-performance.json`。全36条件と、wave比較adapterの時間/コンパイル費用も保存した。
byteはXLA memory_analysisで、プロセスRSSや微分コンパイル全体のピークではない。
任意のSVD状態検査のコストはこのforward表に含めない。通常経路にはその検査も配列保持も追加しない。
通常の演算グラフは不変だが、大規模GPU学習の実時間・メモリは未測定。

## Eへの移行順序（今回は未実施）

1. checkpoint・nspins・feature/lattice・Window/Trim/介入の実効設定・dtypeを復元する読取り専用adapterを解析側に用意し、別JITの公開診断入口へ渡す。
2. 層番号とweights_window/aggregation_weights等の意味を解析出力へ明記し、最初は既知の一致条件（cos/Trim2）で旧解析と照合する。
3. Trim4・smooth7・余弦/輪切り等の旧解析の別式を順に置き換える。既知の差を修正された解析として区別し、過去出力を上書きしない。
4. 頭ごとの解析は取得したWeight/aggregateから独立に実施する。head別射影の加算で通常の最終出力を再構成しない。ablationはparamsコピーを使い、Productionへ介入しない。
5. 特異/非有限/未検査の状態と「独立診断再評価」の表記を図・保存データへ伝える。HF/λの波動関数全体の診断接続、大きいN・複数seed/座標・GPUは追加検証する。

CPUの今回の範囲ではEの小さい移行に進める。別環境や節近傍への精度保証は拡張していない。
**今回は正式API化で停止し、screening_analysisを変更していない。**

---

# 段階D'：Productionを保持した独立診断の試作・誤差評価（2026-10-23・履歴）

**D'の設計・最小試作・測定まで実施。本体Pythonファイルは一切変更していない。**
Productionの固定基準・許容誤差・コンパイル設定は維持した。
診断は同じソースから別JITで評価する値であり、Production実行内部のビット列ではない。
段階Dの公開API完成、Eの解析移行には進んでいない。

## 変更と設計

追加したコードは次の3ファイルだけ。既存テストの式・許容誤差は無変更。

| ファイル | 役割 |
|---|---|
| `tests/diagnostic_layers.py` | 現行ソースから診断factoryを生成し、既存軌道処理に接続する最小試作 |
| `tests/measure_diagnostic_layers.py` | 分離JITの値・微分・物理的影響・時間・メモリを測定。新規出力先だけを許可 |
| `tests/check_diagnostic_layers.py` | ソース／設定分離、検証checkpoint、36条件の記録・接続・ハッシュを検証する4メソッド |

このほか `tests/diagnostic_validation/` に測定記録、入力・全初期パラメータ・内部量NPZ、HLOを追加し、本記録を更新した。

Productionは既存Network.applyを呼ぶ。診断factoryは現行CustomScreenformer.pyを読み、
保存済み `layers_validation/candidate.patch` を**一時ファイルだけ**にfuzzなしで適用する。
変更対象の3 factoryだけを独立した名前空間で生成するため、folx再登録やProductionのモンキーパッチを行わない。
Attentionの式やTransformerループを手書きでコピーしていない。パッチの変更は参照取得分岐のみで、
生成した候補のSHA-256は保存済みD候補と一致する。現行ソース／差分に変更があれば適用を失敗させる。

これは**本体を触らず分離コンパイルを評価する試験用の生成方法**であり、恒久的なメタプログラミングAPIの提案ではない。
正式なDでは静的分岐付きの共有層評価を通常・診断の小さい入口から別コンパイルする設計を推奨する。
今回その本体変更は行っていない。

`make_diagnostic(net, nspins, data, namespace)` は `evaluate(params, positions)` を返す。
戻り値は `(stack([phase, logabs]), layers_tuple)`。Network構造は変更しない。
初期化はProductionのnet.initだけをseed=7で呼び、同じparamsを両側に渡す。
診断の最終Xを既存 `networks.make_orbitals` の依存注入点へ渡し、既存logdet_matmulを呼ぶ。
幾何特徴の準備は二度通るが、Attention／Transformerは一度だけである。
通常経路と診断経路を同じJIT関数で同時に評価しない。

各層tupleのindexがlayer index。辞書には input、B2のattention辞書、attention_output、
attention_residual、ln1_output、mlp_output、mlp_residual、ln2_output、output を保持する。
層境界は[N,E]、Attention内部shape・数式は既存B2と同じ（下のB2節参照）。
LNなしではLN出力は恒等。次層には同じループで計算したxを渡す。
診断の有無はPythonの静的分岐で、パラメータや乱数キーは追加しない。

設定は構築時の実効値を独立名前空間へ固定し、初回traceで同じ設定を使用する。
Window／Trimを変える測定では、未変更の通常factoryソースも別の設定名前空間で評価し、両側の設定を揃えた。
ProductionモジュールのグローバルやJITキャッシュには介入していない。
構築後の外部グローバル変更を自動追随するAPIではない。設定を変えたら両側を再構築する。

## 実際に実行したProduction回帰検証

環境は既存の独立CPU環境のまま：Python 3.13.14、JAX/jaxlib 0.7.2、folx 0.2.19、
KFAC-JAX 0.0.7、Optax 0.2.6、NumPy 2.3.3、Apple M5、CPU 1 device。
matmul precision=float32、x64はケースごとに切替。依存更新・XLAフラグ追加なし。
float64実行でも元実装が明示するfloat32パラメータはそのまま保持し、実効dtypeを各JSONへ記録した。

| 対象 | 結果 |
|---|---|
| float64固定snapshot、5経路・11 Attention設定 | 902配列・21651要素を厳密比較、PASS |
| float32固定snapshot、5経路・11設定、Adam/KFAC・状態・復元 | 1319配列・31142要素を厳密比較、PASS |
| 既存33メソッド | **25 PASS / 4 XFAIL / 2 FAIL / 2 ERROR / 0 SKIP**。前回からstatus変更なし |
| Window6、LayerNorm/JIT/checkpoint4、B1比較4、B2比較5 | 追加19メソッドすべてPASS |
| 新しいD'検証 | 4メソッドすべてPASS（測定記録の検査であり診断とProductionの厳密一致を意味しない） |
| 静的再現性検査 | 10 PASS / 1 XFAIL |
| 診断実行前後のProduction | 全36測定条件で同じコンパイル済みProduction出力が厳密一致 |
| 診断のauxを破棄して別JITした出力 | 全36条件でProductionと厳密一致 |
| eager値・残差/LN/MLPの関係、JITでの次層接続 | 全36条件で厳密一致 |
| 保存済みScreenformer検証checkpoint | 両精度で復元、分離診断も実行。今回の1層・幅8条件では出力差0。ファイルhash不変 |

33件のraw unittest出力はsubtestを個別に数えるためERRORが4件に見えるが、
plane-waveの1メソッドとKFACの1メソッド（3経路のsubtest）で計2 ERROR。
frozenメソッドは最初の全体実行でSKIPし、別プロセスで実際に再実行してPASSを確認した。
`diagnostic_validation/final-results.json` にメソッド単位の集約・前回との差分なしを保存。
既存のFAIL/ERRORは修正していない。既知のAttention解析、複素folx、VMC係数等のXFAILも維持。
KFACはfloat32全5経路を確認。float64 screen/lambda/HFのmixed dtype ERRORは従来どおり。
実際の過去研究checkpointは未入手で互換性未検証、確認したのは保存済み検証checkpointである。

## XLA融合の追加確認

保存済み62失敗subtest・diagnosis.json・HLOを再読した。
2層LNありの保存済みHLOは両精度とも、出力1配列のOFFでfusion instruction 38、
内部量も返すONで54、root出力も多数へ変化している（`prior-hlo-review.json`）。
以前の局所的なfusion無効化試験では差0だったが、今回その設定をProductionにも診断にも適用していない。

今回もaux破棄なら差0、同じソースのeagerなら差0、aux保持JITでのみ差が生じる。
これは中間配列の実体化による融合境界の変更と整合する。
新しい波動関数全体のoptimized HLOも1組ずつ両精度で保存した。
個別の機械語FMAまで差の起点を一意に特定したという意味ではない。

## 誤差分布と物理的影響

まず1層・2層を両精度で試作し、その後1/2/3層、LNあり・なし、電子数2/4、
cos/smooth7/Trim4/sin階段/Gaussian階段/Windowなし/固定Valueへ拡大した。
`matrix-v1` は30条件、`matrix-wide-v1` は幅8の4電子追加6条件。
いずれもseed=7、座標・spin・params・設定・source hash・dtype・NPZ hashを保存。
全36条件を完走したが、下の特異6条件を数値的に適合したものとしては扱わない。

非特異30条件（各精度15条件）の最大絶対誤差：

| 対象 | float32 | float64 |
|---|---:|---:|
| 位相 | 1.19209e-6 | 1.66533e-15 |
| log振幅 | 8.64267e-7 | 6.66134e-16 |
| 座標勾配（位相/log振幅の最大） | 1.71661e-5 | 3.10862e-15 |
| Hessian（両成分の最大） | 5.79834e-4 | 4.26326e-14 |
| パラメータ勾配（両成分の最大） | 1.09673e-5 | 3.55271e-15 |
| 層入力／出力（別prefix参照との最大） | 1.43051e-6 | 6.66134e-16 |
| 集約用Weight（別Weight JIT参照との最大） | 2.98023e-7 | 2.77556e-16 |
| 局所運動／全エネルギーの実部 | 2.84643e-4 | 1.06581e-14 |
| 局所運動／全エネルギーの虚部 | 7.49432e-5 | 2.13163e-14 |

最大絶対・RMS・相対RMS・参照RMSスケール、ケース別の成分相対誤差と除外数は
[`diagnostic_validation/ERRORS.md`](diagnostic_validation/ERRORS.md) と `error-summary.json` に記載。
相対成分の計算では |reference| < 1e-6×reference RMS を除外する。
相対誤差だけで判定しない。例えばfloat32エネルギー虚部は打消しにより相対RMSが最大約8.2e-4になる。
参照スケールと絶対誤差を併記した。
位相は返り値のラジアンをそのまま比較し、2πの枝補正は行っていない。行列式の節では位相自体が未定義となる。

局所エネルギーへの影響は、測定した g=(∇theta,∇a)、H=(H_theta,H_a)、a=log|psi| から

`T = -1/2 [trace H_a + |grad a|² - |grad theta|² + i(trace H_theta + 2 grad a·grad theta)]`

を両側とも同じNumPy float64縮約で計算し、同じ既存Ewaldポテンシャルを加えて分離評価した。
`ERRORS.md` にΔT_H=-1/2(Δtrace H_a+iΔtrace H_theta)と勾配項の寄与を全36ケースで記録。
既存folx本体の局所エネルギーは修正していない。この恒等式の測定値を、虚部を落とす既知のfolx経路の
出力そのものと混同しない。Productionの実際のfolx出力は固定snapshotの厳密比較で別途確認した。

### 4電子・幅4・最終LN初期状態は特異（診断の許容誤差で覆わない）

最終LNの初期scale=1、offset=0なので、幅4のXは行平均ゼロ、rank(X)≤3。
現行make_orbitalsはspinをまとめた4×4行列式を作り、今回はnull envelope・biasなしである。
したがってその行列式は数学的にゼロ。丸めで得た極小の行列式からlog微分を取ると極端に不安定になる。
これは試作の小さい幅と電子数の組合せに起因する条件で、本体へ導入した数式バグではない。

最小/最大特異値比はfloat32で約1e-8、float64で約1e-17以下（`nodal-stress-analysis.json`）。
位相差はラジアン単位、log振幅差はO(1)、Hessian差はfloat32で最大約2.3e7、float64で約3.9e18。
float64にするだけでは解決しない。これら6条件は**定量診断に不適合**として保持する。
幅8の4電子条件を追加して非特異な場合を別途測定し、元の結果を削除・置換していない。
既知のVMC/folx FAILとは別問題であり、Productionの改造や初期化変更で対処していない。

## 時間・メモリと暫定的な許容誤差案

Productionは診断試作をimportしない。全48本体Pythonファイルのhashが不変で、通常時の処理・配列保持は追加ゼロ。
診断のみ複数層のO(L H N²)の重みやO(L N E)の中間量を返す。
CPUの小規模・幅8・LNあり・4電子条件の一例（warmup3回、9組×10回、各回同期）：

| 条件 | Production / Diagnostic時間ms | 出力byte | 一時領域byte |
|---|---:|---:|---:|
| f32 1層 | 0.0265 / 0.0380 | 8 / 2904 | 1432 / 536 |
| f32 2層 | 0.0336 / 0.0470 | 8 / 5792 | 1752 / 536 |
| f64 1層 | 0.0274 / 0.0371 | 16 / 5600 | 2392 / 1048 |
| f64 2層 | 0.0339 / 0.0616 | 16 / 11176 | 2776 / 1048 |

別コンパイル費用もJSONへ保存（上記Production約0.13–0.18秒、Diagnostic約0.11–0.13秒）。
出力・一時領域はXLA memory_analysisの値で、プロセスRSS／コンパイラピークメモリではない。
alias・argument byteも生記録にある。派生微分コンパイル全体のピークメモリは未測定。
表は他の検証ジョブ終了後のwide追加測定を使用。最初のmatrixの一部・探索smokeの時間には並行テストの影響があるため使わない。
実規模GPU性能は未検証。この小さいCPU時間比から学習全体の性能比を推定しない。

**診断専用の提案**：同じCPU環境・非特異な測定条件に限り、次を再測定時の絶対誤差警告上限の候補とする。
実測最大から約1.2–2倍の余裕を取り、丸め差の大きさを追跡するための値である。

| 対象 | float32案 | float64案 |
|---|---:|---:|
| 波動関数の位相/log振幅、層入出力 | 2e-6 | 3e-15 |
| Attention Weight | 5e-7 | 1e-15 |
| 座標・パラメータ勾配 | 2e-5 | 5e-15 |
| Hessian | 7e-4 | 6e-14 |
| エネルギー実部 / 虚部 | 3.5e-4 / 1e-4 | 3e-14 / 3e-14 |

相対RMS・参照スケールも同時に監視し、ノード付近・悪条件の行列式にはこの表を適用しない。
これは一般的な物理的精度保証でも本体テストの新しい合否条件でもない。
実際に必要なエネルギー精度がこの差より厳しければfloat32診断は使えない。
定量的な微分解析はfloat64を優先し、学習・VMC・最適化・エネルギー採用値は引き続きProductionを使う。
既存テストや固定snapshotの許容誤差は一切緩めておらず、この提案値を自動テストへ導入していない。

## 再実行・制約・次の判断

新規ディレクトリを指定し、既存測定記録を上書きしない：

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -u -m tests.measure_diagnostic_layers /tmp/pw-dprime-new --smoke
JAX_PLATFORMS=cpu .venv-pw/bin/python -u -m tests.measure_diagnostic_layers /tmp/pw-dprime-matrix-new
JAX_PLATFORMS=cpu .venv-pw/bin/python -u -m tests.measure_diagnostic_layers /tmp/pw-dprime-wide-new --wide-electrons
JAX_PLATFORMS=cpu .venv-pw/bin/python -m unittest tests.check_diagnostic_layers -v
```

利用試作は `diagnostic_layers.make_diagnostic` が返す関数をProductionとは別にjax.jitし、
同じparamsとpositionsで別々に呼ぶ。dataのspin/atom/chargesは診断構築時に固定される。
診断結果をoptimizerやVMCへ入力しない。解析スクリプトへの接続は今回行っていない。

探索段階の `smoke` はEwald factoryの引数誤り、`smoke-v2` はattr設定をdataclassとして扱った試作コードの誤りで未完走。
試作側だけ修正し、`smoke-v3`／最終smoke／両matrixは完走した。未完走記録をPASSに数えていない。
基準・検証checkpoint・既存ファイルのhash比較は `scope-check.json`。RESULTS以外の既存ファイル変更なし。

**判断：分離方針を採用してDを完成させることを推奨する。ただしProduction内部の完全一致値を取得する機能とは位置づけない。**
今回のsource-derived patchは試作限定。正式API化時は共有ソースの静的な診断分岐と小さい入口へ整理し、
Productionの同じ厳密ゲートを再度満たすことを条件にする。
複数seed/座標・学習済み非特異checkpoint・大きいN・HF/λ診断接続・GPU、設定介入の扱いを追加検証する必要がある。
現在の5経路厳密回帰の成功は、それらすべての診断経路の誤差を測定したことを意味しない。
行列式の悪条件検出・エネルギー誤差予算・表示上の「診断再評価」表記を決めてからEを検討する。
**今回はD'で停止。解析コード、Hamiltonian/VMC/MCMC/学習、checkpoint形式、固定snapshotは変更していない。**

---

# 段階D：多層取得を試行したが厳密JIT一致に失敗、候補撤回（2026-10-23・履歴）

**段階Dは未完了。新しい共有ループ候補はJITの取得OFF/ONで厳密一致せず、本体を作業開始時のB2へ完全復元した。Eには進まない。**
従来B2までの成果・固定snapshotは保持。以下の候補APIは現行本体には提供していない。

## 試した変更と最終状態

候補では `CustomScreenformer.py` の以下だけを拡張した：

- `make_screening_attention` の返すapply：静的return_auxをB2 kernelへ渡す。
- `make_self_attention_block` の返すapply：同じ通常ループの各計算後に参照を記録。
- `make_screenformer_layers` の返すapply：同じ特徴生成・埋め込みから上記へ静的フラグを渡す。

別の本体Transformerループは作らず、次層は従来のxをそのまま使う。
初期化・乱数キー・パラメータPyTree・Network.apply・軌道接続・Window・attention.pyは変更しなかった。
通常OFFのASTは静的取得分岐と追加引数/docstringを除くと旧版と一致した
（`layers_validation/candidate-scope.json`）。

**最終的には上記の本体変更を全て撤回**し、保存した `before-source.txt` から作業開始時のファイルを復元した。
HEADへcheckoutしたのではなく、LayerNorm/A/B1/B2の未commit差分を含むB2のバイト列へ戻している。
候補差分は `tests/layers_validation/candidate.patch` に保存し、現行B2上で `git apply --check` が通ることだけを確認。
本体へ再適用していない。候補source hashは `candidate.sha256`。

今回残したもの：

| ファイル | 役割 |
|---|---|
| `tests/check_layers.py` | 候補の3検証。現行B2ではAPIがないため明示SKIP。成功とは数えない |
| `tests/layers_reference.py` | 試行前の層factoryの固定参照。検証専用で本体からは使わない |
| `tests/diagnose_layers.py` | 候補適用時のeager/JIT/微分・融合の切り分け。候補なしでは明示エラーで停止 |
| `tests/layers_validation/` | 候補、失敗記録、HLO、診断結果、復元確認 |
| `tests/RESULTS.md` | 今回の未完了と根拠を記録 |

## 候補のインターフェースと取得情報（撤回済み）

候補は既存のブロックapply／層applyへkeyword-only `return_aux=False` を追加し、
Trueなら `(final_X, layers)` を返した。layersは層順のtuple、`layers[i]` がi番目の層。
各層に以下の値を保持した（層境界の配列は[N,E]）：

- `input`：その層が実際に受け取ったx。
- `attention`：B2辞書をそのまま格納。Q/K/V、正規化、Similarity、Trim、Window、実集約重み、集約、TanhNorm/Gate等。
- `attention_output`：残差へ実際に加えたAttention出力。
- `attention_residual`：x + attention_output直後。
- `ln1_output`：1回目LN後。LNなしならattention_residualそのもの。
- `mlp_output`：上記を入力とする既存MLPの出力。
- `mlp_residual`：ln1_output + mlp_output直後。
- `ln2_output` / `output`：2回目LN後と次層入力。LNなしならmlp_residual。

通常OFFでは辞書・リストを作らない。ONでも式の再計算やhead別射影による再構成はしていない。
しかし、計算グラフから途中配列を返すこと自体がJIT最適化へ影響し、下記の不一致が生じた。

## 成功した確認と、失敗した確認を分ける

環境は同じ専用CPU環境：Python 3.13.14、JAX/jaxlib 0.7.2、folx 0.2.19、KFAC-JAX 0.0.7、
Optax 0.2.6、NumPy 2.3.3、Apple M5、CPU 1 device、matmul precision=float32。環境・依存更新なし。

| 対象 | 候補の結果 |
|---|---|
| 通常OFFの両精度・5経路・11設定 | 既存の固定snapshot全項目に厳密一致 |
| 通常OFFのfloat32 Adam/KFAC更新・状態・復元 | 固定基準に厳密一致 |
| 既存33メソッド | 25 PASS / 4 XFAIL / 2 FAIL / 2 ERROR / 0 SKIP。全メソッドのstatus不変 |
| 既存追加19件 | Window6 / LayerNorm・JIT・checkpoint4 / B1比較4 / B2比較5、全PASS |
| 多層eager | 試行した1/2/3層・LN有無・各設定で通常と取得ONが厳密一致。通常forwardの各層入力・Attention出力をテスト内で記録して照合した段階も通過 |
| 多層JITで途中配列を保持 | **不一致。候補3メソッドともFAIL、計62 subtest失敗** |
| 本体trunkとのJIT比較 | Screen/lambda/HF・両精度の6 subtestで不一致。後続の波動関数assertへは到達していない |
| 取得ONの多層Adam/KFAC・長期再開 | 数値ゲート失敗のため未実行。成功とは扱わない |
| 多層取得のメモリ・時間 | 数値ゲート失敗のためベンチマーク未実施 |

33件の生記録は `numerical-results.json`（frozenのみ初回SKIP）。
固定x64基準に合わせ、PW_RUN_KFACを設定しない別プロセスでfrozenを実際に再実行しPASS。
両記録を統合した `final-results.json` をB2時点の結果と比較した。
runner終了コード2は従来FAIL/ERRORを含むためであり、全件正常ではない。
KFACの3つのsubtest ERRORは1メソッドのERRORとして集計している。
候補OFFの結果を、失敗したONの結果の代用にはしていない。

追加テストの範囲は `layers-output.txt`：

- 1/2/3層、LNあり／なし、Q/K次元3・Value次元2・固定対特徴次元5。
- cos/smooth7/Trim⁴/band/sin/Gaussian、固定Value、Windowなし、Gateなし、倍率なし、TanhNormなし、凍結、介入。
- 3層の候補を含む実際のScreen/lambda/HF trunk接続。
- 通常とONで各層の入力・出力を照合し、JIT・座標微分・Hessian・params勾配を検証。
- 最初の失敗assertより後の検査は未到達。全項目を検証完了したとは報告しない。

新規FAILの内訳：層forward比較40、微分比較16、実trunk比較6 subtest。
これは既存の物理検証FAIL/ERRORとは別の、候補で新たに発生した失敗。
XFAILへ変更せず、許容誤差も緩めていない。`stage-d-result.json` に明示した。

## 原因の切り分け

`diagnosis.json` に1/2/3層・LN有無・両精度の10例を保存した。
ブロックの入力は既存周期特徴／周期距離を座標から作り、微分用scalarは `sum(sin(final_X))`。
以下の微分差はこの検査用scalarに対するもの。エネルギー差や波動関数Hessianの差ではない。

全10例で：

- 旧ブロックJITと候補通常OFFは厳密一致。
- 旧ブロックと通常OFFのJAXPR文字列が一致。
- eagerのOFF/ONは厳密一致。
- ONを呼んでもauxを捨てるJITは、通常OFFと厳密一致し、JAXPRも同じ。
- **途中配列をJITの出力として保持したときだけ差が発生**。

代表例（N=2、H=2、E=4、Q/K=3、Value=2、3層・LNあり）：

| 最大絶対差 | float32 | float64 |
|---|---:|---:|
| 最終trunk出力 OFF対ON | 6.25848770e-7 | 2.22044605e-16 |
| params勾配 | 4.76837158e-6 | 4.44089210e-15 |
| 座標勾配 | 1.33514404e-5 | 7.10542736e-15 |
| 座標Hessian | 5.49316406e-4 | 1.36424205e-12 |

最適化済みHLO：`hlo-False-off.txt` / `hlo-False-on.txt`（float32）、同True版（float64）。
2層・LNありの診断だけで、両モードのfusion passを局所的に無効化すると、両精度とも出力差が0になった。
これはXLAの演算融合変更に伴う丸め差と、その微分への伝播に整合する。
このcompiler optionは診断呼出にだけ使い、本体や環境変数へ適用していない。

「同じPython演算順・同じ共有ループ」と「途中配列を追加返却したJITの厳密一致」は別の条件であり、
今回の実装は後者を満たさなかった。通常forwardへbarrierを追加したりコンパイル設定を変えると
従来の数値・性能にも影響し得るため、今回の条件下で自動採用しなかった。

## 復元後の状態と再現手順

最終本体は開始時B2とhash一致し、新しい多層APIは残していない。
従来のLayerNorm/A/B1/B2の未commit変更は保持。固定snapshot・manifest・研究データ・解析結果・checkpointは不変。
復元後のB2内部量5件と検証checkpoint関連4件は `restored-b2-output.txt`、
静的検証と候補なしSKIPの確認は `restored-static-output.txt`。
候補なしの3 SKIPをDのPASSとは数えない。過去研究checkpoint・GPU・複数deviceは未検証。

失敗を再現する場合は、**B2の未commit／未追跡ファイルを含む作業ツリーの検証用コピー**で行う。
HEADだけのcloneではB2のattention.pyやtestsが揃わない。
現行の研究用作業ツリーへ勝手に再適用しない。

```sh
# 以下は検証用コピー内だけで実行する。
git apply --check tests/layers_validation/candidate.patch
git apply tests/layers_validation/candidate.patch
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_layers
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.diagnose_layers NEW_DIAGNOSIS.json
```

現行B2上でcheck_layersは撤回理由を示してSKIPし、diagnose_layersは候補がない旨を明示して停止する。
候補の失敗ログとHLOは今回すでに保存済みなので、確認のためだけに本体を変える必要はない。

## 段階Eへの判断

**段階Eへは進めない。Dの完了条件が未達。**
続行する前に、取得ONの比較を精度相当の丸め差まで許す設計とするか、
通常側も含む実行境界変更を別の数値変更として検証するか、条件・設計の判断が必要。
今回はどちらも承認された変更とは扱わず、基準を変えずに候補を撤回した。

その判断後に、実波動関数・微分のON/OFF、複数層ONのAdam/KFAC、checkpoint復元、
大きいNでの返却メモリ・性能を再検証する必要がある。
解析スクリプトの移行、softmax分離、Hamiltonian/VMC修正、学習ループ統合は未実施。
既存の物理検証FAIL/ERRORも全て維持している。

---

# 段階B2：単一Screening Attentionの内部量取得（2026-10-23・D試行前の安定版）

**B2のみ完了。通常forwardの固定snapshotは両精度で厳密一致。内部量ON/OFFも検証範囲で厳密一致。C/D/Eには進んでいない。**

## 変更ファイルとインターフェース

- `periodicwave/attention.py`：`screening_attention` にkeyword-only `return_aux=False` を追加。
  Falseは従来の出力配列のみ、Trueは `(out, aux)` を返す。auxは単純な辞書。
  中間値を上書きする前に参照を保持するだけで、Similarity/Trim/Window/集約を再計算しない。
  追加のstop_gradientや、head別射影による出力の再構成はない。
- 新規 `tests/check_attention_aux.py`：OFF/ON、独立式、微分、Network接続、Window境界の5検証。
- 新規 `tests/measure_attention_aux.py`：B1固定参照とOFF/ONのCPU JAXPR・メモリ・時間測定。
- 新規 `tests/aux_validation/`：開始時hash/AST、全実行結果・ログ・性能記録。
- `tests/RESULTS.md`：今回の結果を追記。

既存 `CustomScreenformer.make_screening_attention`、返される通常apply、Network.apply、Network構造は変更していない。
初期化・乱数キー・パラメータPyTree・Transformer・checkpointも変更なし。
通常経路は従来どおりreturn_auxを指定しない。辞書構築・中間値の保持はTrueの静的分岐内だけ。
`scope-check.json` でauxの分岐・追加keyword・docstringを除いたASTがB1と一致することを確認した。

## 内部量の定義とshape

N=電子数、H=head数、E=埋め込み次元、K=attn_dim、V=value_dim、P=pair_dim。
D=通常ValueならV、固定ValueならP。ε=1e-6。

| キー／戻り値 | shape | 数式・取得位置 |
|---|---|---|
| `q`, `k` | [N,H,K] | XW_Q / XW_Kを既存のreshapeでheadに分割した直後 |
| `v` | [N,H,V]またはNone | XW_V直後。固定Value時はVを作らずNone |
| `q_normalized`, `k_normalized`, `v_normalized` | 元と同じ | z/sqrt(sum(z*z)+ε²)。vは_v_mode介入前。固定ValueのvはNone |
| `v_effective` | [N,H,V]またはNone | _v_modeの電子平均置換後。実際に集約するvbar |
| `pair_features` | [N,N,P]またはNone | 固定Value時に実際に渡された対特徴量。通常Value時はNone |
| `similarity` | [H,N,N] | s_ij=qbar_i・kbar_j |
| `alpha` | [H,N,N] | (relu(s-(1-r))/r)^p、r=trim_max*sigmoid(trim_rho)、p=2/4 |
| `window` | [H,N,N]またはNone | 既存screening_windowが返したm。窓なしはNoneで、onesを新規作成しない |
| `weights_window` | [H,N,N] | alpha*m直後、介入前。窓なしはalphaそのもの |
| `weights` | [H,N,N] | _diag_only、距離clip介入後。固定Valueの対角除去前 |
| `aggregation_weights` | [H,N,N] | 固定Valueならweights*(1-I)、通常Valueならweights。そのまま集約に使用した値 |
| `aggregate` | [N,H,D] | sum_j W_ij*vbar_j または sum_j W_ij*pair_feat_ij。行和で割らない |
| `tanh_output` | [N,H,D] | n=sqrt(sum(aggregate²)+ε²)、aggregate*tanh(n)/nの直後。無効時はaggregate |
| `gate_output` | [N,H*D] | 既存のhead連結後、tanh(SiLU(XW_G))を掛けた直後。Gateなしなら連結した値 |
| `projected_output` | [N,E] | gate_output W_O。headを分けず従来どおり一度の線形射影 |
| タプル第1要素 `out` | [N,E] | 出力倍率exp(out_scale)適用後。倍率なしはprojected_output |

出力の重複保持を避けるため、最終outをaux辞書へ再度入れていない。
存在しない段階はNone、無効な変換は恒等として記録する。
辞書の構造は固定した構成の下でJAX PyTreeとして扱えるが、固定Value等の異なる構成は別traceとなる。
各配列の微分経路は元のまま。解析値を損失へ使う場合も本機能による勾配停止はない。

## 利用方法と境界

既存の単一Attention入力・構成を使い、return_auxを静的に指定する。
xは座標そのものではなく、そのAttentionへ入る埋め込み [N,E]。
optionsはB1と同じfactory構成値の辞書、p_attentionは対応するhead群のparamsである。

```python
from periodicwave import attention, CustomScreenformer as CS
import jax

def inspect_one(p_attention, x, d_ee, pair_feat=None):
    return attention.screening_attention(
        p_attention, x, d_ee, pair_feat, options=options,
        trim_max=CS.TRIM_MAX, trim_pow=CS.TRIM_POW,
        win_shape=CS.WIN_SHAPE, freeze_w=CS.FREEZE_W,
        clip_r=CS.CLIP_R, clip_beta=CS.CLIP_BETA, return_aux=True)

out, aux = jax.jit(inspect_one)(p_attention, x, d_ee, pair_feat)
# aux['weights_window'] と aux['aggregation_weights'] は異なる段階の値。
```

return_auxをJITの動的引数にしない。上のような固定指定、またはstatic_argnamesを使う。
既存グローバルの読取場所・タイミングはB1のまま。Networkから全層の入力や内部量を集める機能は追加していない。
複数層の取得は段階D、解析スクリプトの移行・過去解析との比較は段階Eとして別途行う。

## 数値検証

環境は前回と同じ `.venv-pw`、Python 3.13.14、JAX/jaxlib 0.7.2、folx 0.2.19、KFAC-JAX 0.0.7、
Optax 0.2.6、NumPy 2.3.3、Apple M5、CPU 1 device、matmul precision=float32。
既存環境・依存は変更していない。版・環境・tracebackは `aux_validation/numerical-results.json`。

| 比較 | 結果 |
|---|---|
| 通常OFF、x64=True、snapshot-v1 | 全902配列・21,651要素が厳密一致（`frozen-result.json` / `frozen-output.txt`） |
| 通常OFF、x64=False、snapshot-float32-v1 | 全1,319配列・31,142要素が厳密一致（`compare-float32.txt`） |
| 取得ONの単一kernelを通常API内で使用、float32 | 全5経路・11設定・Adam/KFACの更新/状態/復元値が同じ固定基準と厳密一致（`aux-training-output.txt`） |
| 通常OFFとON、eager/JIT | 両精度・Screening12設定で出力が厳密一致 |
| 内部量を返り値として保持したAD | 両精度・12設定・凍結有無でparams/入力特徴/距離の勾配、距離Hessianが厳密一致 |
| 全Network接続 | 両精度×5経路で波動関数、座標Jacobian/Hessian、params勾配が厳密一致 |

既存固定基準は5経路（Psiformer/SlaterNet/Screenformer/λ-jellium/合成HF）と11設定。
追加の12設定はScreeningの10設定に倍率なし・解析介入を加えたもの。
Trim²/⁴、cos/smooth7/band/sin/gauss、固定Value、窓なし、Gateなし、TanhNormなし、倍率なし、
_v_mode/_diag_only/CLIP_Rを含む。

内部量は既存の独立NumPy式に照合した。未正規化Q/K/V、正規化値、Similarity、alpha、Window、
各Weight、集約、TanhNorm後、Gate後、射影後、最終出力を検査。
独立式との比較には従来のrtol=2e-6/atol=2e-7をそのまま使用し、許容誤差を緩めていない。
OFF/ON出力・微分と固定基準には引き続き厳密一致を要求する。
取得した実際の集約用Weight/Valueから集約を検算し、射影直後の値と倍率から最終出力との整合性も確認した。
これらの再計算はテスト側の検算だけで、本体は既存変数を返すだけである。

## 既存テストとの比較

| 対象 | B1後 | B2後 |
|---|---|---|
| 既存33メソッド | 25 PASS / 4 XFAIL / 2 FAIL / 2 ERROR / 0 SKIP | 同じ。全メソッドのstatus変更なし |
| Window追加6件 | 6 PASS | 6 PASS |
| LayerNorm/JIT/checkpoint 4件 | 4 PASS | 4 PASS |
| forward比較4件 | 4 PASS | 4 PASS |
| 新規B2検証5件 | — | 5 PASS |
| 再現性・静的11件 | 10 PASS / 1 XFAIL | 同じ |

33件の生記録は `numerical-results.json`（frozenのみ環境変数未指定でSKIP）。
基準条件に合わせPW_RUN_KFACを指定しない別プロセスでfrozenを実際に再実行し厳密比較PASS。
両記録を統合した `final-results.json` をB1の結果とメソッド名ごとに比較した。
runner終了コード2は既存FAIL/ERRORのため。KFACの3つのsubtest ERRORは1メソッドのERRORとして集計。

既存追加14件のログは `existing-extra-output.txt`。Window境界近傍の微分・Hessian、勾配停止、folx、
固定対特徴量の微分、グローバル変更/JIT cache、保存済みcheckpoint10個の復元・評価も再確認。
新規の初回4件は `aux-output.txt`。独立式検証へ追加した各段階のassertは同メソッドを再実行してPASS
（`aux-values-output.txt`）。さらにWindow境界の専用1件を追加してPASS（`aux-boundary-output.txt`）。
これは原点・窓幅・帯窓端・輪境界と遷移帯端・d_refの位置とその両側について、
Windowを返り値に保持したまま出力・1階/2階微分をOFFと両精度で厳密比較する。
合計5メソッドであり、独立式の重複実行を追加1メソッドとは数えない。

取得ONの学習比較はテスト内だけでscreening_attentionを包み、同じ関数のreturn_aux=Trueの第1要素を
通常Networkへ渡した。Network/APIの本体変更や多層収集機能の実装ではない。
この状態で既存のfloat32数値基準生成を再実行し、Adam/KFACを含む全配列を--exact相当で比較した。
過去研究checkpointとの互換性、長期再開、GPU、複数device、実HF研究入力、長い既存検算3件は未検証。

## メモリと性能

`measure_attention_aux.py` を他の主な検証プロセス終了後に実行。
N=64/H=4/K=V=16/E=64、通常Value、cos窓、Gate・TanhNorm・倍率ありで測定。
B1固定参照とB2通常OFFのJAXPRは両精度で文字列一致し、コンパイル時メモリ見積りも完全一致。
引数/返却/一時領域の全てを比較した。通常OFFにaux用の行列や返却バッファは追加されない。

| 精度・モード | JIT中央値 ms | 出力bytes | 一時領域bytes |
|---|---:|---:|---:|
| float32 B1参照 | 0.17418 | 16,384 | 114,820 |
| float32 B2 OFF | 0.17353 | 16,384 | 114,820 |
| float32 B2 ON | 0.35351 | 589,968 | 1,028 |
| float64 B1参照 | 0.27387 | 32,768 | 229,508 |
| float64 B2 OFF | 0.27369 | 32,768 | 229,508 |
| float64 B2 ON | 0.36357 | 1,114,256 | 2,052 |

5回warmup、各50回×9組をモード交互に完了待ちして測定。compile時間は除外。
全サンプルと環境・メモリ値は `performance.json`、実行ログは `performance-output.txt`。
時間差はこのCPU小系の測定値であり、全shape/GPUの性能保証ではない。
メモリ値はXLAのコンパイル時見積りであり、プロセスRSSやGPU実測ピークではない。
ONは途中配列を出力するため明確にメモリ・時間が増える。大きなNや多数配置の一括取得は避ける必要がある。
通常の学習はOFFのままなのでこの取得コストを負わない。eagerに追加されるのは静的bool分岐のみで、
本体の追加配列演算はない（eager時間は今回ベンチマークしていない）。

## 未解決問題と次の段階

既存のVMC位相勾配、folx対Hessian差、定数logabsのfolxエラー、x64 KFACのScreen/lambda/HFエラー、
Attention解析2設定・複素局所エネルギー・実数VMC係数のXFAILは維持。修正していない。
固定snapshotとmanifest、既存テスト、研究データ・解析結果・checkpointは変更していない。
本体の今回の変更はattention.pyのみ。CustomScreenformerと他本体のhashはB1後から不変。

**CPU検証範囲では段階C（softmax分離）・D（多層取得）を検討可能。今回はB2で停止。**
解析側は旧実装のままであり、今回の機能追加で過去解析の不一致が解消したとは扱わない。

再実行（JSON保存先には未使用のパスを指定）：

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_attention_aux
JAX_PLATFORMS=cpu .venv-pw/bin/python -m unittest tests.check_window tests.check_layernorm tests.check_screening -v
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.measure_attention_aux NEW_PERFORMANCE.json
JAX_PLATFORMS=cpu PW_BASELINE_DIR=tests/cpu_validation/snapshot-v1 .venv-pw/bin/python -m unittest tests.test_wavefunction.WavefunctionChecks.test_frozen_numerical_baseline -v
JAX_PLATFORMS=cpu PW_TEST_X64=0 PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.baseline compare tests/cpu_validation/snapshot-float32-v1 --exact
JAX_PLATFORMS=cpu PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.run_numerical NEW_REPORT.json
```

取得ONのAdam/KFAC比較は、float32環境で次を実行（比較のみでsnapshotを保存しない）：

```python
from unittest.mock import patch
from tests.baseline import numerical_snapshot, compare
from periodicwave import attention
original = attention.screening_attention
def with_aux(*args, **kwargs):
    return original(*args, **kwargs, return_aux=True)[0]
with patch.object(attention, 'screening_attention', with_aux):
    arrays, info = numerical_snapshot()
    compare('tests/cpu_validation/snapshot-float32-v1', arrays, info, exact=True)
```

---

# 段階B1：Screening Attentionのforward移動（2026-10-23・B2前の記録）

**forward評価のみ移動完了。両精度の固定snapshotが全項目厳密一致。新規FAIL/ERRORなし。B1で停止。**

## 変更した計算とファイル

| ファイル・関数 | 変更 |
|---|---|
| `periodicwave/attention.py`：`screening_attention` | Q/K/V、長さ正規化、Similarity、Trim、既存Window呼出、重み、Value集約、TanhNorm、Gate、連結後の出力射影・倍率を移動 |
| 同：`_unit_length` | 元の演算・eps=1e-6のまま移動 |
| `periodicwave/CustomScreenformer.py`：`make_screening_attention`の`apply` | 同じ公開署名で共通評価を呼ぶ。factoryとinitは元の場所に保持 |
| 同：`_unit_length` | 既存解析向けに共通関数のaliasを保持 |
| `tests/check_screening.py` | 旧forwardと比較する4テストを追加（既存33件とは別に実行） |
| `tests/screening_reference.py` | B1開始時のforwardを固定参照として保存。共有forwardを使わず評価する |
| `tests/screening_validation/` | 開始時hash・AST、旧forward、実行結果・ログ・変更範囲照合 |
| `tests/RESULTS.md` | 今回の結果を追記 |

変更前：factoryのinitとapplyに初期化・評価が同居。
変更後：factoryは従来の初期化と公開applyを保持し、apply → attention.screening_attention → screening_window。
共通評価には計算順を示す10段階のコメントを付けた。Windowの演算部分は変更していない。
内部量を返すAPI、softmax分離、解析更新、Transformer統合は追加していない。
前回までのLayerNorm・Window変更はそのまま保持し、今回の既存本体変更は上記2ファイルだけ。

## 引数・設定・PyTree

長い寸法・フラグ引数列を増やさず、factoryで元から保持していた10個の構成値を
通常の局所辞書 `attention_options` にまとめた。既存の寸法4値・d_ref・機能フラグ5値だけであり、
新しいクラス、設定システム、registry、学習パラメータではない。checkpointにも加わらない。

`TRIM_MAX/TRIM_POW/WIN_SHAPE/FREEZE_W/CLIP_R/CLIP_BETA` はCustomScreenformerに残し、
従来のapply実行時（JITならtrace時）に読み、評価関数へ明示的に渡す。
factory構築時への読取移動はしていない。環境変数読取とfolxのerf登録は元の位置のまま。
既存のcached JITはtrace時の設定を保持し、新しいtraceでは変更後の設定を読むことを比較した。

`scope-check.json` にてfactoryの引数、init全体、評価部分以外のASTが不変と確認。
forwardはグローバル参照名・Window呼出名を対応づけると移動前の演算ASTと一致する。
reshape、einsum、epsilon、dtype、stop_gradient、clip/where/ReLU、自己項とablationの位置を保持。
headごとに出力射影を分けたり、その和で通常forwardを作ったりしていない。
通常Valueと固定Valueの分岐、行和で割らない集約、出力倍率も維持。
乱数分割・初期化方式・PyTreeのキー／構造／leaf値は固定snapshotでも一致した。

## 固定snapshotと実行環境

前回と同じ `.venv-pw`、Python 3.13.14、JAX/jaxlib 0.7.2、folx 0.2.19、KFAC-JAX 0.0.7、
Optax 0.2.6、NumPy 2.3.3、Apple M5、CPU 1 device、matmul precision=float32。
依存更新・既存環境変更なし。版・環境は `screening_validation/numerical-results.json` に収録。

| 固定基準 | 今回の結果 |
|---|---|
| x64=True `cpu_validation/snapshot-v1` | 902配列・21,651要素、全項目厳密一致（`frozen-result.json` / `frozen-output.txt`） |
| x64=False `cpu_validation/snapshot-float32-v1` | 1,319配列・31,142要素、全項目厳密一致（`compare-float32.txt`、`--exact`） |

全5経路（Psiformer/SlaterNet/Screenformer/λ-jellium/合成HF）、11設定
（softmax/cos/smooth7/pow4/band/sin/gauss/fixed/nomask/nogate/nonorm）を比較。
波動関数の値・位相・logabs、軌道・行列式、Attention内部量、座標grad/Hessian、
VMC params grad、複素局所エネルギー、固定乱数MCMC、Adam更新・状態、学習1 stepと復元結果を含む。
float32の全5経路KFAC更新・状態・checkpoint復元値も厳密一致。
shape/dtype/PyTree・実効設定も比較し、複素数は実部・虚部を別々に厳密比較した。
旧snapshot/metadataの4ファイルのSHA-256は保存済みmanifestと一致。基準値や許容誤差は変更していない。

## テスト集計と比較方法

| 対象 | 段階A後 | B1後 |
|---|---|---|
| 既存33メソッド | 25 PASS / 4 XFAIL / 2 FAIL / 2 ERROR / 0 SKIP | 同じ。全メソッドのstatus変更なし |
| Window追加6件 | 6 PASS | 6 PASS |
| 既存LayerNorm/JIT/checkpoint 4件 | 4 PASS | 4 PASS |
| 新規forward比較 | — | 4 PASS |
| 既存再現性・静的検証11件 | 10 PASS / 1 XFAIL | 同じ |

33件の生記録は `numerical-results.json`。その実行ではfrozenのみPW_BASELINE_DIR未指定のSKIP。
固定x64基準の作成条件に合わせ、PW_RUN_KFACを設定しない別プロセスでfrozenを再実行し厳密比較PASS。
`final-results.json` は両結果を統合し、段階Aの `window_validation/final-results.json` とテスト名ごとに比較。
runner終了コード2は既存FAIL/ERRORのためであり、全件成功とは報告しない。
KFACの3つのsubtest ERRORは1メソッドのERRORとして集計している。

Window6件＋既存4件の今回の記録は `window-checkpoint-output.txt`（10件PASS）。
窓境界近傍の1階/2階微分、凍結有無のparams勾配/Hessian、Gaussianのfolx、
Screening/λ-jellium/HFのJIT・座標微分・params微分、Windowなし、旧補助名の互換性を再確認。
既存4件では全5経路の両精度JITと、保存済み10個の検証checkpointからの復元も確認。

新規4件は `screening-output-final.txt`：

1. 両精度、Screeningの10設定に出力倍率なし・解析介入を加えた12設定でeager/JIT出力を旧forwardと厳密比較。
2. 同じ12設定でparams・入力特徴・距離への勾配、距離へのHessianを旧forwardとJITで厳密比較。
   attn_dim=4/value_dim=3、固定Valueはpair_dim=5とし、次元の違いも検査。
3. 固定Valueの対特徴量への勾配・Hessianを追加比較し、対角の対特徴量への勾配がゼロであることも確認。
4. グローバル変更前のcached JITと、変更後のfresh traceの挙動を旧forwardと厳密比較。

解析介入には `_v_mode`、`_diag_only`、`CLIP_R` を含む。
初回3件の記録は `screening-output.txt`、固定Value微分を追加した最終4件は上記finalログ。
既存テストは変更・削除していない。静的検証は `static-output.txt`。

## checkpoint互換性と未解決事項

保存済み検証checkpoint10個（両精度×5経路、LayerNorm移動前に作成）について、
hash不変、復元paramsと初期化の厳密一致、保存時の位相/logabsの厳密一致を確認した。
これらはopt_state=Noneの初期化checkpoint。Adam/KFAC状態は別途上記の固定基準と学習1 stepで比較。
過去研究checkpointの実データとの互換性や長期学習再開は未検証。形式は変更していない。

既存FAIL/ERRORは同じ対象・原因で残る：位相VMC勾配 -4対-2、Psiformerのfolx対Hessian差
（絶対1.78235943e-7、相対5.32383814e-7）、定数logabsの `.laplacian` エラー、
x64 KFACのScreen/lambda/HFにおけるdtype混在。
x64 KFACのPsiformer/SlaterNetは実行できたが、失敗する3経路の同等性は検証完了と扱わない。
Attention解析2設定・複素局所エネルギー・実数VMC係数のXFAILも維持。不具合を修正していない。
GPU、複数device、実HF研究入力、過去研究checkpoint、長時間の統計比較、長い既存検算3件は今回未実行。

## 残る重複とB2判断

本体forwardと `_unit_length` の計算コピーは除去し、旧名はaliasのみ。
解析の `attn_probe`、`head_contrib/head_cancel`、`structure_probe`、`lj2/readout` 等の別実装は残る。
Psiformer/ScreenformerのMLP・Transformer周辺の重複も維持し、今回の変更に混ぜていない。
固定参照の重複は同等性検証用であり、本体では使用しない。

**同一CPU検証環境ではB2（内部量取得）の設計・実装を検討できる状態。ただしB1で停止した。**
通常forwardの返り値は従来の配列のみ。内部量API、解析probe移行、softmax分離には未着手。
B2でも通常経路の演算順と固定基準を維持し、解析の既知不一致の修正を別変更として扱う。

再実行（JSON出力先には未使用のパスを指定）：

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_screening
JAX_PLATFORMS=cpu .venv-pw/bin/python -m unittest tests.check_window tests.check_layernorm -v
JAX_PLATFORMS=cpu PW_BASELINE_DIR=tests/cpu_validation/snapshot-v1 .venv-pw/bin/python -m unittest tests.test_wavefunction.WavefunctionChecks.test_frozen_numerical_baseline -v
JAX_PLATFORMS=cpu PW_TEST_X64=0 PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.baseline compare tests/cpu_validation/snapshot-float32-v1 --exact
JAX_PLATFORMS=cpu PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.run_numerical NEW_REPORT.json
```

---

# 段階A：Screening Window抽出（2026-10-23・B1前の記録）

**Windowのみ抽出を完了。両精度の固定snapshotが全項目厳密一致し、新規FAIL/ERRORなし。段階Bには進んでいない。**

## 変更したファイルと構造

- 新規 `periodicwave/attention.py`：`screening_window`、元の `_smooth7`、`_ring_steps`。
  単調余弦窓・smooth7・帯域窓・sin/Gaussian輪切りを集約した。
  数式、入力の意味・shape、勾配停止、境界の微分可能性をdocstringに記載。
- `periodicwave/CustomScreenformer.py`：既存の `if use_softmask` 内から新関数を呼ぶ。
  `weights = alpha * m` とWindowなしの `weights = alpha` はそのまま維持。
  既存解析から利用する `_smooth7` / `_ring_steps` の名前は同一関数へのaliasとして保持。
- 新規 `tests/window_reference.py`：抽出前のWindow演算ASTと補助関数の固定参照。
  本体をimportして再利用する参照ではなく、独立に保存した比較用実装。
- 新規 `tests/check_window.py`：下記6テスト。既存33件の列挙には加えず、別コマンドで実行する。
- 新規 `tests/window_validation/`：実行ログ、個別結果、開始時source hash、変更範囲検証。
- `tests/RESULTS.md`：今回の検証結果を追記。

変更前は `make_screening_attention.apply` にWindow全分岐があった。
変更後は同じ位置で `attention.screening_window(params, d_ee, d_ref, win_shape=WIN_SHAPE, freeze_w=FREEZE_W)` を呼ぶ。
Q/K/V、Trim、Value集約、TanhNorm、Gate、Transformer、LayerNorm、初期化、軌道、
Hamiltonian、VMC、MCMC、最適化、checkpoint、解析スクリプトには今回変更を加えていない。
前回のLayerNorm共通化の未commit差分はそのまま保持。

## 固定snapshot比較と環境

共通化前と同じ `.venv-pw`、Python 3.13.14、JAX/jaxlib 0.7.2、folx 0.2.19、
KFAC-JAX 0.0.7、Optax 0.2.6、NumPy 2.3.3、Apple M5、CPU 1 device。
JAX_PLATFORMS=cpu、matmul precision=float32。既存環境や依存は変更していない。
個々の版・環境・各テストのtracebackは `window_validation/numerical-results.json` に記録。

| 比較 | 結果・記録 |
|---|---|
| x64=True、固定 `snapshot-v1` | 902配列・21,651要素が全て厳密一致。`frozen-result.json` / `frozen-output.txt` |
| x64=False、固定 `snapshot-float32-v1` | 1,319配列・31,142要素が全て厳密一致。`compare-float32.txt` |
| 既存5経路 | Psiformer / SlaterNet / Screenformer / λ-jellium / 合成HF版を維持 |
| 既存11設定 | softmax / cos / smooth7 / pow4 / band / sin / gauss / fixed / nomask / nogate / nonormを維持 |
| 比較項目 | 位相・logabs・ψ、軌道・行列式、Attention内部量、座標grad/Hessian、VMC params grad、複素局所エネルギー、固定key MCMC、Adam更新・状態、学習1 stepの復元値・CSV |
| float32 KFAC | 全5経路の更新・状態・checkpoint復元値も厳密一致 |

shape/dtype/PyTreeと実効設定を照合し、複素数の実部・虚部を個別に厳密比較した。
許容誤差は変更していない。旧snapshotの配列・metadata計4ファイルのSHA-256は旧manifestと一致。

## テスト結果

| 集計対象 | 変更前 | 段階A後 |
|---|---|---|
| 既存33メソッド | 25 PASS / 4 XFAIL / 2 FAIL / 2 ERROR / 0 SKIP | 同じ。全メソッドのstatus変更なし |
| Window局所検証 | 新規 | 6 PASS |
| 既存 `check_layernorm` | 4 PASS | 4 PASS |
| 既存再現性・静的検証 | 10 PASS / 1 XFAIL | 同じ |

33件runnerの生記録 `numerical-results.json` ではfrozenだけが環境変数未指定でSKIP。
固定x64基準に合わせてPW_RUN_KFACを設定しない別プロセスでfrozenを実際に再実行し、厳密比較PASS。
`final-results.json` は両記録を統合し、前回 `layernorm_validation/final-results.json` とテスト名ごとに比較した。
runnerの終了コード2は既存FAIL/ERRORによるもの。KFACの3つのsubtest ERRORは1メソッドのERRORとして集計。

`check_window` の6件（`window-output-final.txt`）：

1. 両精度、cos/smooth7/band/sin/gauss、凍結有無の値・距離1階/2階微分を旧式と厳密比較。
   原点・幅・帯窓端・輪境界と遷移帯端・d_refの位置、隣接浮動小数点値と±1e-4を検査。eagerとJITを含む。
2. 両精度・全窓・凍結有無でparams勾配/Hessianを厳密比較。凍結対象と常時固定パラメータの勾配ゼロも確認。
3. グローバル変更とJIT traceタイミングを確認。cached JITは旧設定、fresh traceは新設定を使い、旧式と厳密一致。
4. Screening/λ-jellium/HFモデルのJIT出力、座標Jacobian/Hessian、params勾配を、旧Windowを差し込んだ評価と両精度で厳密比較。
5. Gaussian窓のfolx値・Laplacianを旧Windowと両精度で厳密比較。
6. 旧補助関数名とWindowなし経路を確認。Windowなしでは共通関数を呼ばず、追加の乗算やパラメータも不要。

`checkpoint-jit-output.txt` は既存 `check_layernorm` 4件の今回の再実行記録。
両精度×5経路の全モデルJIT、およびLayerNorm移動前に作成した10個のcheckpointのhash・
復元params・保存時の波動関数位相/logabsが引き続き厳密一致した。
過去研究checkpoint実データとの互換性は依然未検証。形式変更はしていない。

## グローバル・微分・変更範囲の互換性

`WIN_SHAPE` / `FREEZE_W` の定義はCustomScreenformerに残し、従来のapply実行時
（JITならtrace時）の値を新関数へ明示的に渡す。新関数には設定取得やJIT decoratorを追加していない。
環境変数の読取は従来どおりモジュールimport時。
`TRIM_MAX` / `TRIM_POW` は元の初期化・Trim位置、`CLIP_R` / `CLIP_BETA` は元のablation位置に残る。
設定の再設計やfactory構築時への読取移動は行っていない。

幅・中心・輪境界のstop_gradient、指数・Gaussian目印の常時stop_gradient、
w[0]の遷移幅、輪の端の階段上書き、原点の二重where、clip、dtype、演算順を維持。
d_refの算出と自己項の扱いも変更していない。
folxのerf登録は元のCustomScreenformerの同じトップレベル位置にあり、attention.pyは登録処理を持たない。

`scope-check.json`：抽出したブロックを戻したASTを比較し、他の計算が不変であることを確認。
補助2関数のASTも一致。`before-source-sha256.json` に記録した既存ソース・解析ファイルは
CustomScreenformer.py以外すべてhash不変。既存テスト・許容誤差・研究データ・解析結果・基準値を変更していない。

## 未解決問題と次の判断

既存の位相VMC勾配（-4対-2）、Psiformerのfolx対Hessian差（絶対1.78235943e-7）、
定数logabsのfolx `.laplacian` エラー、x64 KFACのScreen/lambda/HFのdtype混在エラーを維持。
Attention解析のTrim⁴/smooth7不一致、複素局所エネルギー、実数VMC係数のXFAILも維持。
x64 KFACはPsiformer/SlaterNetの実行が通るが、失敗する3経路の同等性は完了したと扱わない。
GPU、複数device、実HF研究入力、長時間の統計比較、長い既存検算3件は今回未実行。

Windowを改造するときは `attention.screening_window` の該当分岐を変更する。
多項式は `_smooth7`、輪切り階段は `_ring_steps`。学習初期化や幾何・Hamiltonianを変更する必要はない。
Windowの変更自体は新しいモデルになるため、今回の同等性リファクタリングとは別の実験として扱う。
**このCPU検証範囲では段階Bを検討可能。ただし今回は段階Aで停止し、Screening全体移動・解析更新には進んでいない。**

再実行（JSON出力先には未使用のパスを指定）：

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_window
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_layernorm
JAX_PLATFORMS=cpu PW_BASELINE_DIR=tests/cpu_validation/snapshot-v1 .venv-pw/bin/python -m unittest tests.test_wavefunction.WavefunctionChecks.test_frozen_numerical_baseline -v
JAX_PLATFORMS=cpu PW_TEST_X64=0 PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.baseline compare tests/cpu_validation/snapshot-float32-v1 --exact
JAX_PLATFORMS=cpu PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.run_numerical NEW_REPORT.json
```

---

# LayerNorm共通化の検証結果（2026-10-23・段階A前の記録）

**LayerNormのみ共通化を完了。固定snapshotは両精度とも厳密一致し、新規の数値不一致はない。**
独立した物理検証の既存FAIL/ERRORは残る。下記の履歴にある「本体変更なし」は共通化前の記録。

## 変更範囲

| ファイル | 変更 |
|---|---|
| `periodicwave/network_blocks.py` | 同一だった `make_layer_norm` を1箇所に集約。元の著作権・由来を記載 |
| `periodicwave/CustomPsiformer.py` | 重複factoryを共通関数の別名参照に置換、内部呼出を共通モジュールへ変更 |
| `periodicwave/CustomScreenformer.py` | 同上 |
| `periodicwave/SlaterNet.py` | 同上 |
| `tests/check_layernorm.py` | 移動前の式との局所比較、全5経路のJIT比較、移動前checkpoint評価の4テストを追加 |
| `tests/layernorm_validation/` | 今回のログ、個別結果、変更前に保存した小規模checkpointと評価値・hash |
| `tests/RESULTS.md` | 今回の結果を追記 |

3実装を1実装にし、重複していた2コピー（各19行、空行込み）を除去。
元モジュールの公開名 `make_layer_norm` は同一の関数オブジェクトへのaliasとして保持し、
既存解析スクリプトからのimport/callを維持する。wrapperやクラスは追加していない。

`init(param_shape)` と `apply(params, inputs, axis=-1)` の本体ASTは3つの移動前実装と同一。
scale=ones、offset=zeros、dtypeの指定なし、mean→var→epsilon=1e-5→rsqrt→積和の演算順を保持。
乱数キーを消費しない。ParamTreeの型注釈をそのまま維持するため、循環importを避ける
`networks` のimportだけfactory内に置いた（JIT対象のapply内にはimportを追加していない）。
`scope-check.json` によりLayerNorm以外のAST変更がないことを確認。
170ファイルの旧source hashのうち、変更は上記本体4ファイルだけ。

## 固定snapshotとの比較

環境は共通化前と同じ `.venv-pw`（Python 3.13.14、JAX/jaxlib 0.7.2、folx 0.2.19、
KFAC-JAX 0.0.7、Optax 0.2.6、NumPy 2.3.3）、Apple M5、CPU 1 device。
JAX_PLATFORMS=cpu、matmul precision=float32。既存環境・依存バージョンは変更していない。

| 対象 | 比較 | 結果 |
|---|---|---|
| `cpu_validation/snapshot-v1` | x64=True、902配列・21,651要素 | 全項目厳密一致（frozenテスト） |
| `cpu_validation/snapshot-float32-v1` | x64=False、1,319配列・31,142要素、KFAC込み | 全項目厳密一致（CLI `--exact`） |
| LayerNorm局所比較 | 両精度、axis=-1/0、初期化、非自明なscale/offset、JIT、座標grad/Hessian、params grad、JAXPR | 全て厳密一致 |
| 全モデルJIT | 両精度×全5経路、移動前の式を使うモデルとのJIT出力比較 | 全て厳密一致 |

全5経路はPsiformer、SlaterNet、Screenformer、λ-jellium、合成HF軌道版。
Attentionはsoftmax/cos/smooth7/pow4/band/sin/gauss/fixed/nomask/nogate/nonormの11設定。
両snapshotは波動関数の値・位相・logabs、軌道行列・行列式、Attention内部量、位相/logabsの
座標grad/Hessian、局所エネルギー、VMCパラメータ勾配、固定keyのMCMC結果、Adam更新と状態、
trainer 1 stepのcheckpoint復元値・CSV数値を含む。複素数は実部・虚部を別々に比較。
shape/dtype/PyTree構造・実効設定も比較し、許容誤差は緩めていない。
float32では全5経路のKFAC更新・状態も固定基準と一致。
x64のKFACはPsiformer/SlaterNetの実行が通るが、後述の3経路は既存ERRORで完了しない。
**失敗経路のKFAC同等性・継続学習が検証できたとは扱わない。**

元snapshotの `values.npz` / `metadata.json` 計4ファイルはSHA-256が保存済みmanifestと一致。
snapshotとそのmanifest、既存テスト、許容誤差、研究データは変更していない。

## 既存33メソッドの変更前後

| 結果 | 変更前 | 共通化後 |
|---|---:|---:|
| PASS | 25 | 25 |
| XFAIL | 4 | 4 |
| FAIL | 2 | 2 |
| ERROR | 2 | 2 |
| SKIP | 0 | 0 |

全33メソッドでstatus変更なし。KFACの3つのsubtest ERRORは1メソッドのERRORとして集計。
`layernorm_validation/numerical-results.json` は33メソッド実行時の生記録で、frozenのみ環境変数未指定のSKIP。
別プロセスで `PW_BASELINE_DIR` を指定してfrozenを実際に再実行しPASSを確認した
（`frozen-result.json` / `frozen-output.txt`）。KFACを含まないx64基準の作成条件を守るため、
このプロセスには `PW_RUN_KFAC` を設定していない。
両実行を統合した `layernorm_validation/final-results.json` が最終集計であり、
共通化前の `cpu_validation/final-results.json` とテスト名ごとに照合している。
33件runnerの終了コード2は既存FAIL/ERRORを含むためであり、全件成功を意味しない。

追加の局所検証4件は全PASS（`focused-output-final.txt`）。
静的・既存検算suiteは14件中10 PASS / 1 XFAIL / 3 SKIP（`static-output.txt`）。
この3 SKIPは `PW_RUN_LEGACY=1` を必要とする長い既存検算で、今回未実行。
33メソッドの集計には含めていない。

## checkpoint互換性の確認範囲

本体を変更する前に現行saveで両精度×5経路の初期化checkpointを新規作成した
（`layernorm_validation/before-checkpoints/`、計10ファイル、約58 KB）。
評価値・ファイルSHA-256は同フォルダの `evaluations.json`。
共通化後、現行restoreでこれらを読み、パラメータPyTree・各leafのdtype/値が初期化基準と
厳密一致し、保存時の座標の位相/logabsも厳密一致した。
局所テストはcheckpointを上書きせずhashも照合する。

この10個はopt_state=Noneの初期化checkpoint。Adam/KFAC状態の同等性は別途、上記固定snapshotの
学習1 step・保存/復元結果で確認した。過去研究checkpointの実ファイルは未入手で互換性は未検証。
任意の長期学習の完全再開も未検証。checkpoint形式自体は今回変更していない。

## 残っている問題と停止点

変更前と同じ未解決問題：
- FAIL：位相VMC勾配が解析的-2に対し-4（物理的な係数の不一致）。
- FAIL：Psiformerのfolx対Hessian、絶対差1.78235943e-7・相対差5.32383814e-7（変更前と同じ数値差）。
- ERROR：定数logabsの平面波folx出力に `.laplacian` がない。
- ERROR：x64 KFACのScreen/lambda/HFでfloat64とfloat32の抽象構造が一致しない。
- XFAIL：Attention解析2設定、複素局所エネルギーの虚部、実数VMC係数。
- block MCMC引数不整合などを記録する既存テストも維持。不具合は修正していない。

GPU、複数device、大規模系、実HF入力、過去checkpointは今回の確認範囲外。
固定keyのMCMC経路一致を確認したが、新しい長時間の統計比較は実施していない。
**LayerNorm移動はこのCPU検証範囲で安全と判断し、ここで停止する。**
次の候補はAttention本体と解析の境界整理。既知の式の不一致を保持した構造変更と、
物理的な修正を分け、固定snapshotによる比較を続ける。今回は着手していない。

再実行（JSON出力先は未使用のパスを指定）：

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_layernorm
JAX_PLATFORMS=cpu PW_BASELINE_DIR=tests/cpu_validation/snapshot-v1 .venv-pw/bin/python -m unittest tests.test_wavefunction.WavefunctionChecks.test_frozen_numerical_baseline -v
JAX_PLATFORMS=cpu PW_TEST_X64=0 PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.baseline compare tests/cpu_validation/snapshot-float32-v1 --exact
JAX_PLATFORMS=cpu PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.run_numerical NEW_REPORT.json
```

---

# 段階0・1の実行結果（2026-10-23）

## 独立CPU検証環境での実行（共通化前の記録）

既存環境を変更せず `.venv-pw/` を新設した。本体170 Pythonファイルのhashは変更なし。
**数値基準は確立。ただし独立した正しさの試験には未解決FAIL/ERRORがあり、全件正常ではない。**
以下は共通化前の結果で、後段の「基準未確立」「依存不足」は環境構築前の履歴。

### 環境と選定根拠

`setup.py` は版未固定、folx/KFACは可変Git参照だった。一方、
`README_PeriodicWave.md` 279行以降（commit de45304に収録）にはApple M3 CPUでの実測freezeがあり、
これを候補としてそのまま採用した。GPU用のJAX 0.8.0構成とは混ぜていない。
Python 3.13.14 / JAX・jaxlib 0.7.2 / folx 0.2.19
(commit d05c107028e3f88239ebf9e894d4a8c01abf90f6) /
KFAC-JAX 0.0.7 (commit d9ecae99e588e4abbb0dd3d4e977d1266824e14c) /
Optax 0.2.6 / NumPy 2.3.3 / ml_collections 1.1.0 / chex 0.1.91。
全33依存の固定一覧は `requirements-cpu.txt` と `cpu_validation/requirements-resolved.txt`。
Python 3.13で導入・import・JIT・自動微分・folx・float32 KFACが動作したため、3.11/3.12への変更は不要だった。

macOS arm64 / Apple M5 / CPU backend / CPU device 1。
既定の倍精度試験はx64=True、matmul precision=float32。
標準学習に近い別試験は `PW_TEST_X64=0` でfloat32。x64で構造が変わるので基準は分離した。
GPU・multi-host・大batchは実行していない。

再構築（新しい環境にのみ実行）：

```sh
uv venv --python 3.13.14 .venv-pw
uv pip install --python .venv-pw/bin/python -r tests/requirements-cpu.txt
```

### 33件の実行結果

最小Attentionの3件を先に実行し全てPASS。その後、次で33メソッドを実行した。

```sh
JAX_PLATFORMS=cpu PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.run_numerical NEW_REPORT.json
```

初回は24 PASS / 4 XFAIL / 2 FAIL / 2 ERROR / 1 SKIP（snapshot未作成）。
その後snapshotを作成し、最後のSKIP対象 `test_frozen_numerical_baseline` を実際に再実行して厳密比較PASS。
**最終集計は25 PASS / 4 XFAIL / 2 FAIL / 2 ERROR / 0 SKIP（計33メソッド）。**
KFACの失敗した3経路はsubtestとして個別記録し、メソッド集計では1 ERRORとした。
`final-results.json` は初回記録とfrozen再実行記録を合わせたもので、失敗を成功へ再分類していない。
環境・全テスト名・tracebackは `cpu_validation/initial-results.json`、
`frozen-test-result.json`、`final-results.json`。実行出力も同じフォルダに保存した。

| 結果・対象 | 原因と判定 |
|---|---|
| XFAIL 2件：Attention解析 | Trim⁴/smooth7に対して解析はTrim²/余弦のまま。既知K01を数値再現 |
| XFAIL：複素局所エネルギー | 通常PBCのfolxが虚部を落とすK03を再現 |
| XFAIL：実数VMC係数 | 規格化二状態系の厳密微分2に対して1。K04を再現 |
| ERROR：平面波運動項 | 定数logabsを返すとfolx出力は通常Array。運動項が `.laplacian` を参照してAttributeError。default側は通る。定数出力接続の問題であり丸め誤差ではない |
| ERROR：倍精度KFAC | 通常Psiformer/SlaterNetはPASS、Screen/lambda/HFは失敗。KFAC加重和のmask_omega/out_scaleでfloat64対float32を検出（kfac-cause.txt）。float32では全5経路が学習・復元できた。本体のdtypeを変更していない |
| FAIL：位相のVMC勾配 | H=[[0,i],[-i,0]], ψ=(e^-iθ,e^iθ)の厳密値E'(0)=-2に対し-4。lossの未clip分岐ではclipped_el=2E_L-Eとなり、位相勾配に余分な係数が残る。独立した物理的不一致として保持 |
| FAIL：Psiformerのfolx対Hessian | Laplacian絶対差1.78235943e-7、相対差5.32383814e-7。JAX 0.7.2のdot_product_attentionはsoftmax前にfloat32へcastする（functions.py:966）。混合精度の数値差と整合するが許容値は緩めずFAILを維持。巨大な物理的不一致とは区別し、追加解析は未解決 |

既知のblock MCMC引数エラー、NoneMoireのUnboundLocalError、設定の文字列保存を実際に再現した。
これらの「現行動作を記録するテスト」のPASSは、物理的に正しいとの認定ではない。
失敗したsubtestではその先のassertが実行されない場合がある。snapshotは別に両成分の微分を取得している。
既存3つの長い検算は今回の33件には含まず、今回は未実行。

### 固定した数値基準

- `cpu_validation/snapshot-v1/`：x64=True、全5経路、11種類のAttention、Adamと実trainer 1 step。
- `cpu_validation/snapshot-float32-v1/`：x64=False、同じ対象＋全5経路のKFAC学習1 step。
- 両方に値、位相、logabs、軌道・行列式、Attention内部量、座標grad/Hessian、複素E_L、VMC勾配、
  MCMC更新、optimizerの値・状態、実trainer生成checkpoint復元結果、CSVの数値を収録。
- 入力座標・スピン・格子・初期params・key・実効設定・依存・ソースhash・入力hashをmetadataに保存。
- HFはG=0の2軌道からなる合成Bloch表。表の数値と生成ファイルSHA-256を保存。
  実際の結晶HF解/Chern数や過去研究データの代替とはしない。
- snapshotファイル自体のhashは `cpu_validation/snapshot-sha256.json`。
  `.gitignore`の例外はこの検証用snapshotに限定。基準を自動生成/上書きしない。

倍精度のfrozenテストは別プロセスで全数値を再計算し、実部・虚部を含め要素ごとに厳密一致した。
float32・KFAC込みも別プロセスで全数値の厳密比較に成功した（`compare-float32-output.txt`）。
比較の通常許容値 rtol=1e-8 / atol=1e-9は変更せず、今回の再現検証では `--exact` を使用。
固定key経路の再現であり、長い独立chainの統計的一致や収束を確認したものではない。

### checkpointとLayerNormの判断

全5経路の新規Adam checkpoint、float32の全5経路の新規KFAC checkpointを、現行のsave/restoreで生成・復元できた。
保存5フィールドと復元t+1もPASS。**過去checkpointの互換性は実データ不在により未検証**。
checkpointのPRNG状態欠落、設定保存・環境変数依存、解析キャッシュの制約は残る。

3つのLayerNormはAST、初期化、scale/offset、shape、epsilon=1e-5が一致。
float32/float64それぞれでJIT出力、パラメータ勾配、座標勾配、Hessianを比較し完全一致
（`layernorm-results.json`）。乱数を使わず、呼出元のparameter treeを保持する移動が可能。
ただし、まだ共有後のコードも旧checkpoint実データも検証していないため、統合後の安全性まで認定はしない。

判断：**LayerNormだけの小さな共通化は、固定基準の厳密比較と既知FAIL/ERRORの不変確認を条件に着手可能**。
学習ループ/Attentionの広い整理や、既知の物理的不具合・倍精度KFACの修正を同じ変更に含めない。
今回はLayerNormも本体も変更していない。

---

## 研究用環境の再調査・今回の判定

**数値基準は未確立。次の本体リファクタリングはまだ開始不可。**
既存環境を調べたが、数値計算に必要な依存の揃った環境は確認できなかった。
ライブラリの導入・更新、システム環境の変更、本体コードの変更は行っていない。
以下は再調査の結果で、後段の記録は前回の実行履歴。

### 確認した環境

| Python実行環境 | Python | NumPy | JAX / jaxlib / folx / KFAC / Optax |
|---|---|---|---|
| `/usr/bin/python3` | 3.9.6 | 未導入 | 全て未導入 |
| `~/.local/share/uv/python/cpython-3.13.14-macos-aarch64-none/bin/python3` | 3.13.14 | 未導入 | 全て未導入 |
| `~/academic/.venv-hubbard/bin/python` | 3.9.6 | 2.0.2 | 全て未導入 |
| `~/academic/reserch/code/fermi_machine/.venv/bin/python` | 3.9.6 | 2.0.2 | 全て未導入 |

CPUはarm64、論理CPU数10。ハードウェアにはApple M5の内蔵GPU（10コア）が1つある。
これはJAXで利用可能なGPUがあることを意味しない。JAX自体がないため、JAXのCPU/GPUデバイス数・backend・
実効float32/float64設定は**測定不可**。今回の数値runnerは依存確認で停止し、JAX設定段階には到達しない。
JAX/XLA/PW/CUDA関連とVIRTUAL_ENV/CONDA_PREFIX/OMP_NUM_THREADSの環境変数は、調査開始時に設定されていなかった。
テスト側は依存が揃えばCPU・x64を既定とし、matmul precisionはfloat32にする。

探索範囲は `~/academic`、uvのPython格納先、`~/.venv`、`/usr/local` 等。
ディレクトリ走査には深さ制限を設けており、別の場所やリモート研究サーバの環境が存在しないと断定はしない。
サーバへは接続していない。既存の実行方法は `tests/README.md` のunittestと、元のREADMEの研究環境を確認した。
詳細は [environment_audit.json](execution_2026-10-23/environment_audit.json)。

### 33件の数値テスト

Python 3.13.14で次を実行した。

```sh
/Users/toma/.local/share/uv/python/cpython-3.13.14-macos-aarch64-none/bin/python3 \
  -m tests.run_numerical tests/execution_2026-10-23/numerical_results.json
```

| PASS | FAIL / ERROR | XFAIL / XPASS | SKIP |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 33 |

**33件すべて依存不足で計算未実行**。終了コード2。既知問題も今回は数値XFAILとして確認していない。
各メソッド名・SKIP理由・同じ実行環境識別を [numerical_results.json](execution_2026-10-23/numerical_results.json) に保存した。
数値誤差と物理的不一致の判定に使える数値は取得できていない。
`tests.run_numerical` は今後、PASS/FAIL/ERROR/XFAIL/XPASS/SKIPと失敗tracebackを記録する。
既存の全体suiteも同じPythonで再実行し、静的検査10成功・1期待失敗、6数値モジュールと既存3検算のSKIPを確認した。
静的K02の期待失敗は数値検証の成功ではない。

### 保存した基準と未取得項目

- 保存済み：旧source manifest、今回の環境・LayerNormソース監査、33件の個別SKIP記録、全体suiteログ。
- 未取得：5経路と各窓のψ・軌道・Attention・座標微分・局所エネルギー・VMC勾配・MCMC・optimizer更新の数値基準。
- 過去checkpoint：探索範囲内で該当ファイルなし。依存も不足しており、実データ互換性は未確認。
- 外部HF研究表：場所未指定。今回読み込んだ研究用HF表はなく、そのhashを取得したとは報告しない。
  合成HF表の作成・snapshot収録コードは前回整備済みだが、実行できていない。

比較コードには `compare --exact` を追加した。関数の移動のみでは、まず要素ごとの厳密一致を要求する。
通常比較のrtol=1e-8 / atol=1e-9は変更していない。複素値は実部・虚部を別々に比較し、
shape・dtype・実効設定・依存・デバイス条件も照合する。ただし、この比較の数値実行自体も未検証。
固定keyによる経路比較と、独立chainによる統計比較は別物。後者の統計的再現性も今回は未測定。

### LayerNorm共通化の準備

3ファイルの `make_layer_norm()` は位置情報を除いたASTが完全一致（hashは環境監査JSONに収録）。
初期化は `scale=jnp.ones(param_shape)`、`offset=jnp.zeros(param_shape)`、乱数使用なし。
平均・分散は指定軸（既定-1）、keepdims=True、epsilon=1e-5で、同じrsqrt・乗算順序。
shapeの引数・パラメータ名も同じ。dtypeを明示しておらず、同じJAX/x64設定の下で同じ既定dtypeになる。
したがって「常にfloat32」とは言えない。

呼出元の `params['layers']['ln']` の並びと `scale/offset` の名前、初期化位置を変えずに
関数を共有すれば、パラメータ構造は保存できる見込み。ただし以下は未実証。

- 実際の旧checkpointをそのまま復元できること。
- JIT後の値、座標/パラメータ微分、KFACのタグ認識・状態・1更新の一致。
- 未中断/再開時や複数デバイスの学習経路の再現性。

結論は**ソース上は共通化候補として妥当、実行上の安全性は未認定**。統合は実施していない。

### 不足している作業・開始条件

研究で使用しているPython実行ファイル（または研究サーバ上の環境と実行方法）、
代表的な旧checkpointと対応config/HF表の場所が必要。
それがない場合は、研究で使用した正確な依存版を確認し、別途許可された隔離環境を用意する必要がある。
この段階で最新版をインストールして既存研究環境と同じだと扱うことはしない。

環境が利用可能になった後、33件・KFAC・既存検算の実行→失敗の分類→変更前snapshot採取→
同一コードでcompare/compare --exact→旧checkpointの読取専用復元、の順で確認する。
K01〜K05、設定保存の不完全性、seedの経路差、PRNG未保存、解析キャッシュ等は未修正のまま。
この条件が揃うまではLayerNorm・学習ループ・Attentionの構造変更を開始しない。

## 今回の変更

追加は `tests/` のみ。本体・既存テスト・設定・研究データは変更していない。
既存170 Pythonファイルは、記録したSHA-256と全件一致した。

- `source_baseline.json`：commit、170ソースのhash、環境変数、依存の有無。
- `baseline.py`：読取専用の環境調査、新規の数値基準保存、旧基準との比較。
- `support.py`：小系の固定入力、合成HF表、依存確認。
- 8つの `test_*.py`：静的契約、波動関数、Attention、物理、VMC/MCMC、学習/保存、既知問題、既存検算。
- `README.md` / `CONTRACTS.md`：実行方法・範囲・既知問題・保持仕様。

## 実際に実行した内容

環境：macOS arm64、`/usr/bin/python3`、Python 3.9.6。
NumPy、JAX、jaxlib、folx、KFAC、Optax、ml-collections、chex、attrs、absl、typing_extensionsは未導入。
依存の追加・更新は行っていない。

コマンド：

```sh
python3 -m unittest discover -s tests -v
```

結果：`Ran 20 tests`、**通常成功10、期待失敗1、スキップ9、予期しない失敗0**。

スキップ9の内訳は数値依存のないため読込をスキップした6モジュールと、opt-inの既存3検算。
これは全数値テストを実行した件数ではない。数値6モジュールには33のテストメソッドを用意したが、今回は実行していない。
その中でもKFAC統合はopt-in、frozen数値比較は保存済み基準が必要。

成功した静的検査：

- 基準commitとソースmanifestの形式。
- 現行Pythonファイルの構文解析。
- 通常・screen・λ-jelliumの入口とHF切替の存在。
- 計算に関係するPW環境変数の採取。
- 依存をimportせずに実行環境を記録できること。
- 既存の基準保存先を上書きしないこと。
- checkpointの5フィールドと復元時t+1。
- deterministic時の通常seed=42、screen側のみPW_SEED=91を反映すること。
- block呼出の実引数がmh_acceptの定義にbindできないこと。
- λ-jelliumの設定保存がdeterministicの上書きより前にあること。

期待失敗：K02の「block呼出がmh_acceptのインターフェースを満たす」検査。
実コードから抽出した7実引数を9引数の定義へPythonの `Signature.bind` で渡し、TypeErrorを再現した。
数値ライブラリ上でMCMCを実行したという意味ではない。

追加検査：

- 新規Pythonファイルの構文解析：成功。
- 元の170ソースのhash照合：変更0件。
- `tests.baseline capture`：依存不足を表示して終了コード2。保存先は作成されなかった。

既存ソースの構文解析では既存文字列の `\p` / `\s` に対するDeprecationWarningが出る。本体は修正していない。

## 実行していない内容

| 対象 | 理由・次の実行 |
|---|---|
| 波動関数、軌道、Attention、座標微分 | Python 3.10以上と数値依存が必要 |
| 局所エネルギー、VMC勾配、MCMC、Adam更新 | 同上。成功とは判定していない |
| K01/K03/K04/K05の数値再現 | 同上。ソース上の問題とテストの用意まで |
| 5経路の実trainerによる1 step、checkpoint roundtrip | 同上 |
| KFAC 1 step | 同上＋ `PW_RUN_KFAC=1` |
| 既存3検算 | 同上＋ `PW_RUN_LEGACY=1`。Gaussianの旧版比較は参照ファイル未指定で別途省略 |
| frozen数値基準の採取・比較 | 未取得。**本体変更前の必須作業として残る** |
| 実際の過去checkpoint・HF研究表 | リポジトリに含まれていない。合成入力のみ用意 |
| GPU・multi-host、大batch、配布wheel | 今回の小系CPU試験の対象外。GPUで実行したテストは0 |

追加テストにGPU必須のものはないが、GPU版の性能・カーネル・並列性を保証するものではない。

## 次の手順

まず既存の数値計算環境でREADMEのCPU試験、KFAC統合、数値基準の採取を行う。
予期しない失敗は原因を調べ、既知の問題を勝手に修正して基準を作り替えない。
K01〜K05と再現性上の制約の詳細は `CONTRACTS.md`。

その後の最小構造変更は、3つのAnsatzの同一LayerNormを既存 `network_blocks.py` へ集約すること。
学習ループの統合とAttentionの分離はまだ開始していない。
