# 作業引き継ぎ：Periodic Wave Transformerの段階的整理

更新日：2026-10-24。**G3aの自由電子VMC検証まで完了し停止。** **構造整理はF3で停止。G1でloss.pyのVMC主勾配を最小修正。G2bで通常PBC複素folxの虚部欠落を1行修正。lossとλ専用は保持。G2cでoptimizerなしの報告境界を修正。補助JVP等の既知問題は保持。F4は必要な新Ansatzが決まるまで保留。**
最新の実測結果は `tests/RESULTS.md` 先頭、物理/数値契約は `tests/CONTRACTS.md`。
過去の「依存不足」「B2まで」「25 PASS/4 XFAIL」は履歴であり現在の状態ではない。

## G3a追補の最新判定（先に読む）

G3a残存問題の調査・自由系の追加検証・G3b実行計画までで停止。本体変更ゼロ。G3b相互作用学習は未実施。
- 元13条件の4適合/9制約は履歴を保持。詳細分類はtests/g3a_followup/classification-v1.jsonとRESULTS先頭。
- Screenformer64の5.8e−12は旧テストのp+t*vがf32 mask_omega/out_scaleをf64へ昇格させたため。
  元dtypeのnative PyTree JVPでは差~3e−17。Production勾配の修正は不要。
- PsiformerはJAX0.7.2のAttention softmaxのf32 castに局在。テスト限定f64式でN2の1.78e−7差が8e−16へ。
  forwardも変わるので別段階の数値修正が必要。folxだけを原因と断定しない。
- KFAC64はf64係数倍後のmask/out_scaleとf32 velocityでdtype衝突。初期INFO complexエラーも未修正。
- 同じN2 complex24 paramsの独立4seed/209万標本でE=1.234742946±.000034748、固定参照との差1.061SEM。
  同じ学習を96stepへ延長し、最初24stepのcheckpoint全項目が旧版と厳密一致。E_ref=1.2337033123。
- 追加したreal SlaterNet N2/float64/Adam96は全検査適合。G3bはこの0層・null Envelope・Jastrowなし・blocks1を選択。
  L4 square、density.125、rs1.595769、Coulomb scale1、Ewald5、twist0、total Hartree。
  独立な有限Fourier Slater期待値を参照に使う。校正det(1,cos qx)の全E=.00863432013568は厳密基底値ではない。
- 独立seedは凍結測定でのみ検証。全学習RNG/初期化seedの独立再現性は未検証。
- 両精度G2b固定snapshotは再度全配列厳密一致。33件全体は今回再実行せず前回30/0/1/2を履歴として区別。
- 追加コードはtests/check_g3a_followup.py、資産はtests/g3a_followup/。旧コード/旧測定/旧snapshotは不変。

## G3aの判定（以下は追補前の履歴）

本体変更ゼロ。新規3検証CLI（check_g3a_vmc/exact/controls.py）、g3a_validation/の参照・学習・MCMC資産、文書のみ追加。
13条件の短期学習が物理更新/CSV/checkpointまで完走、Screenformer64 KFACは既知ERRORで未完走。
実数SlaterNet両精度、Screenformer32 Adam/KFACは検証範囲PASS。complex通常trainerは初期INFOの%fへcomplexを渡す残存不具合（train.py:640）を捕捉。
Psiformer64は点微分差、Screenformer64は別graph方向勾配差5.834e−12が厳しい基準に不適合で、全経路正常とは言えない。
独立Fourier参照、FFT収束、λ=0 HF固有状態3種×両精度、MCMCの72 histogram bin、blocking、Adam内部状態、checkpoint全leafを検証。
既存33件は30 PASS /0 XFAIL /1 FAIL /2 ERRORを維持。G2b両精度・5経路・11設定snapshotは全配列厳密一致。
Production PRNGはcheckpointに不保存。HF1行loaderのshape制限も検出（本体未修正）。
初回の検証側JSON/HF入力失敗とpilotも保存し、成功条件へ重複計上しない。
G3bは未実施。まず初期ログを別段階で修正/検証し、微分差とKFACは独立課題とするのが安全。
具体的な数値、実行コマンド、制約はtests/RESULTS.mdとtests/README.md先頭。

## 研究上の目標と厳守事項

Attention、Embedding、Wavefunction Construction、Hamiltonianを、研究上独立に改造しやすくする。
MCMC/VMC/Optimizer/Checkpoint/Loggingは原則共通基盤として維持する。
内部状態とMCMCの到達可能状態、lambda/DeltaとHF/Jastrow等の物理的依存は隠さない。

- 数式・初期化・乱数順・dtype・演算順・微分経路を勝手に変えない。
- 単純な関数中心。クラス階層、registry、plugin、汎用hookを追加しない。
- 物理問題の修正と数値同等な構造変更は別作業。既知FAIL/ERRORを通すため許容誤差を緩めない。
- snapshotを自動更新しない。研究データ・図・cache・checkpointを変更/削除しない。
- 指定段階だけ実施し停止。「次へ進める」は次段階の実装許可ではない。
- 未実行を成功扱いしない。日本語で結果と制約を報告する。

## Workspaceと未commitの作業

- `/Users/toma/academic/reserch/projects/Screening_develop`、基点HEAD `038730ef3bd06e738356e2a6f8d8b0adbb3e77cc`。
- 適用される上位指示は `/Users/toma/academic/reserch/AGENTS.md`。READMEも確認する。
- **これまでの実装・テストは未commit。commit/pushは行っていない。** ブランチと差分は再開時にgit statusで確認。
- 本体既存差分：CustomPsiformer/CustomScreenformer/SlaterNet/network_blocks/networks、G1のloss、G2bのpbc/hamiltonian、G2cの両trainer報告境界/statistics doc、新規attention.py/wavefunction.py。
- 解析既存差分：screening_analysis/attn_probe.py、head_contrib.py、head_cancel.py。
- tests/全体・このHANDOFF.md等も未追跡。git cloneだけでは移らない。作業ツリーと検証資産を保持する。
- `git reset/clean`やcheckoutで既存差分を捨てない。過去段階の作業を自分の変更と混同しない。
- F0はREADME/HANDOFF/tests内の文書・追加テスト・記録だけを変更。本体/解析入口は触っていない。

## 到達点

| 段階 | 内容 | 記録 |
|---|---|---|
| 0/1、CPU | 数値33件、独立CPU環境、両精度固定snapshot | tests/cpu_validation/ |
| LayerNorm | 3 Ansatzで共通化、公開名は保持 | tests/layernorm_validation/ |
| A/B1/B2 | Window/Screening評価の抽出、任意return_aux | tests/window_validation/, screening_validation/, aux_validation/ |
| D候補→D'→正式D | 多層診断を独立JITに分離して正式化 | tests/diagnostic_validation/、check_formal_diagnostic.py、RESULTS |
| E1/E2a/E2b | 主要3解析入口を正式診断へ移行 | tests/e1_validation/, e2a_validation/, e2b_validation/ |
| F0 | 4領域の現行契約、独立検証、不具合調査計画、文書整合 | tests/CONTRACTS.md、check_interfaces.py、f0_validation/ |
| F1 | 学習可能Embeddingのinit/applyだけ共通化。キー分割と裸のembedを保持 | network_blocks.py、check_embedding.py、f1_validation/ |
| F2a | 2つのperiodic_normを調査。数式は同じだが独立の研究変更境界として分離維持 | CONTRACTS/RESULTSのF2a欄、f2a_validation/ |
| F2b | 生変位・距離・特徴・Jastrow/Envelope/Hamiltonianの依存表。ファイル移動不要。I01の誤broadcastも調査 | CONTRACTS/RESULTSのF2b欄、f2b_validation/ |
| F3 | 通常make_orbitalsをwavefunction.pyへ移動。networksは同じ関数の再export、数値本体を保持 | check_wavefunction_construction.py、f3_validation/、RESULTS |
| VMC独立調査（修正前） | clip無効のK07・実数K04の原因を特定。旧式の数値記録は保持 | check_vmc_gradients.py、vmc_gradient_validation/、CONTRACTS末尾、RESULTS |
| G2（履歴） | 通常PBC複素folxの234行のreal投影を原因特定。root/λ専用は虚部保持、定数出力問題は別 | g2_validation/、CONTRACTS/RESULTS |
| G2b（履歴） | 同234行をreturn resultへ変更。複素E_LとG1勾配を検証。R08を別問題として発見 | check_g2_local_energy.py、check_g2b_downstream.py、g2b_validation/ |
| G2c | 両trainerで報告用Re(mean E_L)をEW/CSV/logへ渡す。複素aux・勾配・保存状態を維持 | check_g2c_reporting.py、g2c_validation/、CONTRACTS/RESULTS |

Dの最初の案は追加出力によるXLA融合差で厳密比較に失敗し撤回した。
現在はProductionとDiagnosticを**別々のJIT関数**で呼ぶ。
Diagnosticは同じソース・params・入力の再評価でありProduction内部のビット列ではない。
Production/VMC/最適化に診断結果を使わず、同じJITへProductionとDiagnosticをまとめない。

## APIと物理契約

- `Network.apply(params,pos,spins,atoms,charges)`はcomplexなら(位相角,log振幅)、realなら(sign,log振幅)。
- 通常軌道の実装はwavefunction.make_orbitals。networks.make_orbitalsは直接再export。行列式和・位相はnetwork_blocks.logdet_matmul、HFはlj_orbitals。型注釈は遅延評価、geometry importはfactory呼出し時。
- 全電子N×N行列式の和。spin別ブロックへ戻さない。
- 通常Jastrowはmerged `(N,0)`、HF専用Jastrowは実際の動的擬スピン対を使用。統合しない。
- 周期ノルムとEwaldの生変位を混同しない。格子は列ベクトル。
- 現行PBC r_ae [N,A]と非周期 [N,A,1]に差があり、PBC isotropic Envelopeでshape不整合を再現。修正していない。
- F2bでI01は例外だけでなく、同じshapeで誤った軌道別因子になる組合せも確認。null Envelopeの主要プリセットとHFの標準Envelope不使用経路を区別する。
- geometrical r_eeはScreening/Jastrowで共有、FeatureLayerのpairは固定Valueで使う。周期特徴だけ/Attention距離だけを変更するときはこの結合を確認する。
- FeatureLayer/Embedding/軌道/Hamiltonianの式・shape・単位・初期化順はCONTRACTSのF0欄を参照。

```python
production = jax.jit(net.apply)
diagnostic = jax.jit(CS.make_screenformer_diagnostic(nspins, net.options))
phase, logabs = production(params,pos,spins,atoms,charges)
x, layers = diagnostic(params,pos,spins,atoms,charges)
weights = layers[0]['attention']['aggregation_weights']
```

診断はtrunkまで。rank検査は別実行。4電子/幅4/最終初期LN等の特異条件では波動関数微分を定量評価しない。
Window積weights_window、介入後weights、固定Valueの対角除去後aggregation_weightsを区別する。行和正規化しない。
グローバル設定はimport/構築/traceの取得タイミングを保持。folx登録やJITコンパイル設定も変えない。

主要3解析は未保存設定/旧文字列格子を推測せず、settings/元精度latticeの明示を要求する。
新結果はattn_probe/head_contrib/head_cancelの`*_diagnostic_v1.npz`。排他保存で既存結果を上書きしない。
Head寄与はGate後のHeadブロック×W_O×倍率。共有biasは別。Head和を次層へ渡さない。
Cpは非線形な自己応答との差で純粋な二体/Head除去/エネルギー寄与ではない。
head_cancelのrhoは平均二乗の比で、head_contribの平均ノルム指標とは異なる。
他の解析には手動再実装と旧cache方式が残る。RESULTSのE2b一覧を参照。

## 環境・検証資産

既存研究環境は変更せず、CPU専用 `.venv-pw` を使用。
Python3.13.14、JAX/jaxlib0.7.2、folx0.2.19、KFAC0.0.7、Optax0.2.6、NumPy2.3.3、ml_collections1.1.0。
依存の根拠と固定値は `tests/requirements-cpu.txt`、`tests/cpu_validation/requirements-resolved.txt`。
隔離2 CPU deviceはG1の24条件で検証。GPU・multi-host・研究用過去HF/checkpointは未検証。HFテストは合成入力。

- float64：tests/cpu_validation/snapshot-v1、902配列/21651要素。
- float32：tests/cpu_validation/snapshot-float32-v1、1319配列/31142要素、全5経路Adam/KFACを含む。
- Productionは両精度・5経路・11設定で厳密比較。保存済みSHA-256と新旧測定を更新しない。
- 既存33件のG2c実行：**30 PASS / 0 XFAIL / 1 FAIL / 2 ERROR**。K03もPASSへ改善。
- G1では旧snapshotの波動関数/軌道/座標微分/局所エネルギー/MCMC/Attentionを両精度各282配列・5158要素で厳密比較。
- 勾配/更新は意図的な変更。旧全配列を上書きせず、g1_validation/へ独立検算と別version snapshotを保存。
- KFACのscreen/lambda/HF 3 subtest ERRORは1メソッドERROR。raw ERROR4とメソッドERROR2を混同しない。
- F0のCurrentContracts/IndependentPhysics/KnownInconsistencyは別集計。PASSの意味と最新完走状況はRESULTS参照。

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m unittest tests.check_interfaces -v
JAX_PLATFORMS=cpu PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.run_numerical NEW_RESULTS.json
JAX_PLATFORMS=cpu PW_BASELINE_DIR=tests/g2b_validation/snapshot-float64-g2b-v1 .venv-pw/bin/python -m unittest tests.test_wavefunction.WavefunctionChecks.test_frozen_numerical_baseline -v
JAX_PLATFORMS=cpu PW_TEST_X64=0 PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.baseline compare tests/g2b_validation/snapshot-float32-g2b-v1 --exact
```

数値runnerは既知FAIL/ERRORがあるため非0終了が現行基準。依存が欠けたSKIPをPASSにしない。
新しい結果JSONを指定し、既存測定を上書きしない。

## 最優先の未解決問題と次段階

G1でloss.pyの主勾配を `2 real(vdot(psi_tangent,diff))/B` へ最小修正した。
K04の実振幅1→2、K07の位相−4→−2を正規化完全和・FD・AD・独立共分散式で検証した。
clip方式/中心化規則は維持。clip有効時は既存diffに対するスコアで、元のエネルギーの厳密微分と同一視しない。
単一CPU 7 PASS（主勾配と独立式6件、K10保持再現1件）、隔離2 CPU device 1 PASS/24条件。
旧不具合の再現8件も旧lossの保存ソースに固定してPASS。Productionをテストで旧式へpatchしない。
補助JVPの零方向非零tangent/線形性・primal dtypeのK10は未修正。主出力の一階勾配と分離した。
float32の零近傍勾配をAdamが増幅する例も記録した。厳密な勾配とoptimizerの丸め感度を区別する。
G1でVMC勾配/Adam/KFAC/学習軌跡は意図的に変わる。旧optimizer状態の継承/初期化は研究上の判断で、勝手にリセットしない。
旧checkpointの形式は保持。保存済み検証checkpointからの評価と研究データによる過去runの検証を区別する。
folx定数log振幅ERROR、Psiformer微分FAIL、float64 KFAC混合dtype、block MCMC、NoneMoire等も未修正。
有限なAttentionや平均エネルギーだけで学習勾配・物理結果を認定しない。

基本計画はF1 Embedding共通化→F2周期ノルム/幾何→F3通常軌道factory→F4必要なAnsatz接続→F5 Hamiltonian境界整理。
F1はnetwork_blocks.init_input_embedding/apply_input_embeddingへ集約し、params['layers']['input'/'embed']とRNG順を保持した。
**構造整理はF3で停止。F4は研究で新Ansatzが必要になるまで保留。G2cの報告境界修正で停止。** 学習ループ統合や物理修正も自動で進めない。

## 新しいチャットへ渡す文

> HANDOFF.md、tests/CONTRACTS.md、tests/RESULTS.mdの先頭を読んで引き継いでください。
> Attentionと主要3解析入口はE2b、契約・不足テストはF0、学習可能Embeddingの共通化はF1まで完了しました。
> 未commit/未追跡のコード・テスト・固定snapshot・checkpointを保持してください。
> G1で主勾配、G2bで虚部欠落、G2cで報告問題R08を修正しました。K10/定数出力/Psiformer微分/倍精度KFAC等は未修正です。
> G2cの検証結果と残る既知問題を読んでください。G2cで停止し、別の修正やF4の新Ansatz追加は次の明示指示まで行わないでください。
