# G3a追補：残存問題とG3b準備（最新）

結果・13条件の分類・G3bの未実行計画は[RESULTS.md](RESULTS.md)先頭。
追加は `check_g3a_followup.py` だけ。Productionと旧検証資産はread-only。
以下は必ず新規出力先を指定する（既存ディレクトリへの再実行は停止）。

```sh
# 同じ旧24step paramsを固定し、4独立seedの系列長を比較
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_g3a_followup sample --case tests/g3a_validation/slater-two64 --sweeps 16384 --output /tmp/NEW_G3A_MEASUREMENT
# 保存済系列のburn/前後半感度。窓は互いに独立ではない
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_g3a_followup stability --output /tmp/NEW_G3A_STABILITY
# dtype/AD/graphの原因観測。Productionや旧testコードを変更しない
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_g3a_followup probes --output /tmp/NEW_G3A_PROBES
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_g3a_followup attention --output /tmp/NEW_G3A_ATTN
JAX_PLATFORMS=cpu PW_SEED=42 .venv-pw/bin/python -m tests.check_g3a_followup kfac --output /tmp/NEW_G3A_KFAC
# 独立Ewald・有限Fourier Slaterの相互作用期待値。相互作用VMC学習ではない
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_g3a_followup reference --output /tmp/NEW_G3B_REFERENCE
```

追加自由系96stepは旧CLIを変更せず `--particles 2 --iterations 96 --batch 128 --sweeps 1024` で実行した。
実数は `--real`。sampleのcaseにはその新規出力先も使える。
サンプリングseedは2303〜2306、学習の初期seedは42固定。全学習の乱数seedを独立化したと解釈しない。
attentionモードのf64 softmaxは原因切り分けの反実仮想であり、Productionへインストール/学習利用しない。
kfacモードは既知のERRORを観測して記録するため、CLI exit0でもKFAC学習がPASSした意味ではない。
今回はG3bの実行には進まず、設定と独立参照を準備したところで停止した。

---

# G3a：自由電子VMCの独立検証（最新）

実行結果・既知不一致は[RESULTS.md](RESULTS.md)先頭、物理定義は[CONTRACTS.md](CONTRACTS.md)末尾。
既存の本体は変更せず、Production trainerを小規模系で呼ぶ。必ず新しい出力先を指定する。

```sh
JAX_PLATFORMS=cpu PW_SEED=42 .venv-pw/bin/python -m tests.check_g3a_vmc --route slater --real --iterations 24 --batch 128 --sweeps 1024 --output /tmp/NEW_SLATER_G3A
JAX_PLATFORMS=cpu PW_SEED=42 .venv-pw/bin/python -m tests.check_g3a_vmc --route screen --float32 --optimizer kfac --iterations 8 --batch 64 --sweeps 1024 --output /tmp/NEW_SCREEN_G3A
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_g3a_exact --output /tmp/NEW_EXACT_G3A
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_g3a_exact --float32 --output /tmp/NEW_EXACT32_G3A
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_g3a_controls --root tests/g3a_validation --output /tmp/NEW_G3A_CONTROLS.json
```

`check_g3a_vmc --help`のrouteはslater/psiformer/screen。既定はcomplex、double、Adam、folx。
`--real` / `--float32` / `--particles 2`（SlaterNetのみ）/ `--laplacian default`を明示する。
実行結果を全面的にPASSと扱わない：通常trainerのcomplex初期INFO、double NNの点微分/縮約差、double Screenformer KFACには不適合がある。
INFO observerがフォーマットエラーを捕捉し記録するが、物理計算を変更しない。捕捉エラーもCLI exit2の対象。
conservative blocked SEMを使用し、locstdやEW分散を標準誤差として扱わない。
controlsは保存済13条件・6固有状態を読み、再学習せず分布とAdam内部状態等を再検証する。
定数folx/単一行HF loaderなどの既知制限を別に記録する。
新規snapshotの自動更新はしない。保存の詳細/初回失敗を含む履歴はg3a_validation/に保持。

---

# Periodic Wave Transformerの数値契約と検証

## G2c：複素物理量を保持した実数報告

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_g2c_reporting --report NEW_G2C_REPORT.json
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_g2b_downstream --report NEW_DOWNSTREAM_REPORT.json
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_g1_vmc --report NEW_G1_REPORT.json
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_g2_local_energy --report NEW_G2B_REPORT.json
JAX_PLATFORMS=cpu PW_BASELINE_DIR=tests/g2b_validation/snapshot-float64-g2b-v1 .venv-pw/bin/python -m unittest tests.test_wavefunction.WavefunctionChecks.test_frozen_numerical_baseline -v
JAX_PLATFORMS=cpu PW_TEST_X64=0 PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.baseline compare tests/g2b_validation/snapshot-float32-g2b-v1 --exact
```

G2cは報告のみの変更。G2bの全配列（勾配・Adam/KFAC更新を含む）に厳密一致を要求し、比較除外は使わない。
既存frozenメソッドのPW_BASELINE_DIRには現在の物理基準G2bを指定する。旧CPU/G1との変更区分比較は過去段階の専用手順。
check_g2c_reportingは保存した旧trainerのtrain関数をテストだけで読み、旧/新の2 step実評価を比較する。
両trainer/実数・複素/両精度/default・folxの16条件とλ2条件で、checkpoint全フィールド・locstd・pmoveを厳密比較。
実数ケースのCSVは全項目が旧値と同じ。複素ケースは報告energyの実部、実数EW統計と実数ログを検証する。
旧コードはg2c_validation/*-before.txtに保存し、旧数値snapshotは上書きしない。
check_g2b_downstreamも現在は正常な実推論報告を要求する。G2b当時の不具合再現ソース/記録は保存したまま。
raw make_loss_stepは今も複素値を返す契約で、直接%fへ渡す試験は境界違反を検出するだけ。実trainerの報告は修正済み。
locstd/EW統計/標準誤差の定義はCONTRACTS末尾。記録はg2c_validation/、結果はRESULTS先頭。

## G2b：通常PBCの複素局所エネルギー

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_g2_local_energy --report NEW_G2B_PHYSICS.json
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 .venv-pw/bin/python -m tests.check_g2_local_energy --device-probe --report NEW_G2B_DEVICE.json
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_g2b_downstream --report NEW_G2B_DOWNSTREAM.json
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_g2b_snapshots tests/g1_validation/snapshot-float64-g1-v1 NEW_G2B_FLOAT64_DIRECTORY
JAX_PLATFORMS=cpu PW_TEST_X64=0 PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.check_g2b_snapshots tests/g1_validation/snapshot-float32-g1-v1 NEW_G2B_FLOAT32_DIRECTORY
```

2deviceの指定は隔離テストだけ。通常のProduction設定・依存は変更しない。
PBC/default・PBC/folxの複素値を同じ独立式で検証し、G1へ渡す勾配も厳密な物理基準と比較する。
新しい基準はg2b_validation/。旧G1/G2/CPU snapshotは上書きしない。
check_g2_local_energy.pyのG2時点の旧コードはg2b_validation/check_g2_local_energy.py-before.txt、旧測定値はg2_validation/に保存。
定数出力の期待ERRORは正常性のPASSではない。K03は修正済みだがK10・定数出力などは残る。
G2b当時のR08はG2cで報告境界を修正した。現在のdownstreamテストはログエラーがないこと・実数CSVを要求する。

比較は全項目/構造/shapeを保持し、PBC3経路のエネルギー・VMC勾配・更新だけを旧G1値から分ける。
許すdtype変更はE_L/energy auxのfloat32→complex64、float64→complex128だけ。パラメータ/stateの複素化は許さない。
lambda/HFの全値と、他経路の波動関数・微分・初期化・MCMC・11 Attention設定は厳密比較する。
旧CPU基準を使う既存frozenメソッドはg2b-legacy区分でG1の勾配修正も考慮する。
将来G2bの**全**配列と比較する場合は通常のtests.baseline compare G2B_DIRECTORY --exactを使う。
G2b後の既存33件は30 PASS / 0 XFAIL / 1 FAIL / 2 ERROR。記録と制約はRESULTS先頭。

## G2調査の履歴

G2の本体未修正時の180記録・6 device条件はg2_validation/へ保持する。
現在のcheck_g2_local_energyはG2bの正しい複素値を要求するため、旧実部化を成功として検査しない。

## G1：修正後のVMC主勾配

Production本体の変更はloss.pyだけ。主勾配の係数・共役・中心化を独立二状態模型で検証する。
完全和/AD/FD/共分散の参照、clippingのバイアス、補助JVP(K10)の残存問題はCONTRACTS末尾を参照。

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_g1_vmc --report NEW_G1_PHYSICS.json
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 .venv-pw/bin/python -m tests.check_g1_vmc --device-probe --report NEW_G1_TWO_DEVICE.json
```

2device用XLA_FLAGSは隔離テスト専用。通常Productionの設定は変更しない。
G1の保持対象は両精度の波動関数・軌道・座標微分/Hessian・初期params・既存エネルギー・MCMC・Attentionで、旧snapshotと厳密一致を要求する。
旧VMC勾配・更新後params/state/checkpoint/統計は一致対象から明示的に分け、項目名と差の有無を記録する。
全項目のtree/shape/dtype/実効設定の比較は維持する。許容誤差で旧不具合への不一致を隠さない。
新しい参照はg1_validation/physics-final.json（独立物理基準）とsnapshot-*-g1-v1/。
旧基準はcpu_validation/に残る。将来G1の**全**配列と比較するときは従来の`tests.baseline compare G1_DIRECTORY --exact`を使用する。
新規保存先のみ許可し、自動更新はしない。既存33件はG1で29 PASS / 1 XFAIL / 1 FAIL / 2 ERROR。
修正前の旧コード・旧テストはg1_validation/*-before.txt、旧挙動の再現は次節。

## VMC勾配：独立した物理検証と現行不具合の再現

この節はG1前の原因調査の再現用。CurrentBehaviorは明示的に保存した旧loss.pyを読み、Productionモジュールを変更しない。
ExactPhysics（2件）とCurrentBehavior（6件）は分離する。CurrentBehaviorのPASSは不具合の再現成功であり物理的正しさを意味しない。

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.check_vmc_gradients --report NEW_SINGLE_REPORT.json
# 以下のXLA_FLAGSは隔離した検証プロセスのみ。通常Productionの設定を変更しない。
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 .venv-pw/bin/python -m tests.check_vmc_gradients --device-probe --report NEW_TWO_DEVICE_REPORT.json
JAX_PLATFORMS=cpu .venv-pw/bin/python -m unittest tests.test_vmc_mcmc.VMCEstimator -v
```

JSONはopen('x')で保存し、既存結果を上書きしない。最後の既存7件はG1で修正対象がPASSになった。
通常のunittestで新モジュール全体を実行すると、単一deviceではDeviceAggregationの1件がSKIPになる。
上の専用CLIは単一device 8件と隔離2device 1件を分けて実行する。実GPU/multi-hostは未検証。
結果はRESULTSのVMC欄、厳密式・原因・最小修正案はCONTRACTS末尾、保存データはvmc_gradient_validation/。
最新単一device計測はsingle-cpu-final.json（8件）、初回single-cpu.json（7件）は調査履歴。

## F3：通常軌道構成の移動

通常factoryの実装はwavefunction.make_orbitals、旧networks.make_orbitalsは直接再export。
HF/通常Jastrow/Envelope/行列式reducer/Attentionはそれぞれ独立のまま保持する。

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m unittest tests.check_wavefunction_construction tests.check_interfaces -v
```

追加3件はimport順・呼出しsignature・数値本体ASTと、移動前factoryの初期化/PyTree、
軌道・行列式・位相/log振幅・Jastrow前後、座標/パラメータ微分とHessian・JITを両精度で厳密比較する。
加えてfloat64で実行可能なPsiformer/SlaterNetのKFAC 1ステップと復元state/data/params/CSVを旧factoryと厳密比較する。
非周期isotropic/PBC nullを使い、I01の既知不整合はF0再現テストとF2bの8 probeで別に保持する。
固定snapshotは更新しない。記録はf3_validation/、最新の完走結果はRESULTSのF3欄。

## F1：学習可能Embeddingの共通化

3 Ansatzの初期化・concat→dotは既存network_blocks.pyの2関数で共有する。
周期幾何・特徴・固定Value・Transformer・Diagnostic APIは変更しない。
変更前factoryはf1_validation/*-before.py.txtへ保存し、import/folx登録を再実行せずfactoryだけをテストで読む。

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m unittest tests.check_embedding tests.check_interfaces -v
```

F1追加3メソッドは両精度のweight初期化、裸のPyTree、下流を含むRNG割当、
通常/JIT出力、座標勾配/Hessian、パラメータ勾配、Embedding単体のJAXPR/HLOを厳密比較する。
PBC・電子交換はF0の独立テストも再実行する。既存snapshotは更新しない。
実測結果・実行コマンドは[RESULTS.md](RESULTS.md)のF1欄、記録はf1_validation/。

## F0：4領域のインターフェースと不足テスト

物理定義は新しい文書へ分散させず、[CONTRACTS.md](CONTRACTS.md)のF0欄に集約した。
周期幾何→特徴→Embedding、全電子軌道/Jastrow/HF、モデル別Hamiltonianと単位、
最優先の位相VMC勾配の独立調査計画を記載する。本体と主要3解析入口は変更しない。

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m unittest tests.check_interfaces -v
```

- `CurrentContracts`：現在のshape、RNG/tree、軌道packing、merged Jastrow、節の挙動を保持する。
- `IndependentPhysics`：解析微分、特徴共変性/PBC、行列式、HF補正、lambda演算子、Ewald、モアレを独立式で検算。
- `KnownInconsistency`：PBC isotropic Envelopeのshapeエラーを再現。PASSはその物理経路の正しさを意味しない。

`check_interfaces.py`は明示実行し、既存33件のrunnerへ混ぜない。新旧の失敗を別に報告する。
両精度を直接検査する項目は幾何/特徴/Embedding。残るF0独立式はfloat64で実行し、
全モデルのfloat32/float64は既存固定snapshotでも厳密検査する。GPU/実研究checkpointは未検証。
既存33件の基準は27 PASS/2 XFAIL/2 FAIL/2 ERROR。rawのsnapshot SKIPは別プロセスで指定実行して区別する。
最新の結果は[RESULTS.md](RESULTS.md)、記録はf0_validation/。

## E2a/E2b：Head寄与と相殺

head_contrib/head_cancelは正式Diagnosticを共用する。z=gate_output、c_h=exp(out_scale)*z_h W_h。
共有biasを分離し、Head和を次層へ戻さない。Cpは非線形な自己応答との差で、純粋な二体/Head除去/エネルギー寄与ではない。
head_cancelのrhoは平均二乗の比で、head_contribの平均ノルム指標とは別。分母0は未定義として保存する。
主要3入口の新規保存名は`attn_probe_diagnostic_v1.npz`、`head_contrib_diagnostic_v1.npz`、`head_cancel_diagnostic_v1.npz`。
旧cache・新しい既存ファイルも上書きしない。settings/元精度latticeが必要で、softmax解析はこれらのScreenformer入口では未対応。
詳細なCLIと旧比較はRESULTSのE1/E2a/E2b欄を参照する。

## E1：attn_probeの移行

`screening_analysis/attn_probe.py` は正式な独立Diagnostic APIを使う。
`W`は`aggregation_weights`、`W_window`はalpha×Window、`W_intervened`は介入後・固定Value対角除去前。
これらと`S`/`Alpha`/`Window`は[B,L,H,N,N]、`d`は[B,N,N]。
行和正規化は行わない。Windowなしでは`Window`配列はなく、幅の表示値はNaN。

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python screening_analysis/attn_probe.py RUN_FOLDER \
  --settings KNOWN_RUNTIME_SETTINGS.json \
  --lattice '[[3.2,-1.6],[0.0,2.7712812921102037]]' --nconf 16
```

格子例を無関係なrunへ流用しないこと。settingsファイルは、そのrunについて確認した
`TRIM_MAX`, `TRIM_POW`, `WIN_SHAPE`, `FREEZE_W`, `CLIP_R`, `CLIP_BETA`,
`jax_enable_x64`, `jax_default_matmul_precision`のflat JSON。
通常のcheckpointはこれらを保存していないため、現在の環境やfolder名から推測しない。
configに`screening_attention_settings`の記録があれば使用し、明示値と矛盾すれば停止する。
旧writerで文字列化された格子も元の精度を復元できないため、元の共通格子を明示する。
構造化配列がconfigに残っている場合はその値を使う。独立したfeature格子を持つ旧文字列configは
共通格子指定では復元せず、元精度の構造化configが必要となる。

ライブラリ呼出しは`probe(folder, settings=known_settings, lattice=known_lattice)`。
返却値のmetadataに手法名、Weight定義、実効設定と指定元、dtype、checkpoint/config/設定ファイル/
実行ソースのhash、使用walker indexを記録する。元のcheckpointの全deviceのwalkerを対象にする。
元の設定が未記録であれば、明示指定値は利用者による当時の値の申告であり、checkpointからの自動復元とは主張しない。

保存先は`attn_probe_diagnostic_v1.npz`。NPZには`metadata_json`も含み、`allow_pickle=False`で読める。
既存ファイルと旧名`attn_probe.npz`は上書きしない。別名は`--output`で指定する。
一時的なCSグローバル設定とJAX精度は処理後・例外時に元へ戻す。同一プロセスでProductionのtraceと並行実行しない。
他の解析の移行は未実施で、旧呼出しに未保存設定があれば明示エラーになる。旧キャッシュは保持する。

E1の直接検証：

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m unittest tests.check_attn_probe -v
```

数値差と既存33件のstatus変更は[RESULTS.md](RESULTS.md)を参照。

## 正式な多層Screenformer診断API（段階D）

通常のNetwork.applyと診断は**別々にJITし、別々に呼ぶ**。診断値は同じソースの再評価であり、
Production実行内部のビット列ではない。学習・VMC・optimizerにはProductionだけを使う。

```python
import jax
from periodicwave import CustomScreenformer as CS

production = jax.jit(net.apply)
diagnostic = jax.jit(CS.make_screenformer_diagnostic(nspins, net.options))
args = (params, positions, spins, atoms, charges)  # checkpointのPyTreeをそのまま使用
phase, logabs = production(*args)
x, layers = diagnostic(*args)                  # 単一walker、x.shape == (N, E)
w_layer1 = layers[1]['attention']['aggregation_weights']

# 状態検査も診断forwardとは別に実行。追加のSVDでforwardの融合を変えない。
state = CS.screening_diagnostic_status(x)
# state['feature_rank_deficient'] は特徴のランク警告。
# 軌道・波動関数を渡さない場合、wavefunction_unstable_or_undefinedはNone（未検査）。
orbitals = net.orbitals(*args)  # 追加の独立評価。診断Xの軌道を返すAPIではない。
state = CS.screening_diagnostic_status(x, orbitals=orbitals, phase_logabs=(phase, logabs))
```

層はtupleのindexで参照する。境界配列はすべて[N,E]。

| キー | 数式・位置 |
|---|---|
| `input` | 層入力X |
| `attention` | B2 return_aux辞書。下表参照 |
| `attention_output` | Attentionの出力倍率適用後、残差へ加える値 |
| `attention_residual` | X + Attention(X)、最初のLNの入力 |
| `ln1_output` | 最初のLN後、MLPの入力 |
| `mlp_output` | MLP(ln1_output)、残差加算前 |
| `mlp_residual` | ln1_output + mlp_output、2回目のLNの入力 |
| `ln2_output`, `output` | 2回目のLN後、次層へ渡すX（同じ配列） |

LNなしではLN前後は恒等。`layers[i]['output']` がそのまま次層の `input` になる。

| `attention` 内の主なキー | shape・定義 |
|---|---|
| `q`, `k`, `q_normalized`, `k_normalized` | [N,H,K]、射影直後と正規化後 |
| `v`, `v_normalized`, `v_effective` | [N,H,D]、射影・正規化・介入後。固定ValueではNone |
| `similarity`, `alpha` | [H,N,N]、正規化Q/Kの内積とTrim係数 |
| `window` | [H,N,N]、距離Window。WindowなしではNone |
| `weights_window` | [H,N,N]、alpha×Window（Windowなしではalpha） |
| `weights`, `aggregation_weights` | [H,N,N]、介入後、さらに固定Valueの対角除去を反映した実集約重み |
| `pair_features` | 固定Valueの[N,N,P]特徴。通常ValueではNone |
| `aggregate`, `tanh_output` | [N,H,D]（固定ValueではD=P）、集約直後とTanhNorm後 |
| `gate_output` | [N,H*D]、head結合・Gate後 |
| `projected_output` | [N,E]、出力射影後、倍率適用前 |

通常のAttention値は行和で正規化しない。各式・演算順・stop_gradientは本体と共有する。
設定グローバルは従来どおり初回trace時に読まれる。同じ設定で両側を構築・traceすること。
設定変更後に既存JITキャッシュが自動で無効になるとは限らない。

`screening_diagnostic_status` は別評価のランク警告器であり、値や勾配を修正しない。
4電子・幅4・初期最終LN・null envelope・軌道biasなしでは数学的に行列式がゼロになる。
この場合は波動関数のlog微分・局所エネルギーを定量診断へ使わず、Attentionの有限な内部量と分ける。
軌道を渡すと各行列のrcond、`singular_determinants`、`all_determinants_rank_deficient` を返す。
`wavefunction_unstable_or_undefined`／`local_energy_unstable_or_undefined` のTrueは警告、Falseは精度保証ではない。
full-rank軌道間の行列式の打消しなど、すべての節をこのランク検査で検出できるわけではない。
特徴ランクだけでは非一様なenvelope適用後の軌道ランクを断定しない。

診断の誤差評価と実行コストは [RESULTS.md](RESULTS.md) 冒頭を参照。
単一walkerの診断関数なので、batch解析では通常のjax.vmapを別途使用する。

本体の数式・初期化・アルゴリズムは変更しない。既存の結果、checkpoint、HF表は読み書きしない。
テストの書込みは `TemporaryDirectory`、数値基準の作成は明示した**新規ディレクトリ**だけ。
以前の実行方法と物理系の説明は [README_PeriodicWave.md](../README_PeriodicWave.md) を参照。

## 現在の到達点

- ソース基準：commit `038730ef3bd06e738356e2a6f8d8b0adbb3e77cc`。
- `source_baseline.json` に現行170 PythonファイルのSHA-256、Python、依存バージョン、実行に関係する環境変数を記録。
- **数値基準を取得済み。** 独立CPU環境 `.venv-pw`（Python 3.13.14）で実行した最新結果は `RESULTS.md` 冒頭を参照。
- 実行結果は [RESULTS.md](RESULTS.md)、既知問題と保存すべき仕様は [CONTRACTS.md](CONTRACTS.md)。
- 静的検査が通っても、数値検証が完了したとはみなさない。

独立環境は元READMEのApple CPU用freezeを固定した `requirements-cpu.txt` で再構築できる。

```sh
uv venv --python 3.13.14 .venv-pw
uv pip install --python .venv-pw/bin/python -r tests/requirements-cpu.txt
```

既存の固定基準と比較（captureは基準を新設するときだけ使用し、自動更新しない）：

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.baseline compare tests/cpu_validation/snapshot-v1 --exact
JAX_PLATFORMS=cpu PW_TEST_X64=0 PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.baseline compare tests/cpu_validation/snapshot-float32-v1 --exact
```

数値33メソッドの再実行は、次のように新規レポートへ記録する。

```sh
JAX_PLATFORMS=cpu PW_RUN_KFAC=1 \
  .venv-pw/bin/python -m tests.run_numerical /tmp/pw-new-report.json
JAX_PLATFORMS=cpu PW_BASELINE_DIR=tests/cpu_validation/snapshot-v1 \
  .venv-pw/bin/python -m unittest tests.test_wavefunction.WavefunctionChecks.test_frozen_numerical_baseline -v
```

最初の全体実行ではfrozen比較1件だけSKIPとなり、2番目のコマンドで実行する。
基準採取時のKFAC有無を合わせるため、snapshot-v1の比較は `PW_RUN_KFAC` なしの別プロセスにする。
KFACを含むfloat32基準とも混ぜない。倍精度KFAC自体には未解決ERRORがある。

## 実行

追加のテストフレームワークは不要。リポジトリのルートから実行する。

```sh
.venv-pw/bin/python -m unittest discover -s tests -v
```

数値33メソッドの結果を個別JSONへ記録する場合（既存ファイルは上書きしない）：

```sh
PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.run_numerical /tmp/pw-numerical-results.json
```

依存不足でもメソッドごとのSKIPを記録して終了コード2で停止する。
環境を再調査した最新状況は `RESULTS.md` 冒頭を参照。

数値テストは本体が動く既存環境（Python 3.10以上、元のREADMEでは3.13）で実行する。
ライブラリを勝手に更新せず、実際に研究計算に使用しているJAX・folx・KFAC等の組合せを使う。
パッケージが存在するのにimport/API互換性で失敗した場合は、スキップせずエラーになる。

```sh
JAX_PLATFORMS=cpu PW_WIN_SHAPE=cos PW_TRIM_POW=2 PW_FREEZE_W=0 \
  .venv-pw/bin/python -m unittest discover -s tests -v
```

すべての追加テストは小系で、CPUで実行できる設計。GPU必須の追加テストはない。
KFACの1 step統合と既存の長い検算は明示的に追加する。

```sh
JAX_PLATFORMS=cpu PW_RUN_KFAC=1 PW_RUN_LEGACY=1 \
  .venv-pw/bin/python -m unittest discover -s tests -v
```

`PW_RUN_KFAC` / `PW_RUN_LEGACY` / `PW_BASELINE_DIR` はテストだけの設定。
小系CPU試験は、実際の大batch、多GPU、multi-host、GPU固有カーネルの検証を代替しない。

## 数値基準の保存と比較

現行コードの移動には**既存の固定基準をそのまま使う**。次が標準の厳密比較であり、自動captureし直さない。

```sh
JAX_PLATFORMS=cpu .venv-pw/bin/python -m tests.baseline compare tests/cpu_validation/snapshot-v1 --exact
JAX_PLATFORMS=cpu PW_TEST_X64=0 PW_RUN_KFAC=1 .venv-pw/bin/python -m tests.baseline compare tests/cpu_validation/snapshot-float32-v1 --exact
```

以下のcaptureは新しいモデル/環境の基準を**別に**採取する場合の手順。F0では実施しない。
例はfloat32/KFACあり。float64の全経路KFACには既知ERRORがあり、float64既存基準はKFACなし。
`/tmp/pw-before-refactor` は未存在の保存先に置き換える。

```sh
JAX_PLATFORMS=cpu PW_TEST_X64=0 PW_WIN_SHAPE=cos PW_TRIM_POW=2 PW_FREEZE_W=0 PW_RUN_KFAC=1 \
  .venv-pw/bin/python -m tests.baseline capture /tmp/pw-before-refactor
```

`values.npz` と `metadata.json` を新規保存する。既存ディレクトリは上書きしない。
KFACを含めない場合は `PW_RUN_KFAC=1` を外す。ただし標準学習はKFACなので、その基準も取得すること。
依存不足は終了コード2で停止し、数値ファイルを作成しない。

変更後、同じ依存・環境変数・backendで比較する。

```sh
JAX_PLATFORMS=cpu PW_TEST_X64=0 PW_WIN_SHAPE=cos PW_TRIM_POW=2 PW_FREEZE_W=0 PW_RUN_KFAC=1 \
  .venv-pw/bin/python -m tests.baseline compare /tmp/pw-before-refactor
```

単なる関数移動では、同じコマンドに `--exact` を加えてまず厳密な要素一致を確認する。
複素数は実部・虚部を別々に検証する。

通常のテストにも比較を加えられる。

```sh
JAX_PLATFORMS=cpu PW_TEST_X64=0 PW_WIN_SHAPE=cos PW_TRIM_POW=2 PW_FREEZE_W=0 PW_RUN_KFAC=1 \
  PW_BASELINE_DIR=/tmp/pw-before-refactor .venv-pw/bin/python -m unittest discover -s tests -v
```

基準を変更後のコードで自動再生成してはいけない。旧版からの基準と比較して初めて同等性を検証できる。
ソースhashは由来の記録であり、リファクタリング後も全ファイルのhashが一致することは要求しない。
項目・木構造・shape・dtype・実効設定を照合する。Productionの移動は`--exact`を必須とし、
比較関数の近似既定値`rtol=1e-8, atol=1e-9`へ落として通さない。
値にNaN/infがあれば保存を止める。閾値を緩めて失敗を消す前に、対象項目・差・条件を調べる。
GPUや異なる依存への比較は別検証であり、この厳密な同等性比較に混ぜない。

数値基準には以下を含む。

- 5経路の初期パラメータ、入力、格子、主要な設定、固定key。
- ψ、位相角、log|ψ|、軌道行列。
- 位相・log|ψ|の座標勾配と全Hessian。
- 局所エネルギー（複素数をそのまま）、loss、付随量、VMCパラメータ勾配。
- MCMC後の配置・スピン・受理率、Adamの1更新後のパラメータ・状態。
- softmax、余弦、smooth7、Trim⁴、帯窓、sin/gauss輪切り、固定value、mask/gate/TanhNorm無効のAttention内部量。
- 実際のtrainerでMCMC→Adam更新→保存を1 step実行したcheckpoint内容とCSV数値。
- 合成HF表。`PW_RUN_KFAC=1` の場合は実際のtrainerによるKFAC 1 step後のcheckpoint内容とCSV数値も含む。

HFケースの表はG=0の2本のBloch平面波。HF自己無撞着解を求めた研究データではない。
HF表読込・スピノル・NN補正・Jastrow・局所エネルギーの接続を検証するもので、実際の結晶解やChern数の検証は含まない。

## テストの区分

| ファイル | 区分・対象 |
|---|---|
| `test_reproducibility.py` | 依存不要の静的契約。入口、seed、保存形式、既知の引数不整合、基準の上書き防止 |
| `test_wavefunction.py` | 軌道・行列式の一致、反対称性、1電子ずつのPBC、1/2階微分、HF平面波、保存済み数値基準との比較 |
| `test_attention.py` | 本体と独立したNumPy式の一致。内部量、Gaussian輪切りの恒等式 |
| `test_known_attention.py` | 実際の `attn_probe.probe()` を合成checkpointで実行。E1以後のTrim⁴/smooth7一致2件はPASS |
| `test_physics.py` | 独立解析式。複素Gaussian/平面波の運動エネルギー、λ-jelliumバンド、λ=0、Ewaldの周期性・線形性 |
| `test_vmc_mcmc.py` | 二状態の規格化された期待値の微分、位相勾配、chunk/clipping、明示的MH、両trainerのAdam 1更新 |
| `test_training_checkpoint.py` | 実際の5経路の学習1 step、checkpoint復元、設定の配列保存 |
| `test_legacy.py` | 既存3検算を別プロセス・一時出力先で実行（長いのでopt-in） |

`test_physics.py` は旧値に合わせる試験ではなく、数式に照らした独立の検算。
一方、修正前のVMC勾配1/−4とPBC folxの虚部欠落は保存済み旧版/旧数値として保持する。現在の本体に旧不具合を要求しない。
この挙動を物理的に正しいと認定しているわけではない。

一部の既知問題の望ましい振る舞いは`unittest.expectedFailure`で明記する。
G1で振幅/位相VMCの既知問題は改善する。残る微分のFAILやfolx/KFACのERRORを一律XFAILへ変更しない。
`expected failure` は正常性の確認ではない。修正後の `unexpected success` も失敗扱いになるため、
問題を修正する別変更で、旧挙動の試験と期待失敗の指定を見直す。

既存 `test_gauss_rings.py` の旧版比較は参照ソース未指定のため省略される。
現在の比較には保存済みfrozen baselineを使用する。旧テストの連続性試験が作るNPZは、一時領域内に保存される。
