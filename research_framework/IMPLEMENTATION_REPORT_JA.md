# 実装報告: Screening研究の共通比較基盤

実装・検証日: 2026-11-05〜2026-11-06 (Asia/Tokyo)。既存Screening_topology_gpu内に研究packageを追加した。
元のリポジトリには作業中の変更が多数あったため、調査開始時の状態・対象source hash・CustomScreenformerの全文を `results/implementation_audit/` に保存した。
元のcheckpoint、以前の解析結果、既存Slurmジョブは変更していない。

## 納品と実行範囲

主pilot73条件: 連続電子23、TFIM30、QWZ20。
学習可能Fourier候補に結晶shellも含める追加対照6条件を別planで実行した。計79条件。
主pilotの最初の失敗試行も保存し、同一configの別フォルダで再試行した。`retries.json` が対応を記録する。
N=12/16の長期TFIM6条件と、元の実格子・HF表を参照するN36の連続系1条件は、production計画だけで未実行。
新規のGPU/Slurmジョブはsubmitしていない。CPU専用仮想環境を追加し、既存GPU環境を変更していない。

これは以前の8万更新済みN36モデルを改良して再学習した結果ではない。
今回の小規模モデルと、以前の物理相候補を混同しない。

## 再利用と新規実装

| 領域 | 再利用した実装 | 新規の差分 |
|---|---|---|
| 連続電子の特徴 | periodicwave/pbc/feature_layer.py、network_blocks.py | 整数結晶逆格子、one-hot、学習Fourier、複素carrierの実2channel |
| 連続電子のAttention | CustomScreenformer.make_screening_attention、attention.screening_attention/window | factoryの差し替え口、固定/共有/各headのbounded Window、局所性cost |
| 軌道・反対称性 | wavefunction.make_orbitals、lj_orbitals.make_lj_orbitals、logdet_matmul | 組立adapter、振幅/位相API、twist条件付けと準周期位相 |
| 連続電子のH・MC | lambda_jellium.local_energy、mcmc.mh_update/mh_spin_flip | 小規模VMC driver、保存/再開、介入後のfresh Born評価 |
| TFIM | Lattice_screening/lattice/model.py、physics.py、vmc.py、window_research.py | 設定adapter、厳密Born学習、Born MI・Quantum MI、共通介入・保存 |
| QWZ物理 | wannier_ph.geometry、canonical_atlas、cylinder_frame | 新しい実数/複素/chargedモデルを共通設定へ接続 |
| QWZ学習 | 前研究のProjector/atlasという教師の分離原則 | 位相を保つembedding、native complex、輸送を含む共変Attention、rank1出力 |
| PH | screening_analysis.topology.persistence、Ripser、既存phase_lift | task別接続、PH workerのプロセス隔離、同一checkpoint比較 |
| 実験管理 | 既存のcheckpoint構造・解析の由来を調査 | strict config、3 adapter、run store、source snapshot、入力hash、再試行ledger |

PH、Hamiltonian、周期距離、行列式、Jastrow、TFIMのED/CMIを別の数式で書き直していない。
QWZのGauge-covariant blockは既存コードになかったため新規実装した。

## 元のコードへの変更

既存ファイルで編集したのは `periodicwave/CustomScreenformer.py` だけ。
`input_embedding=None` と `attention_factory=None` を追加し、既定値では以前の初期化・forwardをそのまま通す。
差し替え口はoptions→trunk→attention blockへ渡す。通常forwardの返値や既存parameter keyは変えていない。

既存attention.py、network_blocks.py、lj_orbitals.py、wavefunction.py、feature_layer.py、lambda_jellium.py、hamiltonian.pyは調査開始時と同じSHA256。
既存Lattice_screeningとwannier_phのsourceも編集していない。
今回だけのpatchは `results/implementation_audit/framework_only.patch`。以前のユーザーの変更を含むgit diffとは分けた。
旧factoryとの数値照合には調査開始時の保存sourceを使い、旧snapshotを再生成していない。

## 数値検証

- 初期parameter、波動関数、座標勾配、Hessianが旧factoryと一致。
- baselineの位置4成分＋scalar spinが従来の入力と一致。
- 結晶逆格子はL=A Sから導出。S=diag(6,6)で指定の6θ、一般Sで周期性を維持。
- Fourier値・座標微分、one-hot対応、粒子交換共変性、Windowの有限差分勾配・tail・保存復元を検査。
- T_ijのU(1)変換則、複素Value輸送、出力射影・非線形・残差を含むゲージ共変性、零overlapのmaskを検査。
- 連続系のフェルミオン交換符号、twist境界位相、既知plane-waveの局所運動エネルギー、overlap求積状態の正規化を検査。
- ProjectorのHermiticity・冪等性・rank、QWZのC=0,±1、ゲージ不変性、共役符号反転を検査。
- 既存phase-liftのH1 plateauと、相対尺度を変えた際の失敗を両方検査。
- TFIMの局所EとED演算子期待値、Fidelity/S2の既存実装、Z2パリティ、Born CMI/MIとQuantum MIの区別を検査。
- TFIMとTorchの中断再開が、非中断計算とparameter/RNG/出力まで一致することを検査。

最終テストは35件PASS。ログは `results/implementation_audit/pytest_release.txt`。
初期の失敗ログも削除していない。テストPASSは数学的なPH再構成定理や物理相の認定を意味しない。

## 実装中の修正と再解析

1. TFIMのsigmoid幅2.0を既存整数radius欄へ渡していた不備を修正。soft Windowでは未使用のhard radiusをFullとし、元のwindow_initへ実数幅を渡した。失敗した12試行を保持して再試行し、regression testを追加した。
2. real Slaterに存在しない「NN虚部除去」を実行して行列式を特異にする不備を修正。実数Ansatzではその介入を拒否し、適用可能な介入だけで再試行した。
3. twist overlap helperの構文不備を修正。境界/局所E/求積正規化をテストした上で同一configを再試行した。
4. Full Attentionに対する幅介入がFull窓を残していた不備を修正。20個のQWZ checkpointの通常出力が変わらないことを照合し、再学習せず幅介入だけを再評価した。元の診断を保持し、補正後を `interventions_width_corrected.json` と `reports_corrected/` に保存した。
5. PHは元の小規模workerがforkを使うため、JAX初期化後の多threadプロセスでは別Pythonプロセスへ隔離した。PHの距離・係数・数学的定義は変更していない。

修正前後で計算の内容が変わるものを黙って同じ結果として扱わない。各runのsource snapshot、失敗ログ、再試行先、幅介入の再解析provenanceを保持する。

## 実装済みと検証範囲の境界

- Gauge-covariant構成では物理Berryリンクを入力する。リンクの発見やトポロジーの自発学習とは呼ばない。
- HF表を与えないpilotは合成plane-wave Slater。`pure_hf`フラグはNN補正なし/Jastrowなしを意味し、自己無撞着HF最適解を意味しない。
- 連続系は規格化前のlog波動関数を扱い、期待値はBorn分布によって規格化する。一般の連続多体系の規格化定数Zを厳密に求める機能はない。
- 新しい形式へ既存N36 checkpointを自動変換してwarm-startする機能は未実装。既存の正式loaderは保持し、新形式のcheckpointは同一config/sourceで再利用・再開できる。
- TFIMは既存のpositive real Ansatz・site embeddingに限定。一般の複素TFIM Ansatzや別site embeddingの実験は未実装。
- 連続電子のnative complex/covariant Attentionは未実装。複素/Gauge輸送比較は、その局所U(1)構造が定義されたQWZ adapterで行う。
- 連続電子のエンタングルメント、N>2の多体overlapの一般推定、全twistトーラスのsewingとgappedな固有状態追跡は未実装/未検証。
- Phase-sensitivity PHは連続系のlog辺倍率に対する波動関数位相微分で実施した。TFIMのpositive real位相は恒等0であり、QWZのGauge依存な局所位相の同じ微分PHは実装していない。QWZではBerry holonomyと通常内部PHを別に測定する。
- atlasの内部PHは全2Nスロットを対象とし、有効chart maskを別途保存している。無効chartの孤立ノードを除いた比較は未実施。このgraph PHを物理bundleのホモロジーと解釈しない。
- H2の追加大規模計算、有限点群からのWannier障害の普遍的検出、内部PHとChernの一対一対応は未検証。
- Stage Fの最良複合構成は未選択・未実行。短いMCや2seedだけで効果を確定しない。

## 報告

`COMPARISON_REPORT_JA.md` に実計算の条件・数値・制限を記載した。
研究上の問いへの実測回答と介入比較は `results/2026-11-06_pilot/reports_corrected/RESEARCH_FINDINGS_JA.md`。
79条件の履歴・入力・内部量・checkpointを保存し、大型7条件は計画のみと明記した。
