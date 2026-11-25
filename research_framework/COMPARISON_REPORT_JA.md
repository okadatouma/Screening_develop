# Embedding・Screening・Wavefunction比較: 実行結果

計画 73 条件、完了 73 条件。失敗/途中/未実行はcomparison.csvに残した。
この報告の数値表・図はCOMPLETE.jsonのある実計算だけから作成している。
小さいCPU pilotであり、WC/AHC認定、熱力学極限、一般的な最良モデルの決定は行わない。
既存の長時間計算や以前のQWZ/Wannier結果を上書き・置換していない。

## 同条件比較の定義

Hamiltonian、N、境界twist、seed、step、optimizer、sampling設定をconfig.jsonに保存した。
task間は異なるHilbert空間であり、エネルギーやPHの値を横断して同じ量として比較しない。
連続系では自然な入力次元の比較に加え、埋め込み入力を12成分へ揃えた比較を実施した。
後者ではbaseline側に能動的feature gainを置き、learned Fourier振幅と名目パラメータ数も合わせる。
ただしパラメータ数の一致は表現能力や最適化条件の完全一致を意味しない。
TFIMではR=4とFullはN=8のring上で同じsupportであり、独立な物理的範囲とは数えない。
head数の比較ではheads×head_dimを揃えるが、headごとのTrim/幅のパラメータ数は変わる。
深さ比較は自然な容量の比較で、容量差を除去した因果比較ではない。

## 連続電子系

Screenformerのcosine QK、Trim、Window、分母なしValue集約、TanhNorm、Gate、射影を再利用した。
通常のsoftmax Attentionへ置き換えていない。以下のEはfresh Bornサンプリングによる推定。
MC誤差・学習不足が大きく、単一seedの小さい差を改善と認定しない。
非相互作用pilotの参照は合成plane-wave表であり、自己無撞着HF解とは呼ばない。
Coulomb/Jastrow比較は同じCoulomb Hamiltonianの独立groupで行う。
twistあり条件は境界条件の検証用であり、θ=0条件とのE差をAnsatz改善効果とは扱わない。

| run | seed | E | block SEM | parameter数 | Jastrow |
|---|---|---|---|---|---|
| electron_baseline | 101 | 1.1168 | 0.10334 | 341 | False |
| electron_six_theta | 101 | 1.2349 | 0.13673 | 389 | False |
| electron_spin_onehot | 101 | 0.66263 | 0.41021 | 349 | False |
| electron_six_theta_onehot | 101 | 1.2808 | 0.1249 | 397 | False |
| electron_learned_fourier | 101 | 1.6958 | 0.36985 | 392 | False |
| electron_baseline | 202 | 1.2825 | 0.068772 | 341 | False |
| electron_six_theta | 202 | 1.3658 | 0.094422 | 389 | False |
| electron_spin_onehot | 202 | 1.4993 | 0.21157 | 349 | False |
| electron_six_theta_onehot | 202 | 1.8621 | 0.41422 | 397 | False |
| electron_learned_fourier | 202 | 1.3855 | 0.060179 | 392 | False |
| electron_padded_baseline | 101 | 1.1548 | 0.058543 | 400 | False |
| electron_padded_six_theta_onehot | 101 | 1.497 | 0.18873 | 400 | False |
| electron_padded_learned_fourier | 101 | 0.96391 | 0.30359 | 400 | False |
| electron_trainable | 101 | 1.1185 | 0.1023 | 343 | False |
| electron_locality | 101 | 1.1189 | 0.10194 | 343 | False |
| electron_hf | 101 | 1.2337 | 4.1962e-17 | 341 | False |
| electron_complex_slater | 101 | 4.2121 | 3.8593 | 309 | False |
| electron_real_slater | 101 | -1.0521 | 5.1536 | 293 | False |
| electron_twist | 101 | 1.0679 | 0.071131 | 357 | False |
| electron_coulomb_HF | 101 | 0.010295 | 0.017305 | 341 | False |
| electron_coulomb_HF_J | 101 | 0.01145 | 0.01684 | 344 | True |
| electron_coulomb_HF_NN | 101 | -0.11841 | 0.095343 | 341 | False |
| electron_coulomb_HF_NN_J | 101 | -0.13519 | 0.098203 | 344 | True |

Fourier波数は整数逆格子候補だけを用いる。結晶モードはL=A Sの整数Sから導出するので、6×6以外へ固定6を持ち込まない。
初期・学習後のQK/Screening Fourier、同異spin選択性、Spin×Fourier、学習振幅、Born構造因子を保存した。
振幅には即時のa_q W_q再スケール自由度を抑える入力射影の正規化を使うが、後段との冗長性は残る。
振幅の大きさだけを物理秩序の自発的発見とは解釈しない。
uniform probeの構造因子とBorn状態の構造因子は別ファイル・別測定として扱う。
今回N=2の自由系はWC/AHC候補ではないので、one-hotがWC/AHC識別を改善したという検証にはなっていない。

## TFIM: 物理量とWindow

Hamiltonian・局所エネルギー・ED・S2・CMI・相関長・p_-は既存Lattice_screeningから再利用した。
小規模pilotは全配置の厳密なBorn重みでエネルギー勾配を計算し、有限鎖のEDと比較した。
損失にはパリティやFidelityを加えていない。通常Born MI、全残り変数で条件付けたBorn CMI、Quantum MIは別量。
相関長は有限周期鎖のsecond-momentまたは有限鎖fitであり、熱力学的な相関長ではない。

| run | seed | ΔE/(NJ) | Fidelity | p_- | S2 half | xi(second moment) | 幅 |
|---|---|---|---|---|---|---|---|
| tfim_g1.0_R1 | 101 | 0.0074813 | 0.88267 | 0.10136 | 0.25952 | 2.0578 | 1,1 |
| tfim_g1.0_R2 | 101 | 0.0013014 | 0.99712 | 0.00022572 | 0.45776 | 3.0624 | 2,2 |
| tfim_g1.0_R4 | 101 | 0.00051879 | 0.99932 | 0.00019095 | 0.48185 | 3.2948 | 4,4 |
| tfim_g1.0_Rfull | 101 | 0.00051879 | 0.99932 | 0.00019095 | 0.48185 | 3.2948 | 4,4 |
| tfim_g1.0_smooth_fixed | 101 | 0.0030259 | 0.99412 | 0.00020673 | 0.45229 | 2.9885 | 2,2 |
| tfim_g1.0_trainable | 101 | 0.0027932 | 0.99423 | 0.00023385 | 0.4504 | 2.9753 | 2.0261,1.6817 |
| tfim_g1.0_regularized | 101 | 0.002796 | 0.99417 | 0.00022015 | 0.44997 | 2.9699 | 2.0166,1.6431 |
| tfim_g1.5_R1 | 101 | 0.0010007 | 0.99802 | 1.9941e-06 | 0.13886 | 1.1857 | 1,1 |
| tfim_g1.5_R2 | 101 | 0.00024432 | 0.99967 | 6.3426e-05 | 0.14608 | 1.3137 | 2,2 |
| tfim_g1.5_R4 | 101 | 0.0001985 | 0.99979 | 6.1457e-05 | 0.1481 | 1.3428 | 4,4 |
| tfim_g1.5_Rfull | 101 | 0.0001985 | 0.99979 | 6.1457e-05 | 0.1481 | 1.3428 | 4,4 |
| tfim_g1.5_smooth_fixed | 101 | 0.00094619 | 0.99837 | 3.531e-05 | 0.14168 | 1.216 | 2,2 |
| tfim_g1.5_trainable | 101 | 0.00066866 | 0.99888 | 8.9752e-06 | 0.1435 | 1.2446 | 1.637,1.617 |
| tfim_g1.5_regularized | 101 | 0.00064784 | 0.9989 | 8.2901e-06 | 0.14348 | 1.2453 | 1.6289,1.5685 |
| tfim_g1.0_R1 | 202 | 0.0068472 | 0.96076 | 0.020857 | 0.37505 | 2.4494 | 1,1 |
| tfim_g1.0_R2 | 202 | 0.0011096 | 0.99756 | 0.00011384 | 0.45752 | 3.0789 | 2,2 |
| tfim_g1.0_R4 | 202 | 0.0011704 | 0.99836 | 0.00015702 | 0.47273 | 3.245 | 4,4 |
| tfim_g1.0_Rfull | 202 | 0.0011704 | 0.99836 | 0.00015702 | 0.47273 | 3.245 | 4,4 |
| tfim_g1.0_smooth_fixed | 202 | 0.0030607 | 0.99393 | 0.00021005 | 0.45196 | 2.9728 | 2,2 |
| tfim_g1.0_trainable | 202 | 0.0027377 | 0.99436 | 0.00016616 | 0.44858 | 2.9723 | 1.634,2.0805 |
| tfim_g1.0_regularized | 202 | 0.0028177 | 0.99412 | 0.00018227 | 0.4478 | 2.9628 | 1.6311,2.0416 |
| tfim_g1.5_R1 | 202 | 0.0010008 | 0.99801 | 1.3109e-06 | 0.13983 | 1.1854 | 1,1 |
| tfim_g1.5_R2 | 202 | 0.00022771 | 0.99971 | 3.3542e-05 | 0.14617 | 1.3243 | 2,2 |
| tfim_g1.5_R4 | 202 | 0.00024817 | 0.99976 | 5.7813e-05 | 0.14923 | 1.3535 | 4,4 |
| tfim_g1.5_Rfull | 202 | 0.00024817 | 0.99976 | 5.7813e-05 | 0.14923 | 1.3535 | 4,4 |
| tfim_g1.5_smooth_fixed | 202 | 0.00079216 | 0.99871 | 5.8085e-05 | 0.14306 | 1.2379 | 2,2 |
| tfim_g1.5_trainable | 202 | 0.00077876 | 0.99863 | 2.7155e-05 | 0.14171 | 1.2268 | 1.625,1.6067 |
| tfim_g1.5_regularized | 202 | 0.00077037 | 0.99862 | 2.8948e-05 | 0.1415 | 1.2253 | 1.6172,1.5908 |
| tfim_depth2 | 101 | 0.00016645 | 0.99983 | 5.3586e-05 | 0.48177 | 3.3195 | 4,4,4,4 |
| tfim_heads4 | 101 | 0.00081757 | 0.99911 | 0.00037898 | 0.48305 | 3.3415 | 4,4,4,4 |

幅の学習軌跡、実効接続距離、ED/NNのCMI、S2、p_-を同じcheckpointで比較する図を生成した。
正則化ありで幅が縮むことを、自発的な物理的相関長の発見とは呼ばない。
pilotの短い学習で残るp_-や誤差は長時間学習で再確認が必要。

## QWZ: 複素表現とGauge transport

QWZ状態、Projector、局所atlas、FHS、phase-liftは既存wannier_phを再利用した。
教師はProjector/局所フレーム/遷移位相であり、Chern数をlossに使用していない。
以下のゲージ誤差は、同じcheckpoint・同じ入力にランダム局所U(1)を適用したProjector出力の最大差。
実数2channelとbaselineの生Bloch実虚成分は同じ情報・同じ演算になり、差がないことも対照の一部である。

| run | seed | P MSE | gauge誤差 | C一致 | parameter数 |
|---|---|---|---|---|---|
| qwz_real | 101 | 0.0023635 | 0.60889 | 3/3 | 939 |
| qwz_projector | 101 | 0.00067371 | 1.1921e-07 | 3/3 | 987 |
| qwz_complex_twochannel | 101 | 0.0023635 | 0.60889 | 3/3 | 939 |
| qwz_native_complex | 101 | 0.0023911 | 0.43135 | 3/3 | 902 |
| qwz_gauge_covariant | 101 | 0.0017148 | 2.9802e-07 | 3/3 | 828 |
| qwz_atlas_real | 101 | 0.011809 | 0.94367 | 3/3 | 952 |
| qwz_atlas_complex | 101 | 0.0020751 | 0.31712 | 3/3 | 902 |
| qwz_atlas_covariant | 101 | 0.0038897 | 4.1723e-07 | 3/3 | 828 |
| qwz_real | 202 | 0.0029144 | 0.42562 | 3/3 | 939 |
| qwz_projector | 202 | 0.00057059 | 1.1921e-07 | 3/3 | 987 |
| qwz_complex_twochannel | 202 | 0.0029144 | 0.42562 | 3/3 | 939 |
| qwz_native_complex | 202 | 0.0027022 | 0.36966 | 3/3 | 902 |
| qwz_gauge_covariant | 202 | 0.00064786 | 3.5763e-07 | 3/3 | 828 |
| qwz_atlas_real | 202 | 0.0094762 | 0.66482 | 3/3 | 952 |
| qwz_atlas_complex | 202 | 0.009467 | 0.32008 | 3/3 | 902 |
| qwz_atlas_covariant | 202 | 0.0029352 | 3.5763e-07 | 3/3 | 828 |
| qwz_covariant_fixed | 101 | 0.0020028 | 3.5763e-07 | 3/3 | 828 |
| qwz_covariant_shared | 101 | 0.0015307 | 3.5763e-07 | 3/3 | 829 |
| qwz_covariant_per_head | 101 | 0.001543 | 3.5763e-07 | 3/3 | 830 |
| qwz_covariant_locality | 101 | 0.0014302 | 3.5763e-07 | 3/3 | 830 |

Gauge-covariant設計はcharged入力、物理Berryリンク、atlas coverを入力として受け取る。
共変な輸送と出力射影によるゲージ不変性は設計の検証であり、Berryリンクやトポロジーをネットワークが発見した証拠ではない。
特に非零で連続なcharged出力が入力bundleのChernを保つことは、未学習でも起こり得る。
projectorとatlasは異なる出力型で、atlasには2つの局所フレームだけを有効領域で教師として与える。
大域的に滑らかな周期Blochフレームは要求しない。ProjectorのHermiticity・冪等性・rank残差も各行へ保存した。
最小overlap以下のリンクは0にして接続を無効化し、有効リンクmaskを記録する。

## PHと内部量

Attention/QK/Alpha/Window/実集約重み、特徴点群、phase-liftのPHを同じモデルの学習前後で保存した。
Screenformer/TFIMのPHは既存F2 weighted flag reducerを使い、重み0の辺は欠落させる。
QWZのQKはtanhで有界化して同じflag化を行う。QWZ Attentionはsoftmaxであり、CSV/図のScreening graphという共通キーは「実集約重み」の意味で使う。
特徴PHはEuclidean VR、物理phase-lift PHも8次元Euclidean VR。異なる対象・距離を同じ不変量とは扱わない。
既存研究で確認した標準phase-liftのplateauと、距離・尺度を変えた際の失敗条件は回帰テストの基準にした。
本比較だけで内部PHとWannier障害の普遍的対応を認定しない。

## 介入

共通apply_intervention APIから、head/接続/距離帯/幅/Fourier/6θ/spin/位相/transport/patchを適用可能。
各taskに存在しない介入は拒否する。連続電子の番号だけで特定辺を除く物理的介入も反対称性を壊すため拒否する。
TFIMは介入後の全状態を再正規化して厳密に再評価。連続系は変更後の波動関数からfresh Born chainsで再評価し、元分布の局所Eを流用しない。
連続系のzero_phase_channelはNN軌道の虚部を除く介入で、HF軌道の複素位相は残る。
QWZは介入後のProjector誤差・Berry曲率・Chern・atlas位相を再評価し、元/倍メッシュ、復号norm、overlap、plaquette branchのscreenを記録した。
NNの復号normは物理エネルギーgapではない。実gapや連続性が証明されたとは報告しない。
介入で誤差が増えても、そのheadが位相障害を選択的に担うことの証明にはならない。

## 大型実験・未検証

- 小さい連続系のMCは未収束の可能性がある。WC/AHCの識別改善、学習Fourierによる真の結晶波数の自発選択は未検証。
- N=12/16の長期TFIM、実N=36のembedding比較はproduction planだけを作成し、既存Slurm/GPUジョブには変更・追加を行っていない。
- 連続系twistの小さな矩形でperiodic partのoverlapを測れるが、全twistトーラスのsewing・gappedな固有状態追跡・Hall/Chern認定は未実施。
- 連続系のエンタングルメント、N>2の多体overlap推定、厳密な同一表現能力比較は未実装。
- Stage Fの複合改良は、独立seedと誤差に対して単独効果が確認された後に限る。このpilotだけから最良条件を確定して追加学習しない。
- 結果は相関・介入感度として報告し、内部構造と物理量の一般的な因果対応を断定しない。

全runにconfig、入力hash、checkpoint、optimizer/RNG/MC状態、学習ログ、内部配列、コードrevisionとsource snapshotを保存した。
comparison.csvには失敗・途中・未実行も残している。未実行の条件を実行済みの平均へ混ぜない。

[研究上の問いへの実測回答](/home/tomax/projects/Screening_topology_gpu/research_framework/results/2026-11-06_pilot/reports/RESEARCH_FINDINGS_JA.md)

元データ・図・全設定の保存先: /home/tomax/projects/Screening_topology_gpu/research_framework/results/2026-11-06_pilot

## 幅介入の再解析

Full Attentionに対するwindow_width介入が元のFull窓を残していた実装不備を修正した。同じ学習済み20 checkpointを再評価し、通常出力の差が3e-6以下であることを確認した。再学習は行っていない。元の診断は保持し、補正した幅介入は各runのinterventions_width_corrected.jsonへ保存した。

介入の比較には [補正後の実測回答](/home/tomax/projects/Screening_topology_gpu/research_framework/results/2026-11-06_pilot/reports_corrected/RESEARCH_FINDINGS_JA.md) とreports_corrected/matched_checkpoint_interventions.csvを用いる。元のreports/は修正前の記録として保持する。全ての学習結果・エネルギー・Projector再構成表はこの修正の影響を受けない。
