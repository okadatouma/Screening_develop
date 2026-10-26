# C〜F追加実験・物理推定器の検証（2026-10-26）

**距離スケール曲面、向き付きループ、Hidden点群、固定基底の複素一体密度行列・物理ループを実装し、
数値実験した。独立模型では正しいtwist Hamiltonianから多体Chern数も検証した。
ScreenformerによるWC/AHCの相識別と、その多体Chern数との関係は引き続き未検証。**

実装・推定器の導出：[README_topology_cdef.md](README_topology_cdef.md)。
この追加作業では既存の未コミット変更を保持し、Production、学習、Hamiltonian、微分を変更していない。
既存checkpointは、前回pilotで生成した42個を含む563個すべてhash不変。

## 実行した実験と検証

|対象|実施内容・結果|
|---|---|
|C/D/E|前回の保存probe54個を解析。自由1電子12 checkpointと未学習N9の42条件|
|新規C〜F検査|21テストPASS。A/Bの10テストと合わせて31 PASS|
|既存G2b float64 Snapshot再実行|902配列・21,651要素、全て厳密一致|
|同float32、Adam/KFAC込み|1,319配列・31,142要素、全て厳密一致|
|連続空間の物理1RDM|既知2電子Slater、25個の固定空間軌道で最大差3.89e−16、trace=2|
|実checkpointでの1RDM|固定した自由1電子の学習済み状態から8独立MC chain、8,192配置|
|多体Chernの独立模型|QWZ模型、実際の境界twist、2×2/3×3/5×5セル、6×6/10×10 twist格子|
|独立Chern検算|Kubo積分、twist微分、Fourier spectrum、Fock/Slater重なり、ゲージ・精度検査が一致|
|C3推定器|lambda-jelliumのpseudospin変換則と3つの解析的C3固有状態で検査|

新規検査は閾値・同値、独立F2ランク、周期表現、電子・Head置換、鏡映、直交QK変換、GL非不変性、
ε依存、mean poolingの情報消失、Hiddenの距離単位、1RDMの複素共役・粒子数・spin coherence、
基底位相ゲージ、波動関数全体位相、電子交換、独立chain積、twist Hamiltonian・C3を含む。

環境・hash・全ログは[verification.json](topology_cdef_results_2026-10-26_final/validation/verification.json)。
新しい依存は追加していない。全図はSVG、XML整合を検査した。描画環境が利用できず、ラスタ化した目視検査は未実施。

## C：距離スケールとHead

`s_sym>a AND d_minimum_image<b`の7×9格子で、全配置・層・Headのbeta0/beta1と巻き付きRankを測った。
Windowのsupport、統合Window、Head union/consensus、距離帯別Betti・Window質量、
HeadペアのJaccard、出力の相殺も保存した。これは二パラメータの曲面で、1D Diagramやrank invariantではない。

図・数値の例：

- [Betti・巻き付き2D曲面](topology_cdef_results_2026-10-26_final/synthetic-cos-seed11-ordered/multiscale.svg)
- [距離cutoffごとの巻き付き確率](topology_cdef_comparison_2026-10-26_final/distance-winding.svg)
- [Window統合・support冗長性](topology_cdef_results_2026-10-26_final/synthetic-cos-seed11-ordered/windows-redundancy.svg)
- [cos/sin/gaussのHead別距離帯域](topology_cdef_comparison_2026-10-26_final/head-distance-bands.svg)
- [相殺・冗長性の全数値](topology_cdef_comparison_2026-10-26_final/head-redundancy-cancellation.json)

Head番号はrun内の診断図にだけ使い、分類には平均・標準偏差・最大・unionや対平均を使った。
同一probeのHeadを入れ替えた全C/D/E出力特徴が一致することも、ファイル出力を含むテストで確認した。

規則配置／一様配置のseed holdout分類は、距離だけ、QK＋距離、QKから距離baselineを引いた特徴、
Window統合のいずれも全7条件で6/6だった。
**距離だけで同じ成績を達成するため、Cの6/6はQK固有の情報や相識別の有効性を示さない。**
特にa=−1付近の曲面にはほぼ純粋な幾何グラフが含まれ、これだけでも分類できる。
残差特徴でもsupport/densityなどの幾何依存が残り得る。

Headの結合例（seed11・規則配置、L0/L1、平均二乗の比）：

|条件|rho L0|rho L1|平均support Jaccard L0/L1|
|---|---:|---:|---:|
|cos|1.0164|0.9859|0.0316 / 0.0116|
|sinリング|1.0477|1.0005|0.0913 / 0.0594|
|gaussリング|1.0491|1.0005|0.4650 / 0.2777|
|fixed Value|0.99994|0.999999|0.0316 / 0.0119|

この未学習例では出力相殺は小さい。Gaussian tailは微小でも正であり、supportの重なりを増やす。
高いsupport Jaccardを、そのまま大きな出力の冗長性と解釈してはいけない。
距離帯別Head出力の値は非線形性を無視したproxyと明記し、厳密な加法分解とは扱っていない。

## D：向き付きループと正則化依存

周期最短像で可縮な三角形を実空間CCWへ向き付けた。
N9合成配置では全84三角形のうち保持されたのは配置により36〜57個。
最短像縮退・非可縮・面積0の除外数を別々に記録した。
符号付きCCW循環、逆向き、周長・最大辺・面積、Head/層・周長帯別の分布を保存した。

図：[向き付きループの周長別統計](topology_cdef_results_2026-10-26_final/synthetic-cos-seed11-ordered/directed-loops.svg)、
[ε依存](topology_cdef_comparison_2026-10-26_final/epsilon-study.svg)。

Screeningの非対称量`(Wij-Wji)/(Wij+Wji+epsilon)`を使った合成幾何の分類：

|条件|ε=1e−4|ε=1e−8|ε=1e−12|循環の最大変化量|
|---|---:|---:|---:|---:|
|cos|6/6|6/6|6/6|1.738|
|wide|5/6|5/6|5/6|1.686|
|smooth7|6/6|6/6|6/6|1.868|
|band|6/6|6/6|6/6|1.946|
|sin|5/6|5/6|4/6|1.559|
|gauss|6/6|4/6|4/6|2.807|
|fixed|6/6|6/6|6/6|1.420|

全値：[epsilon-study.json](topology_cdef_comparison_2026-10-26_final/epsilon-study.json)。
これは3seed・6判定の小さな合成実験で、有意な精度差とは主張しない。
ただし、ほぼ零の重みに対するεで個々の循環が大きく変わり、Gaussian条件では判定も変わった。
**単一εの結果だけで頑強なカイラル指標と認定できない。**
log比はW+εから計算し、W=0の場合も有限であることを検査した。

QK由来のD特徴も合成幾何を5〜6/6で識別したが、これは実数trunk内部の向き付き相関である。
これを時間反転の破れや多体Chern符号へ読み替えてはいない。
共通の直交変換には不変だが、正規化のため一般GL双対変換はSを変えることを実際に確認した。

## E：Hidden表現をグラフと分けて評価

各配置を1点とし、各層でmean/mean_std/moments/Gram spectrumの4種を比較した。
electron graphのPHではない。各層の実Euclidean距離によるH0/H1 Diagramを保存した。
層間の包含写像を与えていないため、Zigzagとは呼ばない。

図：[Hidden表現・PH・距離だけの比較](topology_cdef_comparison_2026-10-26_final/hidden-comparison.svg)。
全層・全pooling・H0/H1別：[hidden-feature-controls.json](topology_cdef_comparison_2026-10-26_final/hidden-feature-controls.json)。

通常のmean pooling平均ベクトルだけでは、L0の全7条件が3/6だった。
一方、点群の分布を使う特徴は識別できた。
ただしPHと普通の点間距離を混ぜて評価しないよう、元の記述ベクトルを**PHだけ／H0だけ／H1だけ／距離分位点だけ**へ分けた。

Gram spectrum点群のL0の例：

|条件|PHのみ|H0のみ|H1のみ|距離分位点のみ|
|---|---:|---:|---:|---:|
|cos|6/6|6/6|5/6|6/6|
|wide|5/6|6/6|4/6|6/6|
|smooth7|6/6|6/6|3/6|6/6|
|band|5/6|6/6|3/6|6/6|
|sin|6/6|6/6|3/6|6/6|
|gauss|5/6|6/6|5/6|6/6|
|fixed|5/6|6/6|4/6|6/6|

L1のH1のみはcos/smooth7で1/6、wide/gauss/fixedで5/6と不安定だった。
H0だけ・距離だけで高率に分類でき、H1が特に優れているという証拠はない。
mean poolingで消える情報があることは確認できたが、これは点群の広がり・分布の差でも説明できる。
学習済みWC/AHCのHidden topologyの有効性とは別の結果である。

## F：連続空間・擬スピンの物理的1RDM

固定された物理基底に対して次を実装した。

```
G_ab = E_|Psi|² sum_i phi_a(x_i) integral dy phi_b*(y) Psi(R_i->y)/Psi(R)
```

基底ラベルは電子番号ではない。擬スピンを含む積分は離散成分の和で、off-diagonal spin coherenceも検査した。
複素共役、Hermiticity、Nのtrace、全体位相、基底ごとの位相ゲージ、電子交換を独立式で照合した。

### 既知2電子状態での検算とWindow距離分解

三角セル内の25個の固定DVR軌道（対称な5×5 Fourier基底）に、既知の2つの占有軌道を作った。
推定式を独立の`G=U.conj()@U.T`と照合し、
最大誤差3.89e−16、trace=2.000000000000000、Hermiticity残差9.07e−16。
これは厳密重み付きquadratureの検算で、MC精度の主張ではない。
基底ゲージでGの各成分は変わるが、三角積は不変。scalarな複素共役でImループの符号は反転した。

図：[解析解・複素密度行列・推定誤差](topology_f_results_2026-10-26_final/exact-density.svg)。

前段の3×3空間基底では、保持された最短像・可縮三角形が同じ長さだけだった。
距離比較には情報が不足するため5×5でも検算し、600個・5種類の周長を持つ三角形を得た。
同じ物理的Gへ、前回の未学習ネットワークのWindowを固定した距離フィルタとして適用した。
電子トークンのalphaは物理基底へ移植していない。

Head・層のWindow質量を足してから規格化したImループ平均：

|Window|同一GのWindow付きループ平均|
|---|---:|
|cos|−2.627e−6|
|wide|+1.340e−7|
|smooth7|−4.292e−6|
|band|+2.055e−6|
|sin|−1.407e−5|
|gauss|−1.350e−5|
|fixedのWindow|−2.627e−6|

距離周長帯・質量・零分母の状態：
[physical-window-scales.json](topology_cdef_comparison_2026-10-26_final/physical-window-scales.json)。
Windowだけで符号も変わる。同一の物理状態が別のChern相へ変わったのではない。
基底中心間距離に基づく記述的分解であり、この平均を量子化不変量とは呼べない。

### 学習済みcheckpointでの実MC

既存`screen-complex64/qmcjax_ckpt_000015.npz`のparamsを固定し、
独立RNG streamを持つ8chain、burn 256、記録間4更新、各1,024記録の新しいMCを実行した。
位置受理率0.847〜0.867。計8,192配置。trainerや保存walkerは変更していない。
独立した16/32/64/128格子の波動関数積分も実行した。

|量|結果|
|---|---:|
|raw MC trace|0.999022708 + 7.13e−5 i|
|独立積分の射影trace|0.999264526|
|GのFrobenius誤差|0.03223|
|独立chain複素SEMのFrobeniusノルム|0.03001|
|要素誤差／複素SEM＋観測quadrature差の最大|2.92|
|raw Hermiticity残差|0.04558|
|局所推定器16→32格子、同じ128配置での最大差|1.69e−5|
|独立参照64→128格子の最大差|2.80e−9|

図：[MC G・参照値・誤差・SEM](topology_f_results_2026-10-26_final/checkpoint-density.svg)。
有限MCのraw GはHermitianからずれる。これを平均化・trace規格化で隠していない。
参照値との誤差は観測したchain SEMと整合する大きさだった。
局所積分格子8→16で差が見えたため、最終結果は16格子とし32格子への感度も保存した。
128共通配置でのquadrature感度は全系列に対する厳密な誤差上限とは扱わない。
cos(x)とG01実部のblocking plateauは今回の有限系列で確認できたが、収束の数学的証明ではない。

N1の真のGはrank oneなので、三角積の虚部は0である。
独立chainを3因子に割り当てたU統計量のループ平均は4.18e−7、
三角形ごとのjackknife SEMの最大は8.90e−6、独立参照の虚部は最大1.47e−19だった。
生のループ推定・chainごとのG・MC配置は
[checkpoint-density.npz](topology_f_results_2026-10-26_final/checkpoint-density.npz)に保存した。
これは自由1電子のtrial stateの検証で、WC/AHCの測定ではない。

## F：独立した多体ChernとC3の検証

QWZ模型では境界を跨ぐhoppingへtwistを入れ、そのHamiltonianごとに占有部分空間を求めた。
単に同じ状態へ位相を後付けしたのではない。
2×2セルは70次元の固定Fock基底で多体波動関数を明示し、
`det(V_theta†V_theta+dtheta)`との重なりを照合した。
3×3・5×5セルでも厳密な多体Slater重なりを使い、占有軌道の局所U(Nocc)ゲージを検査した。
twist微分・Fourier spectrum・独立Kubo積分と、格子精度・gapを確認した。

5×5セル、10×10 twist格子の結果：

|mass|多体Chern|独立Kubo（2×2）|最小gap|CCW物理ループ平均|
|---|---:|---:|---:|---:|
|−1|−1.0000000000000002|−0.9999999999999999|約2|−7.210e−5|
|+1|+1.0000000000000002|+1.0000000000000000|約2|+6.555e−5|
|+3|−7.91e−17|+7.27e−18|約2|+1.163e−6|

符号規約は`<Psi(theta)|Psi(theta+dtheta)>`のCCW plaquette。
非自明な2条件の最小隣接重なりは約0.897、最大plaquette位相は約0.0715。
6×6/10×10 grid、2×2/3×3/5×5セルでChernの結論は一致した。
図：[Berry曲率](topology_f_results_2026-10-26_final/chern-benchmark.svg)、
全条件：[物理検証summary](topology_f_results_2026-10-26_final/summary.json)。

この模型ではループ平均の符号とChern符号が対応した例を得たが、**C=0でもループは非零**。
ループ量が量子化Chernではないこと、単独で相判定できないことも数値的に確認できた。
これは自由フェルミオン模型での検証で、相互作用するlambda-jelliumへの成立を示さない。

C3はlambda-jelliumのspinor作用`diag(1,exp(i 2pi/3))`と座標の逆回転を含む推定器を追加した。
セル対称性を確認し、解析的な3つのC3固有状態で期待値とroot残差を検査した。
実checkpointのWC/AHC状態でのC3固有値は未測定。
正方セルの今回のN1 checkpointへC3を誤適用せず、ゼロtwistの固有値だけからChernも割り当てない。

## 研究目的に対する結論と未完了部分

今回確立したのは、C〜Fの解析経路と物理推定器の独立検証である。
合成実験の識別は距離・分布でも説明でき、Dにはε依存がある。
物理ループもWindow選択で値・符号が変わり、Chernが0でも非零になり得る。
よって、これらの量を相識別に使うには対照と独立物理ラベルが不可欠である。

|未完了項目|理由・必要なもの|
|---|---|
|WC/AHCのC/D/E比較|独立判定済みcheckpoint、設定、Hamiltonian/seed/収束記録が未入手|
|WC/AHCの物理1RDMと距離帯比較|同じ物理基底・同じ精度で測定する実状態とMC系列が必要|
|Screenformerの多体Chern|正しいtwist Hamiltonian、全twistの状態最適化／追跡、sewing、gap検証が未提供|
|C3による対象状態の認定|数値Hamiltonianの対称性、対象状態の固有state残差、必要なtwistでの回転データがない|
|正則化によるVMC改善|物理ラベル・同一予算baselineを伴う学習実験は未実施|

Zigzag PH／二パラメータrank invariantは今回は不要として実装していない。
実データが来れば現在の独立した各解析を同一状態群へ適用できるが、今回の数字をWC/AHC精度へ流用しない。

## 再実行

```sh
.venv-pw/bin/python -m unittest tests.test_topology tests.test_topology_run tests.test_topology_cdef -v
.venv-pw/bin/python -m screening_analysis.topology_cde_run \
  --previous-results screening_analysis/topology_results_2026-10-26_final --output NEW_CDE
JAX_PLATFORMS=cpu .venv-pw/bin/python -m screening_analysis.topology_f_pilot --output NEW_F
.venv-pw/bin/python -m screening_analysis.topology_cdef_evaluate NEW_CDE --physical NEW_F --output NEW_COMPARISON
```

解析途中の出力も保護して残した。採用するのは`topology_cdef_results_2026-10-26_final/`、
`topology_f_results_2026-10-26_final/`、`topology_cdef_comparison_2026-10-26_final/`。
その他の同名prefixのディレクトリは精度・対照追加前の開発記録である。
