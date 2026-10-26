# C〜F：距離スケール・向き付きループ・Hidden・物理的複素相関

追加結果：[TOPOLOGY_CDEF_RESULTS.md](TOPOLOGY_CDEF_RESULTS.md)。
A/Bの定義：[README_topology.md](README_topology.md)。既存のProduction、VMC、微分、checkpointは変更しない。

## C：二パラメータ曲面

`multiscale_topology.analyze_multiscreen(S, Window, positions, lattice, a_grid, b_grid)`
は単一配置の`S,Window [L,H,N,N]`を受け取る。

```
A_h(a,b) = (S_h+S_h.T)/2 > a  AND  d_minimum_image < b
```

全頂点を残し、対角を除き、F2三角形境界のランクからflag complexのbeta0/beta1を計算する。
距離は既定で最短像Euclidean。`periodic_distance=`を明示した場合だけ、ネットワークの周期ノルム等を
距離cutoffに使える。巻き付きの辺の埋込みは常に実空間最短像である。
strictな不等号をそのまま判定する。同値辺は同じグラフに含まれるか、まとめて除外される。
最短像の縮退があるactive edgeを含むグラフの巻き付きはNaN。

保存量はHead別のBetti、巻き付きRank、window-support付きグラフ、距離帯別Betti、Window質量である。
配置平均のRank確率は`[L,H,rank,a,b]`。`a`を下げ、`b`を上げると辺が増えるが、
この**二パラメータ曲面を1変数Persistence Diagramと呼ばない**。rank invariantは未計算。

Head置換に不変な集約も保存する。

- Headに関するmean/std/max Betti。
- `s_integrated = sum_h m_h*s_sym,h / sum_h m_h`による統合グラフ。Window質量0の辺は不在。
- いずれかのHeadが選ぶ辺のunion、半数以上のHeadが選ぶconsensus。
- Headペアのsupport Jaccard平均。両方空のペアは未定義。Gaussianの微小tailもsupportに含むため、
  Jaccardの大きさだけで出力の冗長性と解釈しない。
- QKを除いた距離だけのbaseline。同じ`b`におけるBetti/Rankと、QK曲面からbaselineを引いた記述特徴。

各層の`head_gram=<c_h.c_k>_electron`から相殺も測る。多数配置のrhoは
`<||sum_h c_h||²> / <sum_h ||c_h||²>`、**平均二乗の比**であり、配置ごとの比の平均とは区別する。
共有biasは含めない。Head番号を分類器の座標として固定しない。

距離帯域は実空間距離の`[b_k,b_k+1)`。既存cos/smooth7/帯窓/sin・gaussリングの実際のmを
その帯域内で積算する。固定Valueにも同じ診断を使う。
`band_energy_proxy`はHead出力の平均二乗×Window質量割合という補助記述量。
TanhNorm/Gateが非線形なので、出力・エネルギーの厳密な帯域加法分解ではない。
自己項のみのHeadの出力を非対角距離帯へ割り振ったとは主張しない。

## D：幾何学的に向き付けた三角ループ

`directed_loops.oriented_triangles`は最短像3辺が閉じる可縮三角形だけを、実空間CCWに並べる。
辺の最短像縮退、非可縮な3辺ループ、面積0の三角形を別理由で除外する。
非可縮ループへ任意の「内側」を選んで面の向きを与えない。
電子置換・周期表現変更・周期並進に不変で、幾何的鏡映では向きが反転する。

```
Omega_QK = (s_ij-s_ji)+(s_jk-s_kj)+(s_ki-s_ik)
A_ij     = (W_ij-W_ji)/(W_ij+W_ji+epsilon)
Omega_W  = A_ij+A_jk+A_ki
```

Wは既存診断の`W_window=alpha*m`。QKは単位ノルム正規化後のS。
epsilon>0を必須とし、W=0同士ならA=0。補助的な対数循環は`log(W+epsilon)`を先に作り、
log(0)を中間値にも発生させない。既定ε=1e−8、比較ε=1e−4/1e−8/1e−12を全て保存する。
固定した相対εに自動置換して差を隠してはいない。

CCWとCWの量、周長・最大辺長・面積、Head/層別mean/abs/RMS/std、周長帯別統計、
全ループの生の分布を保存する。ループは3辺で、周長は物理距離である。
共通の直交基底変換`q->qO,k->kO`とHead置換を検査する。
`q->qA,k->kA^{-T}`は生の内積を保っても長さ正規化後のSを保たないため、一般GL不変とは扱わない。

これらは量子化されず、実数のQ/Kでも非零になる。非零だから時間反転が破れているとは言えない。
出力側の複素位相と実数trunkは別であり、物理的Chern符号との対応は独立の状態群で検証する必要がある。

## E：Hidden点群

`hidden_topology`は`hidden_output [B,L,N,E]`から、各層で以下を比較する。

|pooling|置換不変な特徴|
|---|---|
|mean|電子平均|
|mean_std|電子平均と標準偏差|
|moments|平均・標準偏差・min・max|
|gram_spectrum|中心化hiddenの電子Gram固有値＋平均ベクトルのノルム|

現行のHiddenは実数。Gram特徴は電子置換と特徴空間の直交変換にも不変。
他のpoolingの座標軸は異なる初期化間で整列されているとは仮定しない。

点はMC配置、距離はpooled feature間のEuclidean距離。N点の電子グラフとは分けて解析する。
既存参照PHへ`B=1-d/(2*dmax)`で渡し、Diagramの座標を実際のEuclidean距離へ戻す。
最遠辺も保持し、点群ごとの尺度を捨てて一律の単位図へ変換しない。重複点は寿命0として処理する。
参照実装のため最大64配置。大きい集合を使う場合は明示的に標本化するか標準backendが必要。

層ごとに点群・Diagram・指標・分類結果を保存する。層間で距離が変わるだけではcomplexの包含写像は
定義できないため、Zigzag PHとは呼ばない。今回はZigzagを実装していない。
分類では学習groupを丸ごとholdoutし、標準化はtrain側だけでfitする。

## F：固定物理基底の一体密度行列

`physical_loops.TorusBasis`は格子列ベクトルL、面積A=|det L|、整数逆格子モードnに対して
`phi_n(r)=exp(2*pi*i*n.L^{-1}r)/sqrt(A)`を作る。モードは異なり、基底回転はunitaryに限る。
`dvr_basis`は対称な奇数Fourier格子を固定した空間中心へユニタリ変換した基底。
連続空間のdelta関数ではなく、有限幅の直交する空間軌道である。
擬スピンを有効にすると`phi_(a,sigma)(r,s)=phi_a(r) delta_(sigma,s)`。
このラベルaとsigmaは固定され、電子トークンiとは異なる。

### 推定器の導出

正規直交基底で`c_a†=integral dx phi_a(x) psi†(x)`、
`c_b=integral dy phi_b*(y) psi(y)`とする。x,yは位置と、必要なら離散擬スピンを含む。
座標表示でO=c_a†c_bをPsiに作用させると、

```
G_ab = <c_a†c_b>
     = E_[|Psi(R,S)|²] sum_i phi_a(x_i)
           integral dy phi_b*(y) Psi(R_(i->y))/Psi(R).
```

電子iを元のslotで置換しているため追加の交換符号はない。sum_iが因子Nを既に含む。
`phi_a(x_i)`は**複素共役しない**。比も共役しない。
既知のSlater軌道`Phi_j=sum_a phi_a U_aj`なら`G=U.conj() @ U.T`となり、これを独立の解析解に使う。

`one_body_samples`は任意のbatch callback
`evaluate(positions [M,N,2], spins [M,N]) -> (phase ANGLE [M], logabs [M])`を受け取る。
`make_network_evaluator`は既存`Network.apply`をそのままvmap/JITし、forwardを複製しない。
実数ネットワークのsign出力を使う場合だけ`phase_convention='sign'`を明示する。

連続積分をuniform cell quadratureで近似し、擬スピンでは+/-の両成分を**和**で列挙する。
spinless/fixed-component測定では元のspinを保持する。
MCサンプリング点のPsi=0はエラー、置換点のPsi=0は比0とする。
極端な比をclipして誤差を隠さず、overflowは明示停止する。

raw Gを保存し、Hermiticity、trace、Hermitian部分の固有値、独立chain平均・SEMを報告する。
推定値を勝手にHermitizeしたりtrace=Nへ規格化したりしない。
**有限基底のtraceは射影された粒子数**であり、占有部分空間を完全に含む場合だけNになる。
トーラス3モード等の単純な打切りではN未満でも推定器の失敗とは限らない。
擬スピンなしで複数の固定spinを測る場合は空間1RDMのspin和となる。

### 物理ループ・誤差・距離分解

`Im(G_ab G_bc G_ca)`を固定空間基底のCCW可縮三角形で測る。
基底軌道の位相`phi_a->exp(i chi_a) phi_a`に対し、
`G_ab->exp(i(chi_a-chi_b))G_ab`となり、積の位相は相殺される。
任意unitary基底混合に対する個々のループ不変性は主張しない。物理基底は固定する。

ノイズを含むGの平均を3回掛けるplug-in推定は一般に有限標本biasを持つ。
`independent_loop_products`は、異なる独立MC chainから各因子を取り、
相異なる3chainの全順列について平均するU統計量も保存する。
chainが4本以上ならdelete-one-chain jackknifeを計算する。
同じchainの時系列点を独立chainと呼び替えない。独立性はサンプラー側の要件である。
これは一体平均の積であり、相互作用系の六体演算子平均やHall応答そのものとは同一視しない。

`basis_windows`は学習時と同じ`attention.screening_window`と周期ノルムを、
**物理基底の中心間距離**に評価する。電子トークンのalphaやW行列を軌道番号へ移植しない。
`m_ab*m_bc*m_ca*Im(G_ab G_bc G_ca)`を距離周長帯・Window・Head・層ごとに保存する。
中心間距離による記述フィルタであり、有限幅軌道の厳密な点距離分解でも、Chernの加法分解でもない。
Head統合は質量を足してから正規化する。分母0は未定義。

発想上の参照：[Zhang & Kim, Quantum Loop Topography](https://arxiv.org/abs/1611.01518)。
本実装の物理的検算は、既知Slater、擬スピン、ゲージ、粒子数、MCと独立空間積分による。

## F：回転とBerry

`rotation_probe`はlambda-jelliumのband行列
`h(k)=[[lambda² k²,-lambda k_-],[-lambda k_+,1]]`について
`h(R_theta k)=U_theta h(k) U_theta†, U_theta=diag(1,exp(i theta))`を使う。
従ってC3作用は`exp(i theta n_minus)*Psi(R_-theta R,S)`。theta=2pi/3。
セルにC3対称性がなければ拒否する。3つの既知C3固有状態とband行列の変換式を独立検査した。
出力は回転**期待値**と候補rootからの残差。unit modulusの固有値と認定するには状態の残差・MC誤差が必要。
実際の数値Hamiltonianの回転対称性も別検査が必要であり、ゼロtwistの回転固有値だけでChernを決めない。

`berry_validation.berry_links`は共通Hilbert基底の状態vectorからFukui–Hatsugai–Suzuki型リンクを作る。
規約は`<Psi(theta)|Psi(theta+dtheta)>`、theta_x/theta_yのCCW plaquette。
ゼロ近傍の隣接重なり、ゼロ状態、非対応のsewingはエラー。
必要なsewing行列は利用者が正しいfiber同定から与える。非可換sewingのcorner cocycleは未対応。
数値的整数が出ても入力状態の物理的由来までは認定しない。

独立検証用に有限サイズQWZ格子模型を実装した。
twistは境界を跨ぐhoppingだけにPeierls位相として入れ、**各twistのHamiltonianを対角化**する。
占有Slater状態を固定Fock基底で評価する方法と、
`det(V(theta)†V(theta+dtheta))`という厳密多体重なりを使う方法を照合する。
後者は個別の1粒子固有ベクトル番号を追跡せず、占有部分空間全体を追う。
別のKubo曲率積分、Hamiltonianの解析twist微分、Fourier spectrum、gap、grid/size精度で検算する。
参照：[Fukui, Hatsugai & Suzuki](https://arxiv.org/abs/cond-mat/0503172)。

これは**QWZ模型の多体Chern検証**であり、Screenformer/lambda-jelliumの多体Chernではない。
lambda-jelliumではperiodic gaugeのkappa=L^{-T}thetaを用い、kineticとspin-flip微分の両方へ
`nabla->nabla+i*kappa`を入れたHamiltonianが必要になる。
各twistの状態最適化／追跡、共通Hilbert基底での重なり、2pi境界でのlarge-gauge sewing、gapとgrid精度が要る。
既存状態へ後から位相を掛けるだけでは検証できない。現在その状態族はなく、WC/AHCのChernは未検証。

## 実行

```sh
# 保存済み正式probeでC/D/Eを実行。既存出力を上書きしない
.venv-pw/bin/python -m screening_analysis.topology_cde_run --probe INTERNAL_PROBE.npz --output NEW_CDE
# 前回pilotの全54保存probeで比較
.venv-pw/bin/python -m screening_analysis.topology_cde_run \
  --previous-results screening_analysis/topology_results_2026-10-26_final --output NEW_CDE_BATCH
# F：厳密Slater、独立Chern模型、既存checkpointでの新規MC
JAX_PLATFORMS=cpu .venv-pw/bin/python -m screening_analysis.topology_f_pilot --output NEW_F
.venv-pw/bin/python -m screening_analysis.topology_cdef_evaluate NEW_CDE_BATCH --physical NEW_F --output NEW_COMPARISON
.venv-pw/bin/python -m unittest tests.test_topology tests.test_topology_run tests.test_topology_cdef -v
```

CDEの`summary.json`はgraph/loop/Hidden別の特徴を区別する。Headはmean/std/maxや対平均で集約する。
実相を評価する場合はrun metadataに`phase`、独立の`phase_evidence`、`group`、比較する`condition`を与える。
各相3group以上がなければ分類を実行しない。checkpoint反復は同じgroup/phaseへ集約する。
今回のsyntheticラベルは規則配置／一様配置であり、WC/AHCに読み替えない。
