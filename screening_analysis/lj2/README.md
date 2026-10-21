# lj2 の結果の解析（2026-10-21、Claude のコンテナで実行）

データ：`ws_pw_kit/pulled_lj2/`（`_log_get_lj2.bat` で回収した 12 本の最後のチェックポイントと train_stats）。
スクリプトの中のパスはコンテナのもの（/home/claude/lj2res/results/... と /home/claude/ljhf/hf_N36_*.txt。harmonics.py は lj2net.py を /home/claude/lj2res/ana から読む）。PC で回すときは
`lj2net.BASE` を `ws_pw_kit/pulled_lj2/results/lambda-jellium/CustomScreenformer`、HF の表を `ws_pw_kit/lj2_hf/` に読み替える。
窓と Trim の形は `PW_WIN_SHAPE=smooth7`、`PW_TRIM_POW=4`（lj2net.py が import の前に設定）。

| ファイル | 中身 | 出力 |
|---|---|---|
| summarize.py | 最後の 10,000 step の平均エネルギー（500 step のブロック誤差）、σ(E_L)、n_-/N | summary_energies.csv |
| order.py | 密度の変調 \|<ρ_G>\|/N、S(G)、単位胞の密度の山での n_-/n | order.json |
| c3hf.py, c3centers.py | HF の行列式の C3 固有値（厳密、det M）と C3 中心 | c3_hf_centers.json |
| lj2net.py | チェックポイントから網を作り直す | |
| c3vmc.py | VMC の状態の <C3>（ウォーカー 1024 本） | c3_vmc.json |
| readout.py | 1 層目のアテンションの中身（網の attention の出力と 5e-7 で一致を確認） | |
| run_readouts.py | 擬スピンの q·k 差、ρ_p、寄与率 q_h、Trim の幅 | readouts.json |
| pstype.py | 擬スピンの組ごとの内容の重み ᾱ_{σσ'}（窓で重みづけ、距離ごとも） | pstype.json |
| netshare.py | 軌道の行列のうち網の部分と HF の部分の比 | netshare.json |
| table2.py | 12 本のアテンション側の量と、同じ λ の 2 つの型の差（報告の表 2） | 画面に出力 |
| harmonics.py | q·k と α を相手（または自分）の位置の関数としてセル全体で Fourier 分解し、セル全体の波長の成分と結晶の周期の成分の割合を出す（報告 3 節の「理由」） | harmonics.json |
| flip_shift_check.py | 1 層目のアテンションの重みが「全員の擬スピンを反転し、セルの半分ずらす」操作で変わらないことの確認（報告 3 節の「理由」、12 本とも差 ≤ 8e-6） | 画面に出力 |

図は `../src/make_figs.py`（同じデータを読む）。
