# lj3（入力 12 成分・Trim 固定）の解析（2026-10-30）

run：λ = 0.33・0.72 の WC 型（hw33, hw72）と AHC 型（ha33, ha72）、各 690 分（約 4.3–4.4 万 step）。ジョブ 2452・2453（10-28）。
比べる lj2：同じ λ・型（cw33 ca33 cw72 ca72）の、lj3 の step を超えない最後のチェックポイント。回収は `ws_pw_kit/pulled_lj3/`。
解析はクラウドのコンテナ（`/home/claude/lj3res/`、コードは wada_dev の f4ddf23）で行った。パスはコンテナのもの。

| ファイル | 中身 |
|---|---|
| energies.py → energies_lj3_vs_lj2.csv | 最後の 10,000 step の平均（500 step のブロックの誤差）、同じ step の窓、途中の step |
| lj3net.py | チェックポイントから網を作り直す（lj2・lj3 両方） |
| readout3.py | 1 層目のアテンションの写し（網の出力と相対 5e-7 で一致）。入力は CS.one_electron_inputs |
| analyze_all.py → ana_all.json | C3 固有値、擬スピンの模様、擬スピンの組ごとの窓つきの重み A_{σσ'}、ρ_p、⟨w⟩_q |
| ../lj3/input12_check.py → input12_check.json | 反転＋半分ずらしの対称性、q̄·k̄ の結晶の周期の成分、埋め込みの重み |
| transfer.py → transfer.json | WC 型と AHC 型の差の向きが 2 つの λ で揃うか（64 個の特徴の差の余弦、LDA の AUC） |
| moved.py → moved.json | 学習でパラメータがどれだけ動いたか（初期値は PRNGKey(42) を 1 回 split して作り直す。窓は 0.000 で一致） |
| swap.py → swap.json | 同じ網で WC 型と AHC 型の配置を比べる（網を固定すれば違いは配置だけから来る） |
