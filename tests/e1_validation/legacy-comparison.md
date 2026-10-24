# E1：旧probeとDiagnostic probeの差

同じcheckpoint・設定・入力で比較。新probeのS/W/Alpha/Windowは正式APIに厳密一致。
旧WはTrim²×旧Windowで、固定Valueの対角除去やdiag/clip介入を反映しない。新Wはaggregation_weights。
以下は独立したfloat32/float64ケースで、両精度間を引き算したものではない。
相対RMS=RMS(new-old)/RMS(old)。成分相対誤差のゼロ近傍除外閾値・除外数・参照スケールはJSONへ保存。

| 精度 | 条件 | W最大絶対差 | W RMS差 | W相対RMS差 | 旧W参照RMS |
|---|---|---:|---:|---:|---:|
| float32 | cos | 1.192093e-07 | 2.740685e-08 | 7.341475e-08 | 0.3733153 |
| float32 | pow4 | 0.2497322 | 0.1495068 | 0.4048361 | 0.3693019 |
| float32 | smooth7 | 0.03442216 | 0.01014857 | 0.02719049 | 0.3732396 |
| float32 | fixed | 0.8716617 | 0.3264875 | 0.9302066 | 0.3509839 |
| float32 | noln | IndexError: list index out of range | — | — | — |
| float32 | nomask | KeyError: 'mask_omega' | — | — | — |
| float32 | band | 1.788139e-07 | 3.799066e-08 | 1.248199e-07 | 0.3043638 |
| float32 | sin | 3.72529e-09 | 6.585445e-10 | 2.342341e-09 | 0.281148 |
| float32 | gauss | 1.192093e-07 | 2.984485e-08 | 1.063467e-07 | 0.2806372 |
| float32 | freeze | 1.192093e-07 | 2.740685e-08 | 7.341475e-08 | 0.3733153 |
| float32 | hooks | 0.4379343 | 0.1253627 | 0.340385 | 0.3682968 |
| float32 | features | TypeError: dot_general requires contracting dimensions to have the same shape, got (5,) and (6,). | — | — | — |
| float64 | cos | 3.330669e-16 | 8.380946e-17 | 2.435369e-16 | 0.3441345 |
| float64 | pow4 | 0.2491245 | 0.1335941 | 0.3859075 | 0.3461816 |
| float64 | smooth7 | 0.01355948 | 0.00337368 | 0.009803016 | 0.3441472 |
| float64 | fixed | 0.8758546 | 0.3493808 | 0.989431 | 0.3531129 |
| float64 | noln | IndexError: list index out of range | — | — | — |
| float64 | nomask | KeyError: 'mask_omega' | — | — | — |
| float64 | band | 5.551115e-17 | 1.032784e-17 | 6.610482e-17 | 0.1562344 |
| float64 | sin | 2.63678e-16 | 5.79009e-17 | 3.928122e-16 | 0.147401 |
| float64 | gauss | 1.110223e-16 | 1.964294e-17 | 1.335333e-16 | 0.1471014 |
| float64 | freeze | 3.330669e-16 | 8.380946e-17 | 2.435369e-16 | 0.3441345 |
| float64 | hooks | 0.161335 | 0.04098968 | 0.1181517 | 0.3469241 |
| float64 | features | TypeError: dot_general requires contracting dimensions to have the same shape, got (5,) and (6,). | — | — | — |

## 原因と対象範囲

- cos/band/sin/Gaussian/Freeze_W：意味は一致し、旧vmap eagerと単一walker JITによる丸め、演算順の違いがある。float32では旧d_refのNumPy float64計算とNativeのfloat32計算にも約1.63e-8の差があった。新probeはNativeと同じdtypeの格子を使い、距離ノルムも本体関数を使う。
- Trim4：旧解析のu²に対して本体はu⁴。新しい式を導入したのではなく、本体の係数を取得するようになった。
- smooth7：旧解析は余弦窓、本体は1-S7。新しい値は本体の窓を取得している。
- 固定Value：旧Wには自己項が残る。新Wの対角は厳密に0。Window積のW_windowは別に保存する。
- hooks：旧Wは_diag_only/CLIP_Rを反映していない。新Wは実際の集約用重みであり、W_intervened/W_windowと区別できる。
- LNなし：旧probeの無条件LN参照でIndexError。新probeは既存の層評価と共有し、そのまま動作する。
- Windowなし：旧probeの表示用mask_omega参照でKeyError。新probeはWindowを保存せず、表示幅をNaNとする。
- features：include_r_ae=Trueで、旧probeはFalseを強制してembeddingの内積shape不一致。新probeは保存済みfeature factory/kwargsを使用する。

旧解析のERRORはこのE1比較で再現した旧方式の結果であり、ProductionのFAIL/ERRORや新方式の失敗ではない。
歴史的な実研究runは未入手。これらは実際の保存形式で生成した小規模の合成checkpoint/configによる結果である。
