"""Truthful follow-up report: completed evidence and queued work stay distinct."""
import json
from pathlib import Path
import numpy as np
from ..experiments.store import csv_save,json_save


def _cka(x,y):
    x=x-x.mean(0);y=y-y.mean(0)
    a=np.linalg.norm(x.T@y)**2
    return float(a/max(np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y),1e-20))


def report(root,n36=None):
    root=Path(root).resolve();n36=Path(n36) if n36 else root/'n36_cluster'
    lines=['# 追加実験報告','',
        '既存のEmbedding・Attention・Wavefunction、Hamiltonian、VMC、診断を再利用した。追加コードは `followup/` と新規テストのみ。既存checkpoint・CSV・レポートを上書きしていない。',
        '', '## 監査と既存結果との差分','',
        '`audit_before.json` に旧結果79条件、旧ソース・成果物のSHA256、監査時ジョブ状態を保存した。既存のN=2 pilotをN=36の物理検証には数えない。',
        '', '|項目|実装済み・既存検証|今回の追加・未解決|','|---|---|---|',
        '|Embedding/Attention/Wavefunction切替|既存実装・小規模検証済み|再実装なし|',
        '|N=36|HF参照とbaseline単独の既存ジョブ2881|5方式の共通初期状態・3 seed比較を別ジョブで実行|',
        '|Learned Fourier|振幅・疎性項実装済み、旧N=2 pilot|候補ごとの除去・Born再標本化を追加|',
        '|QWZ|Projector/Atlas再構成・ゲージ共変性は既存検証済み|局所情報を隠した大域holonomy合成課題を追加|',
        '|PH/CMI/相関長/S2/パリティ|旧結果を参照|再実装・再計算なし。今回の因果的主張の根拠に転用しない|',
        '', '旧報告の「大規模未投入」は作成時点の記録。現在の旧ジョブと今回の新ジョブを混同しない。既存の物理相診断・PHは `audit_before.json: archive_references` に列挙した資料を読む。',
        '', '## N=36 / Fourier','',
        '同じλ=0.72、rs=25、N=36、セル、収束済みHF表、10000更新、64 walkers、学習率0.002を使う。seed=101/202/303。5方式にlearned_fourierのL1=0.01を加えた18条件。候補は(1,0),(0,1),(1,-1),(6,0),(0,6),(6,-6)。疎性係数は事前固定し、結果による選択はしない。',
        '', '最終NN軌道射影だけをゼロ初期化するため、全方式の物理的初期状態は同じ供給HF状態。これは既存baselineのランダムNN初期状態とは異なる新しい比較条件。padded入力20で名目パラメータ数を合わせるが、活性特徴と関数クラスは異なる。',
        '', '学習前後のQK結晶波数・擬スピン選択性・位相のedge感度は既存diagnosticsを使用。EとS(q)はBorn標本から測定。候補除去では当該cos/sin対だけを消し、各状態で独立に再標本化してE・S(q)・位相応答を保存する。振幅履歴は毎更新保存する。振幅の大小だけから波数の物理的重要性を主張しない。']
    if (root/'initial_smoke.json').exists():
        smoke=json.loads((root/'initial_smoke.json').read_text())
        lines+=['',f'ローカル初期状態検査: {len(smoke["conditions"])}条件、最大log|ψ|差={smoke["max_logabs_delta"]:.3g}、最大位相差={smoke["max_phase_delta"]:.3g}。']
    if (root/'n36_pipeline_smoke_retry1/candidate_response.json').exists():
        lines+=['','N=36の2更新・短鎖smokeで学習、前後診断、保存checkpointの読込、6候補の個別除去と再標本化まで完走した。初回smokeではhost配列の`.at`非対応を追加コード内で修正し、失敗ログと初回成果物を保持した。smokeのエネルギーや候補応答は収束結果として集計しない。']
    if (n36/'submission.json').exists():
        jobs=json.loads((n36/'submission.json').read_text());done=[p for p in n36.glob('*/FOLLOWUP_COMPLETE.json')]
        lines+=['',f'投入済みSlurm: {jobs}。解析まで完了した条件は **{len(done)}/18**。',
                'pilot全18条件が2更新を終え、同一seed内の初期波動関数・burn-in後walkers・第1更新のエネルギー・パラメータ数が一致した場合だけ本計算を開始する。']
        if (n36/'PILOT_GATE_PASSED.json').exists():lines+=['','共通初期状態・有限更新のpilot gate: **通過済み**。']
        else:lines+=['','pilot gateは未完了。N=36の学習後比較・候補選択の物理的結論はまだ出ていない。']
        physical=[]
        for case in json.loads((n36/'plan.json').read_text()):
            folder=n36/case['name']
            if (folder/'summary.json').exists():
                v=json.loads((folder/'summary.json').read_text())
                physical.append(dict(name=case['name'],seed=case['config']['experiment']['seed'],embedding=case['config']['embedding']['type'],
                                     penalty=case['config']['embedding']['sparse_penalty'],energy=v['energy'],sem=v['block_sem'],energy_imag=v['energy_imag'],
                                     before_energy=v['before']['energy'],before_sem=v['before']['block_sem']))
        if physical:
            csv_save(root/'n36_completed_summary.csv',physical,replace=True)
            lines+=['','完了条件の数値は `n36_completed_summary.csv`。全seedの整合性とMonte Carlo収束を確認するまで、優劣・結晶化・AHCを確定しない。']
    else:lines+=['','N=36本計算は未投入。数値的な物理比較は未完了。']
    lines+=['','## QWZの新課題と入力監査','',
        'QWZ占有状態と既存canonical atlasから局所リンクを作る。生のフレーム・Projector・物理座標・質量は隠し、2閉路の辺トークンと指定閉路の情報だけを入力する。正解は規約T_ij=〈u_j|u_i〉/|〈u_j|u_i〉|による閉路積（通常の正方向Berry holonomyの複素共役）。',
        '', '遠方に滑らかな周期的SU(2)回転を加え、ギャップとChernを保ちながら局所Berry幾何を変える。同一局所入力・異なるholonomyの対を同一splitに保持する。異なるsplitは独立した物理生成seedを使う。全リンクの積oracleが入力から正解を復元し、局所MLP/2層局所Attentionの観測範囲は対を区別できない。正解一意性は全入力についての性質であり、局所入力からの一意性ではない。',
        '', 'ゲージ共変Attentionの輸送には入力carrierの平坦なフレーム変換を使う。隠したQWZフレームから非局所リンクを供給しない。realモデルとMLPには同じcarrierを使う点ごとの厳密なゲージ除去を与える。これは全方式に同じ局所情報を与えるための既知の前処理であり、学習成果には数えない。',
        '', '主比較は同じデータ、固定最終更新、Adam、batch64、学習率0.003、幅24・2層・3heads。複素モデルは同じ幅でも実自由度が多いため、性能差をAttention形式だけに帰属しない。全パッチ集約MLPを強い対照として追加した。']
    qr=root/'qwz_main'
    if (qr/'TASK_CONTRACT.json').exists():
        checks=json.loads((qr/'TASK_CONTRACT.json').read_text())['test']
        lines+=['',f'学習前に保存したtest課題監査: oracle最大誤差={checks["oracle_max_error"]:.3g}、局所対の入力差={checks["identical_local_twin_max_error"]:.3g}、局所対によるMSE下限={checks["local_twins_mse_lower_bound"]:.6f}。']
    summaries=[json.loads(p.read_text()) for p in qr.glob('*/summary.json')]
    if summaries:
        lines+=['','|モデル|完了seed数|学習後test MSE 平均±seed標準偏差|学習前MSE|長い閉路MSE|実パラメータ自由度|','|---|---:|---:|---:|---:|---:|']
        for kind in ['local_mlp','pooled_mlp','local_attention','full_attention','gauge_covariant','trainable_window']:
            vs=[v for v in summaries if v['kind']==kind]
            if not vs:continue
            after=[v['after']['mse'] for v in vs]
            lines.append(f'|{kind}|{len(vs)}|{np.mean(after):.6f} ± {np.std(after,ddof=1) if len(vs)>1 else 0:.6f}|{np.mean([v["before"]["mse"] for v in vs]):.6f}|{np.mean([v["longer_loop"]["mse"] for v in vs]):.6f}|{vs[0]["real_parameter_dof"]}|')
        lines+=['',f'学習データの円平均を使う定数予測のtest MSE: {summaries[0]["constant_test_mse"]:.6f}。',
                f'完了は{len(summaries)}/18。未完了分を平均に含めていない。MSEはcos/sin各成分の平均。長い閉路は12→16辺の未学習長で、外挿性能の指標。',
                '', '各モデルの前後内部表現・QK・重み・head寄与は `before_internal.npz` / `after_internal.npz`。各層各headの除去、パッチ間接続除去、輸送無効化は `interventions.csv`。遠方トークンを対と交換した予測応答もsummaryに保存。これらを単なるhead振幅からの寄与判定と区別する。']
        reps=[]
        for p in qr.glob('*/after_internal.npz'):
            before=np.load(p.parent/'before_internal.npz');after=np.load(p)
            reps.append(dict(run=p.parent.name,before_target_CKA=_cka(before['representation'],before['target']),
                             after_target_CKA=_cka(after['representation'],after['target'])))
        csv_save(root/'representation_alignment.csv',reps,replace=True)
        lines+=['','`representation_alignment.csv` は同一固定16標本での内部表現とholonomyのCKA（探索的、因果証明ではない）。head介入による誤差変化と合わせて読む。',
                '', 'この課題で得られるのは、供給された局所リンクを選択・合成する処理の証拠である。Berry接続・Chern・Wannier障害をモデルが発見したという証拠ではない。局所モデルの限界は観測範囲で証明できるが、それだけでAttention固有の必要性は証明できない。全体MLP、解析的積oracleとの比較、および未学習長への汎化が必要。']
        if len(summaries)==18:
            lines+=['','### 今回得られた知見','',
                '同一ギャップ・Chernを保つ遠方の幾何変化に対し、Full Attentionは局所モデルよりholonomy合成誤差を下げた。これは単純なProjector再構成で測れなかった、パッチ間通信の効果を示す。今回の小さな全体集約MLPも定数予測付近に留まった。ただし集約MLPは2090自由度でFull Attentionの5690自由度より小さいため、これだけでAttention演算固有の必要性を確定しない。',
                '', 'ゲージ共変モデルはゲージ整合性を構造として満たすが、今回の同一学習予算ではreal Full Attentionより精度が低かった。学習可能Windowにも一律の改善はなかった。対称性を入力・設計として与えることと、大域の合成処理を学習できることは別の性質である。',
                '', '全層を局所半径1に制限した介入は `qwz_main/all_layers_local_control.csv` に保存した。性能低下は通信を使っていた証拠だが、介入に伴う入力分布変化も含む。head除去の効果はモデル・seed・層で異なるので、単一headを普遍的な位相処理器とは呼ばない。']
        controls=[json.loads(p.read_text()) for p in (root/'qwz_capacity_controls').glob('*/summary.json')]
        if controls:
            lines+=['','容量を近づけた追加対照（同じ保存データ・batch順・1200更新・3 seed）:','',
                    '|全体MLP対照|完了seed|実自由度|test MSE 平均±標準偏差|','|---|---:|---:|---:|']
            for kind in ['global_flat_mlp','pooled_mlp_matched']:
                vs=[v for v in controls if v['kind']==kind]
                if vs:
                    values=[v['after']['mse'] for v in vs]
                    lines.append(f'|{kind}|{len(vs)}|{vs[0]["real_parameter_dof"]}|{np.mean(values):.6f} ± {np.std(values,ddof=1) if len(vs)>1 else 0:.6f}|')
            lines+=['','flat MLPは全トークンを平坦化して見る固定長モデルで、長さ外挿は定義しない。pool MLPは幅42でFull Attentionより少し多い自由度を持つ。これらも今回の予算ではFull Attentionに及ばなかったため、単なる局所観測制限や小さいMLP容量だけでは結果を説明できない。ただし万能近似器としてのMLPの不可能性を示したわけではなく、最適化・アーキテクチャ・データ量への依存は残る。']
    else:lines+=['','本比較の学習は未完了。小規模pilotの数値から結論を出さない。']
    lines+=['','## 未解決と結果の扱い','',
        '- N=36の全条件・候補除去の完了、独立seedとBorn鎖の収束が必要。有限chainのblock SEMは収束証明ではない。',
        '- 共通HF初期状態には既に物理構造がある。学習前QKに結晶波数が見えても、入力Embeddingに与えた周期性やHF由来の構造と、学習で獲得した構造を区別する。',
        '- QWZの局所MLP失敗は通信範囲の不足を示す。Attentionという演算形式だけが必要だったという結論にはならない。Chernを保存した変形なので位相分類の性能とも異なる。',
        '- 未完了・失敗条件は保存し、既存結果へ混入しない。本計算終了後のcollectジョブがこの報告を同じ新規成果物として更新する。',
        '', '再生成: `python -c "from research_framework.followup.report import report; report(\'research_framework/results/2026-11-06_followup\')"`。環境は既存 `research_framework/.venv`。']
    target=root.parents[1]/'FOLLOWUP_REPORT_JA.md'
    target.write_text('\n'.join(lines)+'\n')
    return target
