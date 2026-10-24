"""比較条件をJSONへ固定し、Slurm arrayの各taskで1条件だけ実行する。"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-task', type=int, help='既存plan中のindexを実行')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out-root', type=Path)
    p.add_argument('--cases', type=Path, help='[{lam,wc_hf,wc_meta,ahc_hf,ahc_meta},...]')
    p.add_argument('--lam', type=float, default=.64)
    for name in ('wc-hf', 'ahc-hf', 'wc-meta', 'ahc-meta'): p.add_argument('--'+name)
    p.add_argument('--phase', choices=['inputs', 'windows', 'pairs', 'recommended', 'all'], default='inputs')
    p.add_argument('--chosen-input', choices=['E0', 'E1', 'E2', 'E3'], default='E2')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--python', default=sys.executable, help='大学の研究環境Pythonの絶対パス推奨')
    args, extra = p.parse_known_args()
    if args.execute_task is not None:
        plan = json.loads(args.plan.read_text())
        if not 0 <= args.execute_task < len(plan['jobs']): raise ValueError('task index範囲外')
        job = plan['jobs'][args.execute_task]
        subprocess.run(job['argv'], cwd=plan['repo'], check=True)
        return
    controlled = {'--out', '--resume', '--smoke', '--hf', '--hf-meta', '--label', '--lam',
                  '--preset', '--input', '--window', '--pair-windows', '--no-pair-windows', '--freeze-window'}
    if any(token.split('=')[0] in controlled for token in extra):
        p.error('比較条件を追加引数で上書きできません。phase/chosen-input/casesで指定してください')
    if not args.out_root: p.error('--out-rootが必要です')
    cases = json.loads(args.cases.read_text()) if args.cases else [dict(lam=args.lam,
        wc_hf=args.wc_hf, wc_meta=args.wc_meta, ahc_hf=args.ahc_hf, ahc_meta=args.ahc_meta)]
    conditions = []
    phases = ['inputs', 'windows', 'pairs', 'recommended'] if args.phase == 'all' else [args.phase]
    for phase in phases:
        if phase == 'inputs':
            conditions.extend((f'inputs_{e}', ['--input', e, '--window', 'W0', '--freeze-window']) for e in ('E0', 'E1', 'E2', 'E3'))
        if phase == 'windows':
            conditions.extend((f'windows_{w}', ['--input', args.chosen_input, '--window', w]) for w in ('W0', 'W1', 'W2', 'W3'))
        if phase == 'pairs':
            # 同一のValue/Gateで共有窓と対依存窓を比較。提案Bとの混同を避ける。
            conditions.extend((f'pairs_{w}_{pair}', ['--input', 'E3', '--window', w,
                '--pair-windows' if pair else '--no-pair-windows']) for w in ('W1', 'W2') for pair in (False, True))
        if phase == 'recommended': conditions.extend((f'preset_{s}', ['--preset', s]) for s in ('A', 'B', 'C', 'D'))
    jobs = []
    for case in cases:
        for label, prefix in [('WC', 'wc'), ('AHC', 'ahc')]:
            hf = case.get(prefix+'_hf'); meta = case.get(prefix+'_meta')
            if not hf or not meta: p.error('WC/AHCそれぞれのHFパス・メタデータパスが必要です。存在検査はtask実行時にも行います')
            for seed in map(int, args.seeds.split(',')):
                for name, condition in conditions:
                    folder = args.out_root.resolve()/f'lam{case["lam"]:g}'/label/name/f'seed{seed}'
                    command = [args.python, '-m', 'periodicwave.window_experiments.run', '--require-gpu', '--out', str(folder),
                        '--hf', str(Path(hf).expanduser().absolute()), '--hf-meta', str(Path(meta).expanduser().absolute()),
                        '--lam', str(case['lam']), '--label', label, '--seed', str(seed)] + condition + extra
                    jobs.append(dict(name=f'{label}_{name}_{seed}', argv=command))
    if len({job['argv'][job['argv'].index('--out')+1] for job in jobs}) != len(jobs):
        raise ValueError('出力先の重複。λ条件やseedが重複していませんか')
    plan = dict(version='window-suite-v1', repo=str(Path(__file__).resolve().parents[2]), jobs=jobs,
        note='WC/AHCはHF初期状態ラベル。相判定ではない。未存在HF/条件不一致はtask開始時に停止')
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    with args.plan.open('x') as f: json.dump(plan, f, indent=2)
    print(f'{len(jobs)} jobs: {args.plan.resolve()}')
    script = Path(__file__).with_name('submit.slurm')
    print('sbatch --array=0-'+str(len(jobs)-1)+'%1 --export=ALL,PW_PLAN='+shlex.quote(str(args.plan.resolve()))+' '+shlex.quote(str(script)))
    print('通常実行: '+shlex.join([args.python, '-m', 'periodicwave.window_experiments.suite', '--plan', str(args.plan), '--execute-task', '0']))


if __name__ == '__main__': main()
