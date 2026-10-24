"""Head射影ベクトルの相殺（独立Diagnostic再評価、Production内部のビット列ではない）。

c_h = exp(out_scale) z_h W_h、zはGate後。共有biasは相殺指標から分離する。
rho = <||sum_h c_h||²> / <sum_h ||c_h||²>。
c^p_h = c_h - c_h(self) に対して同じ比をrho_pとする。
平均はwalkerと電子。比の平均や、平均ノルムの二乗の比ではない。

rho<1は平均交差項が負、rho>1は正。0<=rho<=H（分母が正の場合）。
TanhNormの非線形性によりc^pは純粋な非対角項の応答とは限らない。
自己項除去・Head除去で層列全体を再評価するablation、波動関数/エネルギー変化は測らない。
Head和は診断値の分解に限り、残差接続・LayerNorm・後続層に戻さない。

CLI: head_cancel.py RUN [RUN ...] --settings KNOWN.json --lattice '[[...],[...]]'
新規head_cancel_diagnostic_v1.npzに保存。既存結果は再利用/上書きしない。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from screening_analysis import head_contrib


def cancellation_statistics(head_vectors, self_head_vectors):
    """実数c/c_self [B,L,N,H,E] -> 旧R [L,4]、rho/rho_p [L]と定義域。

    Rの列は <Σ||c||²>, <||Σc||²>, <Σ||c-c_self||²>, <||Σ(c-c_self)||²>。
    自己応答は非線形変換を通した後に差し引く。biasはここへ含めない。
    分母0や非有限値では比をNaNとし、defined=Falseを返す（epsilonで隠さない）。
    """
    c, cd = np.asarray(head_vectors), np.asarray(self_head_vectors)
    if c.ndim != 5 or cd.shape != c.shape or c.dtype.kind != 'f' or cd.dtype.kind != 'f':
        raise ValueError('Headベクトルは同じshape [B,L,N,H,E]の実数配列が必要です')
    cp = c - cd
    terms = np.stack((np.sum(c*c,axis=(-1,-2)), np.sum(c.sum(axis=-2)**2,axis=-1),
                      np.sum(cp*cp,axis=(-1,-2)), np.sum(cp.sum(axis=-2)**2,axis=-1)),axis=-1)
    moments = terms.mean(axis=(0,2))
    defined = np.isfinite(moments[:,0]) & np.isfinite(moments[:,1]) & (moments[:,0]>0)
    defined_p = np.isfinite(moments[:,2]) & np.isfinite(moments[:,3]) & (moments[:,2]>0)
    rho = np.divide(moments[:,1],moments[:,0],out=np.full_like(moments[:,0],np.nan),where=defined)
    rho_p = np.divide(moments[:,3],moments[:,2],out=np.full_like(moments[:,2],np.nan),where=defined_p)
    return moments, rho, rho_p, defined, defined_p


def analyze(folder,nconf=64,checkpoint=None,sel=None,*,settings=None,lattice=None):
    """E2aを介して正式APIを評価し、実際のHeadベクトルから相殺統計だけを導く。

    未保存設定/文字列格子は明示指定が必要。全保存deviceと選択indexを記録する。
    同一プロセスでProductionのtraceと設定一時適用を並行しないこと。
    """
    result = head_contrib.analyze(folder,nconf,checkpoint,sel,settings=settings,lattice=lattice)
    # E2aのR=<||Σc||>とは別物。旧head_cancelのRは平均二乗を並べた4列。
    result['mean_head_sum_norm'] = result['R']
    result['R'],result['rho'],result['rho_p'],result['rho_defined'],result['rho_p_defined'] = (
        cancellation_statistics(result['head_vectors'],result['self_head_vectors']))
    # 非対角Weightが全て0なら、数学的にはc-c(self)=0。別JITの丸め残差で
    # 小さい正の分母が出ても、比を物理的に有意と解釈しない（値を丸め直さない）。
    weights = result['aggregation_weights']
    off_diagonal = ~np.eye(weights.shape[-1],dtype=bool)
    result['rho_p_self_only_warning'] = ~np.any(weights[...,off_diagonal] != 0,axis=(0,2,3))
    metadata = dict(result['metadata'])
    metadata.update(method='head-cancel-diagnostic-v1',parent_method=result['metadata']['method'],
        rho_definition='<||sum_h c_h||^2>/<sum_h ||c_h||^2>; ratio of means over walkers/electrons',
        rho_p_definition='same ratio for c_h-c_h(self); difference AFTER TanhNorm/Gate/projection',
        moment_columns=['mean_sum_head_norm_squared','mean_head_sum_norm_squared',
                        'mean_sum_self_response_difference_norm_squared','mean_self_response_difference_sum_norm_squared'],
        bias_interpretation='shared projected bias saved separately; excluded from both cancellation ratios',
        zero_denominator='undefined: NaN ratio and False defined flag; no epsilon regularization',
        defined_flags='finite floating-point numerator and positive denominator, not a mathematical stability certificate',
        self_only_warning='all off-diagonal aggregation weights zero: rho_p mathematically undefined; raw rounding-residue ratio retained')
    metadata['source_sha256'] = dict(metadata['source_sha256'])
    metadata['source_sha256'][str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['metadata'] = metadata
    return result


def compute(folder,nconf=64,checkpoint=None,sel=None,*,settings=None,lattice=None):
    """旧8要素tupleとR [L,4]を保持。計算/設定復元はE2aと共有する。"""
    r = analyze(folder,nconf,checkpoint,sel,settings=settings,lattice=lattice)
    meta = dict(shell=r['shell'],sigma=r['w_sigma'],centre=r['w_centre'],outer=r['w_outer'],
                A=r['A'],Ap=r['Ap'],fixv=r['fixv'],metadata=r['metadata'])
    return *(r[k] for k in ('C','Cd','Cp','X','R','w','softmax')),meta


def save(result,output):
    """新規NPZのみ保存。旧解析名への保存も拒否。allow_pickle=Falseで読める。"""
    output = Path(output)
    if output.name in ('head_cancel.npz','head_cancel.npy','head_contrib.npz','head_contrib.npy'):
        raise ValueError('旧解析名には保存しません')
    # E2aの保存関数はmethodを固定せず、入力metadataをそのまま保存する。
    head_contrib.save(result,output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders',nargs='+'); parser.add_argument('--nconf',type=int,default=64)
    parser.add_argument('--settings'); parser.add_argument('--lattice',type=json.loads)
    parser.add_argument('--checkpoint'); parser.add_argument('--output')
    args = parser.parse_args()
    if len(args.folders)>1 and (args.checkpoint or args.output):
        parser.error('--checkpoint/--outputは単一folderに対して指定してください')
    # 多folderでも書き込む前に、出力先の重複・既存ファイルを検出する。
    targets = [Path(args.output) if args.output else Path(f)/'head_cancel_diagnostic_v1.npz' for f in args.folders]
    if len({p.resolve() for p in targets}) != len(targets):parser.error('出力先が重複しています')
    for target in targets:
        if target.exists():parser.error('既存結果は上書きしません: '+str(target))
    for folder,target in zip(args.folders,targets):
        r = analyze(folder,args.nconf,args.checkpoint,settings=args.settings,lattice=args.lattice)
        save(r,target)
        print('head-cancel-diagnostic-v1: '+str(target))
        for l,(rho,rho_p) in enumerate(zip(r['rho'],r['rho_p'])):
            print(f'{Path(folder).name} 層{l+1}: rho={rho:.6g} rho_p={rho_p:.6g} '
                  f'defined={bool(r["rho_defined"][l])}/{bool(r["rho_p_defined"][l])}')
        if np.any(r['rho_p_self_only_warning']):
            print('WARNING: 自己項のみの層ではrho_pは数学的に未定義。微小丸め残差の比も解釈しない: '+
                  str(np.flatnonzero(r['rho_p_self_only_warning']).tolist()))
        print('rho_pは非線形な自己応答との差。Head除去・波動関数・エネルギーの寄与ではない。')


if __name__=='__main__':main()
