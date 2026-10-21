# 表 2（12 本のアテンション側の量）と、同じ λ の 2 つの型の差（AHC 型 − WC 型）を出す。
# 入力：同じフォルダの pstype.json（pstype.py の出力）と readouts.json（run_readouts.py の出力）。
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ps = json.load(open(os.path.join(HERE, 'pstype.json')))['runs']
ro = json.load(open(os.path.join(HERE, 'readouts.json')))

def quantities(key):
    A = np.array(ps[key]['A'])                                   # [head, σ, σ']
    same = 0.5 * (A[:, 0, 0] + A[:, 1, 1]).mean()               # 同じ向きの ᾱ（ヘッド平均）
    opp = 0.5 * (A[:, 0, 1] + A[:, 1, 0]).mean()                # 逆向きの ᾱ
    r = ro[key]
    wq = float(np.sum(np.array(r['q']) * np.array(r['w'])))     # 寄与率で重みをつけた窓の幅
    return dict(same=same, opp=opp, diff=same - opp, rho=r['rho_p'], wq=wq)

for lam in ['0.33', '0.4', '0.48', '0.56', '0.64', '0.72']:
    wc, ahc = quantities(f'{lam}_cw'), quantities(f'{lam}_ca')
    for name, q in (('WC', wc), ('AHC', ahc)):
        print(f"{float(lam):.2f} {name:3s}  same {q['same']:.3f}  opp {q['opp']:.3f}  diff {q['diff']:.3f}"
              f"  rho_p {q['rho']:.3f}  <w>_q {q['wq']:.2f}")
    print('      AHC-WC: ' + '  '.join(f"{k} {ahc[k] - wc[k]:+.3f}" for k in ('diff', 'rho', 'wq')))
