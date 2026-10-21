"""1 層目のアテンションの重みが「全員の擬スピンを反転し、セルの半分（格子 3 個ぶん）ずらす」操作で変わらないことの確認（報告 3 節の「理由」）。
入力は (sin θ, cos θ, s) で定数項がないので、θ→θ+π・s→−s で入力が丸ごと符号を変え、q̄·k̄ は偶なので変わらない。対照として擬スピンだけの反転も見る。"""
import sys
sys.path.insert(0, '/home/claude/lj2res/ana')
import numpy as np, jax.numpy as jnp
import lj2net as LN, readout as RO

for lam in LN.LAMS:
    for typ in ('cw', 'ca'):
        run = LN.load(lam, typ)
        internals, _ = RO.make_parts(run)
        L = run['L']; t = 0.5 * (L[:, 0] + L[:, 1])
        ds = dw = dflip = 0.0
        for k in range(4):
            R = run['R'][k].reshape(36, 2); S = run['S'][k]
            A = internals(jnp.asarray(R.reshape(-1), jnp.float32), jnp.asarray(S, jnp.float32))
            B = internals(jnp.asarray((R + t).reshape(-1), jnp.float32), jnp.asarray(-S, jnp.float32))
            C = internals(jnp.asarray(R.reshape(-1), jnp.float32), jnp.asarray(-S, jnp.float32))
            ds = max(ds, float(jnp.max(jnp.abs(A['s'] - B['s']))))
            dw = max(dw, float(jnp.max(jnp.abs(A['wgt'] - B['wgt']))))
            dflip = max(dflip, float(jnp.max(jnp.abs(A['s'] - C['s']))))
        print(f'{lam} {typ}  反転＋ずらし: max|Δs| {ds:.1e}  max|Δ(α m)| {dw:.1e}   反転だけ: max|Δs| {dflip:.2f}')
