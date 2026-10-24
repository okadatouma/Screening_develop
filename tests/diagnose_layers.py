"""D候補のJIT差を調査する。候補適用時のみ実行。基準や本体は変更しない。"""
import argparse
import json
from pathlib import Path

from tests.check_layers import fixture, CANDIDATE_AVAILABLE
from tests.support import require_numerics
require_numerics()
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import patch
from periodicwave import CustomScreenformer as CS


def difference(a,b):
    return max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in
               zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if not CANDIDATE_AVAILABLE:parser.error('D候補は撤回済み。検証コピーにcandidate.patchを適用してから実行する')
    if args.output.exists():parser.error('既存記録を上書きしない')
    reports=[]
    for x64 in (False,True):
        jax.config.update('jax_enable_x64',x64)
        for n,ln in ((1,False),(1,True),(2,False),(2,True),(3,True)):
            p,_,apply,previous,settings,inputs,pos=fixture(n,ln,'cos')
            x,d,pair=inputs(pos)
            normal=lambda pp,xx,dd,pf:apply(pp,xx,dd,pf)
            aux=lambda pp,xx,dd,pf:apply(pp,xx,dd,pf,return_aux=True)
            discard=lambda pp,xx,dd,pf:apply(pp,xx,dd,pf,return_aux=True)[0]
            with patch.multiple(CS,**settings):
                args4=(p,x,d,pair)
                a=jax.jit(normal)(*args4);b,info=jax.jit(aux)(*args4)
                c=jax.jit(discard)(*args4);before=jax.jit(previous)(*args4)
                graph=str(jax.make_jaxpr(normal)(*args4))
                row=dict(x64=x64,num_layers=n,use_layer_norm=ln,
                    eager_off_on=difference(normal(*args4),aux(*args4)[0]),
                    old_jit_vs_off=difference(before,a),jit_off_on=difference(a,b),
                    jit_off_aux_discarded=difference(a,c),
                    normal_jaxpr_matches_old=graph==str(jax.make_jaxpr(previous)(*args4)),
                    normal_jaxpr_matches_aux_discarded=graph==str(jax.make_jaxpr(discard)(*args4)))
                def f(pp,r):return jnp.sum(jnp.sin(apply(pp,*inputs(r))))
                def f_aux(pp,r):
                    out,records=apply(pp,*inputs(r),return_aux=True)
                    return jnp.sum(jnp.sin(out)),records
                v,g=jax.jit(jax.value_and_grad(f,argnums=(0,1)))(p,pos)
                (va,_),ga=jax.jit(jax.value_and_grad(f_aux,argnums=(0,1),has_aux=True))(p,pos)
                h=jax.jit(jax.hessian(f,argnums=1))(p,pos)
                ha,_=jax.jit(jax.hessian(f_aux,argnums=1,has_aux=True))(p,pos)
                row.update(loss_difference=difference(v,va),params_grad_difference=difference(g[0],ga[0]),
                           coordinate_grad_difference=difference(g[1],ga[1]),hessian_difference=difference(h,ha))
                if n==2 and ln:
                    for label,fun in [('off',normal),('on',aux)]:
                        text=jax.jit(fun).lower(*args4).compile().as_text()
                        args.output.with_name(f'hlo-{x64}-{label}.txt').write_text(text)
                    # 局所的な原因切分けだけ。通常環境・本体に適用しない。
                    try:
                        compiler_options={'xla_disable_hlo_passes':'fusion'}
                        af=jax.jit(normal,compiler_options=compiler_options)(*args4)
                        bf=jax.jit(aux,compiler_options=compiler_options)(*args4)[0]
                        row['diagnostic_fusion_disabled_off_on']=difference(af,bf)
                    except Exception as error:row['diagnostic_fusion_disabled_error']=str(error)
                reports.append(row)
    args.output.write_text(json.dumps(dict(backend=jax.default_backend(),jax=jax.__version__,
        cases=reports,conclusion='Diagnostic only; no tolerance or production compiler setting changed'),indent=2))
    print('Saved:',args.output)


if __name__=='__main__':main()
