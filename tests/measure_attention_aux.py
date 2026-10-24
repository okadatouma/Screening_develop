"""B1固定参照とB2取得OFF/ONのCPUメモリ・時間を測る。JSONは新規保存のみ。"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from tests.support import require_numerics
require_numerics()
import jax
import jax.numpy as jnp
import numpy as np
from periodicwave import attention, CustomScreenformer as CS
from tests import screening_reference as old


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.output.exists(): parser.error('既存結果は上書きしません')
    reports=[]
    for x64 in (False,True):
        jax.config.update('jax_enable_x64',x64)
        opts=dict(num_electrons=64,num_heads=4,attn_dim=16,value_dim=16,d_ref=1.4,
                  use_softmask=True,use_fixed_value=False,use_tanh_norm=True,use_gate=True,use_output_scale=True)
        init,_=CS.make_screening_attention(**opts,embed_dim=64)
        p=init(jax.random.PRNGKey(29))
        x=jnp.sin(jnp.arange(4096).reshape(64,64)/13.)
        d=jnp.abs(jnp.arange(64)[:,None]-jnp.arange(64)[None,:])/64.
        kw=dict(options=opts,trim_max=2.,trim_pow=2,win_shape='cos',freeze_w=False,clip_r=None,clip_beta=0.3)
        functions=dict(before=lambda p,x,d:old.screening_attention(p,x,d,**kw),
                       off=lambda p,x,d:attention.screening_attention(p,x,d,**kw),
                       on=lambda p,x,d:attention.screening_attention(p,x,d,**kw,return_aux=True))
        graphs={name:str(jax.make_jaxpr(fn)(p,x,d)) for name,fn in functions.items()}
        assert graphs['before']==graphs['off'],'通常経路のJAXPRが異なる'
        compiled={name:jax.jit(fn).lower(p,x,d).compile() for name,fn in functions.items()}
        before=compiled['before'](p,x,d);off=compiled['off'](p,x,d);on=compiled['on'](p,x,d)
        np.testing.assert_array_equal(before,off)
        np.testing.assert_array_equal(off,on[0])
        records={}
        for name,fn in compiled.items():
            for _ in range(5):jax.block_until_ready(fn(p,x,d))
            mem=fn.memory_analysis()
            memory=(None if mem is None else {k:getattr(mem,k) for k in
                ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')})
            records[name]=dict(memory=memory,samples_ms=[])
        # 9回の交互測定。compileを含めず、各50回を完了待ちする。
        names=list(compiled)
        for sample in range(9):
            for name in names[sample%3:]+names[:sample%3]:
                start=time.perf_counter()
                for _ in range(50):jax.block_until_ready(compiled[name](p,x,d))
                records[name]['samples_ms'].append((time.perf_counter()-start)*1000/50)
        for record in records.values():record['median_ms']=statistics.median(record['samples_ms'])
        assert records['before']['memory']==records['off']['memory'],'通常経路のメモリ見積りが異なる'
        reports.append(dict(x64=x64,shape=dict(N=64,H=4,D=16,E=64),jaxpr_equal=True,
            normal_jaxpr_sha256=hashlib.sha256(graphs['off'].encode()).hexdigest(),measurements=records))
    args.output.write_text(json.dumps(dict(jax=jax.__version__,backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],warmup=5,samples=9,calls_per_sample=50,results=reports),indent=2))
    print('Memory/JAXPR/performance record:',args.output)


if __name__=='__main__':main()
