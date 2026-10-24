"""Screenformer Head寄与（独立Diagnostic再評価、Production内部のビット列ではない）。

c_h = exp(s_O) z_h W_O[h]、zはAPIのgate_output。biasは共有項として別保存。
C=<||c_h||>, Cd=<||c_h(self)||>, Cp=<||c_h-c_h(self)||>。
selfは同じ層入力で自己項だけを通す反実仮想。Cpは純粋な二体/Head除去/エネルギー寄与ではない。
R=<||sum_h c_h||>、A=<||aggregate_h||>、Ap=<||aggregate_h-self_h||>。
平均はwalkerと電子に取る。C²のshareは平均ノルムの二乗（平均二乗ではない）。
Head分解の和は解析用であり、次層入力には使用しない。

CLI: head_contrib.py RUN --settings known.json --lattice '[[...],[...]]'
新規head_contrib_diagnostic_v1.npzだけに保存し、既存ファイルを上書きしない。
旧softmax解析はこの入口の対象外。他の解析スクリプトは変更しない。
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import attr
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from periodicwave import networks, CustomScreenformer as CS
from screening_analysis.attn_probe import SETTINGS, _effective_settings, lattice_from_config, _centres_from


def projected_heads(gate_output, projection, num_heads, scale=1.):
    """[N,H*D]をc[N,H,E]へ分解。共有bias[E]はどのHeadにも割り当てない。

    実数演算ではsum(c,head)+bias=scale*(z@W+b)。加算順の違う解析的分解で、
    forwardや次層更新には使用しない。現行checkpointのprojectionはbiasなしの配列。
    """
    w = projection['w'] if isinstance(projection, dict) else projection
    b = projection.get('b') if isinstance(projection, dict) else None
    n, hd = gate_output.shape
    c = scale * jnp.einsum('ihd,hde->ihe', gate_output.reshape(n,num_heads,hd//num_heads),
                          w.reshape(num_heads,hd//num_heads,-1))
    bias = jnp.zeros(w.shape[-1],dtype=c.dtype) if b is None else scale*b
    return c, bias


def analyze(folder, nconf=64, checkpoint=None, sel=None, *, settings=None, lattice=None):
    """全保存deviceから選択。未保存の実効設定・元精度格子は明示指定が必要。

    各walkerの正式APIを独立JIT。自己項統計だけは、同じ層入力を固定して既存B2の
    _diag_only介入を別途評価する（層列を再評価/更新しない）。本体の数式はコピーしない。
    設定一時適用中に同一プロセスでProductionのtraceを並行しないこと。
    """
    folder = Path(folder)
    cfg_path = folder / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg["network"]["network_type"] != "CustomScreenformer" or cfg["system"]["ndim"] != 2:
        raise ValueError("この入口は2D CustomScreenformer専用です")
    effective, provenance = _effective_settings(cfg, settings, None)
    stored_lattice = lattice_from_config(cfg)
    if isinstance(cfg["system"]["pbc_lattice"], str) and lattice is None:
        raise ValueError("文字列化格子は元精度を保証できません。lattice=または--latticeで元の格子を指定してください")
    L = stored_lattice if lattice is None else lattice_from_config({"system": {"pbc_lattice": lattice}})
    candidates = sorted(folder.glob("qmcjax_ckpt_*.npz"))
    if checkpoint is None and not candidates:
        raise FileNotFoundError("checkpointがありません: " + str(folder))
    ck = Path(checkpoint) if checkpoint is not None else candidates[-1]
    with np.load(ck, allow_pickle=True) as z:
        raw_params, data, step = z["params"].item(), z["data"].item(), int(z["t"])
    devices, walkers = data["positions"].shape[:2]
    for leaf in jax.tree_util.tree_leaves(raw_params):
        a = np.asarray(leaf)
        if a.ndim == 0 or a.shape[0] != devices or any(not np.array_equal(a[0], v) for v in a[1:]):
            raise ValueError("checkpointのパラメータreplicaが不一致または未対応のshapeです")
    P = jax.tree_util.tree_map(lambda a: np.asarray(a)[0], raw_params)
    batch = {}
    for key in ("positions", "spins", "atoms", "charges"):
        a = np.asarray(data[key])
        if a.shape[:2] != (devices, walkers):
            raise ValueError("checkpoint dataのdevice/walker軸が不一致: " + key)
        batch[key] = a.reshape((-1,) + a.shape[2:])
    if nconf is not None and (type(nconf) is not int or nconf <= 0):
        raise ValueError("nconfは正の整数です")
    count = min(nconf or devices*walkers, devices*walkers)
    indices = np.linspace(0, devices*walkers-1, count).astype(int)
    batch = {key: value[indices] for key, value in batch.items()}
    if sel is not None:
        indices = np.asarray(sel)
        if indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in 'iu' or np.any(indices < 0) or np.any(indices >= devices*walkers):
            raise ValueError("selは全保存walkerに対する空でない整数index列です")
        # nconfによる予備選択ではなく、元のdevice-major indexから選ぶ。
        batch = {key: np.asarray(data[key]).reshape((-1,)+np.asarray(data[key]).shape[2:])[indices]
                 for key in ("positions","spins","atoms","charges")}
        count = len(indices)
    if not effective["jax_enable_x64"] and any(np.asarray(a).dtype == np.float64
            for a in jax.tree_util.tree_leaves((P, batch))):
        raise ValueError("float64 checkpointをx64無効で切り捨てることはできません")

    globals_before = {k: getattr(CS, k) for k in SETTINGS[:6]}
    try:
        for key in SETTINGS[:6]:
            setattr(CS, key, effective[key])
        with jax.experimental.enable_x64(effective["jax_enable_x64"]), jax.default_matmul_precision(effective["jax_default_matmul_precision"]):
            nspins = tuple(cfg["system"]["electrons"])
            network = cfg["network"]
            feature_kwargs = dict(network.get("make_feature_layer_kwargs", {}))
            if "lattice" in feature_kwargs:
                value = feature_kwargs["lattice"]
                feature_L = lattice_from_config({"system": {"pbc_lattice": value}})
                if isinstance(value, str) and lattice is None:
                    raise ValueError("feature latticeも文字列です。元精度の共通latticeを指定してください")
                if lattice is not None:
                    if not np.array_equal(feature_L, stored_lattice):
                        raise ValueError("共通格子の上書きに対してfeature latticeが異なります。元の構造化configが必要です")
                    feature_L = L
                feature_kwargs["lattice"] = jnp.asarray(feature_L)
            if any(isinstance(v, str) and v.startswith("unserializable object:") for v in feature_kwargs.values()):
                raise ValueError("feature kwargsに復元不能な配列があります")
            if not network.get("make_feature_layer_fn") or not isinstance(network.get("make_feature_layer_kwargs"),dict):
                raise ValueError("feature factoryとkwargsをconfigに明示してください。特徴の定義を推測しません")
            feature_name = network["make_feature_layer_fn"]
            module, name = feature_name.rsplit(".", 1)
            feature_factory = getattr(importlib.import_module(module), name)
            fl = feature_factory(natoms=batch["charges"].shape[-1], nspins=nspins, ndim=2, **feature_kwargs)
            opts = dict(network["CustomScreenformer"])
            required = ("num_layers", "num_heads", "attn_dim", "value_dim", "mlp_dim",
                        "num_perceptrons_per_layer", "use_layer_norm", "mlp_activation_fct",
                        "use_softmask", "use_gate", "use_tanh_norm", "use_fixed_value", "use_output_scale")
            if any(k not in opts for k in required):
                raise ValueError("必須のScreenformer構成がconfigにありません")
            # HF/波動関数構成はtrunk診断には不要。外部HF表もロードしない。
            hf_keys = {"lj_hf_path", "lj_lam", "lj_cusps", "lj_net_scale", "lj_jastrow"}
            unknown = set(opts) - {a.name for a in attr.fields(CS.ScreenformerOptions)} - hf_keys
            if unknown:
                raise ValueError("未対応のScreenformer設定: " + str(sorted(unknown)))
            opts = {k: v for k, v in opts.items() if k not in hf_keys}
            dref = CS._max_periodic_distance(jnp.asarray(L))
            # 初期値だけの設定。診断はinitを呼ばず、学習済み配列をそのまま使う。
            opts["init_softmask_centres"] = _centres_from(P, dref, opts["num_layers"])
            ap0 = P["layers"]["attention"][0]
            opts["init_softmask_rings"] = None
            if "mask_rings" in ap0:
                delta = float(np.logaddexp(0., np.asarray(ap0["mask_omega"]))[0])
                opts["init_softmask_rings"] = (np.asarray(ap0["mask_rings"]), delta,
                    float(ap0["mask_envp"]), "gauss" if "mask_gauss" in ap0 else "sin")
            options = CS.ScreenformerOptions(feature_layer=fl, pbc_lattice=jnp.asarray(L), ndim=2, **opts)
            if len(P["layers"]["attention"]) != options.num_layers:
                raise ValueError("configとcheckpointの層数が不一致")
            ln = P["layers"]["ln"]
            if len(ln) != (options.num_layers if options.use_layer_norm else 0):
                raise ValueError("configとcheckpointのLayerNorm構造が不一致")
            for ap, mlp in zip(P["layers"]["attention"], P["layers"]["mlp"]):
                flags = {"v_w": not options.use_fixed_value, "g_w": options.use_gate,
                         "out_scale": options.use_output_scale, "mask_omega": options.use_softmask}
                if any((key in ap) != enabled for key, enabled in flags.items()):
                    raise ValueError("configとcheckpointのValue/Gate/Window/出力倍率の構造が不一致")
                if (ap["q_w"].shape != (options.mlp_dim, options.num_heads*options.attn_dim)
                        or ap["k_w"].shape != ap["q_w"].shape
                        or ap["trim_rho"].shape != (options.num_heads,)
                        or len(mlp) != options.num_perceptrons_per_layer):
                    raise ValueError("configとcheckpointのAttention/MLP次元が不一致")
            diagnostic = jax.jit(CS.make_screenformer_diagnostic(nspins, options))
            geometry = jax.jit(networks.make_periodic_construct_input_features(jnp.asarray(L)), static_argnames=("ndim",))
            core_options = dict(num_electrons=sum(nspins),num_heads=options.num_heads,
                attn_dim=options.attn_dim,value_dim=options.value_dim,d_ref=dref,
                use_softmask=options.use_softmask,use_fixed_value=options.use_fixed_value,
                use_tanh_norm=options.use_tanh_norm,use_gate=options.use_gate,
                use_output_scale=options.use_output_scale)
            # 自己項だけの反実仮想も本体kernelを使う。QKV/Trim/窓/非線形の別実装はない。
            self_attention = jax.jit(lambda ap,x,d,pair: CS.attention.screening_attention(
                dict(ap,_diag_only=jnp.ones(options.num_heads)),x,d,pair,options=core_options,
                trim_max=effective['TRIM_MAX'],trim_pow=effective['TRIM_POW'],
                win_shape=effective['WIN_SHAPE'],freeze_w=effective['FREEZE_W'],
                clip_r=effective['CLIP_R'],clip_beta=effective['CLIP_BETA'],return_aux=True))
            samples=[]
            for i in range(count):
                args=[jnp.asarray(batch[k][i]) for k in ('positions','spins','atoms','charges')]
                final_x,layers=diagnostic(P,*args)
                distance=geometry(args[0],args[2],ndim=2)[3][...,0]
                rows=[]
                for record,ap in zip(layers,P['layers']['attention']):
                    aux=record['attention']
                    _,self_aux=self_attention(ap,record['input'],distance,aux['pair_features'])
                    scale=jnp.exp(ap['out_scale']) if options.use_output_scale else 1.
                    c,bias=projected_heads(aux['gate_output'],ap['attn_output'],options.num_heads,scale)
                    cd,_=projected_heads(self_aux['gate_output'],ap['attn_output'],options.num_heads,scale)
                    a=aux['aggregate']; ad=self_aux['aggregate']
                    norm=lambda value: np.linalg.norm(np.asarray(value),axis=-1)
                    rows.append(dict(C=norm(c),Cd=norm(cd),Cp=norm(c-cd),X=norm(record['input']),
                        R=norm(c.sum(axis=1)),A=norm(a),Ap=norm(a-ad),
                        head_vectors=np.asarray(c),self_head_vectors=np.asarray(cd),bias=np.asarray(bias),
                        attention_output=np.asarray(record['attention_output']),
                        decomposition_sum=np.asarray(c.sum(axis=1)+bias),
                        aggregation_weights=np.asarray(aux['aggregation_weights']),
                        weights_window=np.asarray(aux['weights_window']),aggregate=np.asarray(a),
                        self_aggregate=np.asarray(ad),gate_output=np.asarray(aux['gate_output']),
                        tanh_output=np.asarray(aux['tanh_output']),weights_intervened=np.asarray(aux['weights']),
                        layer_input=np.asarray(record['input']),layer_output=np.asarray(record['output'])))
                samples.append({key:np.stack([row[key] for row in rows]) for key in rows[0]})
            result={key:np.stack([s[key] for s in samples]) for key in samples[0]}
            for key in ('C','Cd','Cp','X','R','A','Ap'):
                result[key]=result[key].mean(axis=(0,2))
            # 旧CLIの派生指標も定義を維持する。ゼロ総量は未定義としてNaNを明示する。
            for source,target in (('C','q_C'),('Cp','q_Cp'),('Ap','q_Ap')):
                power=result[source]**2
                result[target]=power/power.sum() if power.sum()>0 else np.full_like(power,np.nan)
            A=result['A']
            unsat=A/np.maximum(np.tanh(A),1e-12)
            power=(result['Cp']*unsat)**2
            result['q_g_heuristic']=power/power.sum() if power.sum()>0 else np.full_like(power,np.nan)
            result['compression_heuristic']=np.divide(np.tanh(A),A,out=np.full_like(A,np.nan),where=A>1e-12)
            result['alignment_ratio']=np.divide(result['C'].sum(axis=-1),result['R'],
                out=np.full_like(result['R'],np.nan),where=result['R']>0)
            # 表示用の長さだけをcheckpointから読む。輪切りsigmaは旧head_contribと同じ全幅。
            widths=[];centres=[];outer=[];representative=[];shells=[]
            for ap in P['layers']['attention']:
                sigma=np.asarray(jax.nn.softplus(ap['mask_omega'])) if 'mask_omega' in ap else np.full(options.num_heads,np.nan)
                centre=np.asarray(ap['mask_centre']) if 'mask_centre' in ap else np.zeros_like(sigma)
                shell='mask_centre' in ap or 'mask_rings' in ap
                edge=centre+sigma
                if 'mask_rings' in ap:
                    rings=np.asarray(ap['mask_rings']);centre=.5*(rings[:-1]+rings[1:])
                    sigma=np.diff(rings);edge=rings[1:]
                widths.append(sigma);centres.append(centre);outer.append(edge)
                representative.append(centre if shell else sigma);shells.append(shell)
            result.update(w=np.asarray(representative),w_sigma=np.asarray(widths),
                w_centre=np.asarray(centres),w_outer=np.asarray(outer),shell=np.asarray(all(shells)),
                softmax=np.asarray(False),fixv=np.asarray(options.use_fixed_value),lattice=L,
                layer_index=np.arange(options.num_layers),**batch)
    finally:
        for key, value in globals_before.items():
            setattr(CS, key, value)
    result["metadata"] = dict(method="head-contrib-diagnostic-v1", weight_definition="aggregation_weights",
        legacy_weight_definition="Trim^2 * legacy cosine window; fixed-Value diagonal removed; other interventions ignored",
        interpretation="independent diagnostic reevaluation; not Production-internal bit patterns",
        checkpoint=str(ck.resolve()), checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(cfg_path.read_bytes()).hexdigest(), step=step,
        settings=effective, settings_provenance=provenance, lattice_override=lattice is not None,
        selected_indices=indices.tolist(), saved_devices=devices,
        feature_factory=feature_name, feature_kwargs=network.get("make_feature_layer_kwargs", {}),
        network_options=network["CustomScreenformer"], jax=jax.__version__, jaxlib=jaxlib.__version__, numpy=np.__version__,
        parameter_dtypes=sorted({str(np.asarray(a).dtype) for a in jax.tree_util.tree_leaves(P)}),
        input_dtypes={key: str(value.dtype) for key, value in batch.items()},
        backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
        xla_flags=os.environ.get("XLA_FLAGS"), runtime_settings_recovered_from_checkpoint=False,
        source_sha256={str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in {__file__, CS.__file__, CS.attention.__file__, networks.__file__,
                         CS.network_blocks.__file__, importlib.import_module('screening_analysis.attn_probe').__file__,
                         importlib.import_module(feature_factory.__module__).__file__}},
        wavefunction_status="not evaluated; Head norms do not certify wavefunction stability")
    result['metadata'].update(head_definition='c_h=exp(s_O)*gate_output_h@W_O[h]; shared bias separate',
        self_definition='same layer input, native B2 _diag_only=1 counterfactual; no layer update',
        Cp_definition='mean norm(c_h-c_h(self)); nonlinear self-response difference, not pure two-body',
        statistic='mean norm over walkers/electrons; squared shares square the mean over ALL layers/heads, not mean-square',
        q_g_definition='normalize((Cp*A/max(tanh(A),1e-12))**2); legacy heuristic, not exact unsaturation',
        decomposition='sum_h c_h + bias; analysis only, different floating-point accumulation order')
    return result


def compute(folder,nconf=64,checkpoint=None,sel=None,*,settings=None,lattice=None):
    """旧8要素tupleのshapeを保持。Screening専用、意味はmodule docstring参照。"""
    r=analyze(folder,nconf,checkpoint,sel,settings=settings,lattice=lattice)
    meta=dict(shell=r['shell'],sigma=r['w_sigma'],centre=r['w_centre'],outer=r['w_outer'],
              A=r['A'],Ap=r['Ap'],fixv=r['fixv'],metadata=r['metadata'])
    return *(r[k] for k in ('C','Cd','Cp','X','R','w','softmax')),meta


def save(result,output):
    """旧キャッシュも新規結果も上書きしない。object配列を含まないNPZ。"""
    output=Path(output)
    if output.name in ('head_contrib.npz','head_contrib.npy'):
        raise ValueError('旧解析名には保存しません')
    payload={k:v for k,v in result.items() if k!='metadata'}
    payload['metadata_json']=np.asarray(json.dumps(result['metadata'],ensure_ascii=False))
    with output.open('xb') as handle:np.savez(handle,**payload)


def cached(folder,nconf=64,*,settings=None,lattice=None,checkpoint=None,sel=None,output=None):
    """新方式で計算・保存。旧cacheを読まず、既存新cacheも再利用/削除しない。

    条件不明のcacheを黙って再利用しない。再実行は明示的な別outputを指定する。
    """
    target=Path(output) if output else Path(folder)/'head_contrib_diagnostic_v1.npz'
    if target.exists():raise FileExistsError('既存結果は上書きしません: '+str(target))
    result=analyze(folder,nconf,checkpoint,sel,settings=settings,lattice=lattice)
    save(result,target)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder');parser.add_argument('--nconf',type=int,default=64)
    parser.add_argument('--checkpoint');parser.add_argument('--settings');parser.add_argument('--lattice',type=json.loads)
    parser.add_argument('--output');args=parser.parse_args()
    z=cached(args.folder,args.nconf,checkpoint=args.checkpoint,settings=args.settings,lattice=args.lattice,output=args.output)
    C,Cp=z['C'],z['Cp']
    print('head-contrib-diagnostic-v1: 独立診断。Cpは非線形な自己項との差（純粋な二体/エネルギー寄与ではない）')
    total=np.sum(C**2)
    for l in range(C.shape[0]):
        ratio=np.sum(C[l])/z['R'][l] if z['R'][l]>0 else np.nan
        print('layer %d: mean-norm alignment ratio=%.6g (等ノルム・直交Headならsqrt(H))'%(l,ratio))
        for h in range(C.shape[1]):
            print('  head %d: C=%.6g Cd=%.6g Cp=%.6g square-of-mean share=%.6g'%
                (h,C[l,h],z['Cd'][l,h],Cp[l,h],C[l,h]**2/total if total else np.nan))
    delta=z['decomposition_sum']-z['attention_output']
    print('sum(heads)+shared bias vs diagnostic Attention: max_abs=%.6g RMS=%.6g'%
          (np.max(np.abs(delta)),np.sqrt(np.mean(delta**2))))
    print('q_Ap / q_Cp / q_g_heuristic（最後は平均後の飽和補正heuristic、正確な逆変換ではない）')
    for l in range(C.shape[0]):
        for h in range(C.shape[1]):
            print('  layer %d head %d: %.6g / %.6g / %.6g'%(l,h,z['q_Ap'][l,h],z['q_Cp'][l,h],z['q_g_heuristic'][l,h]))


if __name__=='__main__':main()
