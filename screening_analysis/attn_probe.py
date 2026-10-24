"""Screenformerの独立Diagnostic APIをcheckpoint上で評価する（E1）。

S/W/Alpha/Windowは[B,L,H,N,N]、dは[B,N,N]。SはSimilarity、Wは実際の
aggregation_weights。旧probeのTrim²×余弦窓とは異なる。行和正規化はしない。
Window積はW_window、介入後はW_intervenedとして別に残す。各値は診断再評価であり、
Production内部のビット列ではない。位置・spinを同時に並べ替えれば電子軸も共変。

configの未保存グローバルは--settingsのflat JSONで明示する（SETTINGS参照）。
文字列化され丸められた格子には--latticeで既知の元の数値を指定する。
出力はattn_probe_diagnostic_v1.npz。旧解析・既存出力は上書きしない。
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys

import attr
import numpy as np
import jax
import jax.numpy as jnp
import jaxlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from periodicwave import networks, CustomScreenformer as CS  # noqa: E402

SETTINGS = ("TRIM_MAX", "TRIM_POW", "WIN_SHAPE", "FREEZE_W", "CLIP_R", "CLIP_BETA",
            "jax_enable_x64", "jax_default_matmul_precision")


def lattice_from_config(cfg):
    """名前付きsystem.pbc_latticeを読む。旧形式は丸め精度のまま（元精度とは限らない）。"""
    value = cfg["system"]["pbc_lattice"]
    if isinstance(value, str):
        if not value.startswith("unserializable object:") or "..." in value:
            raise ValueError("復元不能なpbc_lattice。省略のない元の格子が必要です")
        value = re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value)
    result = np.asarray(value, dtype=np.float64)
    if result.size != 4 or not np.isfinite(result).all():
        raise ValueError("attn_probeは有限な2×2格子を対象とします")
    return result.reshape(2, 2)


def _effective_settings(cfg, settings, trim_max):
    recorded = dict(cfg.get("screening_attention_settings", {}))
    provenance = {"recorded_keys": sorted(recorded)}
    if isinstance(settings, (str, os.PathLike)):
        path = Path(settings)
        provenance.update(settings_file=str(path.resolve()), settings_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        settings = json.loads(path.read_text())
    supplied = dict(settings or {})
    conflict = [k for k in recorded.keys() & supplied.keys() if recorded[k] != supplied[k]]
    if conflict:
        raise ValueError("保存済み設定と明示設定が矛盾: " + ", ".join(conflict))
    effective = {**recorded, **supplied}
    provenance["explicit_keys"] = sorted(supplied)
    if trim_max is not None:
        effective["TRIM_MAX"] = trim_max
        provenance["analysis_override"] = {"TRIM_MAX": trim_max}
    missing, extra = set(SETTINGS)-effective.keys(), effective.keys()-set(SETTINGS)
    if missing or extra:
        raise ValueError("未保存設定を環境から推測しません。settingsに必要: "
                         + str(sorted(missing)) + "; 未対応キー: " + str(sorted(extra)))
    if effective["TRIM_POW"] not in (2, 4) or effective["WIN_SHAPE"] not in ("cos", "smooth7"):
        raise ValueError("TRIM_POWは2/4、WIN_SHAPEはcos/smooth7")
    if type(effective["FREEZE_W"]) is not bool or type(effective["jax_enable_x64"]) is not bool:
        raise ValueError("FREEZE_Wとjax_enable_x64はJSON booleanで指定してください")
    if not np.isfinite(effective["TRIM_MAX"]) or effective["TRIM_MAX"] <= 0:
        raise ValueError("TRIM_MAXは有限の正数が必要です")
    if effective["CLIP_R"] is not None and (not np.isfinite(effective["CLIP_R"]) or effective["CLIP_R"] <= 0):
        raise ValueError("CLIP_Rはnullまたは有限の正数が必要です")
    if not np.isfinite(effective["CLIP_BETA"]) or effective["CLIP_BETA"] <= 0:
        raise ValueError("CLIP_BETAは有限の正数が必要です")
    return effective, provenance


def _centres_from(P, dref, NL):
    """checkpoint に mask_centre があれば (centres, sigmas)/d_ref を返す。

    make_screening_attention は centres が None かどうかで mask_centre を
    作る/作らないを決めるので、学習済みの構造を再現するために必要。値は
    どのみち P から上書きされるので、形さえ合っていればよい。
    """
    a = P["layers"]["attention"][0]
    if "mask_centre" not in a:
        return None
    c = np.asarray(a["mask_centre"], dtype=np.float64)
    sg = np.log1p(np.exp(np.asarray(a["mask_omega"], dtype=np.float64)))
    return (c, sg)


def probe(folder, checkpoint=None, trim_max=None, nconf=None, *, settings=None, lattice=None):
    """保存walkerを評価する。paramsを初期化せず、返却値Wはaggregation_weights。

    settingsはSETTINGS全項目のdict/JSONファイル、またはconfig内の同名記録を使う。
    当時の未保存値は利用者が確認して指定する。latticeは旧文字列格子に対する元精度の
    共通格子指定。設定の一時適用中に同じプロセスでProductionのtraceを並行しないこと。
    """
    folder = Path(folder)
    cfg_path = folder / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if cfg["network"]["network_type"] != "CustomScreenformer" or cfg["system"]["ndim"] != 2:
        raise ValueError("この入口は2D CustomScreenformer専用です")
    effective, provenance = _effective_settings(cfg, settings, trim_max)
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
            feature_name = network.get("make_feature_layer_fn")
            if feature_name:
                module, name = feature_name.rsplit(".", 1)
                feature_factory = getattr(importlib.import_module(module), name)
            else:
                feature_factory = networks.make_ferminet_features
            fl = feature_factory(natoms=batch["charges"].shape[-1], nspins=nspins, ndim=2, **feature_kwargs)
            opts = dict(network["CustomScreenformer"])
            required = ("num_layers", "num_heads", "attn_dim", "value_dim", "mlp_dim",
                        "num_perceptrons_per_layer", "use_layer_norm", "mlp_activation_fct")
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
            results = {key: [] for key in ("S", "W", "W_window", "W_intervened", "Alpha", "d", "feature_rank_deficient", "finite_features")}
            if options.use_softmask:
                results["Window"] = []
            # walkerだけのループ。Transformerのループ/式は正式APIと共有し、再実装しない。
            for i in range(count):
                pos, spins, atoms, charges = (jnp.asarray(batch[k][i]) for k in ("positions", "spins", "atoms", "charges"))
                x, layers = diagnostic(P, pos, spins, atoms, charges)
                for key, source in (("S", "similarity"), ("W", "aggregation_weights"),
                                    ("W_window", "weights_window"), ("W_intervened", "weights"), ("Alpha", "alpha")):
                    results[key].append(np.stack([np.asarray(layer["attention"][source]) for layer in layers]))
                if options.use_softmask:
                    results["Window"].append(np.stack([np.asarray(layer["attention"]["window"]) for layer in layers]))
                results["d"].append(np.asarray(geometry(pos, atoms, ndim=2)[3][..., 0]))
                state = CS.screening_diagnostic_status(x)
                results["feature_rank_deficient"].append(bool(state["feature_rank_deficient"]))
                results["finite_features"].append(bool(state["finite_features"]))
            result = {key: np.asarray(value) for key, value in results.items()}
            widths, centres, representative, trim, shells = [], [], [], [], []
            # 以下は表示用のcheckpointパラメータ読出しのみ。Weightは再計算しない。
            for ap in P["layers"]["attention"]:
                sigma = np.asarray(dref * jax.nn.softplus(ap["mask_omega"])) if "mask_omega" in ap else np.full(options.num_heads, np.nan)
                shell = "mask_centre" in ap or "mask_rings" in ap
                c = np.asarray(dref * ap["mask_centre"]) if "mask_centre" in ap else np.zeros_like(sigma)
                if "mask_rings" in ap:
                    edges = dref * np.asarray(ap["mask_rings"])
                    c = .5 * (edges[:-1] + edges[1:]); sigma = edges[1:] - c
                centres.append(c); widths.append(sigma); representative.append(c if shell else sigma); shells.append(shell)
                trim.append(np.asarray(effective["TRIM_MAX"] * jax.nn.sigmoid(ap["trim_rho"])))
            result.update(w=np.asarray(representative), r=np.asarray(trim), dref=dref,
                w_sigma=np.asarray(widths), w_centre=np.asarray(centres),
                w_outer=np.asarray(centres)+np.asarray(widths), shell=all(shells), shell_by_layer=np.asarray(shells),
                spins=batch["spins"], positions=batch["positions"], atoms=batch["atoms"], charges=batch["charges"], lattice=L,
                NE=sum(nspins), NL=options.num_layers, NH=options.num_heads, checkpoint=ck.name, folder=str(folder))
    finally:
        for key, value in globals_before.items():
            setattr(CS, key, value)
    result["metadata"] = dict(method="screenformer-diagnostic-v1", weight_definition="aggregation_weights",
        legacy_weight_definition="Trim^2 * legacy window, before fixed-Value diagonal removal and interventions",
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
                         CS.network_blocks.__file__, importlib.import_module(feature_factory.__module__).__file__}},
        wavefunction_status="not evaluated; feature rank flag alone does not certify wavefunction stability")
    return result


def summarize(a):
    print("  method=%s; W=%s（旧解析Wとは定義が異なる）" % (
        a.get("metadata", {}).get("method", "unversioned"),
        a.get("metadata", {}).get("weight_definition", "unverified")))
    if np.any(a.get("feature_rank_deficient", False)):
        print("  特徴ランク警告あり。波動関数・局所エネルギーの安定性はこのprobeでは未検査。")
    NE, off = a["NE"], ~np.eye(a["NE"], dtype=bool)
    d = a["d"][:, off]
    si = a["spins"][:, np.where(off)[0]]
    sj = a["spins"][:, np.where(off)[1]]
    par = (si * sj > 0).ravel()
    db = np.clip(np.digitize(d.ravel(), np.linspace(d.min(), d.max(), 21)) - 1, 0, 19)

    def r2(y, g):
        tot = y.var()
        return sum(((g == u).mean() * (y[g == u].mean() - y.mean()) ** 2)
                   for u in np.unique(g) if (g == u).sum() > 10) / tot

    print("  checkpoint = %s   d_ref = %.3f   d = %.2f 〜 %.2f   電子 %d 個"
          % (a["checkpoint"], a["dref"], d.min(), d.max(), NE))
    print("  %3s%4s%8s%9s%12s%11s%9s%9s%9s"
          % ("層", "H", "w", "w/d_ref", "自己重み", "実効到達", "sd(s)", "R2距離", "R2spin"))
    for l in range(a["NL"]):
        for h in range(a["NH"]):
            W = a["W"][:, l, h]
            Wo, Ws = W[:, off], np.einsum("cii->ci", W)
            s = a["S"][:, l, h][:, off].ravel()
            reff = (Wo * d).sum(1).mean() / max(Wo.sum(1).mean(), 1e-12)
            print("  %3d%4d%8.2f%9.2f%11.1f%%%11.2f%9.3f%9.1f%9.1f"
                  % (l, h, a["w"][l, h], a["w"][l, h] / a["dref"],
                     100 * (Ws.sum(1) / (Ws.sum(1) + Wo.sum(1))).mean(), reff,
                     s.std(), 100 * r2(s, db), 100 * r2(s, par.astype(int))))
    print("  ※ 自己重み: 一様配分なら 1/%d = %.1f%%。R2 は s の分散の説明率[%%]。"
          % (NE, 100.0 / NE))
    print("  ※ 残差 100 - (R2距離 + R2spin) は、そのペア以外の %d 電子の配置に由来する成分。"
          % (NE - 2))


def save_probe(result, output):
    """版名・Weight定義をNPZへ保存する。既存ファイルと旧キャッシュ名は使わない。"""
    output = Path(output)
    if output.name == "attn_probe.npz":
        raise ValueError("旧解析名attn_probe.npzは使用できません")
    payload = {k: v for k, v in result.items() if k != "metadata"}
    payload["metadata_json"] = json.dumps(result["metadata"], sort_keys=True, ensure_ascii=False)
    with output.open("xb") as handle:
        np.savez_compressed(handle, **payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument("--checkpoint")
    parser.add_argument("--settings", help="未保存の実効設定を明示するflat JSON")
    parser.add_argument("--lattice", type=json.loads, help="元精度の共通格子 [[...],[...]]")
    parser.add_argument("--trim-max", type=float)
    parser.add_argument("--nconf", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path(args.folder) / "attn_probe_diagnostic_v1.npz"
    if output.exists():
        parser.error("既存出力は上書きしません: " + str(output))
    a = probe(args.folder, args.checkpoint, args.trim_max, args.nconf, settings=args.settings, lattice=args.lattice)
    summarize(a)
    save_probe(a, output)
    print("  saved:", output)
