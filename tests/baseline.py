"""変更前に capture、変更後に compare。保存先は新規ディレクトリのみ。"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import unittest

from tests.support import PACKAGES, ROOT, ROUTES, require_numerics


def source_inventory(root=ROOT):
    """数値ライブラリを import せず、実行に影響する環境変数とソースを記録。"""
    paths = sorted([*root.glob("*.py"), *root.joinpath("periodicwave").rglob("*.py"),
                    *root.joinpath("screening_analysis").rglob("*.py")])
    hashes, env = {}, set()
    for path in paths:
        raw = path.read_bytes()
        hashes[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        tree = ast.parse(raw.decode("utf-8"))
        for node in ast.walk(tree):
            # PW_* は get() に加え、代入/setdefault/ヘルパー経由も含めて抽出する。
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith("PW_") and node.value.replace("_", "").isalnum():
                    env.add(node.value)
    return {"sha256": hashes, "pw_environment_names": sorted(env)}


def metadata():
    versions = {}
    for name in PACKAGES:
        distribution = {"attr": "attrs", "absl": "absl-py"}.get(name, name)
        try:
            versions[name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    inventory = source_inventory()
    names = set(inventory["pw_environment_names"]) | {
        "JAX_PLATFORMS", "JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION",
        "CUDA_VISIBLE_DEVICES", "XLA_FLAGS", "OMP_NUM_THREADS", "PW_TEST_X64"}
    return {"commit": commit, "python": platform.python_version(), "platform": platform.platform(),
            "versions": versions, "environment": {k: os.environ.get(k) for k in sorted(names)},
            **inventory}


def numerical_snapshot():
    """値・微分・推定量・MCMC・Adam更新を保存。正しさの証明とは別の基準。"""
    require_numerics()
    import tempfile
    import attr
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from periodicwave import CustomScreenformer as CS, loss, mcmc, train, train_screen
    from tests.support import make_case, batch_data
    from tests.test_attention import attention_values

    if jax.device_count() != 1:
        raise RuntimeError("数値基準の小系は単一device用です。XLA_FLAGS等を確認してください")
    arrays, structures, configurations, input_hashes = {}, {}, {}, {}

    def put(name, value):
        structures[name] = str(jax.tree_util.tree_structure(value))
        for i, leaf in enumerate(jax.tree_util.tree_leaves(value)):
            array = np.asarray(leaf)
            if array.dtype == object or not np.all(np.isfinite(array)):
                raise ValueError("基準として保存できない値: " + name)
            arrays[f"{name}/{i}"] = array

    with tempfile.TemporaryDirectory(prefix="pw-baseline-") as tmp:
        for route in ROUTES:
            net, params, data, lattice, options, energy = make_case(route, tmp)
            put(route + "/input", data)
            put(route + "/lattice", lattice)
            effective = dict(attr.asdict(net.options, recurse=False), **options)
            effective = {k: v for k, v in effective.items()
                         if k not in ("envelope", "feature_layer", "lj_hf_path")}
            configurations[route] = json.loads(json.dumps(
                effective, default=lambda v: v.tolist() if hasattr(v, "tolist") else v.name))
            configurations[route].update(nspins=[1, 1] if route in ("psiformer", "slater", "screen") else [2, 0],
                                         interaction_energy_scale=2., convergence_radius=2,
                                         laplacian_method="folx", init_key=7, loss_key=13,
                                         mcmc_key=17, update_key=19, adam_rate=0.001)
            if route in ("lambda", "hf"):
                configurations[route].update(lam=0.4, delta=2., flip_batch=1, spin_flips=1)
            args = (data.spins, data.atoms, data.charges)
            f = lambda x: net.apply(params, x, *args)
            put(route + "/params", params)
            put(route + "/wavefunction", f(data.positions))
            phase, logabs = f(data.positions)
            put(route + "/psi", jnp.exp(logabs + 1j * phase))
            orbital_matrices = net.orbitals(params, data.positions, *args)
            put(route + "/orbitals", orbital_matrices)
            put(route + "/determinants", [jnp.linalg.det(m) for m in orbital_matrices])
            input_hashes[route] = {name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
                                   for name, value in dict(data).items()}
            for component in (0, 1):
                scalar = lambda x: f(x)[component]
                put(f"{route}/grad{component}", jax.grad(scalar)(data.positions))
                put(f"{route}/hessian{component}", jax.hessian(scalar)(data.positions))
            put(route + "/energy", energy(params, jax.random.PRNGKey(11), data))
            batch = batch_data(data)
            logpsi = lambda p, x, s, a, c: (lambda ph, la: la + 1j * ph)(*net.apply(p, x, s, a, c))
            objective = loss.make_loss(logpsi, energy, complex_output=True)
            put(route + "/loss_gradient", jax.value_and_grad(objective, has_aux=True)(
                params, jax.random.PRNGKey(13), batch))
            bn = jax.vmap(lambda p, x, s, a, c: net.apply(p, x, s, a, c)[1],
                          in_axes=(None, 0, 0, 0, 0))
            flips = 1 if route in ("lambda", "hf") else 0
            step = mcmc.make_mcmc_step(bn, 2, steps=2, ndim=2, spin_flips=flips)
            put(route + "/mcmc", step(params, batch, jax.random.PRNGKey(17), 0.2))
            opt = optax.adam(0.001)
            trainer = train_screen if route in ("screen", "lambda", "hf") else train
            update = trainer.make_opt_update_step(objective, opt)
            put(route + "/adam", update(params, batch, opt.init(params), jax.random.PRNGKey(19)))
            # 固定バッチの更新とは別に、実際の学習ループの乱数・MCMC接続も固定する。
            import csv
            from tests.test_training_checkpoint import training_config
            from periodicwave import checkpoint
            optimizers = ("adam", "kfac") if os.environ.get("PW_RUN_KFAC") == "1" else ("adam",)
            for optimizer_name in optimizers:
                cfg = training_config(route, tmp, optimizer_name)
                trainer.train(cfg)
                restored = checkpoint.restore(str(Path(cfg.log.save_path) / "qmcjax_ckpt_000000.npz"), 2)
                put(route + "/" + optimizer_name + "_training_checkpoint", restored)
                with Path(cfg.log.save_path, "train_stats.csv").open() as handle:
                    row = next(csv.DictReader(handle))
                put(route + "/" + optimizer_name + "_training_stats",
                    {k: np.asarray(complex(v)) for k, v in row.items()})
            # 合成表の内容を保存し、絶対パスには依存しない。
            if route == "hf":
                put("hf/table", np.loadtxt(options["lj_hf_path"]))
                input_hashes[route]["synthetic_hf_file_sha256"] = hashlib.sha256(
                    Path(options["lj_hf_path"]).read_bytes()).hexdigest()
        for mode in ("softmax", "cos", "smooth7", "pow4", "band", "sin", "gauss", "fixed",
                     "nomask", "nogate", "nonorm"):
            values, _ = attention_values(mode)
            put("attention/" + mode, values)
    info = metadata()
    info.update(structures=structures, routes=list(ROUTES),
                input_sha256=input_hashes,
                configurations=configurations, kfac_included=os.environ.get("PW_RUN_KFAC") == "1",
                harness_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(ROOT.joinpath("tests").glob("*.py"))},
                mcmc_pin=(None if mcmc.PIN is None else
                          {k: np.asarray(v).tolist() for k, v in mcmc.PIN.items()}),
                devices=[str(d) for d in jax.devices()], backend=jax.default_backend(),
                x64=bool(jax.config.x64_enabled),
                screening_globals={k: getattr(CS, k) for k in
                                   ("TRIM_MAX", "TRIM_POW", "WIN_SHAPE", "FREEZE_W", "CLIP_R", "CLIP_BETA")})
    return arrays, info


def compare(directory, arrays, info, exact=False, vmc_gradient_version="legacy"):
    """Compare preserved values; keep intentionally changed physical results frozen.

    The explicit G1 exception concerns values only: keys, PyTrees, shapes, dtypes
    and effective settings must still match for gradients and optimizer states.
    G2b permits matching real-to-complex promotion only for PBC energy/aux leaves,
    and compares all lambda/HF values against G1 exactly. g2b-legacy additionally
    accounts for G1's old gradient correction when using pre-G1 snapshots.
    Independent physics is checked by check_g1_vmc/check_g2_local_energy.
    """
    import numpy as np
    if vmc_gradient_version not in ("legacy", "g1", "g2b", "g2b-legacy"):
        raise ValueError("unknown VMC gradient version")
    counts = dict(exact_arrays=0, exact_elements=0, changed_gradient_update_arrays=[],
                  gradient_update_arrays=0, dtype_changes=[])
    directory = Path(directory)
    old_info = json.loads(directory.joinpath("metadata.json").read_text())
    if vmc_gradient_version == "g2b":
        # This mode compares with G1, whose loss is explicitly protected in G2b.
        if old_info['sha256']['periodicwave/loss.py'] != info['sha256']['periodicwave/loss.py']:
            raise AssertionError('G2b requires the unchanged G1 loss reference')
    for key in ("versions", "python", "devices", "backend", "x64", "screening_globals", "structures", "environment",
                "configurations", "kfac_included", "mcmc_pin"):
        old_value, new_value = old_info[key], info[key]
        if key == "environment":
            # 未設定の環境変数と、監査項目追加前の「キーなし」は同じ実行条件。
            old_value = {k: v for k, v in old_value.items() if v is not None}
            new_value = {k: v for k, v in new_value.items() if v is not None}
        if old_value != new_value:
            raise AssertionError("比較条件が異なる: " + key)
    with np.load(directory / "values.npz", allow_pickle=False) as old:
        if set(old.files) != set(arrays):
            raise AssertionError("数値基準の項目が異なる")
        for name, value in arrays.items():
            expected = old[name]
            route, group, leaf = name.split('/')
            updates = ("loss_gradient", "adam", "adam_training_checkpoint", "adam_training_stats",
                       "kfac_training_checkpoint", "kfac_training_stats")
            pbc_change = (vmc_gradient_version.startswith('g2b')
                          and route in ('psiformer', 'slater', 'screen')
                          and group in ('energy', *updates))
            old_g1_change = (vmc_gradient_version in ('g1', 'g2b-legacy')
                             and route in ROUTES and group in updates)
            # Only EL and energy-valued auxiliary leaves may become complex.
            # Optimizer parameter/state leaves must retain their real dtype.
            promote = False
            if pbc_change:
                nparams = sum(k.startswith(route+'/params/') for k in arrays)
                nprimal = sum(k.startswith(route+'/loss_gradient/') for k in arrays)-nparams
                nadam = sum(k.startswith(route+'/adam/') for k in arrays)
                promote = ((group == 'energy' and int(leaf) == 0)
                           or (group == 'loss_gradient' and 0 < int(leaf) < nprimal)
                           or (group == 'adam' and int(leaf) > nadam-nprimal))
            if expected.shape != value.shape:
                raise AssertionError("shape/dtype が異なる: " + name)
            if expected.dtype != value.dtype:
                if not (promote and expected.dtype.kind == 'f' and value.dtype.kind == 'c'
                        and value.real.dtype == expected.dtype):
                    raise AssertionError("unexpected dtype change: " + name)
                counts['dtype_changes'].append(dict(name=name, old=str(expected.dtype), new=str(value.dtype)))
            if pbc_change or old_g1_change:
                counts['gradient_update_arrays'] += 1
                if not np.array_equal(value, expected):
                    counts['changed_gradient_update_arrays'].append(name)
                continue
            parts = (("real", value.real, expected.real), ("imag", value.imag, expected.imag)) if np.iscomplexobj(value) else (("value", value, expected),)
            for part, actual_part, expected_part in parts:
                if exact:
                    np.testing.assert_array_equal(actual_part, expected_part, err_msg=name + '/' + part)
                else:
                    np.testing.assert_allclose(actual_part, expected_part, rtol=1e-8, atol=1e-9,
                                               err_msg=name + '/' + part)
            counts['exact_arrays'] += 1
            counts['exact_elements'] += int(value.size)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "capture", "compare"))
    parser.add_argument("directory", nargs="?")
    parser.add_argument("--exact", action="store_true", help="compare時に実部・虚部の要素ごとの厳密一致を要求")
    args = parser.parse_args()
    if args.action == "inspect":
        print(json.dumps(metadata(), indent=2, ensure_ascii=False))
        return
    if not args.directory:
        parser.error("capture/compare はディレクトリを指定してください")
    target = Path(args.directory)
    if args.action == "capture" and target.exists():
        parser.error("既存の保存先は上書きしません: " + str(target))
    try:
        arrays, info = numerical_snapshot()
    except unittest.SkipTest as error:
        print("数値基準は未作成・未比較: " + str(error), file=sys.stderr)
        raise SystemExit(2)
    if args.action == "capture":
        import numpy as np
        target.mkdir(parents=True, exist_ok=False)
        np.savez(target / "values.npz", **arrays)
        target.joinpath("metadata.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
        print("数値基準を新規保存: " + str(target))
    else:
        compare(target, arrays, info, exact=args.exact)
        print("数値基準との比較に成功: " + str(target))


if __name__ == "__main__":
    main()
