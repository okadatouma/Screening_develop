"""D'の保存済み試作の再現と、比較用の既存軌道処理への接続。

過去のfactoryは保存済みB2ソースとD差分から再現する。正式APIの生成には使わない。
正式APIのwave比較でも軌道処理への接続だけを再利用する。Transformerの式・ループを
このファイルへコピーしない。本体APIはCustomScreenformer.make_screenformer_diagnostic。
"""
import ast
import hashlib
from pathlib import Path
import subprocess
import tempfile

from periodicwave import CustomScreenformer as CS, networks, network_blocks


def diagnostic_namespace():
    """一時ファイルだけにD差分を適用。モジュール全体の再import/folx再登録はしない。"""
    # D'の過去の候補を再現する。正式APIの生成には使わない。
    source = (Path(__file__).parent / 'layers_validation/before-source.txt').read_text()
    patch = Path(__file__).parent / 'layers_validation/candidate.patch'
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'CustomScreenformer.py'
        path.write_text(source)
        result = subprocess.run(['patch', '--fuzz=0', '--silent', str(path), str(patch.resolve())],
                                text=True, capture_output=True, check=True)
        if result.stdout or result.stderr:
            raise RuntimeError('候補差分を無変更で適用できない: ' + result.stdout + result.stderr)
        candidate = path.read_text()
    candidate_hash = hashlib.sha256(candidate.encode()).hexdigest()
    expected_hash = patch.with_name('candidate.sha256').read_text().strip()
    if candidate_hash != expected_hash:
        raise RuntimeError('D候補の固定ソースと不一致。試作の参照取得差分を再確認する必要がある')
    names = {'make_screening_attention', 'make_self_attention_block', 'make_screenformer_layers'}
    tree = ast.parse(candidate)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(tree.body) == 3
    namespace = dict(CS.__dict__)  # 設定は構築時に同じ実効値をコピーする。
    namespace.pop('make_screenformer_diagnostic', None)  # 過去の試作だけを再現する。
    exec(compile(tree, '<D-prime: source-derived diagnostic factories>', 'exec'), namespace)
    namespace['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
    namespace['candidate_sha256'] = candidate_hash
    return namespace


def make_diagnostic(net, nspins, data, namespace):
    """(phase/logabs [2], layers tuple)を返す純粋関数。呼出側で別個にjax.jitする。

    同じ層factoryから得た最終Xを、既存make_orbitalsの層入力として渡す。
    そのため幾何特徴の準備は二度通るが、Attention/Transformerは一度だけ評価する。
    通常Network、パラメータ、乱数キーは変更しない。HFは本試作の対象外。
    """
    import jax.numpy as jnp
    init, apply = namespace['make_screenformer_layers'](nspins, len(data.charges), net.options)
    geometry = networks.make_periodic_construct_input_features(net.options.pbc_lattice)
    formal = (namespace['make_screenformer_diagnostic'](nspins, net.options)
              if 'make_screenformer_diagnostic' in namespace else None)

    def evaluate(params, pos):
        if formal is not None:
            x, layers = formal(params, pos, data.spins, data.atoms, data.charges)
        else:
            ae, ee, ra, re = geometry(pos, data.atoms, ndim=net.options.ndim)
            x, layers = apply(params['layers'], ae=ae, ee=ee, r_ae=ra, r_ee=re,
                              spins=data.spins, charges=data.charges, return_aux=True)
        # 既存の依存注入点。軌道/行列式の数式を再実装せず、得たXをそのまま使う。
        _, orbitals = networks.make_orbitals(
            nspins=nspins, charges=data.charges, options=net.options,
            equivariant_layers=(init, lambda *args, **kwargs: x))
        phase, logabs = network_blocks.logdet_matmul(
            orbitals(params, pos, data.spins, data.atoms, data.charges))
        return jnp.stack((phase, logabs)), layers

    return evaluate
