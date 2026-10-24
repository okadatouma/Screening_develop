# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Neural network building blocks."""

# 複数Ansatzで数式・初期化が共通の小さな演算を置く。
# Embedding、LayerNorm、線形層、行列式の和を変更する際の共通入口。
import functools
import itertools
from typing import Mapping, MutableMapping, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp


# LayerNorm from CustomPsiformer.py, CustomScreenformer.py and SlaterNet.py:
# Copyright 2023 DeepMind Technologies Limited.
# Modifications Copyright (c) 2025 Max Geier, Khachatur Nazaryan
# Massachusetts Institute of Technology, MA, USA
# Distributed under the Apache License, Version 2.0 stated above.
def make_layer_norm() ->...:
  """Implementation of LayerNorm."""
  # Import here to preserve ParamTree annotations without a module import cycle.
  # 電子ごとの特徴軸を正規化する。epsilon・dtype・演算順は固定snapshotの比較対象。
  from periodicwave import networks

  def init(param_shape: int) -> Mapping[str, jnp.ndarray]:
    params = {}
    params['scale'] = jnp.ones(param_shape)
    params['offset'] = jnp.zeros(param_shape)
    return params

  def apply(params: networks.ParamTree,
            inputs: jnp.ndarray,
            axis: int = -1) -> jnp.ndarray:
    mean = jnp.mean(inputs, axis=axis, keepdims=True)
    variance = jnp.var(inputs, axis=axis, keepdims=True)
    eps = 1e-5
    inv = params['scale'] * jax.lax.rsqrt(variance + eps)
    return inv * (inputs - mean) + params['offset']

  return init, apply


# Learned embedding from the three Ansatz modules (same provenance as LayerNorm
# above). Geometry and fixed features remain the responsibility of FeatureLayer.
def init_input_embedding(key: chex.PRNGKey,
                         one_electron_feature_dim: int,
                         output_dim: int) -> jnp.ndarray:
  """Return bare W[F+1,E], without bias; caller owns the PRNG split order.

  The existing linear initializer uses normal(key1)/sqrt(F+1), with its original
  dtype and two-key split. No extra parameter nesting is introduced.
  """
  # 1体特徴の末尾にspinを1成分足すため、重みの入力次元はone_electron_feature_dim+1。
  return init_linear_layer(
      key, in_dim=one_electron_feature_dim + 1, out_dim=output_dim,
      include_bias=False
  )['w']


def apply_input_embedding(weight: jnp.ndarray,
                          one_electron_features: jnp.ndarray,
                          spins: jnp.ndarray) -> jnp.ndarray:
  """X = concat(f_i, s_i) W: [N,F], [N] -> [N,E], W[F+1,E].

  Features and spin are concatenated before the single dot, preserving dtype,
  operation order and electron-permutation equivariance. No geometry is computed.
  """
  # X_i = concat(1体特徴_i, spin_i) @ weight。電子の行順を保つため交換共変。
  features = jnp.concatenate((one_electron_features, spins[..., None]), axis=-1)
  return jnp.dot(features, weight)


def array_partitions(sizes: Sequence[int]) -> Sequence[int]:
  """Returns the indices for splitting an array into separate partitions.

  Args:
    sizes: size of each of N partitions. The dimension of the array along
    the relevant axis is assumed to be sum(sizes).

  Returns:
    sequence of indices (length len(sizes)-1) at which an array should be split
    to give the desired partitions.
  """
  return list(itertools.accumulate(sizes))[:-1]


def split_into_blocks(block_arr: jnp.ndarray,
                      block_dims: Tuple[int, ...]) -> Sequence[jnp.ndarray]:
  """Split a square array into blocks along the leading two axes.

  Consider the (N,N) array
  A B
  C D
  where A=(N1,N1), B=(N1,N2), C=(N2,N1), D=(N2,N2), and N=N1+N2. Split the array
  into the given blocks.

  Args:
    block_arr: block array to split.
    block_dims: the size of each block along each axis (i.e. N1, N2, ...).

  Returns:
    blocks of the array split along the two leading axes into chunks with
    dimensions given by block_dims.
  """
  partitions = array_partitions(block_dims)
  block1 = jnp.split(block_arr, partitions, axis=0)
  block12 = [jnp.split(arr, partitions, axis=1) for arr in block1]
  return tuple(itertools.chain.from_iterable(block12))


def init_linear_layer(
    key: chex.PRNGKey, in_dim: int, out_dim: int, include_bias: bool = True
) -> MutableMapping[str, jnp.ndarray]:
  """Initialises parameters for a linear layer, x w + b.

  Args:
    key: JAX PRNG state.
    in_dim: input dimension to linear layer.
    out_dim: output dimension (number of hidden units) of linear layer.
    include_bias: if true, include a bias in the linear layer.

  Returns:
    A mapping containing the weight matrix (key 'w') and, if required, bias
    unit (key 'b').
  """
  # 重みとbiasは別subkeyで初期化する。乱数splitを変えると後続の初期値も変わり得る。
  key1, key2 = jax.random.split(key)
  weight = (
      jax.random.normal(key1, shape=(in_dim, out_dim)) /
      jnp.sqrt(float(in_dim)))
  if include_bias:
    bias = jax.random.normal(key2, shape=(out_dim,))
    return {'w': weight, 'b': bias}
  else:
    return {'w': weight}


def linear_layer(x: jnp.ndarray,
                 w: jnp.ndarray,
                 b: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Evaluates a linear layer, x w + b.

  Args:
    x: inputs.
    w: weights.
    b: optional bias.

  Returns:
    x w + b if b is given, x w otherwise.
  """
  y = jnp.dot(x, w)
  return y + b if b is not None else y

vmap_linear_layer = jax.vmap(linear_layer, in_axes=(0, None, None), out_axes=0)


def slogdet(x):
  """Computes sign and log of determinants of matrices.

  This is a jnp.linalg.slogdet with a special (fast) path for small matrices.

  Args:
    x: square matrix.

  Returns:
    sign, (natural) logarithm of the determinant of x.
  """
  if x.shape[-1] == 1:
    if x.dtype == jnp.complex64 or x.dtype == jnp.complex128:
      sign = x[..., 0, 0] / jnp.abs(x[..., 0, 0])
    else:
      sign = jnp.sign(x[..., 0, 0])
    logdet = jnp.log(jnp.abs(x[..., 0, 0]))
  else:
    sign, logdet = jnp.linalg.slogdet(x)

  return sign, logdet


def logdet_matmul(
    xs: Sequence[jnp.ndarray], w: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Combines determinants and takes dot product with weights in log-domain.

  We use the log-sum-exp trick to reduce numerical instabilities.

  Args:
    xs: FermiNet orbitals in each determinant. Either of length 1 with shape
      (ndet, nelectron, nelectron) (full_det=True) or length 2 with shapes
      (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta) (full_det=False,
      determinants are factorised into block-diagonals for each spin channel).
    w: weight of each determinant. If none, a uniform weight is assumed.

  Returns:
    sum_i w_i D_i in the log domain, where w_i is the weight of D_i, the i-th
    determinant (or product of the i-th determinant in each spin channel, if
    full_det is not used).
  """
  # 1x1 determinants appear to be numerically sensitive and can become 0
  # (especially when multiple determinants are used with the spin-factored
  # wavefunction). Avoid this by not going into the log domain for 1x1 matrices.
  # Pass initial value to functools so det1d = 1 if all matrices are larger than
  # 1x1.
  # 通常Ansatzからは全電子N×N行列の配列[ndet,N,N]が渡される。
  # 最大log|det|を引いてから行列式を足し、桁あふれを抑える。複素出力のphaseは角度。
  det1d = functools.reduce(lambda a, b: a * b,
                           [x.reshape(-1) for x in xs if x.shape[-1] == 1], 1)
  # Pass initial value to functools so sign_in = 1, logdet = 0 if all matrices
  # are 1x1.
  phase_in, logdet = functools.reduce(
      lambda a, b: (a[0] * b[0], a[1] + b[1]),
      [slogdet(x) for x in xs if x.shape[-1] > 1], (1, 0))

  # log-sum-exp trick
  maxlogdet = jnp.max(logdet)
  det = phase_in * det1d * jnp.exp(logdet - maxlogdet)
  if w is None:
    result = jnp.sum(det)
  else:
    result = jnp.matmul(det, w)[0]
  # return phase as a unit-norm complex number, rather than as an angle
  if result.dtype == jnp.complex64 or result.dtype == jnp.complex128:
    phase_out = jnp.angle(result)  # result / jnp.abs(result)
  else:
    phase_out = jnp.sign(result)
  log_out = jnp.log(jnp.abs(result)) + maxlogdet
  return phase_out, log_out
