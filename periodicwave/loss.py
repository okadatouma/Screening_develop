# Copyright 2020 DeepMind Technologies Limited.
# Modifications Copyright (c) 2025 Max Geier, Khachatur Nazaryan, Massachusetts Institute of Technology, MA, USA
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

# NOTICE: This file has been modified from the original DeepMind version.
# Changes:
# - Streamlined for materials calculations

"""Helper functions to create the loss and custom gradient of the loss."""

import functools
from typing import Tuple

import chex
from periodicwave import constants
from periodicwave import hamiltonian
from periodicwave import networks
import folx
import jax
import jax.numpy as jnp
import kfac_jax
from typing_extensions import Protocol


@chex.dataclass
class AuxiliaryLossData:
  """Auxiliary data returned by total_energy.

  Attributes:
    energy: mean energy over batch, and over all devices if inside a pmap.
    variance: mean variance over batch, and over all devices if inside a pmap.
    local_energy: local energy for each MCMC configuration.
    clipped_energy: local energy after clipping has been applied
    grad_local_energy: gradient of the local energy.
    local_energy_mat: for excited states, the local energy matrix.
    s_ij: Matrix of overlaps between wavefunctions.
    mean_s_ij: Mean value of the overlap between wavefunctions across walkers.
  """
  energy: jax.Array  # for some losses, the energy and loss are not the same
  variance: jax.Array
  local_energy: jax.Array
  clipped_energy: jax.Array
  grad_local_energy: jax.Array | None = None
  local_energy_mat: jax.Array | None = None
  s_ij: jax.Array | None = None
  mean_s_ij: jax.Array | None = None


class LossFn(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.FermiNetData,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched data elements to pass to the network.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """


def clip_local_values(
    local_values: jnp.ndarray,
    mean_local_values: jnp.ndarray,
    clip_scale: float,
    clip_from_median: bool,
    center_at_clipped_value: bool,
    complex_output: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Clips local operator estimates to remove outliers.

  Args:
    local_values: batch of local values,  Of/f, where f is the wavefunction and
      O is the operator of interest.
    mean_local_values: mean (over the global batch) of the local values.
    clip_scale: clip local quantities that are outside nD of the estimate of the
      expectation value of the operator, where n is this value and D the mean
      absolute deviation of the local quantities from the estimate of w, to the
      boundaries. The clipped local quantities should only be used to evaluate
      gradients.
    clip_from_median: If true, center the clipping window at the median rather
      than the mean. Potentially expensive in multi-host training, but more
      accurate/robust to outliers.
    center_at_clipped_value: If true, center the local energy differences passed
      back to the gradient around the clipped quantities, so the mean difference
      across the batch is guaranteed to be zero.
    complex_output: If true, the local energies will be complex valued.

  Returns:
    Tuple of the central value (estimate of the expectation value of the
    operator) and deviations from the central value for each element in the
    batch. If per_device_threshold is True, then the central value is per
    device.
  """

  batch_mean = lambda values: constants.pmean(jnp.mean(values, axis=0))

  def clip_at_total_variation(values, center, scale):
    tv = batch_mean(jnp.abs(values- center))
    return jnp.clip(values, center - scale * tv, center + scale * tv)

  if clip_from_median:
    # More natural place to center the clipping, but expensive due to both
    # the median and all_gather (at least on multihost)
    all_local_values = constants.all_gather(local_values).real
    shape = all_local_values.shape
    if all_local_values.ndim == 3:  # energy_and_overlap case
      all_local_values = all_local_values.reshape([-1, shape[-1]])
    else:
      all_local_values = all_local_values.reshape([-1])
    clip_center = jnp.median(all_local_values, axis=0)
  else:
    clip_center = mean_local_values
  # roughly, the total variation of the local energies
  if complex_output:
    clipped_local_values = (
        clip_at_total_variation(
            local_values.real, clip_center.real, clip_scale) +
        1.j * clip_at_total_variation(
            local_values.imag, clip_center.imag, clip_scale)
    )
  else:
    clipped_local_values = clip_at_total_variation(
        local_values, clip_center, clip_scale)
  if center_at_clipped_value:
    diff_center = batch_mean(clipped_local_values)
  else:
    diff_center = mean_local_values
  diff = clipped_local_values - diff_center
  return diff_center, diff


def make_loss(network: networks.LogFermiNetLike,
              local_energy: hamiltonian.LocalEnergy,
              clip_local_energy: float = 0.0,
              clip_from_median: bool = True,
              center_at_clipped_energy: bool = True,
              complex_output: bool = False,
              local_energy_chunk: int = 0,
              num_groups: int = 0) -> LossFn:
  """Creates the loss function, including custom gradients.

  Args:
    network: callable which evaluates the log of the magnitude of the
      wavefunction (square root of the log probability distribution) at a
      single MCMC configuration given the network parameters.
    local_energy: callable which evaluates the local energy.
    clip_local_energy: If greater than zero, clip local energies that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. The clipped local energies are only used to
      evaluate gradients.
    clip_from_median: If true, center the clipping window at the median rather
      than the mean. Potentially expensive in multi-host training, but more
      accurate.
    center_at_clipped_energy: If true, center the local energy differences
      passed back to the gradient around the clipped local energy, so the mean
      difference across the batch is guaranteed to be zero.
    complex_output: If true, the local energies will be complex valued.
    local_energy_chunk: [λ-jellium] > 0 なら局所エネルギーをウォーカー local_energy_chunk 本ずつ
      順に計算してメモリの山を抑える（jax.lax.map）。0（既定）なら従来どおり全ウォーカーを
      一度に vmap する。値は変わらない。
    num_groups: [λ-jellium 多点] > 0 なら、ウォーカーは data.charges[2] の点の番号（0, ..., num_groups - 1）で
      組に分かれる（点ごとにハミルトニアンの (r_s, lam) が違う）。E_L の切り詰めと、勾配で E_L から引く平均を
      点ごとに取る：勾配 = (2/B) sum_b Re[(E_c,b - Re Ebar_{c,p(b)}) conj(O_b)]、O = d log psi / d theta
      （全点の平均を引くと 2 (E_p - c) Re<O>_p だけずれる）。各点のウォーカー数が同じなら、
      点ごとのエネルギー E_p の勾配の平均 (1/P) sum_p dE_p/dtheta と同じ。記録する loss は全ウォーカーの平均のまま。
      clip_from_median には対応しない。

  Returns:
    Callable with signature (params, data) and returns (loss, aux_data), where
    loss is the mean energy, and aux_data is an AuxiliaryLossDataobject. The
    loss is averaged over the batch and over all devices inside a pmap.
  """
  # Define function that vmaps local energy computation over the batch of positions 
  # with individual keys for individual random numbers
  # Input: 
  #   params: 
  #   keys: vector of keys of dimension batch_size
  #   data: FermiNetData where positions, spins, atoms, charges have dim (batch_size, x) 
  #     where x is the dimension of the respective element
  batch_local_energy = jax.vmap(
      local_energy,
      in_axes=(
          None,
          0,
          networks.FermiNetData(positions=0, spins=0, atoms=0, charges=0),
      ),
      out_axes=(0, 0)
  )
  if local_energy_chunk and local_energy_chunk > 0:
    def batch_local_energy(params, keys, data):  # pylint: disable=function-redefined
      return jax.lax.map(lambda kd: local_energy(params, kd[0], kd[1]),
                         (keys, data), batch_size=int(local_energy_chunk))
  batch_network = jax.vmap(network, in_axes=(None, 0, 0, 0, 0), out_axes=0)

  @jax.custom_jvp
  def total_energy(
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.FermiNetData,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched MCMC configurations to pass to the local energy function.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """
    # Generate batch_size number of RNG keys from passed key
    keys = jax.random.split(key, num=data.positions.shape[0])
    # Compute e_l and e_l_mat (for excited states) over the whole batch
    e_l, e_l_mat = batch_local_energy(params, keys, data)
    # Take mean of e_l to as loss function
    loss = constants.pmean(jnp.mean(e_l))
    # Get variance of local energy
    loss_diff = e_l - loss
    variance = constants.pmean(jnp.mean(loss_diff * jnp.conj(loss_diff)))
    return loss, AuxiliaryLossData(
        energy=loss,
        variance=variance.real,
        local_energy=e_l,
        clipped_energy=e_l,
        local_energy_mat=e_l_mat,
    )

  if num_groups > 0 and clip_from_median and clip_local_energy > 0.0:
    raise ValueError('num_groups（多点）は clip_from_median に対応しない（cfg.optim.clip_median = False にする）')

  def group_terms(local_energy, charges):
    """[多点] 点ごとに切り詰めた E_L と、勾配に掛ける重み w_b = E_c,b - Re Ebar_{c,p(b)}（ウォーカーごと）。"""
    dt = jnp.real(local_energy).dtype
    onehot = jax.nn.one_hot(charges[:, 2].astype(jnp.int32), num_groups, dtype=dt)   # (B, P)
    count = jnp.maximum(constants.pmean(jnp.sum(onehot, axis=0)), 1.0)             # 点ごとのウォーカー数

    def gmean(v):                     # 点ごとの平均（全デバイス）を、各ウォーカーの位置に戻す
      return onehot @ (constants.pmean(onehot.T @ v) / count)

    def clip(v):
      c = gmean(v)
      tv = gmean(jnp.abs(v - c))
      return jnp.clip(v, c - clip_local_energy * tv, c + clip_local_energy * tv)

    if clip_local_energy > 0.0:
      clipped = (clip(local_energy.real) + 1j * clip(local_energy.imag)) if complex_output else clip(local_energy)
    else:
      clipped = local_energy
    ref = clipped if center_at_clipped_energy else local_energy
    if complex_output:
      return clipped, clipped - gmean(ref.real)
    return clipped, clipped - gmean(ref)

  @total_energy.defjvp
  def total_energy_jvp(primals, tangents):  # pylint: disable=unused-variable
    """Custom Jacobian-vector product for unbiased local energy gradients."""
    params, key, data = primals
    loss, aux_data = total_energy(params, key, data)

    if num_groups > 0:
      aux_data.clipped_energy, weights = group_terms(aux_data.local_energy, data.charges)
      data_tangents = tangents[2]
      psi_primal, psi_tangent = jax.jvp(
          batch_network, (params, data.positions, data.spins, data.atoms, data.charges),
          (tangents[0], data_tangents.positions, data_tangents.spins, data_tangents.atoms,
           data_tangents.charges))
      device_batch_size = jnp.shape(aux_data.local_energy)[0]
      if complex_output:
        kfac_jax.register_normal_predictive_distribution(psi_primal.real[:, None])
        grad_dot = 2.0 * jnp.real(jnp.dot(jnp.conjugate(weights), psi_tangent)) / device_batch_size
        return (loss.real, aux_data), (grad_dot, aux_data)
      kfac_jax.register_normal_predictive_distribution(psi_primal[:, None])
      return (loss, aux_data), (jnp.dot(psi_tangent, weights) / device_batch_size, aux_data)

    if clip_local_energy > 0.0:
      aux_data.clipped_energy, diff = clip_local_values(
          aux_data.local_energy,
          loss,
          clip_local_energy,
          clip_from_median,
          center_at_clipped_energy,
          complex_output)
    else:
      diff = aux_data.local_energy - loss

    # Due to the simultaneous requirements of KFAC (calling convention must be
    # (params, rng, data)) and Laplacian calculation (only want to take
    # Laplacian wrt electron positions) we need to change up the calling
    # convention between total_energy and batch_network
    data = primals[2]
    data_tangents = tangents[2]
    primals = (primals[0], data.positions, data.spins, data.atoms, data.charges)
    tangents = (
        tangents[0],
        data_tangents.positions,
        data_tangents.spins,
        data_tangents.atoms,
        data_tangents.charges,
    )
    psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)
    if complex_output:
      clipped_el = diff + aux_data.clipped_energy
      term1 = (jnp.dot(clipped_el, jnp.conjugate(psi_tangent)) +
               jnp.dot(jnp.conjugate(clipped_el), psi_tangent))
      term2 = jnp.sum(aux_data.clipped_energy*psi_tangent.real)
      kfac_jax.register_normal_predictive_distribution(
          psi_primal.real[:, None])
      primals_out = loss.real, aux_data
      device_batch_size = jnp.shape(aux_data.local_energy)[0]
      tangents_out = ((term1 - 2*term2).real / device_batch_size, aux_data)
    else:
      kfac_jax.register_normal_predictive_distribution(psi_primal[:, None])
      primals_out = loss, aux_data
      device_batch_size = jnp.shape(aux_data.local_energy)[0]
      tangents_out = (jnp.dot(psi_tangent, diff) / device_batch_size, aux_data)
    return primals_out, tangents_out

  return total_energy