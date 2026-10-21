# Copyright 2020 DeepMind Technologies Limited.
# Modifications Copyright (c) 2025 Max Geier Massachusetts Institute of Technology, MA, USA
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
# - Simplified MCMC move width update routine (called directly in train.py)

"""Metropolis-Hastings Monte Carlo.

NOTE: these functions operate on batches of MCMC configurations and should not
be vmapped.
"""

import chex
from periodicwave import constants
from periodicwave import networks
import jax
from jax import lax
from jax import numpy as jnp

# [PROBE] 谷の割り当てを固定して走らせるためのフック（解析用。既定は None で無効）。
# PIN = dict(sites=(ns,2), lattice=(2,2), target=(ns,)) を入れると、
# 各 walker の「どの谷に何個いるか」を target から動かさないよう、
# 変わってしまった walker を元に戻す。深い谷での秩序状態を保ったまま最適化して、
# そのエネルギーを乱れた状態と比べるために使う。
PIN = None


def _pin_pattern(pos, sites, lattice, ndim=2):
  nw = pos.shape[0]
  R = pos.reshape(nw, -1, ndim)
  dr = R[:, :, None, :] - sites[None, None, :, :]
  f = dr @ jnp.linalg.inv(lattice).T
  d = jnp.linalg.norm((f - jnp.round(f)) @ lattice.T, axis=-1)
  who = jnp.argmin(d, axis=-1)
  return jax.nn.one_hot(who, sites.shape[0]).sum(1)


def mh_accept(x1, x2, spins1, spins2, lp_1, lp_2, ratio, key, num_accepts):
  """Given state, proposal, and probabilities, execute MH accept/reject step."""
  key, subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
  cond = ratio > rnd
  x_new = jnp.where(cond[..., None], x2, x1)
  spins_new = jnp.where(cond[..., None], spins2, spins1)
  lp_new = jnp.where(cond, lp_2, lp_1)
  num_accepts += jnp.sum(cond)
  return x_new, spins_new, key, lp_new, num_accepts

def mh_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev=0.02,
    ndim=3,
    blocks=1,
    i=0
    
):
  """Performs one Metropolis-Hastings step using an all-electron move.

  Args:
    params: Wavefuncttion parameters.
    f: Callable with signature f(params, x) which returns the log of the
      wavefunction (i.e. the sqaure root of the log probability of x).
    data: Initial MCMC configurations (batched).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MH move proposals accepted.
    stddev: width of Gaussian move proposal.
    ndim: dimensionality of system.
    blocks: Ignored.
    i: Ignored.
    allow_spin_flips: whether to allow spin flips during training

  Returns:
    (x, key, lp, num_accepts), where:
      x: Updated MCMC configurations.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.
  """
  del i, blocks, ndim  # electron index ignored for all-electron moves
  key, subkey = jax.random.split(key)
  x1 = data.positions
  spins1 = data.spins  

  x2 = x1 + stddev * jax.random.normal(subkey, shape=x1.shape)  # proposal
  spins2 = spins1                                  
  lp_2 = 2.0 * f(
      params, x2, spins2, data.atoms, data.charges
  )  # log prob of proposal
  ratio = lp_2 - lp_1

  x_new, spins_new, key, lp_new, num_accepts = mh_accept(
      x1, x2, spins1, spins2, lp_1, lp_2, ratio, key, num_accepts)
  
  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new, 'spins': spins_new}))
  return new_data, key, lp_new, num_accepts

def mh_spin_flip(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
):
  """[λ-jellium] 擬スピンを 1 本だけ反転する Metropolis の一歩（位置は動かさない）。

  ウォーカーごとに電子 i を一様に選んで s_i -> -s_i を提案する。提案は対称なので
  受理確率は min(1, |Psi(R, S')|^2 / |Psi(R, S)|^2)。

  Returns:
    (data, key, lp, num_accepts)。num_accepts は受理された反転の数の累計。
  """
  key, subkey = jax.random.split(key)
  spins1 = data.spins
  nwalk, nelec = spins1.shape
  idx = jax.random.randint(subkey, (nwalk,), 0, nelec)
  spins2 = spins1 * (1.0 - 2.0 * jax.nn.one_hot(idx, nelec, dtype=spins1.dtype))
  lp_2 = 2.0 * f(params, data.positions, spins2, data.atoms, data.charges)
  ratio = lp_2 - lp_1
  _, spins_new, key, lp_new, num_accepts = mh_accept(
      data.positions, data.positions, spins1, spins2, lp_1, lp_2, ratio, key,
      num_accepts)
  new_data = networks.FermiNetData(**(dict(data) | {'spins': spins_new}))
  return new_data, key, lp_new, num_accepts


def mh_block_update(                                   # SPIN UPDATES ARE NOT IMPLEMENTED
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev=0.02,
    ndim=3,
    blocks=1,
    i=0,
):
  """Performs one Metropolis-Hastings step for a block of electrons.

  Args:
    params: Wavefuncttion parameters.
    f: Callable with LogFermiNetLike signature which returns the log of the
      wavefunction (i.e. the sqaure root of the log probability of x).
    data: Initial MCMC configuration (batched).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MH move proposals accepted.
    stddev: width of Gaussian move proposal.
    ndim: dimensionality of system.
    blocks: number of blocks to split electron updates into.
    i: index of block of electrons to move.

  Returns:
    (x, key, lp, num_accepts), where:
      x: MCMC configurations with updated positions.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.

  Raises:
    NotImplementedError: if atoms is supplied.
  """
  key, subkey = jax.random.split(key)
  batch_size = data.positions.shape[0]
  nelec = data.positions.shape[1] // ndim
  pad = (blocks - nelec % blocks) % blocks
  x1 = jnp.reshape(
      jnp.pad(data.positions, ((0, 0), (0, pad * ndim))),
      [batch_size, blocks, -1, ndim],
  )
  ii = i % blocks
  x2 = x1.at[:, ii].add(
      stddev * jax.random.normal(subkey, shape=x1[:, ii].shape))
  x2 = jnp.reshape(x2, [batch_size, -1])
  if pad > 0:
    x2 = x2[..., :-pad*ndim]
  # log prob of proposal
  lp_2 = 2.0 * f(params, x2, data.spins, data.atoms, data.charges)
  ratio = lp_2 - lp_1

  x1 = jnp.reshape(x1, [batch_size, -1])
  if pad > 0:
    x1 = x1[..., :-pad*ndim]
  x_new, key, lp_new, num_accepts = mh_accept(
      x1, x2, lp_1, lp_2, ratio, key, num_accepts)
  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new}))
  return new_data, key, lp_new, num_accepts

def make_mcmc_step(batch_network,
                   batch_per_device,
                   steps=10,
                   ndim=3,
                   blocks=1,
                   spin_flips=0,
                   ):
  """Creates the MCMC step function.

  Args:
    batch_network: function, signature (params, x), which evaluates the log of
      the wavefunction (square root of the log probability distribution) at x
      given params. Inputs and outputs are batched.
    batch_per_device: Batch size per device.
    steps: Number of MCMC moves to attempt in a single call to the MCMC step
      function.
    atoms: atom positions. If given, an asymmetric move proposal is used based
      on the harmonic mean of electron-atom distances for each electron.
      Otherwise the (conventional) normal distribution is used.
    ndim: Dimensionality of the system (usually 3). Required only for block updates.
    blocks: Number of blocks to split the updates into. If 1, use all-electron
      moves.
    spin_flips: [λ-jellium] 位置の一歩ごとに続けて行う擬スピン反転の提案の数。
      0（既定）なら従来とまったく同じ（乱数の使い方も同じ）。> 0 のとき mcmc_step は
      pmove の代わりに [位置の受理率, 反転の受理率] を返す。

  Returns:
    Callable which performs the set of MCMC steps.
  """
  if spin_flips > 0:
    return _make_mcmc_step_with_flips(batch_network, batch_per_device, steps,
                                      ndim, blocks, spin_flips)
  inner_fun = mh_block_update if blocks > 1 else mh_update

  def mcmc_step(params, data, key, width):
    """Performs a set of MCMC steps.

    Args:
      params: parameters to pass to the network.
      data: (batched) MCMC configurations to pass to the network.
      key: RNG state.
      width: standard deviation to use in the move proposal.

    Returns:
      (data, pmove), where data is the updated MCMC configurations, key the
      updated RNG state and pmove the average probability a move was accepted.
    """
    pos = data.positions

    def step_fn(i, x):
      return inner_fun(
          params,
          batch_network,
          *x,
          stddev=width,
          ndim=ndim,
          blocks=blocks,
          i=i
          )

    nsteps = steps * blocks
    logprob = 2.0 * batch_network(
        params, pos, data.spins, data.atoms, data.charges
    )
    new_data, key, _, num_accepts = lax.fori_loop(
        0, nsteps, step_fn, (data, key, logprob, 0.0)
    )
    if PIN is not None:
      st = jnp.asarray(PIN['sites']); lat = jnp.asarray(PIN['lattice'])
      tgt = jnp.asarray(PIN['target'])[None, :]
      keep = jnp.all(_pin_pattern(new_data.positions, st, lat, ndim) == tgt, axis=-1)
      new_data = networks.FermiNetData(**(dict(new_data) | {
          'positions': jnp.where(keep[:, None], new_data.positions, pos)}))
    pmove = jnp.sum(num_accepts) / (nsteps * batch_per_device)
    pmove = constants.pmean(pmove)
    return new_data, pmove

  return mcmc_step


def _make_mcmc_step_with_flips(batch_network, batch_per_device, steps, ndim,
                               blocks, spin_flips):
  """[λ-jellium] 位置の全電子移動 1 回ごとに、擬スピン反転の提案を spin_flips 回続ける。"""
  if blocks > 1:
    raise NotImplementedError('spin_flips は blocks = 1（全電子移動）のときだけ使える。')
  if PIN is not None:
    raise NotImplementedError('spin_flips と PIN は併用できない。')

  def mcmc_step(params, data, key, width):
    """Performs a set of MCMC steps (positions and pseudospins).

    Returns:
      (data, [pmove, pflip]): pmove は位置の受理率、pflip は擬スピン反転の受理率。
    """
    def step_fn(i, x):
      data, key, lp, n_move, n_flip = x
      data, key, lp, n_move = mh_update(
          params, batch_network, data, key, lp, n_move,
          stddev=width, ndim=ndim, blocks=blocks, i=i)
      for _ in range(spin_flips):
        data, key, lp, n_flip = mh_spin_flip(
            params, batch_network, data, key, lp, n_flip)
      return data, key, lp, n_move, n_flip

    logprob = 2.0 * batch_network(
        params, data.positions, data.spins, data.atoms, data.charges)
    new_data, key, _, n_move, n_flip = lax.fori_loop(
        0, steps, step_fn, (data, key, logprob, 0.0, 0.0))
    pmove = constants.pmean(n_move / (steps * batch_per_device))
    pflip = constants.pmean(n_flip / (steps * spin_flips * batch_per_device))
    return new_data, jnp.stack([pmove, pflip])

  return mcmc_step
