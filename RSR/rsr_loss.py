"""Algorithm-independent RSR distribution loss utilities.

The online action passed to :func:`compute_rsr_loss` is expected to be produced
by the policy being optimized.  This is important: using actions stored in a
rollout or replay buffer makes the RSR term constant with respect to the actor
parameters and therefore produces no policy gradient.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

import RSR.dataset_processor as dp


class RSRData(NamedTuple):
  """Precomputed real/sim distribution statistics used during training."""

  divergence: jnp.ndarray
  reference_density: jnp.ndarray
  reference_data: jnp.ndarray
  grid: jnp.ndarray
  bandwidth: float


def make_grid(
    num_samples: int,
    dimension: int,
    min_value: float = -3.0,
    max_value: float = 3.0,
    seed: int = 0,
) -> jnp.ndarray:
  """Creates the deterministic grid used by the KDE approximation."""
  return jax.random.uniform(
      jax.random.PRNGKey(seed),
      (num_samples, dimension),
      minval=min_value,
      maxval=max_value,
  )


def build_rsr_data(
    real_data: jnp.ndarray,
    previous_sim_data: jnp.ndarray,
    current_sim_data: jnp.ndarray,
    *,
    num_samples: int = 10,
    min_value: float = -3.0,
    max_value: float = 3.0,
    bandwidth: float = 0.1,
    seed: int = 0,
) -> RSRData:
  """Precomputes the fixed part of the RSR distribution objective."""
  if real_data.ndim != 2:
    raise ValueError(f'real_data must be rank 2, got shape {real_data.shape}')
  if previous_sim_data.shape != real_data.shape:
    raise ValueError(
        'previous_sim_data must match real_data: '
        f'{previous_sim_data.shape} != {real_data.shape}'
    )
  if current_sim_data.shape != real_data.shape:
    raise ValueError(
        'current_sim_data must match real_data: '
        f'{current_sim_data.shape} != {real_data.shape}'
    )
  if num_samples <= 0:
    raise ValueError(f'num_samples must be positive, got {num_samples}')
  if bandwidth <= 0:
    raise ValueError(f'bandwidth must be positive, got {bandwidth}')

  grid = make_grid(
      num_samples,
      real_data.shape[-1],
      min_value=min_value,
      max_value=max_value,
      seed=seed,
  )
  real_density = dp.evaluate_kde(real_data, grid, bandwidth)
  previous_sim_density = dp.evaluate_kde(
      previous_sim_data, grid, bandwidth
  )
  reference_density = dp.evaluate_kde(current_sim_data, grid, bandwidth)
  divergence = dp.kl_divergence(real_density, previous_sim_density)
  return RSRData(
      divergence=divergence,
      reference_density=reference_density,
      reference_data=current_sim_data,
      grid=grid,
      bandwidth=bandwidth,
  )


def _as_rsr_data(past_data: Any) -> RSRData:
  """Accepts the new RSRData format and the legacy 3-tuple format."""
  if isinstance(past_data, RSRData):
    return past_data

  if not isinstance(past_data, (tuple, list)):
    raise TypeError('past_data must be RSRData or a tuple/list')
  if len(past_data) == 5:
    return RSRData(*past_data)
  if len(past_data) != 3:
    raise ValueError(
        'legacy past_data must contain (KLD, density, reference_data)'
    )

  divergence, reference_density, reference_data = past_data
  grid = make_grid(
      int(reference_density.shape[0]),
      int(reference_data.shape[-1]),
  )
  return RSRData(
      divergence=divergence,
      reference_density=reference_density,
      reference_data=reference_data,
      grid=grid,
      bandwidth=0.1,
  )


def compute_rsr_loss(
    observations: jnp.ndarray,
    policy_actions: jnp.ndarray,
    next_observations: jnp.ndarray,
    past_data: Any,
    *,
    loss_scale: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes the RSR transition-distribution penalty.

  The three online tensors may have any number of leading dimensions; they are
  flattened into a transition batch.  A positive ``loss_scale`` minimizes the
  distribution discrepancy.  Set it to zero to disable RSR while retaining
  standard PPO/SAC behavior.

  Returns:
    A tuple ``(scaled_loss, distribution_distance)``.
  """
  if past_data is None or loss_scale == 0.0:
    zero = jnp.asarray(0.0, dtype=observations.dtype)
    return zero, zero

  rsr_data = _as_rsr_data(past_data)
  observation_size = observations.shape[-1]
  action_size = policy_actions.shape[-1]
  next_observation_size = next_observations.shape[-1]

  flat_observations = jnp.reshape(observations, (-1, observation_size))
  flat_actions = jnp.reshape(policy_actions, (-1, action_size))
  flat_next_observations = jnp.reshape(
      next_observations, (-1, next_observation_size)
  )
  current_data = jnp.concatenate(
      [flat_observations, flat_actions, flat_next_observations], axis=-1
  )

  if current_data.shape[-1] != rsr_data.reference_data.shape[-1]:
    raise ValueError(
        'online transition width does not match RSR reference data: '
        f'{current_data.shape[-1]} != {rsr_data.reference_data.shape[-1]}'
    )

  augmented_data = jnp.concatenate(
      [rsr_data.reference_data, current_data], axis=0
  )
  current_density = dp.evaluate_kde(
      augmented_data, rsr_data.grid, rsr_data.bandwidth
  )
  distance = dp.wasserstein_distance(
      current_density, rsr_data.reference_density
  )
  loss = jnp.asarray(loss_scale, dtype=distance.dtype)
  loss *= rsr_data.divergence * distance
  return loss, distance
