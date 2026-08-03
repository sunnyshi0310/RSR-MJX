"""Soft Actor-Critic losses with the RSR distribution objective.

This module follows ``brax.training.agents.sac.losses`` from Brax 0.12.1.
Only the actor objective is extended; alpha and critic updates retain the
upstream SAC equations.
"""

from typing import Any

from brax.training import types
from brax.training.agents.sac import networks as sac_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp

import RSR.rsr_loss as rsr


Transition = types.Transition


def make_losses(
    sac_network: sac_networks.SACNetworks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
    *,
    past_data: Any = None,
    rsr_loss_scale: float = 1.0,
):
  """Creates SAC losses, optionally adding RSR to the actor objective."""
  target_entropy = -0.5 * action_size
  policy_network = sac_network.policy_network
  q_network = sac_network.q_network
  action_distribution = sac_network.parametric_action_distribution

  def alpha_loss(
      log_alpha: jnp.ndarray,
      policy_params: Params,
      normalizer_params: Any,
      transitions: Transition,
      key: PRNGKey,
  ) -> jnp.ndarray:
    """Temperature loss (SAC equation 18)."""
    dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.observation
    )
    raw_action = action_distribution.sample_no_postprocessing(dist_params, key)
    log_prob = action_distribution.log_prob(dist_params, raw_action)
    alpha = jnp.exp(log_alpha)
    loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
    return jnp.mean(loss)

  def critic_loss(
      q_params: Params,
      policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      alpha: jnp.ndarray,
      transitions: Transition,
      key: PRNGKey,
  ) -> jnp.ndarray:
    """Twin-Q Bellman loss."""
    old_q = q_network.apply(
        normalizer_params,
        q_params,
        transitions.observation,
        transitions.action,
    )
    next_dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation
    )
    next_raw_action = action_distribution.sample_no_postprocessing(
        next_dist_params, key
    )
    next_log_prob = action_distribution.log_prob(
        next_dist_params, next_raw_action
    )
    next_action = action_distribution.postprocess(next_raw_action)
    next_q = q_network.apply(
        normalizer_params,
        target_q_params,
        transitions.next_observation,
        next_action,
    )
    next_value = jnp.min(next_q, axis=-1) - alpha * next_log_prob
    target_q = jax.lax.stop_gradient(
        transitions.reward * reward_scaling
        + transitions.discount * discounting * next_value
    )
    q_error = old_q - jnp.expand_dims(target_q, -1)

    truncation = transitions.extras['state_extras']['truncation']
    q_error *= jnp.expand_dims(1 - truncation, -1)
    return 0.5 * jnp.mean(jnp.square(q_error))

  def actor_loss(
      policy_params: Params,
      normalizer_params: Any,
      q_params: Params,
      alpha: jnp.ndarray,
      transitions: Transition,
      key: PRNGKey,
  ) -> jnp.ndarray:
    """Entropy-regularized actor loss plus the differentiable RSR penalty."""
    dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.observation
    )
    raw_action = action_distribution.sample_no_postprocessing(dist_params, key)
    log_prob = action_distribution.log_prob(dist_params, raw_action)
    action = action_distribution.postprocess(raw_action)
    q_action = q_network.apply(
        normalizer_params,
        q_params,
        transitions.observation,
        action,
    )
    base_actor_loss = jnp.mean(alpha * log_prob - jnp.min(q_action, axis=-1))

    sim2real_loss, _ = rsr.compute_rsr_loss(
        transitions.observation,
        action,
        transitions.next_observation,
        past_data,
        loss_scale=rsr_loss_scale,
    )
    return base_actor_loss + sim2real_loss

  return alpha_loss, critic_loss, actor_loss
