"""Brax SAC training adapter with an injectable RSR actor loss.

Brax 0.12.1 does not expose a public loss-factory argument on ``sac.train``.
This adapter temporarily replaces the loss module referenced by the Brax train
module, delegates the complete training loop to Brax, and restores the module
after training.  No installed-package files are modified.
"""

import functools
from types import SimpleNamespace
import threading
from typing import Any, Callable, Optional, Tuple, Union

from brax import base
from brax import envs
from brax.training import types
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as brax_sac_train
from brax.v1 import envs as envs_v1

import RSR.sac_losses as sac_losses


Metrics = types.Metrics
_LOSS_INJECTION_LOCK = threading.Lock()


def train(
    environment: Union[envs_v1.Env, envs.Env],
    num_timesteps: int,
    episode_length: int,
    past_data: Any = None,
    wrap_env: bool = True,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    discounting: float = 0.9,
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 1,
    normalize_observations: bool = False,
    max_devices_per_host: Optional[int] = None,
    reward_scaling: float = 1.0,
    tau: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    grad_updates_per_step: int = 1,
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[
        sac_networks.SACNetworks
    ] = sac_networks.make_sac_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_logdir: Optional[str] = None,
    eval_env: Optional[envs.Env] = None,
    randomization_fn: Optional[
        Callable[[base.System, Any], Tuple[base.System, base.System]]
    ] = None,
    rsr_loss_scale: float = 1.0,
):
  """Trains a SAC policy with the optional RSR distribution penalty.

  Except for ``past_data`` and ``rsr_loss_scale``, arguments and return values
  match ``brax.training.agents.sac.train.train`` in Brax 0.12.1.

  ``checkpoint_logdir`` follows Brax's naming convention and is a file prefix:
  checkpoints are written as ``<prefix>_sac_<step>.pkl``.
  """
  if rsr_loss_scale < 0:
    raise ValueError(
        f'rsr_loss_scale must be non-negative, got {rsr_loss_scale}'
    )

  train_kwargs = dict(
      environment=environment,
      num_timesteps=num_timesteps,
      episode_length=episode_length,
      wrap_env=wrap_env,
      wrap_env_fn=wrap_env_fn,
      action_repeat=action_repeat,
      num_envs=num_envs,
      num_eval_envs=num_eval_envs,
      learning_rate=learning_rate,
      discounting=discounting,
      seed=seed,
      batch_size=batch_size,
      num_evals=num_evals,
      normalize_observations=normalize_observations,
      max_devices_per_host=max_devices_per_host,
      reward_scaling=reward_scaling,
      tau=tau,
      min_replay_size=min_replay_size,
      max_replay_size=max_replay_size,
      grad_updates_per_step=grad_updates_per_step,
      deterministic_eval=deterministic_eval,
      network_factory=network_factory,
      progress_fn=progress_fn,
      checkpoint_logdir=checkpoint_logdir,
      eval_env=eval_env,
      randomization_fn=randomization_fn,
  )

  if past_data is None or rsr_loss_scale == 0:
    return brax_sac_train.train(**train_kwargs)

  make_losses = functools.partial(
      sac_losses.make_losses,
      past_data=past_data,
      rsr_loss_scale=rsr_loss_scale,
  )
  loss_module = SimpleNamespace(make_losses=make_losses)

  # The referenced module is replaced instead of mutating the global Brax
  # losses module, which keeps unrelated imports untouched.  The lock prevents
  # concurrent train calls from observing another call's RSR data.
  with _LOSS_INJECTION_LOCK:
    original_loss_module = brax_sac_train.sac_losses
    brax_sac_train.sac_losses = loss_module
    try:
      return brax_sac_train.train(**train_kwargs)
    finally:
      brax_sac_train.sac_losses = original_loss_module
