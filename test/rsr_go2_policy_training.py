"""RSR-SAC policy training for Go2 locomotion."""

from __future__ import annotations

import functools
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GO2_TRAINING = REPO_ROOT / 'ppo_train' / 'go2_training'
DATA_DIR = GO2_TRAINING / 'outputs'
OUTPUT_DIR = DATA_DIR / 'rsr_training'

# Must run before importing the local ``RSR`` package.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GO2_TRAINING))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from brax.training.agents.sac import networks as sac_networks
from etils import epath
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

import RSR.rsr_pipeline as rsr_pipeline

ENV_NAME = 'Go2JoystickFlatTerrain'
REQUIRED_DATA_FILES = (
    'real_obs.txt',
    'real_action.txt',
    'past_sim_obs.txt',
    'current_sim_obs.txt',
    'obs.txt',
    'actions.txt',
)

# Training options.
ALGORITHM = 'sac'
MAX_TRANSITIONS = 50
NUM_TIMESTEPS = 50_000
NUM_EVALS = 2
NUM_ENVS = 64
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 256
MAX_REPLAY_SIZE = 10_000
SEED = 0


def _require_data_file(filename: str) -> Path:
  path = DATA_DIR / filename
  if not path.is_file():
    raise FileNotFoundError(
        f'Required dataset file not found: {path}. '
        f'Expected files: {", ".join(REQUIRED_DATA_FILES)}'
    )
  return path


def _load_numeric_table(path: Path) -> np.ndarray:
  data = np.loadtxt(path, delimiter=',')
  if data.ndim == 1:
    data = data.reshape(1, -1)
  if data.size == 0:
    raise ValueError(f'{path.name} is empty.')
  return data


def _load_transition_triplet(
    obs_path: Path,
    action_path: Path,
    max_transitions: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Loads (s_t, a_t, s_{t+1}) with a shared transition count."""
  observations = _load_numeric_table(obs_path)
  actions = _load_numeric_table(action_path)

  transition_count = min(
      len(observations) - 1,
      len(actions),
      max_transitions,
  )
  if transition_count <= 0:
    raise ValueError(
        f'Not enough aligned transitions in {obs_path.name} and '
        f'{action_path.name}. Need at least 2 observations and 1 action.'
    )

  states = jnp.array(observations[:transition_count])
  action_seq = jnp.array(actions[:transition_count])
  next_states = jnp.array(observations[1:transition_count + 1])
  return states, action_seq, next_states


def _validate_observation_sequence(path: Path, transition_count: int) -> np.ndarray:
  observations = _load_numeric_table(path)
  required_rows = transition_count + 1
  if len(observations) < required_rows:
    raise ValueError(
        f'{path.name} needs at least {required_rows} rows for '
        f'{transition_count} transitions, found {len(observations)}.'
    )
  return observations


def _validate_action_sequence(path: Path, transition_count: int) -> np.ndarray:
  actions = _load_numeric_table(path)
  if len(actions) < transition_count:
    raise ValueError(
        f'{path.name} needs at least {transition_count} rows, '
        f'found {len(actions)}.'
    )
  return actions


def _validate_feature_width(
    arrays: dict[str, np.ndarray],
    expected_width: int,
    label: str,
) -> None:
  for name, array in arrays.items():
    if array.shape[1] != expected_width:
      raise ValueError(
          f'{name} must have {expected_width} {label} features, '
          f'found shape {array.shape}.'
      )


def load_rsr_datasets(max_transitions: int):
  """Loads and validates all datasets required by policy_params_training."""
  data_paths = {name: _require_data_file(name) for name in REQUIRED_DATA_FILES}

  past_states, past_actions, past_next_states_real = _load_transition_triplet(
      data_paths['real_obs.txt'],
      data_paths['real_action.txt'],
      max_transitions,
  )
  transition_count = int(past_states.shape[0])
  obs_dim = int(past_states.shape[1])
  action_dim = int(past_actions.shape[1])

  past_sim_obs = _validate_observation_sequence(
      data_paths['past_sim_obs.txt'], transition_count
  )
  current_sim_obs = _validate_observation_sequence(
      data_paths['current_sim_obs.txt'], transition_count
  )
  sim_obs = _validate_observation_sequence(
      data_paths['obs.txt'], transition_count
  )
  sim_actions = _validate_action_sequence(
      data_paths['actions.txt'], transition_count
  )

  _validate_feature_width(
      {
          'real_obs.txt': _load_numeric_table(data_paths['real_obs.txt']),
          'past_sim_obs.txt': past_sim_obs,
          'current_sim_obs.txt': current_sim_obs,
          'obs.txt': sim_obs,
      },
      obs_dim,
      'observation',
  )
  _validate_feature_width(
      {
          'real_action.txt': _load_numeric_table(data_paths['real_action.txt']),
          'actions.txt': sim_actions,
      },
      action_dim,
      'action',
  )

  past_next_states_sim = jnp.array(past_sim_obs[1:transition_count + 1])
  current_next_states_sim = jnp.array(current_sim_obs[1:transition_count + 1])

  print('====== RSR dataset summary ======')
  print(f'data_dir: {DATA_DIR}')
  print(f'transitions: {transition_count}')
  print(f'obs_dim: {obs_dim}, action_dim: {action_dim}')
  for filename in REQUIRED_DATA_FILES:
    print(f'{filename}: {data_paths[filename]}')

  return (
      past_states,
      past_actions,
      past_next_states_real,
      past_next_states_sim,
      current_next_states_sim,
  )


def build_go2_env():
  """Builds the wrapped Go2 environment used for RSR-SAC training."""
  sac_params = locomotion_params.brax_sac_config(ENV_NAME)
  env_cfg = registry.get_default_config(ENV_NAME)
  env = registry.load(ENV_NAME, config=env_cfg)
  env = wrapper.SelectObservationWrapper(env, obs_key=sac_params.policy_obs_key)
  return env, env_cfg, sac_params


def main():
  env, env_cfg, sac_params = build_go2_env()
  print(f'Environment: {ENV_NAME}')
  print(f'observation_size: {env.observation_size}, action_size: {env.action_size}')

  (
      past_states,
      past_actions,
      past_next_states_real,
      past_next_states_sim,
      current_next_states_sim,
  ) = load_rsr_datasets(MAX_TRANSITIONS)

  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  ckpt_path = epath.Path(OUTPUT_DIR / 'checkpoints')
  ckpt_path.mkdir(parents=True, exist_ok=True)
  plot_dir = OUTPUT_DIR / 'plots'
  plot_dir.mkdir(parents=True, exist_ok=True)

  eval_env = registry.load(ENV_NAME, config=env_cfg)
  eval_env = wrapper.SelectObservationWrapper(
      eval_env, obs_key=sac_params.policy_obs_key
  )

  x_data = []
  y_data = []
  ydataerr = []
  times = [datetime.now()]

  def progress_fn(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics['eval/episode_reward'])
    ydataerr.append(metrics['eval/episode_reward_std'])

    plt.clf()
    plt.xlim([0, NUM_TIMESTEPS * 1.25])
    if y_data:
      plt.ylim([min(y_data) * 1.1, max(y_data) * 1.1 + 1e-6])
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.title(f'y={y_data[-1]:.3f}')
    plt.errorbar(x_data, y_data, yerr=ydataerr)
    plt.savefig(plot_dir / f'go2_{num_steps}.png')

  network_factory = functools.partial(
      sac_networks.make_sac_networks,
      hidden_layer_sizes=tuple(sac_params.network_factory.hidden_layer_sizes),
  )

  make_inference_fn, params = rsr_pipeline.policy_params_training(
      env=env,
      algorithm=ALGORITHM,
      restore_checkpoint_path=None,
      past_states=past_states,
      past_actions=past_actions,
      past_next_states_real=past_next_states_real,
      past_next_states_sim=past_next_states_sim,
      current_next_states_sim=current_next_states_sim,
      num_timesteps=NUM_TIMESTEPS,
      num_evals=NUM_EVALS,
      num_envs=NUM_ENVS,
      batch_size=BATCH_SIZE,
      episode_length=env_cfg.episode_length,
      reward_scaling=sac_params.reward_scaling,
      normalize_observations=sac_params.normalize_observations,
      action_repeat=sac_params.action_repeat,
      discounting=sac_params.discounting,
      learning_rate=sac_params.learning_rate,
      tau=sac_params.tau,
      min_replay_size=MIN_REPLAY_SIZE,
      max_replay_size=MAX_REPLAY_SIZE,
      grad_updates_per_step=sac_params.grad_updates_per_step,
      progress_fn=progress_fn,
      network_factory=network_factory,
      checkpoint_logdir=str(ckpt_path / 'rsr'),
      wrap_env_fn=wrapper.wrap_for_brax_training,
      eval_env=eval_env,
      seed=SEED,
  )

  print(f'RSR {ALGORITHM.upper()} training finished.')
  print(f'checkpoints: {ckpt_path}')
  print(f'plots: {plot_dir}')
  return make_inference_fn, params


if __name__ == '__main__':
  main()
