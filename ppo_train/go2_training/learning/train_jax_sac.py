"""Train a SAC agent using JAX on MuJoCo Playground locomotion environments."""
from datetime import datetime
import functools
import json
import os
import time
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.io import model
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
import mujoco
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

logging.set_verbosity(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "Go2JoystickFlatTerrain",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 5_000_000, "Number of timesteps"
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 10, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 1.0, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 3e-4, "Learning rate")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_TAU = flags.DEFINE_float("tau", 0.005, "Target network soft update rate")
_MIN_REPLAY_SIZE = flags.DEFINE_integer(
    "min_replay_size", 100_000, "Replay buffer warm-up size"
)
_MAX_REPLAY_SIZE = flags.DEFINE_integer(
    "max_replay_size", 1_000_000, "Maximum replay buffer size"
)
_GRAD_UPDATES_PER_STEP = flags.DEFINE_integer(
    "grad_updates_per_step", 1, "Gradient updates per environment step"
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Observation key used by SAC"
)
_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "hidden_layer_sizes",
    [256, 256],
    "SAC actor/critic hidden layer sizes",
)
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path",
    None,
    "Optional Brax SAC inference checkpoint (.pkl) for play-only mode",
)


def get_sac_config(env_name: str):
  if env_name not in mujoco_playground.locomotion.ALL_ENVS:
    raise ValueError(
        f"Env {env_name} is not a locomotion task. "
        f"Available locomotion envs: {mujoco_playground.locomotion.ALL_ENVS}"
    )
  return locomotion_params.brax_sac_config(env_name)


def main(argv):
  """Run SAC training and evaluation for the specified locomotion environment."""
  del argv

  sac_params = get_sac_config(_ENV_NAME.value)
  if _NUM_TIMESTEPS.present:
    sac_params.num_timesteps = _NUM_TIMESTEPS.value
  if _PLAY_ONLY.present:
    sac_params.num_timesteps = 0
  if _NUM_EVALS.present:
    sac_params.num_evals = _NUM_EVALS.value
  if _REWARD_SCALING.present:
    sac_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present:
    sac_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    sac_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    sac_params.action_repeat = _ACTION_REPEAT.value
  if _DISCOUNTING.present:
    sac_params.discounting = _DISCOUNTING.value
  if _LEARNING_RATE.present:
    sac_params.learning_rate = _LEARNING_RATE.value
  if _NUM_ENVS.present:
    sac_params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present:
    sac_params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present:
    sac_params.batch_size = _BATCH_SIZE.value
  if _TAU.present:
    sac_params.tau = _TAU.value
  if _MIN_REPLAY_SIZE.present:
    sac_params.min_replay_size = _MIN_REPLAY_SIZE.value
  if _MAX_REPLAY_SIZE.present:
    sac_params.max_replay_size = _MAX_REPLAY_SIZE.value
  if _GRAD_UPDATES_PER_STEP.present:
    sac_params.grad_updates_per_step = _GRAD_UPDATES_PER_STEP.value
  if _POLICY_OBS_KEY.present:
    sac_params.policy_obs_key = _POLICY_OBS_KEY.value
  if _HIDDEN_LAYER_SIZES.present:
    sac_params.network_factory.hidden_layer_sizes = list(
        map(int, _HIDDEN_LAYER_SIZES.value)
    )

  env_cfg = registry.get_default_config(_ENV_NAME.value)
  env = registry.load(_ENV_NAME.value, config=env_cfg)
  env = wrapper.SelectObservationWrapper(env, obs_key=sac_params.policy_obs_key)

  print(f"Environment Config:\n{env_cfg}")
  print(f"SAC Training Parameters:\n{sac_params}")

  now = datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{_ENV_NAME.value}-sac-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  print(f"Experiment name: {exp_name}")

  logdir = epath.Path("logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  checkpoint_prefix = ckpt_path / "sac"
  print(f"Checkpoint prefix: {checkpoint_prefix}")

  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  training_params = dict(sac_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]
  if "policy_obs_key" in training_params:
    del training_params["policy_obs_key"]

  network_factory = functools.partial(
      sac_networks.make_sac_networks,
      hidden_layer_sizes=tuple(sac_params.network_factory.hidden_layer_sizes),
  )

  if _DOMAIN_RANDOMIZATION.value:
    training_params["randomization_fn"] = registry.get_domain_randomizer(
        _ENV_NAME.value
    )

  train_fn = functools.partial(
      sac.train,
      **training_params,
      network_factory=network_factory,
      checkpoint_logdir=str(checkpoint_prefix),
      seed=_SEED.value,
      wrap_env_fn=wrapper.wrap_for_brax_training,
  )

  times = [time.monotonic()]

  def progress(num_steps, metrics):
    times.append(time.monotonic())
    print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")

  eval_env = registry.load(_ENV_NAME.value, config=env_cfg)
  eval_env = wrapper.SelectObservationWrapper(
      eval_env, obs_key=sac_params.policy_obs_key
  )

  make_inference_fn, params, _ = train_fn(
      environment=env,
      progress_fn=progress,
      eval_env=eval_env,
  )

  print("Done training.")
  if len(times) > 1:
    print(f"Time to JIT compile: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

  if _LOAD_CHECKPOINT_PATH.value is not None:
    params = model.load_params(_LOAD_CHECKPOINT_PATH.value)
    print(f"Loaded checkpoint for rollout: {_LOAD_CHECKPOINT_PATH.value}")

  final_model_path = ckpt_path / "final_sac.pkl"
  model.save_params(str(final_model_path), params)
  print(f"Saved final model to: {final_model_path}")

  print("Starting inference...")
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)
  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)

  rng = jax.random.PRNGKey(123)
  state = jit_reset(rng)
  rollout = [state]
  for _ in range(env_cfg.episode_length):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state)
    if state.done:
      break

  render_every = 2
  fps = 1.0 / eval_env.dt / render_every
  print(f"FPS for rendering: {fps}")
  traj = rollout[::render_every]
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
  frames = eval_env.render(
      traj, height=480, width=640, scene_option=scene_option
  )
  rollout_path = logdir / "rollout.mp4"
  media.write_video(str(rollout_path), frames, fps=fps)
  print(f"Rollout video saved as '{rollout_path}'.")


if __name__ == "__main__":
  app.run(main)
