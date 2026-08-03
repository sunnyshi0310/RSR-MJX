"""Train the Airbot task with the native Brax SAC implementation."""

import functools
from datetime import datetime
from pathlib import Path

from brax import envs
from brax.io import model
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
import imageio
import jax

from cube_env import AirbotPlayBase
from domain_randomize import domain_randomize

DOMAIN_RANDOMIZATION = True

OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
CHECKPOINT_PREFIX = OUTPUT_DIR / 'checkpoints' / 'airbot_sac'
MODEL_PATH = OUTPUT_DIR / 'models' / 'airbot_sac.pkl'
VIDEO_PATH = OUTPUT_DIR / 'video' / 'push_sac.mp4'

for path in (CHECKPOINT_PREFIX.parent, MODEL_PATH.parent, VIDEO_PATH.parent):
  path.mkdir(parents=True, exist_ok=True)

envs.register_environment('airbot_sac', AirbotPlayBase)
env = envs.get_environment('airbot_sac')
eval_env = envs.get_environment('airbot_sac')
print(f'domain_randomization={DOMAIN_RANDOMIZATION}')

network_factory = functools.partial(
    sac_networks.make_sac_networks,
    hidden_layer_sizes=(256, 256),
)

train_fn = functools.partial(
    sac.train,
    num_timesteps=500_000,
    episode_length=1200,
    num_evals=10,
    reward_scaling=0.1,
    normalize_observations=True,
    action_repeat=1,
    discounting=0.96,
    learning_rate=1e-4,
    num_envs=1024,
    batch_size=256,
    min_replay_size=100_000,
    max_replay_size=1_000_000,
    grad_updates_per_step=1,
    network_factory=network_factory,
    checkpoint_logdir=str(CHECKPOINT_PREFIX),
    randomization_fn=domain_randomize if DOMAIN_RANDOMIZATION else None,
    seed=0,
)

times = [datetime.now()]


def progress(num_steps, metrics):
  times.append(datetime.now())
  reward = float(metrics['eval/episode_reward'])
  print(f'step={num_steps}, eval_reward={reward:.3f}')


make_inference_fn, params, _ = train_fn(
    environment=env,
    progress_fn=progress,
    eval_env=eval_env,
)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')

model.save_params(str(MODEL_PATH), params)
print(f'saved model to {MODEL_PATH}')
params = model.load_params(str(MODEL_PATH))

inference_fn = make_inference_fn(params, deterministic=True)
jit_inference_fn = jax.jit(inference_fn)
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
rollout = [state.pipeline_state]
for _ in range(1500):
  action_key, rng = jax.random.split(rng)
  action, _ = jit_inference_fn(state.obs, action_key)
  state = jit_step(state, action)
  rollout.append(state.pipeline_state)

imageio.mimwrite(
    str(VIDEO_PATH),
    env.render(rollout),
    fps=120,
    macro_block_size=None,
)
print(f'saved rollout video to {VIDEO_PATH}')
