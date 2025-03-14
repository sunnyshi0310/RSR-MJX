import jax
from jax import numpy as jp
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model
from etils import epath
import functools
from matplotlib import pyplot as plt
from datetime import datetime
import imageio
from cube_env import AirbotPlayBase
from jax import config
# config.update("jax_debug_nans", True)
# jax.config.update('jax_default_matmul_precision',jax.lax.Precision.HIGH)

envs.register_environment('airbot', AirbotPlayBase)
env_name = 'airbot'
env = envs.get_environment(env_name)
print(env.dt)
make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(32, 32, 32, 32))

from orbax import checkpoint as ocp
from flax.training import orbax_utils
ckpt_path = epath.Path('path/to/your/ckpt_folder')
ckpt_path.mkdir(parents=True, exist_ok=True)

def policy_params_fn(current_step, make_policy, params):
  # save checkpoints
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(params)
  path = ckpt_path / f'{current_step}'
  orbax_checkpointer.save(path, params, force=True, save_args=save_args)


ckpt_path_restart = epath.Path('path/to/your/checkpoints')

train_fn = functools.partial(
    ppo.train, num_timesteps=15_000_000, num_evals=10, reward_scaling=0.1,
    episode_length=1200, normalize_observations=True, action_repeat=1,
    unroll_length=10, num_minibatches=32, num_updates_per_batch=8,
    discounting=0.96, learning_rate=1e-4, entropy_cost=2e-2, num_envs=1024,
    batch_size=256, num_resets_per_eval=1,
    network_factory=make_networks_factory, 
    policy_params_fn=policy_params_fn,
    restore_checkpoint_path=ckpt_path_restart,
    seed=0)

x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]

max_y, min_y = 13000,2000
def progress(num_steps, metrics):
  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics['eval/episode_reward'])
  ydataerr.append(metrics['eval/episode_reward_std'])

  plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
  plt.ylim([min_y, max_y])

  plt.xlabel('# environment steps')
  plt.ylabel('reward per episode')
  plt.title(f'y={y_data[-1]:.3f}')

  plt.errorbar(
      x_data, y_data, yerr=ydataerr)
  filename = f'path/to/your/img_folder/push_{num_steps}.png'
  plt.savefig(filename)

make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)



print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')


model_path = 'path/to/your/madel_folder'
model.save_params(model_path, params)
params = model.load_params(model_path)

inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)
eval_env = envs.get_environment(env_name)

jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)
# initialize the state
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
rollout = [state.pipeline_state]


n_steps = 1500
render_every = 2

for i in range(n_steps):
  act_rng, rng = jax.random.split(rng)
  ctrl, _ = jit_inference_fn(state.obs, act_rng)
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)

if state.done:
   print('\n************************************\ndone\n************************************')

imageio.mimwrite('path/to/your/vedio_folder/push.mp4', env.render(rollout), fps = 120, macro_block_size=None)
