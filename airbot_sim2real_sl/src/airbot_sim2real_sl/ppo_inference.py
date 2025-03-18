#!/home/wang/anaconda3/envs/airbot/bin/python
import functools
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from orbax import checkpoint as ocp
from etils import epath
from jax.tree_util import tree_map

class PolicyInference:
    def __init__(self, ckpt_dir: str, env_fn):

        make_networks_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=(32, 32, 32, 32), 
        )

        x_data = []
        y_data = []
        ydataerr = []
        times = [datetime.now()]

        max_y, min_y = 5000,-5000
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
            filename = f'/push_{num_steps}.png'
            plt.savefig(filename)

        ckpt_path = epath.Path(ckpt_dir)  

        train_fn = functools.partial(
            ppo.train,
            num_timesteps=0, 
            network_factory=make_networks_factory,
            num_evals=60, reward_scaling=0.1,
            episode_length=1200, 
            normalize_observations=True, 
            action_repeat=1,
            unroll_length=10, 
            num_minibatches=32, 
            num_updates_per_batch=8,
            discounting=0.96, 
            learning_rate=1e-4, 
            entropy_cost=2e-2, 
            num_envs=512,
            batch_size=128, 
            num_resets_per_eval=1,      
            restore_checkpoint_path=ckpt_path,  
            seed=0 
        )
        self.make_inference_fn, self.params, _ = train_fn(environment=env_fn, progress_fn=progress)
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        self.rng = jax.random.PRNGKey(42)

    def get_action(self, observation: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        
        policy_fn = self.make_inference_fn(self.params, deterministic=deterministic)
        act_rng, self.rng = jax.random.split(self.rng)
        action, extras = policy_fn(observation, act_rng)

        with open('/real_action.txt', 'a') as f_ctrl:
            np.savetxt(f_ctrl, action.reshape(1, -1), fmt='%.6f', delimiter=',')
        ctrl_first_six = action[:6]
        ctrl = ctrl_first_six * 0.02
        return ctrl

