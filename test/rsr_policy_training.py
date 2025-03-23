# Standard libraries
import os
import time
from datetime import datetime
from etils import epath
import itertools
from typing import Callable, NamedTuple, Optional, Union, List, Any
import functools

# Third-party libraries
import jax
import jax.numpy as jnp
from jax import grad
import optax
import numpy as np
import scipy.optimize as sp_opt
import matplotlib.pyplot as plt
import mediapy as media

# Brax and MuJoCo related
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.io import html, mjcf, model
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

# Flax related
from flax import struct
from flax.training import orbax_utils
from orbax import checkpoint as ocp

# MuJoCo
import mujoco
from mujoco import mjx

"""ATTENTION HERE"""
"""rsr_ppo instead of brax ppos"""
# Dataset processing and training(RSR)
import RSR.dataset_processor as dp
import RSR.train as rsr_ppo
import RSR.rsr_pipeline as rsr_pipeline


### define your env here ###
import sys
from airbot import AirbotPlayBase

envs.register_environment('airbot', AirbotPlayBase)
env_name = 'airbot'
env = envs.get_environment(env_name)

### data ###
# past states 真机上的数据(m,t)
file_path = '/real_obs.txt'
past_states = jnp.array(np.loadtxt(file_path, delimiter=',', max_rows=50))

# past actions 根据目前的policy推算的action(n,t)
file_path = '/real_action.txt'
past_actions = jnp.array(np.loadtxt(file_path, delimiter=',', max_rows=50))

# past_next_states_real 真机上的下一步状态(m,t)
file_path = '/real_obs.txt'
past_next_states_real = jnp.array(np.loadtxt(file_path, delimiter=',', skiprows=1, max_rows=50))

# past_next_states_sim 在尚未调整参数的仿真器上的下一步状态(m,t)
file_path = '/past_sim_obs.txt'
past_next_states_sim = jnp.array(np.loadtxt(file_path, delimiter=',', skiprows=1, max_rows=50))

# current_next_states_sim 在已经调整参数的仿真器上的下一步状态(m,t)
file_path = '/current_sim_obs.txt'
current_next_states_sim = jnp.array(np.loadtxt(file_path, delimiter=',', skiprows=1, max_rows=50))


ckpt_path = epath.Path('/checkpoints')
ckpt_path.mkdir(parents=True, exist_ok=True)

def policy_params_fn(current_step, make_policy, params):
    # save checkpoints
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = ckpt_path / f'{current_step}'
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)

x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]
max_y, min_y = 13000, 0
def progress_fn(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics['eval/episode_reward'])
    ydataerr.append(metrics['eval/episode_reward_std'])

    plt.xlim([0, 5000000 * 1.25])
    plt.ylim([min_y, max_y])

    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.title(f'y={y_data[-1]:.3f}')

    plt.errorbar(
        x_data, y_data, yerr=ydataerr)
    filename = f'push_{num_steps}.png'
    plt.savefig(filename)
    # plt.show()

restore_checkpoint_path = epath.Path('')

network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(32, 32, 32, 32))

fn, params = rsr_pipeline.policy_params_training(
    env= env,
    init_policy_params=None,
    past_states = past_states,
    past_actions = past_actions ,
    past_next_states_real= past_next_states_real,
    past_next_states_sim= past_next_states_sim,
    current_next_states_sim= current_next_states_sim,
    progress_fn=progress_fn,
    network_factory=network_factory,
    policy_params_fn=policy_params_fn
)