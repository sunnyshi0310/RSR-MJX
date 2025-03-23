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
import RSR.rsr_pipeline as pipeline


### define your env here ###
import sys
# change to your environment
from airbot import AirbotPlayBase
### read data ###

# convert data from txt
def txt_to_2d_array(file_path):
    """
    将txt文件中的浮点数转换为二维数组
    :param file_path: txt文件路径
    :return: 二维数组 (list of lists)
    """
    result = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去除首尾空白并跳过空行
            stripped_line = line.strip()
            if not stripped_line:
                continue
            
            # 分割字符串并转换为浮点数
            numbers = [float(num) for num in stripped_line.split(',')]
            result.append(numbers)
    return jnp.array(result)




real_obs = txt_to_2d_array("/real_obs.txt") # change to your path of real obs
actions = txt_to_2d_array("/real_action.txt") # change to your path of real action

envs.register_environment('airbot', AirbotPlayBase)
env_name = 'airbot'

env = envs.get_environment(env_name)
import numpy as np


init_env_params = jnp.array(0.4)
env_params_max = init_env_params * 10.0
env_params_min = init_env_params * 0.2

print(init_env_params, env_params_max, env_params_min)


n = 15
index = 0
num_steps = 1000


interval = 3
indices = range(0, len(real_obs), interval)
# sampled_obs = jnp.array([real_obs[idx] for idx in indices][index:index+ n])
# sampled_actions = jnp.array([actions[idx] for idx in indices][index:index+ n])
# sampled_next_obs_true = jnp.array([real_obs[idx+1] for idx in indices][index:index+ n])

sampled_obs = real_obs[index:index+ n]
sampled_actions = actions[index:index+ n]
sampled_next_obs_true = real_obs[1+index:1+index+ n]


log_path='log.txt'

with open(log_path, 'w') as f:
    f.write(f"n_samples = {n}, num_steps = {num_steps}" + '\n')
    f.write(f"init_params = {init_env_params}" + '\n')

new_params, train_log = pipeline.env_params_tuning(
    init_env= env, 
    num_steps= num_steps,
    init_env_params= init_env_params,
    env_params_max= env_params_max,
    env_params_min= env_params_min,
    obs= sampled_obs,
    actions= sampled_actions,
    next_obs_true= sampled_next_obs_true,
    log_path=log_path)


