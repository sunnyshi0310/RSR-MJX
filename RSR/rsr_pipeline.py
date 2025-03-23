# Standard libraries
import os
import time
from datetime import datetime
from etils import epath
import itertools
from typing import Callable, NamedTuple, Optional, Union, List, Any
import functools
from dataclasses import replace

# Third-party libraries
import jax, jaxlib
import jax.numpy as jnp
from jax import grad, vmap
import jaxlib.xla_extension
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
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

# Flax related
from flax import struct
from flax.training import orbax_utils
from orbax import checkpoint as ocp

# Dataset processing and training
import RSR.dataset_processor as dp
import RSR.train as rsr_ppo

# MuJoCo
import mujoco
from mujoco import mjx



# env params tuning
def env_params_tuning(
        init_env: Env, num_steps: int,
        init_env_params: dict, env_params_min: dict, env_params_max: dict,
        obs: Any, actions:Any, next_obs_true: Any,
        log_path: Any,
        ):
    """
    tune env params to approximate the true value

    Args:
    init_env:
    init_env_params:
    env_params_min:
    env_params_max:
    dataset:

    Returns:
    tuned_env:
    tuned_env_params:
    """

    len_data = len(obs)
    params = init_env_params


    # obs2state
    def obs2state(env, obs):
        # initialize the state
        rng = jax.random.PRNGKey(0)
        state_0 = env.reset(rng)
        ctrl = jnp.array([0,0,0,0,0])
        rng = jax.random.PRNGKey(0)

        state_1 = env.step(state_0, ctrl)
        states = []

        for i in range(len(obs)):
            new_qpos = state_0.pipeline_state.qpos
            new_qpos = new_qpos.at[env.joint_id].set(obs[i][0:6])
            new_qpos = new_qpos.at[env.cube_id + 2 : env.cube_id + 5].set(obs[i][12:15])
            new_pipeline_state = replace(
                state_0.pipeline_state,
                qpos=new_qpos,
                xpos=state_0.pipeline_state.xpos.at[env.cube_id].set(obs[i][12:15])
            )
            state_1 = replace(state_1, pipeline_state=new_pipeline_state)
            states.append(state_1)

        print("===data load===")
        return states
    
    # set params
    def set_params(env, params):
        env_new = env
        in_axes = jax.tree_util.tree_map(lambda x: None, env_new.sys)
        in_axes = in_axes.tree_replace({
            param_name: 0
            for param_name, new_value in params.items()
        })
        env_new.sys = env_new.sys.tree_replace({
            param_name: new_value
            for param_name, new_value in params.items()
        })
        
        return env_new
    
    # clip the params
    def clip_params(params, params_min, params_max): 
        return jax.tree_util.tree_map( lambda p, min_val, max_val: jnp.clip(p, a_min=min_val, a_max=max_val), params, params_min, params_max )

    def compute_error(next_obs_pred: jnp.array, next_obs_true: jnp.array):
        w = jnp.array([1,1,1,1,1,1,10,10,10,0,0,0,10,10,10,10,10,0,0,0,0,0,0])
        error = next_obs_pred - next_obs_true
        weighted_error = jnp.dot(w, error)
        return jnp.linalg.norm(weighted_error)
    
    @jax.jit
    def step_with_params(params, state, action):
        env = init_env
        # friction_data = env.sys.geom_friction.at[-7,:].set(params)
        friction_data = env.sys.geom_friction.at[-1,:].set(params)
        all_params = {'geom_friction': friction_data}
        # gravity_data = init_env.sys.gravity.at[-1].set(params)
        # all_params = {'gravity': gravity_data}
        # mass_data = init_env.sys.body_mass.at[-1].set(params)
        # all_params = {'body_mass': mass_data}
        env_new = set_params(env, all_params)
        return env_new.step(state, action), params
    
    # def step_with_params(env, params):
    #     env_new = set_params(env, params)
    #     step_fn = jax.jit(env_new.step)
    #     # step_fn = env_new.step
    #     return step_fn
    
    # loss function
    # @jax.jit
    def loss_fn(params): 
        ############### (2)
        # friction_data = init_env.sys.geom_friction.at[-1].set(params)
        # all_params = {'geom_friction': friction_data}
        # env_new = set_params(init_env, all_params)
       
        # jit_step = jax.jit(env_new.step)
        ##############

        error_sum = jnp.array(0)

        for i in range(len_data):
            next_obs_pred, params = step_with_params(params, states[i], actions[i]) # (1)

            error = compute_error(next_obs_pred.obs,  next_obs_true[i])
            error_sum = error_sum + error
        return error_sum
    
    

    def create_optimizer():
        return optax.adam(learning_rate=0.005)
    
    def update_step(opt_state, params):
        grads = grad(loss_fn)(params)  # 计算梯度
        updates, opt_state = opt_update(grads, opt_state)  # 更新优化器状态
        new_params = optax.apply_updates(params, updates)  # 应用更新到参数
        new_params = clip_params(new_params, env_params_min, env_params_max) # 对new_params clip
        return new_params, opt_state
    
    states = obs2state(env= init_env, obs=obs)
    opt_init, opt_update = create_optimizer()
    opt_state = opt_init(init_env_params)
    print("===optimizer initialized===")
    print(loss_fn(init_env_params))
    

    
    train_time = []
    train_loss = []
    train_params = []
    for i in range(num_steps):
        time_0 = time.time()
        params, opt_state = update_step(opt_state, params)  # 更新参数
        time_step = time.time()-time_0
        loss = loss_fn(params)
        line = f"step {i}: {time_step:.2f}s. params = {params}. loss = {loss}."
        print(line)
        train_time.append(time_step)
        train_loss.append(loss)
        train_params.append(params)
        with open(log_path, 'a') as f:
            f.write(line + '\n')


    tuned_env_params = params
    train_log = {
        "time_cost": train_time,
        "loss": train_loss,
        "params": train_params,}
    return tuned_env_params, train_log

# policy params tuning
def policy_params_training(
        env: Env, restore_checkpoint_path, policy_params_fn, network_factory, progress_fn,
        past_states: jnp.array, past_actions: jnp.array, 
        past_next_states_real: jnp.array, past_next_states_sim: jnp.array, current_next_states_sim: jnp.array,
        num_samples: int =10, min_val: int =-3, max_val: int =3, bandwidth: float = 0.1,
        num_timesteps: int=5_000_000, 
        num_evals:int=10, 
        reward_scaling:float=0.1, 
        episode_length:int=1200, 
        normalize_observations:bool=True, 
        action_repeat:int =1, 
        unroll_length:int =10, 
        num_minibatches:int =32, 
        num_updates_per_batch:int =8, 
        discounting:float=0.96, 
        learning_rate:float =1e-4, 
        entropy_cost:float =2e-2, 
        num_envs:int =512, 
        batch_size: int =128, 
        seed: int=0):
    """
    training policy with ppo_rsr loss

    Args:
    env:
    init_policy_params:

    Returns:
    make_interfernce_fn:
    tuned_policy_params:
    """
    
    len_dataset = len(past_states)

    # D^{k}_{real}
    D1 = jnp.hstack([past_states, past_actions, past_next_states_real])

    # D^{k-1}_{sim}
    D2 = jnp.hstack([past_states, past_actions, past_next_states_sim])

    # D^{k}_{sim}
    D3 = jnp.hstack([past_states, past_actions, current_next_states_sim])

    dim = len(D1[0])
    grid = jax.random.uniform(jax.random.PRNGKey(0), (num_samples, dim), minval=min_val, maxval=max_val)

    p_D1 = dp.evaluate_kde(D1, grid, bandwidth)
    p_D2 = dp.evaluate_kde(D2, grid, bandwidth)
    p_D3 = dp.evaluate_kde(D3, grid, bandwidth)

    KLD = dp.kl_divergence(p_D1, p_D2)
    past_data = (KLD, p_D3, D3) 
    print("====== past_data calculated =======")
    


    train_fn = functools.partial(
        rsr_ppo.train, 
        num_timesteps=num_timesteps, 
        num_evals=num_evals, 
        reward_scaling=reward_scaling, 
        episode_length=episode_length, 
        normalize_observations=normalize_observations, 
        action_repeat=action_repeat, 
        unroll_length=unroll_length, 
        num_minibatches=num_minibatches, 
        num_updates_per_batch=num_updates_per_batch, 
        discounting=discounting, 
        learning_rate=learning_rate, 
        entropy_cost=entropy_cost, 
        num_envs=num_envs, 
        batch_size=batch_size, 
        restore_checkpoint_path=restore_checkpoint_path,
        policy_params_fn=policy_params_fn,
        network_factory=network_factory,
        seed=0)


    make_inference_fn, params, _ = train_fn(
        environment=env, 
        past_data=past_data, 
        progress_fn=progress_fn)
    
    tuned_policy_params = params

    return make_inference_fn, tuned_policy_params