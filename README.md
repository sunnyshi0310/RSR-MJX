# RSR-MJX

This repository contains the codebase for the paper "RSR: An Real-Sim-Real (RSR) Loop Framework for Generalizable Robotic Policy Transfer with Differentiable Simulation" submitted to IROS2025. In this work, we propose a framework that leverages real-world data to adjust the parameters of the differentiable simulator - MJX. 

## Setup Instructions

### 1. Create an environment
To begin, create and activate a new Conda environment:
```bash
conda create -n unienv python=3.10
conda activate unienv
```
### 2. Setup the dependencies
Create a file named ``requirements.txt`` and add the following content:
```bash
numpy==1.26.4
scipy==1.12.0
optax==0.2.4
brax==0.12.1
jax==0.4.29
nvidia-cudnn-cu12==9.1.0.70
nvidia-cuda-cupti-cu12
--find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
jaxlib==0.4.29+cuda12.cudnn91
mujoco==3.2.4
mujoco-mjx==3.2.4
```

Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Tune your environmental parameters in MJX with real-world data

## Usage for ppo_train
In ```train.py```, import AirbotPlayBase from ```cube_env```or```T_shape_env``` and select the corresponding mjcf file in the those env.

If you want to start training base on the previous policy network, set a reasonable ```ckpt_path_restart``` path, otherwise comment it out in the ```train_fn```.

Please set up other paths reasonably, such as:```ckpt_path, image_path, model_path, vedio_folder```

If you're having trouble with ```nan```, you can try uncommenting: ```jax config``` (especially in T_shape task)

You can try modifying the hyperparameters to achieve better training results, such as:```num_timesteps, episode_length, num_minibatches, discounting, learning_rate, num_envs, batch_size```

## Usage for real_robot_inference

The folders `airbot_sim2real_sl` and `airbot_t` correspond to experimental tasks involving cubic objects and T-shaped objects, respectively. Both are structured as standard ROS packages.

### Core Components (airbot_sim2real_sl Implementation)
**`sim2real_sl_control_node.py`**  
  Serves as the main control node for the system.

**`marker_pose_publisher.py`**  
  Implements the node responsible for activating and managing the Intel RealSense depth camera.

**`ppo_inference.py`**  
  Handles inference execution of pre-trained reinforcement learning policies.

**`real_robot_interface.py`**  
  Manages hardware interaction with the Airbot Play robotic arm, including:  
  - Interface invocation for robotic manipulation  
  - Real-world state monitoring through sensor data acquisition from physical environments
