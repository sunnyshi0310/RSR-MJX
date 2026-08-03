# RSR-MJX

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

### SAC support

The repository supports both PPO and SAC:

- `ppo_train/airbot_training/train.py` trains the original PPO baseline.
- `ppo_train/airbot_training/train_sac.py` trains a native Brax SAC baseline.
- `ppo_train/go2_training/learning/train_jax_sac.py` trains locomotion tasks
  such as `Go2JoystickFlatTerrain` with Brax SAC.
- `RSR.rsr_pipeline.policy_params_training` applies the RSR objective with
  either algorithm.  PPO remains the default for backward compatibility.

Select SAC in the shared RSR pipeline:

```python
from brax.training.agents.sac import networks as sac_networks
from RSR.rsr_pipeline import policy_params_training
import functools

network_factory = functools.partial(
    sac_networks.make_sac_networks,
    hidden_layer_sizes=(256, 256),
)

make_inference_fn, params = policy_params_training(
    env=env,
    algorithm="sac",
    network_factory=network_factory,
    past_states=past_states,
    past_actions=past_actions,
    past_next_states_real=past_next_states_real,
    past_next_states_sim=past_next_states_sim,
    current_next_states_sim=current_next_states_sim,
    min_replay_size=100_000,
    max_replay_size=1_000_000,
    grad_updates_per_step=1,
    tau=0.005,
    checkpoint_logdir="path/to/checkpoints/rsr",
)
```

SAC-specific notes:

- The RSR term is added to the SAC actor loss; critic and temperature losses
  use the standard Brax equations.
- `rsr_loss_scale=0.0` disables the RSR term and gives standard Brax SAC.
- In Brax 0.12.1, `checkpoint_logdir` is a filename prefix.  Checkpoints are
  written as `<prefix>_sac_<step>.pkl`.
- Brax 0.12.1 SAC checkpoints contain inference parameters only and cannot
  restore the optimizer and replay buffer for exact training resumption.
- PPO and SAC checkpoint parameter trees are different and are not
  interchangeable.

### Go2 / locomotion SAC training

Run from `ppo_train/go2_training/learning`:

```bash
conda activate mjx
cd ppo_train/go2_training/learning
PYTHONPATH=.. python train_jax_sac.py --env_name=Go2JoystickFlatTerrain
```

Useful flags:

```bash
PYTHONPATH=.. python train_jax_sac.py \
  --env_name=Go2JoystickFlatTerrain \
  --num_timesteps=5000000 \
  --num_envs=1024 \
  --batch_size=256 \
  --min_replay_size=100000 \
  --domain_randomization=true
```

Notes:

- Locomotion tasks expose dictionary observations. SAC training selects the
  `state` vector automatically through `SelectObservationWrapper`.
- Go2 robot assets are bundled locally under
  `mujoco_playground/_src/locomotion/go2/xmls`; no `mujoco_menagerie` download
  is required.
- Checkpoints are written as `logs/<experiment>/checkpoints/sac_sac_<step>.pkl`.
- The final inference checkpoint is also saved as
  `logs/<experiment>/checkpoints/final_sac.pkl`.

### Go2 RSR-SAC training

Run from the repository root:

```bash
conda activate mjx
python test/rsr_go2_policy_training.py
```

Place the six required aligned datasets under
`ppo_train/go2_training/outputs`:

- `real_obs.txt` — real observations `s_t` (48-dim `state` vector)
- `real_action.txt` — real actions `a_t` (12-dim)
- `past_sim_obs.txt` — sim observations before env-parameter tuning
- `current_sim_obs.txt` — sim observations after env-parameter tuning
- `obs.txt` — sim observation rollout for alignment checks
- `actions.txt` — sim action rollout for alignment checks

Checkpoints and plots are written to
`ppo_train/go2_training/outputs/rsr_training/`.

Notes:

- Go2 uses `SelectObservationWrapper` to extract the 48-dim `state` vector
  required by Brax SAC.
- `wrap_for_brax_training` is applied automatically for the MJX environment.
- For full training, increase `NUM_TIMESTEPS`, `NUM_ENVS`, and `MIN_REPLAY_SIZE`
  in `test/rsr_go2_policy_training.py` or pass production-scale values.

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
