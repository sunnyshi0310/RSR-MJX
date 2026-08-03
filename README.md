# RSR-MJX

An **RSR (Real-to-Sim-to-Real)** reinforcement learning framework built on MuJoCo MJX and Brax. The project aligns real-world collected data with simulation environments using distribution-matching constraints (KDE + Wasserstein) to reduce the sim-to-real gap. Both **PPO** and **SAC** are supported.

Two experimental platforms are included:

| Platform | Task | Baseline Training | RSR Training |
|----------|------|-------------------|--------------|
| Airbot Play arm | Cube pushing / T-shaped object | `ppo_train/airbot_training/` | `test/rsr_policy_training.py` |
| Unitree Go2 quadruped | Flat/rough terrain joystick control, etc. | `ppo_train/go2_training/learning/` | `test/rsr_go2_policy_training.py` |

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Full Workflow](#full-workflow)
- [Airbot Experiments](#airbot-experiments)
- [Go2 Experiments](#go2-experiments)
- [RSR Core API](#rsr-core-api)
- [Data File Format](#data-file-format)
- [Real Robot Deployment](#real-robot-deployment)

---

## Environment Setup

### 1. Create a Conda environment

```bash
conda create -n mjx python=3.10
conda activate mjx
```

### 2. Install dependencies

Create `requirements.txt` in the project root:

```text
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
mediapy
matplotlib
etils
flax
orbax-checkpoint
absl-py
ml-collections
```

```bash
pip install -r requirements.txt
```

### 3. Before you run

- Run all RSR training scripts from the **project root**
- Go2 training scripts require `PYTHONPATH` to point to `ppo_train/go2_training`
- Go2 robot assets are bundled under `mujoco_playground/_src/locomotion/go2/xmls/`; no `mujoco_menagerie` download is needed
- A CPU-only JAX build can be used without a GPU, but training will be significantly slower

---

## Full Workflow

A complete RSR experiment typically follows these steps:

### Step 1: Collect real-world data

Run a policy (or teleoperate) on the real robot and record aligned trajectory data as comma-separated text files:

- `real_obs.txt` — real observation sequence \(s_t\)
- `real_action.txt` — real action sequence \(a_t\)

### Step 2: Train a simulation baseline policy (optional)

Train an initial policy in pure simulation as a starting point for RSR fine-tuning:

```bash
# Airbot PPO baseline
cd ppo_train/airbot_training
python train.py

# Airbot SAC baseline
python train_sac.py

# Go2 SAC baseline
cd ppo_train/go2_training/learning
PYTHONPATH=.. python train_jax_sac.py --env_name=Go2JoystickFlatTerrain
```

### Step 3: Align environment parameters

Optimize simulation physics parameters (e.g. friction) using real-world data so that one-step simulation predictions better match reality:

```bash
# Edit data paths and parameter ranges in test/rsr_env_params_tuning.py
python test/rsr_env_params_tuning.py
```

API: `RSR.rsr_pipeline.env_params_tuning(...)`

### Step 4: Prepare RSR policy training data

Execute the real-world action sequence in simulation before and after environment parameter tuning to produce four additional files:

| File | Description |
|------|-------------|
| `past_sim_obs.txt` | Simulation observations **before** parameter tuning |
| `current_sim_obs.txt` | Simulation observations **after** parameter tuning |
| `obs.txt` | Simulation rollout observations (for alignment checks) |
| `actions.txt` | Simulation rollout actions (for alignment checks) |

Together with `real_obs.txt` and `real_action.txt` from Step 1, all **six files are required**.

### Step 5: RSR policy training

```bash
# Airbot (set ALGORITHM = 'ppo' or 'sac' in the script)
python test/rsr_policy_training.py

# Go2 quadruped RSR-SAC
python test/rsr_go2_policy_training.py
```

### Step 6: Deploy on the real robot

Deploy the trained checkpoint to the corresponding ROS package under `real_robot_inference/` (see [Real Robot Deployment](#real-robot-deployment)).

---

## Airbot Experiments

### Environment selection

Switch environments in `ppo_train/airbot_training/train.py`:

```python
from cube_env import AirbotPlayBase      # cube pushing task
# from T_shape_env import AirbotPlayBase  # T-shaped object task
```

`test/airbot.py` provides a test Airbot environment definition (23-dim observations, 5-dim actions).

### PPO baseline training

```bash
cd ppo_train/airbot_training
python train.py
```

Configure the following in `train.py` before training:

- `ckpt_path` — checkpoint save directory
- `ckpt_path_restart` — resume from an existing checkpoint (comment out if not needed)
- Image/video output paths

Key hyperparameters: `num_timesteps=15_000_000`, `episode_length=1200`, `num_envs=1024`.

### SAC baseline training

```bash
cd ppo_train/airbot_training
python train_sac.py
```

Output directory (created automatically):

```
ppo_train/airbot_training/outputs/
├── checkpoints/airbot_sac_sac_<step>.pkl
├── models/airbot_sac.pkl
└── video/push_sac.mp4
```

### RSR training

1. Place all six data files in `ppo_train/airbot_training/outputs/`
2. Edit `ALGORITHM` (`'ppo'` or `'sac'`) and training hyperparameters in `test/rsr_policy_training.py`
3. Run:

```bash
python test/rsr_policy_training.py
```

Output directory:

```
ppo_train/airbot_training/outputs/rsr_training/
├── checkpoints/          # Orbax (PPO) or Brax SAC pkl
└── plots/                # training curves
```

---

## Go2 Experiments

### Available environments

| Environment | Description |
|-------------|-------------|
| `Go2JoystickFlatTerrain` | Flat terrain joystick velocity tracking (primary) |
| `Go2JoystickRoughTerrain` | Rough terrain joystick control |
| `Go2Getup` | Fall recovery |
| `Go2Handstand` | Handstand |
| `Go2Footstand` | Footstand |

Observations are a 48-dim `state` vector; actions are 12-dim (3 joints per leg).

### SAC baseline training

```bash
cd ppo_train/go2_training/learning
PYTHONPATH=.. python train_jax_sac.py --env_name=Go2JoystickFlatTerrain
```

Common flags:

```bash
PYTHONPATH=.. python train_jax_sac.py \
  --env_name=Go2JoystickFlatTerrain \
  --num_timesteps=5000000 \
  --num_envs=1024 \
  --batch_size=256 \
  --min_replay_size=100000 \
  --domain_randomization=true
```

Output directory:

```
ppo_train/go2_training/learning/logs/<experiment>/
├── checkpoints/
│   ├── sac_sac_<step>.pkl
│   └── final_sac.pkl
└── rollout.mp4
```

### PPO baseline training

```bash
cd ppo_train/go2_training/learning
PYTHONPATH=.. python train_jax_ppo.py --env_name=Go2JoystickFlatTerrain
```

### RSR-SAC training

1. Place all six data files in `ppo_train/go2_training/outputs/` (48-dim obs, 12-dim actions)
2. Edit training hyperparameters in `test/rsr_go2_policy_training.py`
3. Run:

```bash
python test/rsr_go2_policy_training.py
```

Output directory:

```
ppo_train/go2_training/outputs/rsr_training/
├── checkpoints/
└── plots/
```

> Go2 environments expose dictionary observations; `SelectObservationWrapper` extracts the `state` vector, and `wrap_for_brax_training` is applied automatically for MJX environments.

---

## RSR Core API

### Environment parameter alignment

```python
from RSR.rsr_pipeline import env_params_tuning

tuned_params, train_log = env_params_tuning(
    init_env=env,
    num_steps=1000,
    init_env_params=init_params,    # initial physics params (e.g. friction)
    env_params_min=params_min,
    env_params_max=params_max,
    obs=sampled_obs,                # real observations s_t
    actions=sampled_actions,        # real actions a_t
    next_obs_true=sampled_next_obs, # real next observations s_{t+1}
    log_path='log.txt',
)
```

### Policy parameter alignment (unified entry point)

```python
from RSR.rsr_pipeline import policy_params_training

make_inference_fn, params = policy_params_training(
    env=env,
    algorithm='sac',               # 'ppo' or 'sac'
    past_states=past_states,
    past_actions=past_actions,
    past_next_states_real=past_next_states_real,
    past_next_states_sim=past_next_states_sim,
    current_next_states_sim=current_next_states_sim,
    rsr_loss_scale=1.0,            # set to 0 for standard Brax RL
    num_timesteps=5_000_000,
    # SAC-specific
    min_replay_size=100_000,
    max_replay_size=1_000_000,
    checkpoint_logdir='path/to/checkpoints/rsr',
    # Go2-specific
    wrap_env_fn=wrapper.wrap_for_brax_training,
    eval_env=eval_env,
)
```

### Algorithm adapters

| Module | Role |
|--------|------|
| `RSR/train.py` | Brax PPO training loop with injected RSR actor loss |
| `RSR/losses.py` | RSR distribution penalty on the PPO actor |
| `RSR/sac_train.py` | Brax SAC training loop with injected RSR actor loss |
| `RSR/sac_losses.py` | RSR distribution penalty on the SAC actor |

---

## Data File Format

All data files are comma-separated plain text (`.txt`), one timestep per row and one feature per column.

```
# real_obs.txt example (Airbot: 23 dims, Go2: 48 dims)
0.12,0.34,0.56,...
0.13,0.35,0.57,...
```

```
# real_action.txt example (Airbot: 5 dims, Go2: 12 dims)
0.01,-0.02,0.03,...
0.02,-0.01,0.04,...
```

### Consistency requirements

RSR training scripts enforce strict validation:

- All **six files must exist**; missing files raise errors
- All observation files must have the same number of columns as `real_obs.txt`
- All action files must have the same number of columns as `real_action.txt`
- `real_obs.txt` must have at least `transition_count + 1` rows
- `real_action.txt` must have at least `transition_count` rows

### Data directories per platform

| Platform | Data directory |
|----------|----------------|
| Airbot | `ppo_train/airbot_training/outputs/` |
| Go2 | `ppo_train/go2_training/outputs/` |

---

## Real Robot Deployment

`real_robot_inference/` contains two standard ROS packages for different manipulation tasks:

| ROS package | Task | Entry script |
|-------------|------|--------------|
| `airbot_sim2real_sl` | Cube pushing (SL task) | `scripts/sim2real_sl_control_node.py` |
| `airbot_t` | T-shaped object | `scripts/sim2real_t_node.py` |

### Core components

| File | Role |
|------|------|
| `sim2real_*_control_node.py` | Main control node coordinating perception and execution |
| `marker_pose_publisher.py` | Intel RealSense depth camera and AprilTag pose publishing |
| `ppo_inference.py` | Load checkpoint and run policy inference |
| `real_robot_interface.py` | Airbot Play hardware interface and state acquisition |
| `config/config.yaml` | Camera intrinsics, workspace bounds, etc. |

### Deployment steps

1. Set the trained checkpoint path in `ckpt_dir` inside the control node
2. Choose the ROS package for your task and configure `config/config.yaml`
3. Build and launch:

```bash
# Cube pushing task
roslaunch airbot_sim2real_sl air_sl_arm_control.launch

# T-shaped object task
roslaunch airbot_t air_t_control.launch
```
