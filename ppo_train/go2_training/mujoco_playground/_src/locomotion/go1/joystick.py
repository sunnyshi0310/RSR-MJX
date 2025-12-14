"""Joystick task for Go1."""
from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np
from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import go1_constants as consts
def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      Kp=60.0,
      Kd=3.0,
      action_repeat=1,
      action_scale=0.5,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              tracking_lin_vel=3.0,
              tracking_ang_vel=1.5,
              lin_vel_z=-0.5,
              ang_vel_xy=-0.05,
              orientation=-3.0,
              dof_pos_limits=-1.0,
              pose=0.0,
              termination=-1.0,
              stand_still=-1.0,
              torques=-0.0002,
              action_rate=-0.01,
              energy=-0.001,
              feet_clearance=-2.0,
              feet_height=-3.5,
              feet_slip=-0.1,
              feet_air_time=0.8,
              all_feet_air=-1.0,
              symmetric_gait=-0.8,
              lr_symmetry=-0.8,
              fb_symmetry=-0.8,
              feet_off_ground_when_still=-1.0,
          ),
          tracking_sigma=0.25,
          max_foot_height=0.12,
      ),
      pert_config=config_dict.create(
          enable=False,
          velocity_kick=[0.0, 3.0],
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      command_config=config_dict.create(
          a=[0.8, 0.0, 2.0],
          b=[0.8, 0.0, 0.8],
          change_interval=12.0,
      ),
      delay_config=config_dict.create(
          action=config_dict.create(
              enable=True,
              steps=3,
          ),
          imu=config_dict.create(
              enable=True,
              steps=3,
          ),
      ),
  )
class Joystick(go1_base.Go1Env):
  """Track a joystick command."""
  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()
  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor
    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )
    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)
    self._cmd_a = jp.array(self._config.command_config.a)
    self._cmd_b = jp.array(self._config.command_config.b)
  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )
    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])
    rng, key1, key2, key3 = jax.random.split(rng, 4)
    time_until_next_pert = jax.random.uniform(
        key1,
        minval=self._config.pert_config.kick_wait_times[0],
        maxval=self._config.pert_config.kick_wait_times[1],
    )
    steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(
        jp.int32
    )
    pert_duration_seconds = jax.random.uniform(
        key2,
        minval=self._config.pert_config.kick_durations[0],
        maxval=self._config.pert_config.kick_durations[1],
    )
    pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(
        jp.int32
    )
    pert_mag = jax.random.uniform(
        key3,
        minval=self._config.pert_config.velocity_kick[0],
        maxval=self._config.pert_config.velocity_kick[1],
    )
    rng, key1, key2 = jax.random.split(rng, 3)
    time_until_next_cmd = jax.random.exponential(key1) * self._config.command_config.change_interval
    steps_until_next_cmd = jp.round(time_until_next_cmd / self.dt).astype(
        jp.int32
    )
    cmd = jax.random.uniform(
        key2, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
    )
    action_delay_steps = self._config.delay_config.action.steps if self._config.delay_config.action.enable else 0
    imu_delay_steps = self._config.delay_config.imu.steps if self._config.delay_config.imu.enable else 0
    action_buffer = jp.zeros((action_delay_steps + 1, self.mjx_model.nu))
    gyro_buffer = jp.zeros((imu_delay_steps + 1, 3))
    linvel_buffer = jp.zeros((imu_delay_steps + 1, 3))
    gravity_buffer = jp.zeros((imu_delay_steps + 1, 3))
    info = {
        "rng": rng,
        "command": cmd,
        "steps_until_next_cmd": steps_until_next_cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(4),
        "feet_contact_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        "swing_peak": jp.zeros(4),
        "steps_until_next_pert": steps_until_next_pert,
        "pert_duration_seconds": pert_duration_seconds,
        "pert_duration": pert_duration_steps,
        "steps_since_last_pert": 0,
        "pert_steps": 0,
        "pert_dir": jp.zeros(3),
        "pert_mag": pert_mag,
        "action_buffer": action_buffer,
        "gyro_buffer": gyro_buffer,
        "linvel_buffer": linvel_buffer,
        "gravity_buffer": gravity_buffer,
    }
    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)
    if self._config.delay_config.action.enable:
      actual_action = state.info["action_buffer"][0]
      new_action_buffer = jp.vstack([
          state.info["action_buffer"][1:],
          action[None, :]
      ])
      state.info["action_buffer"] = new_action_buffer
    else:
      actual_action = action
    motor_targets = self._default_pose + actual_action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    if self._config.delay_config.imu.enable:
      current_gyro = self.get_gyro(data)
      current_linvel = self.get_local_linvel(data)
      current_gravity = self.get_gravity(data)
      state.info["gyro_buffer"] = jp.vstack([
          state.info["gyro_buffer"][1:],
          current_gyro[None, :]
      ])
      state.info["linvel_buffer"] = jp.vstack([
          state.info["linvel_buffer"][1:],
          current_linvel[None, :]
      ])
      state.info["gravity_buffer"] = jp.vstack([
          state.info["gravity_buffer"][1:],
          current_gravity[None, :]
      ])
    contact = jp.array([
        collision.geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)
    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)
    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["steps_until_next_cmd"] -= 1
    state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
    state.info["command"] = jp.where(
        state.info["steps_until_next_cmd"] <= 0,
        self.sample_command(key1, state.info["command"]),
        state.info["command"],
    )
    state.info["steps_until_next_cmd"] = jp.where(
        done | (state.info["steps_until_next_cmd"] <= 0),
        jp.round(jax.random.exponential(key2) * self._config.command_config.change_interval / self.dt).astype(jp.int32),
        state.info["steps_until_next_cmd"],
    )
    state.info["feet_air_time"] += self.dt
    state.info["feet_air_time"] *= ~contact
    state.info["feet_contact_time"] += self.dt
    state.info["feet_contact_time"] *= contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])
    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state
  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_upvector(data)[-1] < 0.0
    return fall_termination
  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
    if self._config.delay_config.imu.enable:
      gyro = info["gyro_buffer"][0]
      linvel = info["linvel_buffer"][0]
      gravity = info["gravity_buffer"][0]
    else:
      gyro = self.get_gyro(data)
      linvel = self.get_local_linvel(data)
      gravity = self.get_gravity(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )
    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )
    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )
    state = jp.hstack([
        noisy_linvel,
        noisy_gyro,
        noisy_gravity,
        noisy_joint_angles - self._default_pose,
        noisy_joint_vel,
        info["last_act"],
        info["command"],
    ])
    accelerometer = self.get_accelerometer(data)
    angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    current_gyro = self.get_gyro(data)
    current_linvel = self.get_local_linvel(data)
    current_gravity = self.get_gravity(data)
    privileged_state = jp.hstack([
        state,
        current_gyro,
        accelerometer,
        current_gravity,
        current_linvel,
        angvel,
        joint_angles - self._default_pose,
        joint_vel,
        data.actuator_force,
        info["last_contact"],
        feet_vel,
        info["feet_air_time"],
        data.xfrc_applied[self._torso_body_id, :3],
        info["steps_since_last_pert"] >= info["steps_until_next_pert"],
    ])
    return {
        "state": state,
        "privileged_state": privileged_state,
    }
  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics
    return {
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "orientation": self._cost_orientation(self.get_upvector(data)),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "termination": self._cost_termination(done),
        "pose": self._reward_pose(data.qpos[7:]),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "all_feet_air": self._cost_all_feet_air(contact, info["command"]),
        "symmetric_gait": self._cost_symmetric_gait(data.qpos[7:], info["command"]),
        "lr_symmetry": self._cost_lr_symmetry(
            info["feet_air_time"], info["feet_contact_time"], contact, info["command"]
        ),
        "fb_symmetry": self._cost_fb_symmetry(
            info["feet_air_time"], info["feet_contact_time"], contact, info["command"]
        ),
        "feet_off_ground_when_still": self._cost_feet_off_ground_when_still(
            contact, info["command"]
        ),
    }
  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)
  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)
  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    return jp.square(global_linvel[2])
  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    return jp.sum(jp.square(global_angvel[:2]))
  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torso_zaxis[:2]))
  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))
  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))
  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act
    return jp.sum(jp.square(act - last_act))
  def _reward_pose(self, qpos: jax.Array) -> jax.Array:
    weight = jp.array([1.0, 1.0, 0.1] * 4)
    return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))
  def _cost_stand_still(
      self,
      commands: jax.Array,
      qpos: jax.Array,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.01)
  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done
  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)
  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(info["command"])
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    return jp.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.01)
  def _cost_feet_clearance(self, data: mjx.Data) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)
  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
      info: dict[str, Any],
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(info["command"])
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact) * (cmd_norm > 0.01)
  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    rew_air_time = jp.sum((air_time - 0.1) * first_contact)
    rew_air_time *= cmd_norm > 0.01
    return rew_air_time
  def _cost_all_feet_air(
      self, contact: jax.Array, commands: jax.Array
  ) -> jax.Array:
    """Penalize when 3 or more feet are in the air (not in contact) when speed command is non-zero."""
    cmd_norm = jp.linalg.norm(commands)
    is_moving_cmd = cmd_norm > 0.01
    num_feet_in_air = jp.sum((~contact).astype(jp.int32))
    too_many_feet_in_air = num_feet_in_air >= 3
    penalty = too_many_feet_in_air.astype(jp.float32) * is_moving_cmd
    return penalty
  def _cost_symmetric_gait(
      self, qpos: jax.Array, commands: jax.Array
  ) -> jax.Array:
    """
    Penalize asymmetric joint angles between diagonal leg pairs.
    Leg order in qpos: FR (0-2), FL (3-5), RR (6-8), RL (9-11)
    Diagonal pairs:
      - FL (left front) and RR (right rear): indices [3-5] vs [6-8]
      - FR (right front) and RL (left rear): indices [0-2] vs [9-11]
    Each leg has 3 joints (hip, thigh, calf).
    """
    cmd_norm = jp.linalg.norm(commands)
    is_moving_cmd = cmd_norm > 0.01
    fr_joints = qpos[0:3]
    fl_joints = qpos[3:6]
    rr_joints = qpos[6:9]
    rl_joints = qpos[9:12]
    diff_pair1 = fl_joints - rr_joints
    error_pair1 = jp.sum(jp.square(diff_pair1))
    diff_pair2 = fr_joints - rl_joints
    error_pair2 = jp.sum(jp.square(diff_pair2))
    total_error = error_pair1 + error_pair2
    penalty = total_error * is_moving_cmd
    return penalty
  def _cost_lr_symmetry(
      self,
      feet_air_time: jax.Array,
      feet_contact_time: jax.Array,
      contact: jax.Array,
      commands: jax.Array,
  ) -> jax.Array:
    """
    Penalize left-right asymmetry in gait timing.
    Foot order: FR (0), FL (1), RR (2), RL (3)
    Left side: FL (1) + RL (3)
    Right side: FR (0) + RR (2)
    Compare average air time and contact time between left and right sides.
    """
    cmd_norm = jp.linalg.norm(commands)
    is_moving_cmd = cmd_norm > 0.01
    left_air_time = (feet_air_time[1] + feet_air_time[3]) / 2.0
    left_contact_time = (feet_contact_time[1] + feet_contact_time[3]) / 2.0
    right_air_time = (feet_air_time[0] + feet_air_time[2]) / 2.0
    right_contact_time = (feet_contact_time[0] + feet_contact_time[2]) / 2.0
    air_time_diff = jp.square(left_air_time - right_air_time)
    contact_time_diff = jp.square(left_contact_time - right_contact_time)
    total_asymmetry = air_time_diff + contact_time_diff
    penalty = total_asymmetry * is_moving_cmd
    return penalty
  def _cost_fb_symmetry(
      self,
      feet_air_time: jax.Array,
      feet_contact_time: jax.Array,
      contact: jax.Array,
      commands: jax.Array,
  ) -> jax.Array:
    """
    Penalize front-back asymmetry in gait timing.
    Foot order: FR (0), FL (1), RR (2), RL (3)
    Front legs: FR (0) + FL (1)
    Rear legs: RR (2) + RL (3)
    Compare average air time and contact time between front and rear legs.
    """
    cmd_norm = jp.linalg.norm(commands)
    is_moving_cmd = cmd_norm > 0.01
    front_air_time = (feet_air_time[0] + feet_air_time[1]) / 2.0
    front_contact_time = (feet_contact_time[0] + feet_contact_time[1]) / 2.0
    rear_air_time = (feet_air_time[2] + feet_air_time[3]) / 2.0
    rear_contact_time = (feet_contact_time[2] + feet_contact_time[3]) / 2.0
    air_time_diff = jp.square(front_air_time - rear_air_time)
    contact_time_diff = jp.square(front_contact_time - rear_contact_time)
    total_asymmetry = air_time_diff + contact_time_diff
    penalty = total_asymmetry * is_moving_cmd
    return penalty
  def _cost_feet_off_ground_when_still(
      self, contact: jax.Array, commands: jax.Array
  ) -> jax.Array:
    """
    Penalize when any foot is off the ground during still command.
    When the robot receives a still command (cmd_norm < 0.01), all four feet
    should be in contact with the ground. Penalize any foot that is not in contact.
    Foot order: FR (0), FL (1), RR (2), RL (3)
    """
    cmd_norm = jp.linalg.norm(commands)
    is_still_cmd = cmd_norm < 0.01
    num_feet_off_ground = jp.sum((~contact).astype(jp.int32))
    penalty = num_feet_off_ground.astype(jp.float32) * is_still_cmd
    return penalty
  def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
    def gen_dir(rng: jax.Array) -> jax.Array:
      angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
      return jp.array([jp.cos(angle), jp.sin(angle), 0.0])
    def apply_pert(state: mjx_env.State) -> mjx_env.State:
      t = state.info["pert_steps"] * self.dt
      u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"])
      force = (
          u_t
          * self._torso_mass
          * state.info["pert_mag"]
          / state.info["pert_duration_seconds"]
      )
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      xfrc_applied = xfrc_applied.at[self._torso_body_id, :3].set(
          force * state.info["pert_dir"]
      )
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state = state.replace(data=data)
      state.info["steps_since_last_pert"] = jp.where(
          state.info["pert_steps"] >= state.info["pert_duration"],
          0,
          state.info["steps_since_last_pert"],
      )
      state.info["pert_steps"] += 1
      return state
    def wait(state: mjx_env.State) -> mjx_env.State:
      state.info["rng"], rng = jax.random.split(state.info["rng"])
      state.info["steps_since_last_pert"] += 1
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state.info["pert_steps"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          0,
          state.info["pert_steps"],
      )
      state.info["pert_dir"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          gen_dir(rng),
          state.info["pert_dir"],
      )
      return state.replace(data=data)
    return jax.lax.cond(
        state.info["steps_since_last_pert"]
        >= state.info["steps_until_next_pert"],
        apply_pert,
        wait,
        state,
    )
  def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
    rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
    y_k = jax.random.uniform(
        y_rng, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
    )
    z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(3,))
    w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
    x_kp1 = x_k - w_k * (x_k - y_k * z_k)
    return x_kp1
