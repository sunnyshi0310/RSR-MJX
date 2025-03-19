import mujoco
import jax
from jax import numpy as jp
from brax.base import State as PipelineState
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from typing import Any
class AirbotPlayBase(PipelineEnv):
    def __init__(
        self, 
        push_reward_weight = 6,
        endpoint_to_target_reward_weight = 0.01,
        ctrl_cost_weight = 0.003,
        box_still_cost_weight = 0.01,
        joint_vel_cost_weight = 0.1,
        siet_to_box_reward_weight = 3.0,
        healthy_reward = 1.0,
        max_vel = 0.4,
        endpoint_min_z_pos = 0.778,
        noise_scale = 1e-2,
        cube_degree_cost_weight = 0.01,
        coll_cost_weight = 0.05,
        cube_vel_cost_weight = 0.001,
        joint_num = 7,
        decimation = 4,
        cube_min_x = 0.29,
        cube_max_x = 0.34,
        cube_min_y = -0.04,
        cube_max_y = 0.01,
        target_min_x = 0.4364427,
        target_max_x = 0.4864427,
        target_min_y = 0.07352592,
        target_max_y = 0.12352592,
        **kwargs,
        ):

        mj_model_path = "cube.xml"
        mj_model = mujoco.MjModel.from_xml_path(mj_model_path)
        kwargs['n_frames'] = kwargs.get(
        'n_frames', decimation)
        kwargs['backend'] = 'mjx'
        sys = mjcf.load_model(mj_model)
        self.sys = sys
        super().__init__(sys, **kwargs)

        
        self.nj = joint_num
        self.push_reward_weight = push_reward_weight
        self.endpoint_to_target_reward_weight = endpoint_to_target_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.box_still_cost_weight = box_still_cost_weight
        self.joint_vel_cost_weight = joint_vel_cost_weight
        self.siet_to_box_reward_weight = siet_to_box_reward_weight
        self.healthy_reward = healthy_reward
        self.degree_cost_weight = cube_degree_cost_weight
        self.coll_cost_weight = coll_cost_weight
        self.cube_vel_cost_weight = cube_vel_cost_weight
        self.max_vel = max_vel
        self.endpoint_min_z_pos = endpoint_min_z_pos
        self.action_scale = jp.array([0.02,0.02,0.02,0.0,0.0])#reset_action_scale#
        self._reset_noise_scale = noise_scale
        self.decimation = decimation
        self._lowers = sys.mj_model.actuator_ctrlrange[:, 0]
        self._uppers = sys.mj_model.actuator_ctrlrange[:, 1]

        self.cube_id = sys.mj_model.body('cube_for_push').id
        self.target_pos_id = sys.mj_model.body('target_pos').id
        self._box_qposadr = sys.mj_model.jnt_qposadr[sys.mj_model.body('cube_for_push').jntadr[0]]
        self._site_qposadr = sys.mj_model.jnt_qposadr[sys.mj_model.body('target_pos').jntadr[0]]  
        self.fixed_gripper_geom_id = sys.mj_model.geom('fixed_gripper').id
        self.cube_geom_id = sys.mj_model.geom('geom_for_push').id
        self.table_id = sys.mj_model.geom('table-b').id
        self.left_finger = sys.mj_model.geom('right_finger').id
        self.right_finger = sys.mj_model.geom('left_finger').id

        # self.cube_id = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "cube_for_push")
        arm_joints_name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_id = jp.array([
            self.sys.mj_model.jnt_qposadr[self.sys.mj_model.joint(j).id] for j in arm_joints_name])
        finger_joints_name = ['endleft']
        self.finger_id = jp.array([
            self.sys.mj_model.jnt_qposadr[self.sys.mj_model.joint(j).id] for j in finger_joints_name])

        # self.site_id = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "endpoint")
        self.site_id = sys.mj_model.site('endpoint').id

        self.cube_min_x = cube_min_x
        self.cube_max_x = cube_max_x
        self.cube_min_y = cube_min_y
        self.cube_max_y = cube_max_y
        self.target_min_x = target_min_x
        self.target_max_x = target_max_x
        self.target_min_y = target_min_y
        self.target_max_y = target_max_y
    def reset(self, rng: jp.ndarray) -> State:
        
        rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qpos = qpos.at[self.joint_id].add(jp.array([0, -0.5422302, 0.45173569, 1.5718, -1.4794435, 1.1731174]))
        qpos = qpos.at[self.finger_id].set(-0.033)
        qvel = jax.random.uniform(
            rng2, (self.sys.nv,), minval=low, maxval=hi
        )
        joint_ctrl = jp.array([0, -0.73151061, 0.455936904, -1.4794435,1.1731174]) + jax.random.uniform(
            rng3, (self.sys.nu,), minval=low, maxval=hi
        )
        target_pos = jax.random.uniform(
        rng4, (3,),
        minval=jp.array([self.target_min_x, self.target_min_y, 0.82]),
        maxval=jp.array([self.target_max_x, self.target_max_y, 0.82]))

        cube_pos = jax.random.uniform(
        rng, (3,),
        minval=jp.array([self.cube_min_x, self.cube_min_y, 0.82]),
        maxval=jp.array([self.cube_max_x, self.cube_max_y, 0.82]))
        
        qpos = jp.array(qpos).at[
            self._box_qposadr : self._box_qposadr + 3].set(cube_pos)
        qpos = jp.array(qpos).at[
            self._site_qposadr : self._site_qposadr + 3].set(target_pos)
        
        data = self.pipeline_init(qpos, qvel)
        data = data.replace(ctrl=joint_ctrl)

        new_cube_pos = jp.array([0.37342, -0.07989])
        reward, done, zero= jp.zeros(3)
        metrics = {
            'push_reward' : zero,
            'ctrl_cost' : zero,
            'siet_to_box_reward' : zero, 
        }
        info = {'target_pos' : data.xpos[self.target_pos_id],
                'new_cube_pos':new_cube_pos,
                'site_pos' : data.site_xpos[self.site_id],
                'cube_pos' : data.xpos[self.cube_id],
                'reached_box' : 0.0,
                }
        obs = self._get_obs(data, info)        

        return State(data, obs, reward, done, metrics,info)

    def step(self, state: State, action: jp.ndarray) -> State:
        data0 = state.pipeline_state

        delta  = jp.multiply(self.action_scale, action)
        action = state.pipeline_state.ctrl + delta
        action =  action.at[3].set(-(1.57 + data0.qpos[self.joint_id[1]] + data0.qpos[self.joint_id][2]))

        cube_pos = data0.xpos[self.cube_id]
        cube_pos_x = cube_pos[0]
        cube_pos_y = cube_pos[1]
        target_pos = state.info['target_pos'][:2]
        delta_x = target_pos[0] - cube_pos_x
        delta_y = target_pos[1] - cube_pos_y
        angle_to_box = jp.arctan2(delta_y, delta_x + 0.00001) 
        action =  action.at[4].set(-angle_to_box + action[0] + 1.5708)

        action = jp.clip(action, self._lowers, self._uppers)
        data1 = self.pipeline_step(data0,action)

        box_target_dis = jp.linalg.norm(state.info['target_pos'] - data1.xpos[self.cube_id])
        box_target_dis = jp.where(box_target_dis < 0.005, 0.0, box_target_dis)
        push_reward = 1 / (1 + 3 * box_target_dis)
        push_reward = push_reward * self.push_reward_weight


        site_pos = data1.site_xpos[self.site_id]
        site_pos_xy = site_pos[0:2]
        site_pos_z = site_pos[2]
        cube_pos = data1.xpos[self.cube_id]
        cube_pos_xy = state.info['new_cube_pos']
        cube_pos_x = cube_pos[0]
        cube_pos_y = cube_pos[1]
        site_z_reward = jp.where(site_pos_z < 0.82, 1.0, 0.0)   

        target_pos = state.info['target_pos'][:2]
        delta_x = target_pos[0] - cube_pos_x
        delta_y = target_pos[1] - cube_pos_y
        angle_to_box = jp.arctan2(delta_y, delta_x + 0.00001)
        distance = jp.sqrt(delta_x**2 + delta_y**2) + 0.04
        y_ = distance * jp.sin(angle_to_box)
        x_ = distance * jp.cos(angle_to_box)
        state.info['new_cube_pos'] = state.info['new_cube_pos'].at[0].set(delta_x - x_ + cube_pos_x)
        state.info['new_cube_pos'] = state.info['new_cube_pos'].at[1].set(delta_y - y_ + cube_pos_y)


        site2cube_dis_xy = jp.linalg.norm(site_pos_xy - cube_pos_xy)
        site2cube_dis_xy = jp.where(site2cube_dis_xy < 0.042, 0, site2cube_dis_xy - 0.042)
        siet2cube_reward_xy = 1 - jp.tanh(5 * site2cube_dis_xy)
        siet2cube_reward = siet2cube_reward_xy * self.siet_to_box_reward_weight
        siet2cube_reward = jp.where(box_target_dis < 0.005, self.siet_to_box_reward_weight, siet2cube_reward)

        health_reward = self.healthy_reward * jp.abs(jp.where(jp.any(site_pos[2] < self.endpoint_min_z_pos), 1.0, 0.0) - 1.0)

        reward = push_reward + siet2cube_reward + health_reward + site_z_reward
        
        done = jp.where(jp.any(data1.xpos[self.cube_id][2] < 0.6), 1.0, 0.0)
        reward = jp.clip(reward, -1e2, 1e2)
        obs = self._get_obs(data1, state.info)
        state.metrics.update(
            push_reward=push_reward,
            siet_to_box_reward=siet2cube_reward,
        )
        state.info.update(
            site_pos=site_pos,
            cube_pos=data1.xpos[self.cube_id],
        )
        return state.replace(
            pipeline_state=data1, obs=obs, reward=reward, done=done
        )
    
    def _get_obs(
        self, data: PipelineState, info: dict[str, Any]
    ) -> jax.Array:

        return jp.concatenate([
                data.qpos[self.joint_id],
                # data.qvel[self.joint_id],
                # data.qfrc_actuator[self.joint_id],
                data.site_xpos[self.site_id],
                info['target_pos'],
                data.xpos[self.cube_id],
                info['new_cube_pos'],
                info['target_pos'] - data.xpos[self.cube_id],
                data.xpos[self.cube_id] - data.site_xpos[self.site_id],
                ],)