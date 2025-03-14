import mujoco
import jax
from jax import numpy as jp
from brax.base import State as PipelineState
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from typing import Any


jax.config.update('jax_default_matmul_precision',jax.lax.Precision.HIGH)
class AirbotPlayBase(PipelineEnv):
    def __init__(
        self, 
        push_reward_weight = 9.0,
        siet_to_box_reward_weight = 3.0,
        endpoint_min_z_pos = 0.78,
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

        mj_model_path = "T_shape.xml"
        mj_model = mujoco.MjModel.from_xml_path(mj_model_path)
        kwargs['n_frames'] = kwargs.get(
        'n_frames', decimation)
        kwargs['backend'] = 'mjx'
        sys = mjcf.load_model(mj_model)
        self.sys = sys
        super().__init__(sys, **kwargs)

        self.nj = joint_num
        self.push_reward_weight = push_reward_weight
        self.siet_to_box_reward_weight = siet_to_box_reward_weight
        self.degree_cost_weight = cube_degree_cost_weight
        self.coll_cost_weight = coll_cost_weight
        self.cube_vel_cost_weight = cube_vel_cost_weight
        self.endpoint_min_z_pos = endpoint_min_z_pos
        self.action_scale = jp.array([0.02,0.02,0.02,0.0,0.0])#reset_action_scale#
        self._reset_noise_scale = noise_scale
        self.decimation = decimation
        self._lowers = sys.mj_model.actuator_ctrlrange[:, 0]
        self._uppers = sys.mj_model.actuator_ctrlrange[:, 1]

        self.T_id = sys.mj_model.body('T_block').id
        self.target_body_id = sys.mj_model.body('T_target').id
        self._box_qposadr = sys.mj_model.jnt_qposadr[sys.mj_model.body('T_block').jntadr[0]]
        self._target_qposadr = sys.mj_model.jnt_qposadr[sys.mj_model.body('T_target').jntadr[0]]  
        self.fixed_gripper_geom_id = sys.mj_model.geom('fixed_gripper').id
        self.T_base_geom_id = sys.mj_model.geom('base_block').id
        self.T_vertical_geom_id = sys.mj_model.geom('vertical_block').id
        self.T_target_base_geom_id = sys.mj_model.geom('base_target').id
        self.T_target_vertical_geom_id = sys.mj_model.geom('vertical_target').id
        self.table_id = sys.mj_model.geom('table-b').id
        self.left_finger = sys.mj_model.geom('right_finger').id
        self.right_finger = sys.mj_model.geom('left_finger').id

        # self.T_id = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "cube_for_push")
        arm_joints_name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_id = jp.array([
            self.sys.mj_model.jnt_qposadr[self.sys.mj_model.joint(j).id] for j in arm_joints_name])
        finger_joints_name = ['endleft']
        self.finger_id = jp.array([
            self.sys.mj_model.jnt_qposadr[self.sys.mj_model.joint(j).id] for j in finger_joints_name])

        # self.site_id = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "endpoint")
        self.site_id = sys.mj_model.site('endpoint').id
        self.T_tail_id = sys.mj_model.site('T_tail').id
        self.T_target_tail_id = sys.mj_model.site('T_target_tail').id
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
        qpos = qpos.at[self.joint_id].add(jp.array([0, -0.57303354, 0.381795, 1.5718, -1.3787, 1.1731174]))
        qvel = jax.random.uniform(
            rng2, (self.sys.nv,), minval=low, maxval=hi
        )
        joint_ctrl = jp.array([0, -0.57303354, 0.381795, -1.3787,1.1731174]) + jax.random.uniform(
            rng3, (self.sys.nu,), minval=low, maxval=hi
        )

        # target_pos = jax.random.uniform(
        # rng4, (3,),
        # minval=jp.array([self.target_min_x, self.target_min_y, 0.82]),
        # maxval=jp.array([self.target_max_x, self.target_max_y, 0.82]))

        # T_pos = jax.random.uniform(
        # rng, (3,),
        # minval=jp.array([self.cube_min_x, self.cube_min_y, 0.82]),
        # maxval=jp.array([self.cube_max_x, self.cube_max_y, 0.82]))
        
        # qpos = jp.array(qpos).at[
        #     self._box_qposadr : self._box_qposadr + 3].set(T_pos)
        # qpos = jp.array(qpos).at[
        #     self._target_qposadr : self._site_qposadr + 3].set(target_pos)
        
        data = self.pipeline_init(qpos, qvel)
        data = data.replace(ctrl=joint_ctrl)

        reward, done, zero= jp.zeros(3)
        metrics = {
            'push_reward' : zero,
            'siet2cube_reward' : zero,
            'health_reward' : zero, 
            'task_complete_reward' : zero,
            'site_z_reward' : zero,
        }        

        info = {'target_base_pos' : data.geom_xpos[self.T_target_base_geom_id],
                'target_vertical_pos' : data.geom_xpos[self.T_target_vertical_geom_id],
                'site_pos' : data.site_xpos[self.site_id],
                'theta' : 0.2876,
                }
        obs = self._get_obs(data, info)        

        return State(data, obs, reward, done, metrics,info)

    def step(self, state: State, action: jp.ndarray) -> State:
        data0 = state.pipeline_state

        delta  = jp.multiply(self.action_scale, action)
        action = state.pipeline_state.ctrl + delta

        action = jp.clip(action, self._lowers, self._uppers)
        data1 = self.pipeline_step(data0,action)
        done = 0.0

        box_target_dis_base = jp.linalg.norm(state.info['target_base_pos'] - data1.geom_xpos[self.T_base_geom_id])
        box_target_dis_base = jp.where(box_target_dis_base < 0.005, 0.0, box_target_dis_base)
        push_reward_base = 1.0 / (1 + 10.0 * box_target_dis_base)
        box_target_dis_vertical = jp.linalg.norm(state.info['target_vertical_pos'] - data1.geom_xpos[self.T_vertical_geom_id])
        box_target_dis_vertical = jp.where(box_target_dis_vertical < 0.005, 0.0, box_target_dis_vertical)
        push_reward_vertical = 1.0 / (1 + 10.0 * box_target_dis_vertical)

        box_array = data1.geom_xpos[self.T_vertical_geom_id] - data1.geom_xpos[self.T_base_geom_id]
        target_array = state.info['target_vertical_pos'] - state.info['target_base_pos']
        theta = jp.dot(box_array, target_array) / (jp.linalg.norm(box_array) * jp.linalg.norm(target_array))
        theta = jp.abs(theta - 1.0)
        theta = jp.arccos(jp.clip(jp.dot(box_array, target_array) / (jp.linalg.norm(box_array) * jp.linalg.norm(target_array)), -1.0, 1.0))
        state.info['theta'] = theta
        push_w_reward = 1.0 / (1 + 6.0 * theta)
        push_reward = (0.333 * (push_reward_base + push_reward_vertical) + 0.666 * push_w_reward) * self.push_reward_weight

        site_pos = data1.site_xpos[self.site_id]

        site2cube_dis = jp.linalg.norm(site_pos - data1.site_xpos[self.T_target_tail_id])
        site2cube_dis = jp.where(site2cube_dis < 0.02, 0, site2cube_dis - 0.02)
        siet2cube_reward_xy = 1 - jp.tanh(5 * site2cube_dis)
        siet2cube_reward = siet2cube_reward_xy * self.siet_to_box_reward_weight

        done = jp.where(jp.any(site_pos[2] < self.endpoint_min_z_pos), 1.0, done)
        done = jp.where(jp.any(site_pos[0] > 1.0), 1.0, done)
        done = jp.where(jp.any(site_pos[0] < -0.6), 1.0, done)
        done = jp.where(jp.any(site_pos[1] > 0.3), 1.0, done)
        done = jp.where(jp.any(site_pos[1] < -0.3), 1.0, done)
        done = jp.where(jp.any(data1.xpos[self.T_id][2] < 0.6), 1.0, done)        

        reward = push_reward + siet2cube_reward


        reward = jp.clip(reward, -1e2, 1e2)
        obs = self._get_obs(data1, state.info)
        state.info.update(
            site_pos=site_pos,
            T_pos=data1.xpos[self.T_id],
        )
        return state.replace(
            pipeline_state=data1, obs=obs, reward=reward, done=done
        )
    
    def _get_obs(
        self, data: PipelineState, info: dict[str, Any]
    ) -> jax.Array:

        return jp.concatenate([
                data.qpos[self.joint_id],
                info['target_base_pos'] - data.geom_xpos[self.T_base_geom_id],
                info['target_vertical_pos'] - data.geom_xpos[self.T_vertical_geom_id],
                jp.array([info['theta']]),
                data.site_xpos[self.T_target_tail_id] - data.site_xpos[self.site_id],
                ],)