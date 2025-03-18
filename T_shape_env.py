import mujoco
import jax
from jax import numpy as jp
from brax.base import State as PipelineState
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from typing import Any

jax.config.update('jax_default_matmul_precision',jax.lax.Precision.HIGH)#set this in training python file
class AirbotPlayBase(PipelineEnv):
    def __init__(
        self, 
        push_reward_weight = 10.0,
        endpoint_to_target_reward_weight = 0.01,
        ctrl_cost_weight = 0.003,
        box_still_cost_weight = 0.01,
        joint_vel_cost_weight = 0.1,
        siet_to_box_reward_weight = 3.0,
        healthy_reward = 1.0,
        max_vel = 0.4,
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
        
        data = self.pipeline_init(qpos, qvel)
        data = data.replace(ctrl=joint_ctrl)

        new_T_pos = jp.array([0.24739072,  -0.00496255])
        reward, done, zero= jp.zeros(3)
        metrics = {
            'push_reward' : zero,
            'siet2cube_reward' : zero,
            'health_reward' : zero, 
            'task_complete_reward' : zero,
            'site_z_reward' : zero,
        }        
        target_quat = data.xquat[self.target_body_id]
        target_quat = target_quat[0] * 10
        info = {'target_base_pos' : data.geom_xpos[self.T_target_base_geom_id],
                'target_vertical_pos' : data.geom_xpos[self.T_target_vertical_geom_id],
                'target_w':target_quat,
                'new_T_pos':new_T_pos,
                'site_pos' : data.site_xpos[self.site_id],
                'T_pos' : data.xpos[self.T_id],
                'xita' : 0.2876,
                }
        obs = self._get_obs(data, info)        

        return State(data, obs, reward, done, metrics,info)

    def step(self, state: State, action: jp.ndarray) -> State:
        data0 = state.pipeline_state

        delta  = jp.multiply(self.action_scale, action)
        action = state.pipeline_state.ctrl + delta
        action =  action.at[3].set(-(1.57 + data0.qpos[self.joint_id[1]] + data0.qpos[self.joint_id][2]))

        T_pos = data0.site_xpos[self.site_id]
        T_pos_x = T_pos[0]
        T_pos_y = T_pos[1]
        target_pos = data0.site_xpos[self.T_tail_id][:2]
        delta_x = target_pos[0] - T_pos_x
        delta_y = target_pos[1] - T_pos_y
        angle_to_box = jp.arctan2(delta_y, delta_x + 0.00001)
        action =  action.at[4].set(-angle_to_box + action[0] + 1.5708)

        action = jp.clip(action, self._lowers, self._uppers)
        data1 = self.pipeline_step(data0,action)

        box_target_dis_base = jp.linalg.norm(state.info['target_base_pos'] - data1.geom_xpos[self.T_base_geom_id])
        box_target_dis_base = jp.where(box_target_dis_base < 0.005, 0.0, box_target_dis_base)
        push_reward_base = 1.0 / (1 + 10.0 * box_target_dis_base)
        box_target_dis_vertical = jp.linalg.norm(state.info['target_vertical_pos'] - data1.geom_xpos[self.T_vertical_geom_id])
        box_target_dis_vertical = jp.where(box_target_dis_vertical < 0.005, 0.0, box_target_dis_vertical)
        push_reward_vertical = 1.0 / (1 + 10.0 * box_target_dis_vertical)

        box_array = data1.geom_xpos[self.T_vertical_geom_id] - data1.geom_xpos[self.T_base_geom_id]
        target_array = state.info['target_vertical_pos'] - state.info['target_base_pos']
        xita = jp.dot(box_array, target_array) / (jp.linalg.norm(box_array) * jp.linalg.norm(target_array))
        xita = jp.abs(xita - 1.0)
        xita = jp.arccos(jp.clip(jp.dot(box_array, target_array) / (jp.linalg.norm(box_array) * jp.linalg.norm(target_array)), -1, 1))
        state.info['xita'] = xita
        push_w_reward = 1.0 / (1 + 6.0 * xita)
        push_reward = (0.1515 * push_reward_base + 0.1515 * push_reward_vertical + 0.66 * push_w_reward) * self.push_reward_weight 

        site_pos = data1.site_xpos[self.site_id]
        site_pos_xy = site_pos[0:2]
        site_pos_z = site_pos[2]
        T_tail_pos = data1.site_xpos[self.T_tail_id]
        cube_pos_xy = state.info['new_T_pos']
        T_pos_x = T_tail_pos[0]
        T_pos_y = T_tail_pos[1]

        site_z_reward = jp.where(site_pos_z < 0.83, 1.0, 0.0)   
        z_dis = jp.abs(site_pos_z - 0.805)
        z_reward = 4.0 / (1 + 3 * z_dis)
        site_z_reward = site_z_reward + z_reward
        #更新new_cube_pos
        target_pos = data1.site_xpos[self.T_target_tail_id][:2]
        delta_x = target_pos[0] - T_pos_x
        delta_y = target_pos[1] - T_pos_y
        angle_to_box = jp.arctan2(delta_y, delta_x + 0.00001)
        distance = jp.sqrt(delta_x**2 + delta_y**2) + 0.025
        y_ = distance * jp.sin(angle_to_box)
        x_ = distance * jp.cos(angle_to_box)
        state.info['new_T_pos'] = state.info['new_T_pos'].at[0].set(delta_x - x_ + T_pos_x)
        state.info['new_T_pos'] = state.info['new_T_pos'].at[1].set(delta_y - y_ + T_pos_y)

        site2cube_dis_xy = jp.linalg.norm(site_pos_xy - cube_pos_xy)
        site2cube_dis_xy = jp.where(site2cube_dis_xy < 0.02, 0, site2cube_dis_xy - 0.02)
        siet2cube_reward_xy = 1 - jp.tanh(5 * site2cube_dis_xy)
        siet2cube_reward = siet2cube_reward_xy * self.siet_to_box_reward_weight

        health_reward = self.healthy_reward * jp.abs(jp.where(jp.any(site_pos[2] < self.endpoint_min_z_pos), 1.0, 0.0) - 1.0)

        reward = push_reward + siet2cube_reward + health_reward + site_z_reward

        done = jp.where(jp.any(data1.xpos[self.T_id][2] < 0.6), 1.0, 0.0)        
        reward = jp.clip(reward, -1e2, 1e2)
        obs = self._get_obs(data1, state.info)
        state.metrics.update(
            push_reward=push_reward,
            siet2cube_reward=siet2cube_reward,  
            health_reward=health_reward,
            site_z_reward=site_z_reward,
        )
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
                jp.array([data.site_xpos[self.site_id][2]]),
                info['target_base_pos'] - data.geom_xpos[self.T_base_geom_id],
                info['target_vertical_pos'] - data.geom_xpos[self.T_vertical_geom_id],
                jp.array([info['xita']]),
                info['new_T_pos'] - data.site_xpos[self.site_id][:2],
                ],)