#!/home/wang/anaconda3/envs/airbot/bin/python
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Float64
import jax.numpy as jnp
import numpy as np

class RealRobotInterface(object):
    def __init__(self):
        self.is_maker_pos_updated = False
        self.pub_target_joint_q = rospy.Publisher(
            "/airbot_play/set_target_joint_q", JointState, queue_size=10
        )
        self.pub_target_pose = rospy.Publisher(
            "/airbot_play/set_target_pose", Pose, queue_size=10
        )
        self.pub_target_joint_v = rospy.Publisher(
            "/airbot_play/set_target_joint_v", JointState, queue_size=10
        )
        self.pub_gripper = rospy.Publisher(
            "/airbot_play/gripper/set_position", Float64, queue_size=10
        )
        self.current_marker_pos = Point()
        self.current_joint_state = JointState()
        self.current_end_pose = Pose()
        self.current_gripper_pos = 0.0
        
        rospy.Subscriber("/airbot_play/joint_states", JointState, self._joint_state_cb)
        rospy.Subscriber("/airbot_play/end_pose", Pose, self._end_pose_cb)
        rospy.Subscriber("/airbot_play/gripper/position", Float64, self._gripper_pos_cb)
        rospy.Subscriber("/qr_coordinates", Point, self._marker_pos_cb)

    def _marker_pos_cb(self, msg):
        self.current_marker_pos.x = msg.x
        self.current_marker_pos.y = msg.y
        self.current_marker_pos.z = msg.z
        self.is_maker_pos_updated = True

    def _joint_state_cb(self, msg):
        self.current_joint_state = msg

    def _end_pose_cb(self, msg):
        self.current_end_pose = msg

    def _gripper_pos_cb(self, msg):
        self.current_gripper_pos = msg.data

    def get_current_observation(self):
        obs_list = []
        obs_list.extend(self.current_joint_state.position)
        end_pos = [
            self.current_end_pose.position.x,
            self.current_end_pose.position.y,
            self.current_end_pose.position.z + 0.78 - 0.025,
        ]
        obs_list.extend(end_pos)

        marker_pos = [self.current_marker_pos.x,self.current_marker_pos.y, 0.82]  # e.g. [mx, my, mz]
        self.marker_pos = marker_pos
        target_pos = [0.455355,0.082943,0.820000]

        obs_list.extend(target_pos)
        obs_list.extend(marker_pos)

        cube_pos_xy = marker_pos[:2].copy()
        target_pos_xy = target_pos[:2].copy()
        direction = jnp.array(cube_pos_xy) - jnp.array(target_pos_xy)
        direction_norm = direction / jnp.linalg.norm(direction)
        distance = 0.04
        new_cube_pos = jnp.array(cube_pos_xy).copy() + direction_norm * distance
        obs_list.extend(new_cube_pos)

        target_minus_marker = jnp.array(target_pos) - jnp.array(marker_pos)
        obs_list.extend(target_minus_marker.tolist())

        marker_minus_end = jnp.array(marker_pos) - jnp.array(end_pos)
        obs_list.extend(marker_minus_end.tolist())

        observation = jnp.array(obs_list)
        rospy.loginfo("Observation: %s", observation.reshape(1, -1))

        with open('/real_obs.txt', 'a') as f_ctrl:
            np.savetxt(f_ctrl, observation.reshape(1, -1), fmt='%.6f', delimiter=',')
        return observation

    def send_joint_position_cmd(self, joint_positions):
        js = JointState()
        js.name = ["joint1","joint2","joint3","joint4","joint5","joint6"]  
        js.position = joint_positions
        self.pub_target_joint_q.publish(js)

    def send_gripper_cmd(self, gripper_val):
        msg = Float64()
        msg.data = gripper_val
        self.pub_gripper.publish(msg)
