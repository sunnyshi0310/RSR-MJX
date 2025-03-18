#!/home/wang/anaconda3/envs/airbot/bin/python
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Float64
import jax.numpy as jnp
import numpy as np

class RealRobotInterface(object):
    def __init__(self):
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

        self.current_marker_point0 = Point()
        self.current_marker_point1 = Point()
        self.current_marker_new_point = Point()
        self.current_joint_state = JointState()
        self.current_end_pose = Pose()
        self.current_gripper_pos = 0.0

        rospy.Subscriber("/airbot_play/joint_states", JointState, self._joint_state_cb)
        rospy.Subscriber("/airbot_play/end_pose", Pose, self._end_pose_cb)
        rospy.Subscriber("/airbot_play/gripper/position", Float64, self._gripper_pos_cb)
        rospy.Subscriber("/point0", Point, self._marker_point0)
        rospy.Subscriber("/point1", Point, self._marker_point1)
        rospy.Subscriber("/new_point", Point, self._marker_new_point)


    def _marker_point0(self, msg):
        self.current_marker_point0.x = msg.x
        self.current_marker_point0.y = msg.y
        self.current_marker_point0.z = msg.z

    def _marker_point1(self, msg):
        self.current_marker_point1.x = msg.x
        self.current_marker_point1.y = msg.y
        self.current_marker_point1.z = msg.z

    def _marker_new_point(self, msg):
        self.current_marker_new_point.x = msg.x
        self.current_marker_new_point.y = msg.y
        self.current_marker_new_point.z = msg.z

    def _joint_state_cb(self, msg):
        self.current_joint_state = msg

    def _end_pose_cb(self, msg):
        self.current_end_pose = msg

    def _gripper_pos_cb(self, msg):
        self.current_gripper_pos = msg.data

    def get_current_observation(self):
        target_base_x = 0.29
        target_base_y = 0.12
        target_base_z = 0.805
        target_vertical_x = 0.343033
        target_vertical_y = 0.066967
        target_vertical_z = 0.805
        obs_list = []

        obs_list.extend(self.current_joint_state.position)
        obs_list.extend([self.current_end_pose.position.z + 0.78 - 0.023])
        obs_list.extend([target_base_x - self.current_marker_point1.x, 
                        target_base_y - self.current_marker_point1.y, 
                        target_base_z - 0.805])
        obs_list.extend([target_vertical_x - self.current_marker_point0.x, 
                        target_vertical_y - self.current_marker_point0.y, 
                        target_vertical_z - 0.805])
        
        target_array = ([target_vertical_x - target_base_x, target_vertical_y - target_base_y, target_vertical_z - target_base_z])
        box_array = ([self.current_marker_point0.x - self.current_marker_point1.x,
                    self.current_marker_point0.y - self.current_marker_point1.y, 
                    0])

        xita = np.dot(box_array,target_array) / (np.linalg.norm(box_array) * np.linalg.norm(target_array))
        xita = np.abs(xita - 1.0)
        obs_list.extend([xita])

        obs_list.extend([self.current_marker_new_point.x - self.current_end_pose.position.x, self.current_marker_new_point.y - self.current_end_pose.position.y])
        
        observation = jnp.array(obs_list)
        rospy.loginfo("Observation: %s", observation.reshape(1, -1))
        rospy.loginfo("current_marker_point1: %s", self.current_marker_point1)

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
