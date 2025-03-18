#!/home/wang/anaconda3/envs/airbot/bin/python

import rospy
import numpy as np
import jax.numpy as jnp
import cv2
import os
import time
from datetime import datetime
import sys

sys.path.insert(0, '/src/airbot_sim2real_sl')

from ppo_inference import PolicyInference
from real_robot_interface import RealRobotInterface
from brax import envs
from dlabsim.envs.airbot import AirbotPlayBase
from std_msgs.msg import Header

ckpt_dir = 'model_path'


def main():
    rospy.init_node("air_arm_control_node")
    step_pub = rospy.Publisher('/airbot_play/step_complete', Header, queue_size=10)

    robot = RealRobotInterface()
    rospy.loginfo("Initialized RealRobotInterface.")

    envs.register_environment('airbot', AirbotPlayBase)
    env_name = 'airbot'
    env = envs.get_environment(env_name)
    inference = PolicyInference(ckpt_dir=ckpt_dir, env_fn=env)
    rospy.loginfo("Policy loaded from: %s", ckpt_dir)

    joint_limits = [
        (-3.14,  2.09),   # joint1
        (-2.96,  0.17),   # joint2
        (-0.087, 3.14),   # joint3
        (-2.96,  2.96),   # joint4
        (-1.74,  1.74),   # joint5
        (-3.14,  3.14),   # joint6
    ]

    JOINT_TOLERANCE = 0.01  # radian
    JOINT_TIMEOUT = 5.0     # second

    max_steps = 10000
    step_count = 0
    last_action5 = 0

    rate = rospy.Rate(10)
    while not rospy.is_shutdown() and step_count < max_steps:

        while not robot.is_maker_pos_updated:
            time.sleep(0.01)
        robot.is_maker_pos_updated = False
        obs = robot.get_current_observation()
        inite_box_target_dis = jnp.linalg.norm(jnp.array([0.455355, 0.082943, 0.82]) - jnp.array(robot.marker_pos))

        ctrl = inference.get_action(obs, deterministic=True)
        ctrl = np.insert(ctrl, 3, 0)

        current_joint_positions = np.array(robot.current_joint_state.position)
        new_joint_positions = current_joint_positions + ctrl
        new_joint_positions = jnp.array(new_joint_positions)
        new_joint_positions = new_joint_positions.at[3].set(1.57)

        cube_pos = robot.marker_pos
        cube_pos_x = cube_pos[0]
        cube_pos_y = cube_pos[1]
        target_pos = [0.455355, 0.082943]
        delta_x = target_pos[0] - cube_pos_x
        delta_y = target_pos[1] - cube_pos_y
        angle_to_box = jnp.arctan2(delta_y, delta_x + 0.00001)
        new_joint_positions = new_joint_positions.at[5].set(
            jnp.where(inite_box_target_dis < 0.01, last_action5, -angle_to_box + new_joint_positions[0] + 1.5708)
        )
        last_action5 = new_joint_positions[5]
        rospy.loginfo("Angle_to_box: %s", angle_to_box)

        joint2 = new_joint_positions[1]
        joint3 = new_joint_positions[2]
        new_joint_positions = new_joint_positions.at[4].set(-(1.57 + joint2 + joint3))
        new_joint_positions = jnp.clip(
            new_joint_positions,
            jnp.array([-3.14, -2.96, -0.087, -2.96, -1.74, -3.14]),
            jnp.array([2.09, 0.17, 3.14, 2.96, 1.74, 3.14])
        )

        dis_to_target = np.linalg.norm(np.array(target_pos) - np.array(cube_pos[:2]))
        if dis_to_target < 0.008:
            rospy.loginfo("Cube reached target position.")
            step_count += 1
            rate.sleep()
            continue

        robot.send_joint_position_cmd(new_joint_positions)
        start_time = rospy.Time.now().to_sec()
        reached = False
        while (rospy.Time.now().to_sec() - start_time) < JOINT_TIMEOUT and not rospy.is_shutdown():
            current_pos = np.array(robot.current_joint_state.position)
            errors = np.abs(current_pos - new_joint_positions)

            if np.all(errors < JOINT_TOLERANCE):
                reached = True
                break

            max_error = np.max(errors)
            rospy.loginfo_throttle(1, f"Max joint error: {max_error:.4f} rad")
            rate.sleep()

        if reached:
            rospy.loginfo("All joints reached target positions.")

            step_pub.publish(
                Header(stamp=rospy.Time.now(),
                       seq=step_count)
            )

        else:
            rospy.logwarn(f"Joint movement timeout after {JOINT_TIMEOUT} sec. Proceeding to next action.")

        step_count += 1

    rospy.loginfo("Reached maximum step limit or ROS is shutting down. Stopping control.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
