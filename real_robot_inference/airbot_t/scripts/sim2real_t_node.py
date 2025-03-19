#!/home/wang/anaconda3/envs/airbot/bin/python
import rospy
import numpy as np
import jax.numpy as jnp
import sys
sys.path.insert(0, '/src/airbot_t')
from ppo_inference import PolicyInference
from real_robot_interface import RealRobotInterface
from brax import envs
from dlabsim.envs.T_nianzhong import AirbotPlayBase

ckpt_dir = 'model_path' 

def main():
    rospy.init_node("air_arm_control_node")

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

    step_count = 0
    rate = rospy.Rate(10) 
    while not rospy.is_shutdown():
        obs = robot.get_current_observation()
        ctrl = inference.get_action(obs, deterministic=True)
        ctrl = np.insert(ctrl, 3, 0)
        current_joint_positions = np.array(robot.current_joint_state.position)
        new_joint_positions = current_joint_positions + ctrl
        new_joint_positions = jnp.array(new_joint_positions)
        new_joint_positions = new_joint_positions.at[3].set(1.57) 

        T_pos_x = robot.current_end_pose.position.x
        T_pos_y = robot.current_end_pose.position.y 
        target_pos = [0.36071068,0.04928932]
        delta_x = target_pos[0] - T_pos_x
        delta_y = target_pos[1] - T_pos_y
        angle_to_box = jnp.arctan2(delta_y, delta_x + 0.00001) 
        new_joint_positions = new_joint_positions.at[5].set(-angle_to_box + ctrl[0] + 1.5708) 
        
        joint2 = new_joint_positions[1]
        joint3 = new_joint_positions[2]
        new_joint_positions = new_joint_positions.at[4].set(-(1.57 + joint2 + joint3)) # joint5

        target_base_x = 0.29
        target_base_y = 0.12
        target_base_z = 0.805
        target_vertical_x = 0.343033
        target_vertical_y = 0.066967
        target_vertical_z = 0.805
        target_array = ([target_vertical_x - target_base_x, target_vertical_y - target_base_y, target_vertical_z - target_base_z])

        box_array = ([robot.current_marker_point0.x - robot.current_marker_point1.x,
                    robot.current_marker_point0.y - robot.current_marker_point1.y, 
                    0])

        xita = np.dot(box_array,target_array) / (np.linalg.norm(box_array) * np.linalg.norm(target_array))
        xita = np.abs(xita - 1.0)
 
        if xita < 0.006:
            rospy.loginfo("T reached target position.")
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
        else:
            rospy.logwarn(f"Joint movement timeout after {JOINT_TIMEOUT} sec. Proceeding to next action.")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
