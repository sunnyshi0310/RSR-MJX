cmake_minimum_required(VERSION 3.0.2)
project(airbot_t)

set(PYTHON_EXECUTABLE "/home/wang/anaconda3/envs/airbot/bin/python")

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  std_msgs
  message_generation
)

catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    scripts/sim2real_t_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
