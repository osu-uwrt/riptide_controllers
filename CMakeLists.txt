cmake_minimum_required(VERSION 3.0.2)
project(riptide_controllers)

find_package(catkin REQUIRED
    COMPONENTS
    dynamic_reconfigure
    geometry_msgs
    sensor_msgs
    trajectory_msgs
    tf
    tf2_geometry_msgs
    actionlib_msgs
)

generate_dynamic_reconfigure_options(
  cfg/NewController.cfg
)

add_action_files(
  DIRECTORY action
  FILES CalibrateBuoyancy.action
  CalibrateDrag.action
  ThrusterTest.action
  FollowTrajectory.action
  ScriptOhio.action
)

generate_messages(
  DEPENDENCIES actionlib_msgs std_msgs geometry_msgs trajectory_msgs # Or other packages containing msgs
)

catkin_package(
  CATKIN_DEPENDS
    dynamic_reconfigure
    actionlib_msgs
)
