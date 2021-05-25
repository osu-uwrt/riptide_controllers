cmake_minimum_required(VERSION 3.0.2)
project(riptide_controllers)

find_package(catkin REQUIRED COMPONENTS
    genmsg
    actionlib_msgs
    rospy
    dynamic_reconfigure
    trajectory_msgs
)

add_action_files(DIRECTORY action FILES 
    CalibrateBuoyancy.action
    CalibrateDrag.action
    FollowTrajectory.action
    ScriptOhio.action
    ShroudTest.action
    SickTrick.action
    ThrusterTest.action
)

generate_dynamic_reconfigure_options(cfg/NewController.cfg)

generate_messages(DEPENDENCIES actionlib_msgs trajectory_msgs)

catkin_package(CATKIN_DEPENDS dynamic_reconfigure)
