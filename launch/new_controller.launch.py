import launch
import launch.actions
from ament_index_python.packages import get_package_share_directory
import launch_ros.actions
import os
from launch.actions import DeclareLaunchArgument
from launch.substitutions import TextSubstitution, LaunchConfiguration
from launch_ros.actions import PushRosNamespace

def generate_launch_description():

    # Read in the vehicle's namespace through the command line or use the default value one is not provided
    launch.actions.DeclareLaunchArgument(
        "robot", 
        default_value="puddles",
        description="Name of the vehicle",
    )
    robot = 'puddles'

    # declare the path to the robot's vehicle description file
    config = os.path.join(
        get_package_share_directory('riptide_descriptions2'),
        'config',
        robot + '.yaml'
    )

    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package="riptide_controllers2",
            executable="controller",
            name="controller",
            output="screen",
            parameters=[{"vehicle_config": config}]
        ),
    ])