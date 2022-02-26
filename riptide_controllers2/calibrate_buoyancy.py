#! /usr/bin/env python3

# Determines the buoyancy parameter of the robot.
# Assumes the robot is upright

import rclpy
import time
import yaml
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from queue import Queue

from geometry_msgs.msg import Vector3, Quaternion, Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Parameter
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.srv import SetParameters
from riptide_msgs2.action import CalibrateBuoyancy
from std_msgs.msg import Empty

from transforms3d import euler, quaternions

import math
import yaml
import numpy as np

WATER_DENSITY = 1000
GRAVITY = 9.81

def msg_to_numpy(msg):
    """Converts a Vector3 or Quaternion message to its numpy counterpart"""
    if hasattr(msg, "w"):
        return np.array([msg.x, msg.y, msg.z, msg.w])
    return np.array([msg.x, msg.y, msg.z])

def changeFrame(orientation, vector, w2b = True):
    """Converts vector into other frame from orientation quaternion. The w2b parameter will
     determine if the vector is converting from world to body or body to world"""

    vector = np.append(vector, 0)
    if w2b:
        orientation = quaternions.qinverse(orientation)
    orientationInv = quaternions.qinverse(orientation)
    newVector = quaternions.qmult(orientation, quaternions.qmult(vector, orientationInv))
    return newVector[:3]


class CalibrateBuoyancyAction(Node):

    _result = CalibrateBuoyancy.Result()
    INITIAL_PARAM_NAMES = ['volume', 'cob', 'linear_damping', 'quadratic_damping']

    def __init__(self):
        super().__init__('calibrate_buoyancy')
        self.declare_parameter("vehicle_config", rclpy.Parameter.Type.STRING)

        self.orientation_pub = self.create_publisher(Quaternion ,"orientation", qos_profile_system_default)
        self.position_pub = self.create_publisher(Vector3 ,"position", qos_profile_system_default)
        self.off_pub = self.create_publisher(Empty ,"off", qos_profile_system_default)
        
        self.odometry_sub = self.create_subscription(Odometry, "odometry/filtered", self.odometry_cb, qos_profile_system_default)
        self.odometry_queue = Queue(1)
        self.requested_accel_sub = self.create_subscription(Twist, "controller/requested_accel", self.requested_accel_cb, qos_profile_system_default)
        self.requested_accel_queue = Queue(1)

        # Get the mass and COM
        with open(self.get_parameter('vehicle_config').value, 'r') as stream:
            vehicle = yaml.safe_load(stream)
            self.mass = vehicle['mass']
            self.com = np.array(vehicle['com'])
            self.inertia = np.array(vehicle['inertia'])

        self.running = False

        self.param_get_client = self.create_client(GetParameters, "controller/get_parameters")
        self.param_set_client = self.create_client(SetParameters, "controller/set_parameters")
        self.param_get_client.wait_for_service()
        self.param_set_client.wait_for_service()

        self._action_server = ActionServer(
            self,
            CalibrateBuoyancy,
            'calibrate_buoyancy',
            self.execute_cb,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup())

    def destroy(self):
        self.destroy_node()
        self._action_server.destroy()

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        if self.running:
            return GoalResponse.REJECT
        else:
            self.running = True
            return GoalResponse.ACCEPT

    def cancel_callback(self, goal):
        return CancelResponse.ACCEPT


    ##############################
    # Message Wait Functions
    ##############################

    def odometry_cb(self, msg: Odometry) -> None:
        if not self.odometry_queue.full():
            self.odometry_queue.put_nowait(msg)

    def wait_for_odometry_msg(self) -> Odometry:
        # Since the queue size is 1, if it has stuff in it just read to clear
        if not self.odometry_queue.empty():
            self.odometry_queue.get_nowait()
            assert self.odometry_queue.empty()
        
        return self.odometry_queue.get(True)

    def requested_accel_cb(self, msg: Twist) -> None:
        if not self.requested_accel_queue.full():
            self.requested_accel_queue.put_nowait(msg)

    def wait_for_requested_accel_msg(self) -> Twist:
        # Since the queue size is 1, if it has stuff in it just read to clear
        if not self.requested_accel_queue.empty():
            self.requested_accel_queue.get_nowait()
            assert self.requested_accel_queue.empty()
        
        return self.requested_accel_queue.get(True)

    ##############################
    # Parameter Utility Functions
    ##############################
    def load_initial_controller_config(self):
        request = GetParameters.Request()
        request.names = self.INITIAL_PARAM_NAMES
        response: GetParameters.Response = self.param_get_client.call(request)
        if len(response.values) != len(self.INITIAL_PARAM_NAMES):
            self.get_logger().error("Unable to retrieve all requested parameters")
            return False
        
        self.initial_config = {}
        for i in range(len(response.values)):
            self.initial_config[self.INITIAL_PARAM_NAMES[i]] = response.values[i].double_value

        return True

    def update_controller_config(self, config: dict):
        parameters = []
        for entry in config:
            param = Parameter()
            param.name = entry
            param_value = ParameterValue()
            if type(config[entry]) == list:
                param_value.type = ParameterType.PARAMETER_DOUBLE_ARRAY
                param_value.double_array_value = config[entry]
            else:
                param_value.type = ParameterType.PARAMETER_DOUBLE
                param_value.double_value = float(config[entry])
            param.value = param_value
            parameters.append(param)
        
        request = SetParameters.Request()
        request.parameters = parameters
        response: SetParameters.Response = self.param_set_client.call(request)
        
        if len(response.results) != len(parameters):
            self.get_logger().error("Unable to set all requested parameters")
            return False
        
        for entry in response.results:
            if not entry.successful:
                self.get_logger().error("Failed to set parameter: " + str(entry.reason))
                return False
        
        return True

    ##############################
    # Calibration Functions
    ##############################

    def tune(self, goal_handle, initial_value, get_adjustment, apply_change, num_samples=10, delay=4):
        """Tunes a parameter of the robot"""
        current_value = np.array(initial_value)
        last_adjustment = np.zeros_like(current_value)
        converged = np.zeros_like(current_value)

        while not np.all(converged):
            # Wait for equilibrium
            time.sleep(delay)

            # Average a few samples
            average_adjustment = 0
            for _ in range(num_samples):
                average_adjustment += get_adjustment() / num_samples
                time.sleep(0.2)

            # Apply change
            current_value += average_adjustment
            apply_change(current_value)

            # Check if the value has converged
            converged = np.logical_or(converged, average_adjustment * last_adjustment < 0)
            last_adjustment = average_adjustment

            if self.check_preempted(goal_handle):
                return None

        return current_value

      
    def execute_cb(self, goal_handle):
        self.running = True
        # Start reconfigure server and get starting config
        if not self.load_initial_controller_config():
            goal_handle.abort()
            self.running = False
            return CalibrateBuoyancy.Result()
        
        # Set variables to defaults
        self.get_logger().info("Starting buoyancy calibration")
        volume = self.mass / WATER_DENSITY
        cob = np.copy(self.com)

        # Reset parameters to default
        self.update_controller_config({
            "volume": volume, 
            "cob": list(cob),
            "linear_damping": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "quadratic_damping": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        })

        # Submerge
        odom_msg = self.wait_for_odometry_msg()
        current_position = msg_to_numpy(odom_msg.pose.pose.position)
        current_orientation = msg_to_numpy(odom_msg.pose.pose.orientation)
        self.position_pub.publish(Vector3(x=current_position[0], y=current_position[1], z=-1.0))
        _, _, y = euler.quat2euler(current_orientation, 'sxyz')
        w,x,y,z = euler.euler2quat(0.0, 0.0, y, axes='sxyz')
        self.orientation_pub.publish(Quaternion(w=w, x=x, y=y, z=z))

        # Wait for equilibrium
        time.sleep(10)

        # Volume adjustment function
        def get_volume_adjustment():
            body_force = self.mass * msg_to_numpy(self.wait_for_requested_accel_msg().linear)
            orientation = msg_to_numpy(self.wait_for_odometry_msg().pose.pose.orientation)
            world_z_force = changeFrame(orientation, body_force, w2b=False)[2]

            return -world_z_force / WATER_DENSITY / GRAVITY

        # Tune volume
        volume = self.tune(goal_handle,
            volume, 
            get_volume_adjustment, 
            lambda v: self.update_controller_config({"volume": v})
        )
        if volume == None:
            return CalibrateBuoyancy.Result()

        self.get_logger().info("Volume calibration complete")
        buoyant_force = volume * WATER_DENSITY * GRAVITY

        # COB adjustment function
        def get_cob_adjustment():
            accel = msg_to_numpy(self.wait_for_requested_accel_msg().angular)
            torque = self.inertia * accel
            orientation = msg_to_numpy(self.wait_for_odometry_msg().pose.pose.orientation)
            body_force_z = changeFrame(orientation, np.array([0, 0, buoyant_force]))[2]

            adjustment_x = torque[1] / body_force_z
            adjustment_y = -torque[0] / body_force_z

            return np.array([adjustment_x, adjustment_y])

        # Tune X and Y COB
        if self.tune(goal_handle,
            cob[:2], 
            get_cob_adjustment, 
            lambda cob: self.update_controller_config({"cob": cob}),
            num_samples = 2,
            delay = 1
        ) == None:
            return CalibrateBuoyancy.Result()

        self.get_logger().info("Buoyancy XY calibration complete")

        # Adjust orientation
        w,x,y,z = euler.euler2quat(0, -math.pi / 4, y, axes='sxyz')
        self.orientation_pub.publish(Quaternion(w=w, x=x, y=y, z=z))
        time.sleep(3)

        # Z COB function
        def get_cob_z_adjustment():
            accel = msg_to_numpy(self.wait_for_requested_accel_msg().angular)
            torque = self.inertia * accel
            orientation = msg_to_numpy(self.wait_for_odometry_msg().pose.pose.orientation)
            body_force_x = changeFrame(orientation, np.array([0, 0, buoyant_force]))[0]

            adjustment = -torque[1] / body_force_x

            return adjustment

        # Tune Z COB
        if self.tune(goal_handle,
            cob[2], 
            get_cob_z_adjustment, 
            lambda z: self.update_controller_config({"cob": cob})
        ) == None:
            return CalibrateBuoyancy.Result()

        self.get_logger().info("Calibration complete")
        self.cleanup()
        self._result.buoyant_force = buoyant_force
        self._result.center_of_buoyancy = cob

        self.running = False
        goal_handle.succeed()
        return self._result

    def check_preempted(self, goal_handle):
        if goal_handle.is_cancel_requested:
            self.get_logger().info('Preempted Calibration')
            self.cleanup()
            goal_handle.canceled()
            self.running = False
            return True

    def cleanup(self):
        self.update_controller_config({
            "linear_damping": self.initial_config['linear_damping'],
            "quadratic_damping": self.initial_config['quadratic_damping'],
        })
        self.off_pub.publish(Empty())


def main(args=None):
    rclpy.init(args=args)

    thruster_test_action_server = CalibrateBuoyancyAction()

    executor = MultiThreadedExecutor()
    rclpy.spin(thruster_test_action_server, executor=executor)

    thruster_test_action_server.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
