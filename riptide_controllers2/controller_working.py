#!/usr/bin/env python3

# controller node
#
# Input topics:
#   odometry/filtered: Current state of the vehicle
#   orientation: Puts the angular controller in position mode. Sets angular target to given orientation
#   angular_velocity: Puts the angular controller in velocity mode. Sets angular target to given body-frame angular velocity
#   disable_angular: Puts the angular controller in disabled mode.
#   position: Puts the linear controller in position mode. Sets linear target to given world-frame position
#   linear_velocity: Puts the linear controller in velocity mode. Sets linear target to given body-frame linear velocity
#   disable_linear: Puts the linear controller in disabled mode.
#   off: Turns off the controller. This will stop all output from the controller and thruster will stop
#
# Output topics:
#   net_force: The force the robot should exert on the world to achieve the given target
#   ~requested_accel: The acceleration requested from the controllers. Used for calibration
#
# This node contains 4 parts. The linear controller, the angular controller, the acceleration calculator, and the trajectory reader.
# The linear and angular controllers return an acceleration the robot should eperience to achieve that controller's target.
# The acceleration calculator takes that acceleration and computes how much force the robot needs to exert to achieve that acceleration.
# The trajectory reader will feed current states to the controllers to follow a trajectory.

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default, qos_profile_sensor_data # can replace this with others
from rclpy.action import ActionServer

import numpy as np
import time
import yaml

from riptide_controllers2.Controllers import msgToNumpy, LinearCascadedPController, AngularCascadedPController, AccelerationCalculator
from riptide_msgs2.action import FollowTrajectory
from riptide_msgs2.msg import RobotState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3, Twist
from std_msgs.msg import Empty, Bool

from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

# assumes order is xyz
def vect3_from_np(np_vect):
    return Vector3(x=np_vect[0], y=np_vect[1], z=np_vect[2])

class ControllerNode(Node):
        
    def __init__(self):
        super().__init__('riptide_controllers2')
        
        self.declare_parameter("vehicle_config", "")
        config_path = self.get_parameter("vehicle_config").value
        if(config_path == ''):
            self.get_logger().fatal("vehicle config file param not set or empty, exiting")

        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
            
        # declare the configuration data
        self.declare_parameters(
            namespace='',
            parameters=[
                ('linear_position_p', config["linear_position_p"]),
                ('linear_velocity_p', config["linear_velocity_p"]),
                ('angular_position_p', config["angular_position_p"]),
                ('angular_velocity_p', config["angular_velocity_p"]),
                ('linear_damping', config["linear_damping"]),
                ('quadratic_damping', config["quadratic_damping"]),
                ('maximum_linear_velocity', config["maximum_linear_velocity"]),
                ('maximum_linear_acceleration', config["maximum_linear_acceleration"]),
                ('maximum_angular_velocity', config["maximum_angular_velocity"]),
                ('maximum_angular_acceleration', config["maximum_angular_acceleration"])
            ]) 

        self.linearController = LinearCascadedPController()
        self.angularController = AngularCascadedPController()
        self.accelerationCalculator = AccelerationCalculator(config)

        self.maxLinearVelocity = config["maximum_linear_velocity"]
        self.maxLinearAcceleration = config["maximum_linear_acceleration"]
        self.maxAngularVelocity = config["maximum_angular_velocity"]
        self.maxAngularAcceleration = config["maximum_angular_acceleration"]

        self.linearController.positionP = config["linear_position_p"]
        self.linearController.velocityP = config["linear_velocity_p"]

        self.angularController.positionP = config["angular_position_p"]
        self.angularController.velocityP = config["angular_velocity_p"]

        self.accelerationCalculator.linearDrag = config["linear_damping"]
        self.accelerationCalculator.quadraticDrag = config["quadratic_damping"]    

        self.linearController.maxVelocity = config["maximum_linear_velocity"]
        self.linearController.maxAccel = config["maximum_linear_acceleration"]

        self.angularController.maxVelocity = config["maximum_angular_velocity"]
        self.angularController.maxAccel = config["maximum_angular_acceleration"]

        self.accelerationCalculator.buoyancy = np.array([0, 0, config["volume"] * self.accelerationCalculator.density * self.accelerationCalculator.gravity  ])
        self.accelerationCalculator.cob = config["cob"]

        self.lastTorque = None
        self.lastForce = None
        self.off = True

        # setup pulbishers 
        self.forcePub = self.create_publisher(Twist, "net_force", qos_profile_system_default)
        self.steadyPub = self.create_publisher(Bool, "steady", qos_profile_system_default)
        self.accelPub = self.create_publisher(Twist, "requested_accel", qos_profile_system_default)

        # setup all subscribers
        self.subs = []
        self.subs.append(self.create_subscription(Odometry, "odometry/filtered", self.updateState, qos_profile_system_default))
        self.subs.append(self.create_subscription(Quaternion, "orientation", self.angularController.setTargetPosition, qos_profile_system_default))
        self.subs.append(self.create_subscription(Vector3, "angular_velocity", self.angularController.setTargetVelocity, qos_profile_system_default))
        self.subs.append(self.create_subscription(Empty, "disable_angular", self.angularController.disable, qos_profile_system_default))
        self.subs.append(self.create_subscription(Vector3, "position", self.linearController.setTargetPosition, qos_profile_system_default))
        self.subs.append(self.create_subscription(Vector3, "linear_velocity", self.linearController.setTargetVelocity, qos_profile_system_default))
        self.subs.append(self.create_subscription(Empty, "disable_linear", self.linearController.disable, qos_profile_system_default))
        self.subs.append(self.create_subscription(Empty, "off", self.turnOff, qos_profile_system_default))
        self.subs.append(self.create_subscription(RobotState, "state/robot", self.switch_cb, qos_profile_sensor_data))

        #create an action server
        self._as = ActionServer(self, FollowTrajectory, "follow_trajectory", self.trajectory_callback)

        # new parameter reconfigure call
        self.add_on_set_parameters_callback(self.paramUpdateCallback)

        self.get_logger().info("Riptide controller initalized")

    def parameters_callback(self, params):
        success = True
        for param in params:
            if param.name == "maximum_linear_velocity":
                self.maxLinearVelocity = param.value
            elif param.name == "maximum_linear_acceleration":
                self.maxLinearAcceleration = param.value
            elif param.name == "maximum_angular_velocity":
                self.maxAngularVelocity = param.value
            elif param.name == "maximum_angular_acceleration":
                self.maxAngularAcceleration = param.value
            elif param.name == "linear_position_p":
                self.linearController.positionP = param.value
            elif param.name == "linear_velocity_p":
                self.linearController.velocityP = param.value
            elif param.name == "angular_position_p":
                self.angularController.positionP = param.value
            elif param.name == "angular_velocity_p":
                self.angularController.velocityP = param.value
            elif param.name == "linear_damping":
                self.accelerationCalculator.linearDrag = param.value
            elif param.name == "quadratic_damping":
                self.accelerationCalculator.quadraticDrag = param.value
            elif param.name == "maximum_linear_velocity":
                self.linearController.maxVelocity = param.value
            elif param.name == "maximum_linear_acceleration":
                self.linearController.maxAccel = param.value
            elif param.name == "maximum_angular_velocity":
                self.angularController.maxVelocity = param.value
            elif param.name == "maximum_angular_acceleration":
                self.angularController.maxAccel = param.value
            elif param.name == "volume":
                self.accelerationCalculator.buoyancy = np.array([0, 0, param.value * self.accelerationCalculator.density * self.accelerationCalculator.gravity  ])
            elif param.name == "cob":
                self.accelerationCalculator.cob = param.value
            else:
                success = False
                
        return SetParametersResult(successful=success)


    def updateState(self, odomMsg):        
        linearAccel = self.linearController.update(odomMsg)
        angularAccel = self.angularController.update(odomMsg)

        #print(linearAccel, angularAccel)

        accelTwist = Twist()
        accelTwist.linear = vect3_from_np(linearAccel)
        accelTwist.angular = vect3_from_np(angularAccel)
        self.accelPub.publish(accelTwist)

        if np.linalg.norm(linearAccel) > 0 or np.linalg.norm(angularAccel) > 0:
            self.off = False

        if not self.off:
            netForce, netTorque = self.accelerationCalculator.accelToNetForce(odomMsg, linearAccel, angularAccel)
        else:
            netForce, netTorque = np.zeros(3), np.zeros(3)

        isSteady = Bool()
        isSteady.data = self.linearController.steady and self.angularController.steady
        self.steadyPub.publish(isSteady)

        if not np.array_equal(self.lastTorque, netTorque) or \
           not np.array_equal(self.lastForce, netForce):

            forceTwist = Twist()
            forceTwist.linear = vect3_from_np(netForce)
            forceTwist.angular = vect3_from_np(netTorque)
            self.forcePub.publish(forceTwist)

            self.lastForce = netForce
            self.lastTorque = netTorque

        self.get_logger().debug("ticked controllers")

    def trajectory_callback(self, goal_handle):
        start = self.get_clock().now()

        for point in goal_handle.goal.trajectory.points:
            # Wait for next point
            while (self.get_clock().now() - start) < point.time_from_start:
                time.sleep(0.01)

            self.linear_controller.targetPosition = msgToNumpy(point.transforms[0].translation)
            self.linear_controller.targetVelocity = msgToNumpy(point.velocities[0].linear)
            self.linear_controller.targetAcceleration = msgToNumpy(point.accelerations[0].linear)
            self.angular_controller.targetPosition = msgToNumpy(point.transforms[0].rotation)
            self.angular_controller.targetVelocity = msgToNumpy(point.velocities[0].angular)
            self.angular_controller.targetAcceleration = msgToNumpy(point.accelerations[0].angular)

        last_point = goal_handle.goal.trajectory.points[-1]
        self.linear_controller.setTargetPosition(last_point.transforms[0].translation)
        self.angular_controller.setTargetPosition(last_point.transforms[0].rotation)

        goal_handle.succeed()
    
    def paramUpdateCallback(self, config):
        pass

        #return config

    def turnOff(self, msg=None):
        self.angularController.disable()
        self.linearController.disable()
        self.off = True

    def switch_cb(self, msg : RobotState):
        if not msg.kill_switch_inserted:
            self.turnOff()
        pass

def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
    