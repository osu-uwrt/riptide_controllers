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
from rclpy.qos import qos_profile_system_default # can replace this with others
from rclpy.action import ActionServer

import numpy as np
import time
import yaml

from riptide_controllers2.Controllers import msgToNumpy, LinearCascadedPController, AngularCascadedPController, AccelerationCalculator
from riptide_msgs2.action import FollowTrajectory
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3, Twist
from std_msgs.msg import Empty, Bool

class ControllerNode(Node):
        
    def __init__(self):
        super().__init__('riptide_controllers2')
        
        self.declare_parameter("vehicle_config", "")
        config_path = self.get_parameter("vehicle_config").value
        if(config_path == ''):
            self.get_logger().fatal("vehicle config file param not set or empty, exiting")

        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)

        self.linearController = LinearCascadedPController()
        self.angularController = AngularCascadedPController()
        self.accelerationCalculator = AccelerationCalculator(config)

        self.maxLinearVelocity = config["maximum_linear_velocity"]
        self.maxLinearAcceleration = config["maximum_linear_acceleration"]
        self.maxAngularVelocity = config["maximum_angular_velocity"]
        self.maxAngularAcceleration = config["maximum_angular_acceleration"]

        self.lastTorque = None
        self.lastForce = None
        self.off = True

        # setup pulbishers 
        self.forcePub = self.create_publisher(Twist, "net_force", qos_profile_system_default)
        self.steadyPub = self.create_publisher(Bool, "steady", qos_profile_system_default)
        self.accelPub = self.create_publisher(Twist, "requested_accel", qos_profile_system_default)

        # setup all subscribers
        self.subs = []
        self.subs.append(self.create_subscription(Quaternion, "orientation", self.angularController.setTargetPosition, qos_profile_system_default))
        self.subs.append(self.create_subscription(Vector3, "angular_velocity", self.angularController.setTargetVelocity, qos_profile_system_default))
        self.subs.append(self.create_subscription(Empty, "disable_angular", self.angularController.disable, qos_profile_system_default))
        self.subs.append(self.create_subscription(Vector3, "position", self.linearController.setTargetPosition, qos_profile_system_default))
        self.subs.append(self.create_subscription(Vector3, "linear_velocity", self.linearController.setTargetVelocity, qos_profile_system_default))
        self.subs.append(self.create_subscription(Empty, "disable_linear", self.linearController.disable, qos_profile_system_default))
        self.subs.append(self.create_subscription(Empty, "off", self.turnOff, qos_profile_system_default))
        self.subs.append(self.create_subscription(Bool, "state/kill_switch", self.switch_cb, qos_profile_system_default))

        #create an action server
        self._as = ActionServer(self, FollowTrajectory, "follow_trajectory", self.execute_cb)

        # new parameter reconfigure call
        self.add_on_set_parameters_callback(self.paramUpdateCallback)

        self.get_logger().info("Riptide controller initalized")

    def updateState(self, odomMsg):        
        linearAccel = self.linearController.update(odomMsg)
        angularAccel = self.angularController.update(odomMsg)

        self.accelPub.publish(Twist(Vector3(*linearAccel), Vector3(*angularAccel)))

        if np.linalg.norm(linearAccel) > 0 or np.linalg.norm(angularAccel) > 0:
            self.off = False

        if not self.off:
            netForce, netTorque = self.accelerationCalculator.accelToNetForce(odomMsg, linearAccel, angularAccel)
        else:
            netForce, netTorque = np.zeros(3), np.zeros(3)

        self.steadyPub.publish(self.linearController.steady and self.angularController.steady)

        if not np.array_equal(self.lastTorque, netTorque) or \
           not np.array_equal(self.lastForce, netForce):

            self.forcePub.publish(Twist(Vector3(*netForce), Vector3(*netTorque)))
            self.lastForce = netForce
            self.lastTorque = netTorque

    def execute_cb(self, goal_handle):
        start = self.get_clock().now()

        for point in goal_handle.goal.trajectory.points:
            # Wait for next point
            while (self.get_clock().now() - start) < point.time_from_start:
                if self._as.is_preempt_requested():
                    self.linear_controller.targetVelocity = np.zeros(3)
                    self.linear_controller.targetAcceleration = np.zeros(3)
                    self.angular_controller.targetVelocity = np.zeros(3)
                    self.angular_controller.targetAcceleration = np.zeros(3)
                    self._as.set_preempted()
                    return

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

        self.goal_handle.set_succeeded()
    
    def paramUpdateCallback(self, config):
        pass
    #     self.linearController.positionP[0] = config["linear_position_p_x"]
    #     self.linearController.positionP[1] = config["linear_position_p_y"]
    #     self.linearController.positionP[2] = config["linear_position_p_z"]
    #     self.linearController.velocityP[0] = config["linear_velocity_p_x"]
    #     self.linearController.velocityP[1] = config["linear_velocity_p_y"]
    #     self.linearController.velocityP[2] = config["linear_velocity_p_z"]
        
    #     self.angularController.positionP[0] = config["angular_position_p_x"]
    #     self.angularController.positionP[1] = config["angular_position_p_y"]
    #     self.angularController.positionP[2] = config["angular_position_p_z"]
    #     self.angularController.velocityP[0] = config["angular_velocity_p_x"]
    #     self.angularController.velocityP[1] = config["angular_velocity_p_y"]
    #     self.angularController.velocityP[2] = config["angular_velocity_p_z"]

    #     self.accelerationCalculator.linearDrag[0] = config["linear_x"]    
    #     self.accelerationCalculator.linearDrag[1] = config["linear_y"] 
    #     self.accelerationCalculator.linearDrag[2] = config["linear_z"]
    #     self.accelerationCalculator.linearDrag[3] = config["linear_rot_x"]
    #     self.accelerationCalculator.linearDrag[4] = config["linear_rot_y"]
    #     self.accelerationCalculator.linearDrag[5] = config["linear_rot_z"]

    #     self.accelerationCalculator.quadraticDrag[0] = config["quadratic_x"]    
    #     self.accelerationCalculator.quadraticDrag[1] = config["quadratic_y"] 
    #     self.accelerationCalculator.quadraticDrag[2] = config["quadratic_z"] 
    #     self.accelerationCalculator.quadraticDrag[3] = config["quadratic_rot_x"]
    #     self.accelerationCalculator.quadraticDrag[4] = config["quadratic_rot_y"]
    #     self.accelerationCalculator.quadraticDrag[5] = config["quadratic_rot_z"]

    #     self.linearController.maxVelocity[0] = config["max_linear_velocity_x"]
    #     self.linearController.maxVelocity[1] = config["max_linear_velocity_y"]
    #     self.linearController.maxVelocity[2] = config["max_linear_velocity_z"]
    #     self.linearController.maxAccel[0] = config["max_linear_accel_x"]
    #     self.linearController.maxAccel[1] = config["max_linear_accel_y"]
    #     self.linearController.maxAccel[2] = config["max_linear_accel_z"]

    #     self.angularController.maxVelocity[0] = config["max_angular_velocity_x"]
    #     self.angularController.maxVelocity[1] = config["max_angular_velocity_y"]
    #     self.angularController.maxVelocity[2] = config["max_angular_velocity_z"]
    #     self.angularController.maxAccel[0] = config["max_angular_accel_x"]
    #     self.angularController.maxAccel[1] = config["max_angular_accel_y"]
    #     self.angularController.maxAccel[2] = config["max_angular_accel_z"]

    #     self.accelerationCalculator.buoyancy = np.array([0, 0, config["volume"] * self.accelerationCalculator.density * self.accelerationCalculator.gravity  ])
    #     self.accelerationCalculator.cob[0] = config["center_x"]
    #     self.accelerationCalculator.cob[1] = config["center_y"]
    #     self.accelerationCalculator.cob[2] = config["center_z"]

    #     return config

    def turnOff(self, msg=None):
        self.angularController.disable()
        self.linearController.disable()
        self.off = True

    def switch_cb(self, msg):
        if not msg.data:
            self.turnOff()
        pass

def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()