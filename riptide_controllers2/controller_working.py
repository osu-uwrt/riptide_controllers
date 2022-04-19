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
        
        self.linearController = LinearCascadedPController()
        self.angularController = AngularCascadedPController()
        self.accelerationCalculator = AccelerationCalculator(config)
            
        self.add_on_set_parameters_callback(self.parameters_callback)
        
        # declare the configuration data
        self.declare_parameters(
            namespace='',
            parameters=[
                ('linear_position_p_x', config["linear_position_p"][0]),
                ('linear_position_p_y', config["linear_position_p"][1]),
                ('linear_position_p_z', config["linear_position_p"][2]),
                ('linear_velocity_p_x', config["linear_velocity_p"][0]),
                ('linear_velocity_p_y', config["linear_velocity_p"][1]),
                ('linear_velocity_p_z', config["linear_velocity_p"][2]),
                ('angular_position_p_x', config["angular_position_p"][0]),
                ('angular_position_p_y', config["angular_position_p"][1]),
                ('angular_position_p_z', config["angular_position_p"][2]),
                ('angular_velocity_p_x', config["angular_velocity_p"][0]),
                ('angular_velocity_p_y', config["angular_velocity_p"][1]),
                ('angular_velocity_p_z', config["angular_velocity_p"][2]),
                ('linear_damping_x', config["linear_damping"][0]),
                ('linear_damping_y', config["linear_damping"][1]),
                ('linear_damping_z', config["linear_damping"][2]),
                ('linear_damping_rot_x', config["linear_damping"][3]),
                ('linear_damping_rot_y', config["linear_damping"][4]),
                ('linear_damping_rot_z', config["linear_damping"][5]),
                ('quadratic_damping_x', config["quadratic_damping"][0]),
                ('quadratic_damping_y', config["quadratic_damping"][1]),
                ('quadratic_damping_z', config["quadratic_damping"][2]),
                ('quadratic_damping_rot_x', config["quadratic_damping"][3]),
                ('quadratic_damping_rot_y', config["quadratic_damping"][4]),
                ('quadratic_damping_rot_z', config["quadratic_damping"][5]),
                ('maximum_linear_velocity_x', config["maximum_linear_velocity"][0]),
                ('maximum_linear_velocity_y', config["maximum_linear_velocity"][1]),
                ('maximum_linear_velocity_z', config["maximum_linear_velocity"][2]),
                ('maximum_linear_acceleration_x', config["maximum_linear_acceleration"][0]),
                ('maximum_linear_acceleration_y', config["maximum_linear_acceleration"][1]),
                ('maximum_linear_acceleration_z', config["maximum_linear_acceleration"][2]),
                ('maximum_angular_velocity_x', config["maximum_angular_velocity"][0]),
                ('maximum_angular_velocity_y', config["maximum_angular_velocity"][1]),
                ('maximum_angular_velocity_z', config["maximum_angular_velocity"][2]),
                ('maximum_angular_acceleration_x', config["maximum_angular_acceleration"][0]),
                ('maximum_angular_acceleration_y', config["maximum_angular_acceleration"][1]),
                ('maximum_angular_acceleration_z', config["maximum_angular_acceleration"][2]),
                ('volume', config["volume"]),
                ('cob_x', config["cob"][0]),
                ('cob_y', config["cob"][1]),
                ('cob_z', config["cob"][2])
            ]) 

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

        self.get_logger().info("Riptide controller initalized")

    def parameters_callback(self, params):
        success = True
        for param in params:
            if param.name == "maximum_linear_velocity_x":
                self.linearController.maxVelocity[0] = param.value
            elif param.name == "maximum_linear_velocity_y":
                self.linearController.maxVelocity[1] = param.value
            elif param.name == "maximum_linear_velocity_z":
                self.linearController.maxVelocity[2] = param.value
            elif param.name == "maximum_linear_acceleration_x":
                self.linearController.maxAccel[0] = param.value
            elif param.name == "maximum_linear_acceleration_y":
                self.linearController.maxAccel[1] = param.value
            elif param.name == "maximum_linear_acceleration_z":
                self.linearController.maxAccel[2] = param.value
            elif param.name == "maximum_angular_velocity_x":
                self.angularController.maxVelocity[0] = param.value
            elif param.name == "maximum_angular_velocity_y":
                self.angularController.maxVelocity[1] = param.value
            elif param.name == "maximum_angular_velocity_z":
                self.angularController.maxVelocity[2] = param.value
            elif param.name == "maximum_angular_acceleration_x":
                self.angularController.maxAccel[0] = param.value
            elif param.name == "maximum_angular_acceleration_y":
                self.angularController.maxAccel[1] = param.value
            elif param.name == "maximum_angular_acceleration_z":
                self.angularController.maxAccel[2] = param.value
            elif param.name == "linear_position_p_x":
                self.linearController.positionP[0] = param.value
            elif param.name == "linear_position_p_y":
                self.linearController.positionP[1] = param.value
            elif param.name == "linear_position_p_z":
                self.linearController.positionP[2] = param.value
            elif param.name == "linear_velocity_p_x":
                self.linearController.velocityP[0] = param.value
            elif param.name == "linear_velocity_p_y":
                self.linearController.velocityP[1] = param.value
            elif param.name == "linear_velocity_p_z":
                self.linearController.velocityP[2] = param.value
            elif param.name == "angular_position_p_x":
                self.angularController.positionP[0] = param.value
            elif param.name == "angular_position_p_y":
                self.angularController.positionP[1] = param.value
            elif param.name == "angular_position_p_z":
                self.angularController.positionP[2] = param.value
            elif param.name == "angular_velocity_p_x":
                self.angularController.velocityP[0] = param.value
            elif param.name == "angular_velocity_p_y":
                self.angularController.velocityP[1] = param.value
            elif param.name == "angular_velocity_p_z":
                self.angularController.velocityP[2] = param.value
            elif param.name == "linear_damping_x":
                self.accelerationCalculator.linearDrag[0] = param.value
            elif param.name == "linear_damping_y":
                self.accelerationCalculator.linearDrag[1] = param.value
            elif param.name == "linear_damping_z":
                self.accelerationCalculator.linearDrag[2] = param.value
            elif param.name == "linear_damping_rot_x":
                self.accelerationCalculator.linearDrag[3] = param.value
            elif param.name == "linear_damping_rot_y":
                self.accelerationCalculator.linearDrag[4] = param.value
            elif param.name == "linear_damping_rot_z":
                self.accelerationCalculator.linearDrag[5] = param.value
            elif param.name == "quadratic_damping_x":
                self.accelerationCalculator.quadraticDrag[0] = param.value
            elif param.name == "quadratic_damping_y":
                self.accelerationCalculator.quadraticDrag[1] = param.value
            elif param.name == "quadratic_damping_z":
                self.accelerationCalculator.quadraticDrag[2] = param.value
            elif param.name == "quadratic_damping_rot_x":
                self.accelerationCalculator.quadraticDrag[3] = param.value
            elif param.name == "quadratic_damping_rot_y":
                self.accelerationCalculator.quadraticDrag[4] = param.value
            elif param.name == "quadratic_damping_rot_z":
                self.accelerationCalculator.quadraticDrag[5] = param.value
            elif param.name == "maximum_linear_velocity_x":
                self.linearController.maxVelocity[0] = param.value
            elif param.name == "maximum_linear_velocity_y":
                self.linearController.maxVelocity[1] = param.value
            elif param.name == "maximum_linear_velocity_z":
                self.linearController.maxVelocity[2] = param.value
            elif param.name == "maximum_linear_acceleration_x":
                self.linearController.maxAccel[0] = param.value
            elif param.name == "maximum_linear_acceleration_y":
                self.linearController.maxAccel[1] = param.value
            elif param.name == "maximum_linear_acceleration_z":
                self.linearController.maxAccel[2] = param.value
            elif param.name == "maximum_angular_velocity_x":
                self.angularController.maxVelocity[0] = param.value
            elif param.name == "maximum_angular_velocity_y":
                self.angularController.maxVelocity[1] = param.value
            elif param.name == "maximum_angular_velocity_z":
                self.angularController.maxVelocity[2] = param.value
            elif param.name == "maximum_angular_acceleration_x":
                self.angularController.maxAccel[0] = param.value
            elif param.name == "maximum_angular_acceleration_y":
                self.angularController.maxAccel[1] = param.value
            elif param.name == "maximum_angular_acceleration_z":
                self.angularController.maxAccel[2] = param.value
            elif param.name == "volume":
                self.accelerationCalculator.buoyancy = np.array([0, 0, param.value * self.accelerationCalculator.density * self.accelerationCalculator.gravity  ])
            elif param.name == "cob_x":
                self.accelerationCalculator.cob[0] = param.value
            elif param.name == "cob_y":
                self.accelerationCalculator.cob[1] = param.value
            elif param.name == "cob_z":
                self.accelerationCalculator.cob[2] = param.value
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
    