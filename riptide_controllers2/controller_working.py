
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
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3, Twist
from std_msgs.msg import Empty, Bool
# from tf.transformations import quaternion_multiply, quaternion_inverse, quaternion_slerp
import numpy as np
from abc import ABC, abstractmethod
import yaml
# from dynamic_reconfigure.server import Server
# from riptide_controllers.cfg import NewControllerConfig
# import actionlib
import riptide_controllers.msg

# def msgToNumpy(msg):
    # if hasattr(msg, "w"):
    #     return np.array([msg.x, msg.y, msg.z, msg.w])
    # return np.array([msg.x, msg.y, msg.z])

# def worldToBody(orientation, vector):
#     """ 
#     Rotates world-frame vector to body-frame

#     Rotates vector to body frame

#     Parameters:
#     orientation (np.array): The current orientation of the robot as a quaternion
#     vector (np.array): The 3D vector to rotate

#     Returns: 
#     np.array: 3 dimensional rotated vector

#     """

#     vector = np.append(vector, 0)
#     orientationInv = quaternion_inverse(orientation)
#     newVector = quaternion_multiply(orientationInv, quaternion_multiply(vector, orientation))
#     return newVector[:3]

# def applyMax(vector, max_vector):
#     """ 
#     Scales a vector to obey maximums

#     Parameters:
#     vector (np.array): The unscaled vector
#     max_vector (np.array): The maximum values for each element

#     Returns: 
#     np.array: Vector that obeys the maximums

#     """

#     scale = 1
#     for i in range(len(vector)):
#         if abs(vector[i]) > max_vector[i]:
#             element_scale = max_vector[i] / abs(vector[i])
#             if element_scale < scale:
#                 scale = element_scale

#     return vector * scale


# class CascadedPController(ABC):

#     def __init__(self):
#         self.targetPosition = None
#         self.targetVelocity = None
#         self.targetAcceleration = None
#         self.positionP = np.zeros(3)
#         self.velocityP = np.zeros(3)
#         self.maxVelocity = np.zeros(3)
#         self.maxAccel = np.zeros(3)
#         self.steadyVelThresh = 0.00001
#         self.steadyAccelThresh = 0.00001
#         self.steady = True

#     @abstractmethod
#     def computeCorrectiveVelocity(self, odom):
#         """ 
#         Computes a corrective velocity.
    
#         If self.targetPosition is not None, will return a corrective body-frame velocity that moves the robot in the direction of self.targetPosiiton. Otherwise returns 0 vector.
    
#         Parameters:
#         odom (Odometry): The latest odometry message

#         Returns: 
#         np.array: 3 dimensional vector representing corrective body-frame velocity.
    
#         """
#         pass

#     @abstractmethod
#     def computeCorrectiveAcceleration(self, odom, correctiveVelocity):
#         """ 
#         Computes a corrective acceleration.
    
#         If self.targetVelocity is not None, will return a corrective body-frame acceleration that transitions the robot twoards the desired self.targetVelocity. Otherwise returns 0 vector.
    
#         Parameters:
#         correctiveVelocity (np.array): Body-frame velocity vector that adds on to the self.targetVelocity. Is used in position correction.
#         odom (Odometry): The latest odometry message

#         Returns: 
#         np.array: 3 dimensional vector representing corrective body-frame acceleration.
    
#         """
#         pass

#     def setTargetPosition(self, targetPosition):
#         """ 
#         Sets target position
    
#         Puts the controller in the Position state and sets self.targetPosition to targetPosition
    
#         Parameters:
#         targetPosition (np.array or Vector3): World-frame vector or quaternion to be achieved by the controller

#         """

#         self.targetPosition = msgToNumpy(targetPosition)
#         # Zeros here because we want velocity correction to 
#         # still happen but don't want it to hold a velocity
#         self.targetVelocity = np.zeros(3)
#         self.targetAcceleration = None

#     def setTargetVelocity(self, targetVelocity):
#         """ 
#         Sets target velocity
    
#         Puts the controller in the Velocity state and sets self.targetVelocity to targetVelocity
    
#         Parameters:
#         targetVelocity (np.array): Body-frame vector to be achieved by the controller

#         """

#         self.targetPosition = None
#         self.targetVelocity = msgToNumpy(targetVelocity)
#         self.targetAcceleration = None
    
#     def disable(self, msg=None):
#         """ 
#         Disables the controller
    
#         Puts the controller in the Disabled state

#         """

#         self.targetPosition = None
#         self.targetVelocity = None
#         self.targetAcceleration = None

#     def update(self, odom):
#         """ 
#         Updates the controller
    
#         Will compute an output acceleration to achieve the desired state
    
#         Parameters:
#         odom (Odometry): The latest odometry message

#         Returns: 
#         np.array: 3 dimensional vector representing net body-frame acceleration.

#         """

#         netAccel = np.zeros(3)

#         if self.targetPosition is not None or self.targetVelocity is not None:
#             correctiveVelocity = self.computeCorrectiveVelocity(odom)
#             netAccel += self.computeCorrectiveAcceleration(odom, correctiveVelocity)

#         if self.targetAcceleration is not None:
#             netAccel += self.targetAcceleration

#         netAccel = applyMax(netAccel, self.maxAccel)
        
#         if self.targetAcceleration is not None:                         # Trajectory mode
#             self.steady = False
#         elif self.targetPosition is not None:                           # Position mode
#             self.steady = np.allclose(correctiveVelocity, np.zeros(3), atol=self.steadyVelThresh)
#         elif self.targetVelocity is not None:                           # Velocity mode
#             self.steady = np.allclose(netAccel, np.zeros(3), atol=self.steadyAccelThresh)
#         else:
#             self.steady = True

#         return netAccel

# class LinearCascadedPController(CascadedPController):

#     def __init__(self):
#         super(LinearCascadedPController, self).__init__()
#         self.steadyVelThresh = 0.02
#         self.steadyAccelThresh = 0.02

#     def computeCorrectiveVelocity(self, odom):

#         if self.targetPosition is not None:
#             currentPosition = msgToNumpy(odom.pose.pose.position) # [1 0 0]
#             outputVel = (self.targetPosition - currentPosition) * self.positionP # [-1 0 1]
#             orientation = msgToNumpy(odom.pose.pose.orientation)
#             return worldToBody(orientation, outputVel)
#         else:
#             return np.zeros(3)

#     def computeCorrectiveAcceleration(self, odom, correctiveVelocity):

#         if self.targetVelocity is not None:          
#             targetVelocity = self.targetVelocity + correctiveVelocity
#             targetVelocity = applyMax(targetVelocity, self.maxVelocity)
#             currentVelocity = msgToNumpy(odom.twist.twist.linear)
#             outputAccel = (targetVelocity - currentVelocity) * self.velocityP
#             return outputAccel       
#         else:
#             return np.zeros(3)

# class AngularCascadedPController(CascadedPController):

#     def __init__(self):
#         super(AngularCascadedPController, self).__init__()
#         self.steadyVelThresh = 0.1
#         self.steadyAccelThresh = 0.1

#     def computeCorrectiveVelocity(self, odom):

#         if self.targetPosition is not None:
#             currentOrientation = msgToNumpy(odom.pose.pose.orientation)

#             # Below code only works for small angles so lets find an orientation in the right direction but with a small angle
#             intermediateOrientation = quaternion_slerp(currentOrientation, self.targetPosition, 0.01)
#             dq = (intermediateOrientation - currentOrientation)
#             outputVel = quaternion_multiply(quaternion_inverse(currentOrientation), dq)[:3] * self.positionP
#             return outputVel        
#         else:
#             return np.zeros(3)

#     def computeCorrectiveAcceleration(self, odom, correctiveVelocity):

#         if self.targetVelocity is not None:   
#             targetVelocity = self.targetVelocity + correctiveVelocity
#             for i in range(3):
#                 if abs(targetVelocity[i]) > self.maxVelocity[i]:
#                     targetVelocity[i] = self.maxVelocity[i] * targetVelocity[i] / abs(targetVelocity[i])           
#             currentVelocity = msgToNumpy(odom.twist.twist.angular)
#             outputAccel = (targetVelocity - currentVelocity) * self.velocityP
#             return outputAccel       
#         else:
#             return np.zeros(3)

# class AccelerationCalculator:
#     def __init__(self, config):
#         self.mass = np.array(config["mass"])
#         self.com = np.array(config["com"])
#         self.inertia = np.array(config["inertia"])
#         self.linearDrag = np.zeros(6)
#         self.quadraticDrag = np.zeros(6)
#         self.volume = np.array(config["volume"])
#         self.cob = np.zeros(3)
#         self.gravity = 9.8 # (m/sec^2)
#         self.density = 1000 # density of water (kg/m^3)
#         self.buoyancy = np.zeros(3)

#     def accelToNetForce(self, odom, linearAccel, angularAccel):
#         """ 
#         Converts vehicle acceleration into required net force.
    
#         Will take the required acceleration and consider mass, buoyancy, drag, and precession to compute the required net force.
    
#         Parameters:
#         odom (Odometry): The latest odometry message.
#         linearAccel (np.array): The linear body-frame acceleration.
#         angularAccel (np.array): The angular body-frame acceleration.

#         Returns: 
#         np.array: 3 dimensional vector representing net body-frame force.
#         np.array: 3 dimensional vector representing net body-frame torque.

#         """

#         linearVelo = msgToNumpy(odom.twist.twist.linear)
#         angularVelo = msgToNumpy(odom.twist.twist.angular)
#         orientation = msgToNumpy(odom.pose.pose.orientation)

#         # Force & Torque Initialization
#         netForce = linearAccel * self.mass
#         netTorque = angularAccel * self.inertia
        
#         # Forces and Torques Calculation
#         bodyFrameBuoyancy = worldToBody(orientation, self.buoyancy)
#         buoyancyTorque = np.cross((self.cob-self.com), bodyFrameBuoyancy)
#         precessionTorque = -np.cross(angularVelo, (self.inertia * angularVelo))
#         dragForce = self.linearDrag[:3] * linearVelo + self.quadraticDrag[:3] * abs(linearVelo) * linearVelo
#         dragTorque = self.linearDrag[3:] * angularVelo + self.quadraticDrag[3:] * abs(angularVelo) * angularVelo
#         gravityForce = worldToBody(orientation, np.array([0, 0, - self.gravity * self.mass]))
                
#         # Net Calculation
#         netForce = netForce - bodyFrameBuoyancy - dragForce - gravityForce
#         netTorque = netTorque - buoyancyTorque - precessionTorque - dragTorque

#         return netForce, netTorque

# class TrajectoryReader:

#     def __init__(self, linear_controller, angular_controller):
#         self.linear_controller = linear_controller
#         self.angular_controller = angular_controller
#         self._as = actionlib.SimpleActionServer("follow_trajectory", riptide_controllers.msg.FollowTrajectoryAction, execute_cb=self.execute_cb, auto_start=False)
#         self._as.start()

#     def execute_cb(self, goal):
#         start = rclpy.get_rostime()

#         for point in goal.trajectory.points:
#             # Wait for next point
#             while (rclpy.get_rostime() - start) < point.time_from_start:
#                 if self._as.is_preempt_requested():
#                     self.linear_controller.targetVelocity = np.zeros(3)
#                     self.linear_controller.targetAcceleration = np.zeros(3)
#                     self.angular_controller.targetVelocity = np.zeros(3)
#                     self.angular_controller.targetAcceleration = np.zeros(3)
#                     self._as.set_preempted()
#                     return

#                 rclpy.sleep(0.01)

#             self.linear_controller.targetPosition = msgToNumpy(point.transforms[0].translation)
#             self.linear_controller.targetVelocity = msgToNumpy(point.velocities[0].linear)
#             self.linear_controller.targetAcceleration = msgToNumpy(point.accelerations[0].linear)
#             self.angular_controller.targetPosition = msgToNumpy(point.transforms[0].rotation)
#             self.angular_controller.targetVelocity = msgToNumpy(point.velocities[0].angular)
#             self.angular_controller.targetAcceleration = msgToNumpy(point.accelerations[0].angular)

#         last_point = goal.trajectory.points[-1]
#         self.linear_controller.setTargetPosition(last_point.transforms[0].translation)
#         self.angular_controller.setTargetPosition(last_point.transforms[0].rotation)

#         self._as.set_succeeded()

class ControllerNode(Node):
    def blob(self, params):
        print('blob')
        
    def __init__(self):
        super().__init__('riptide_controllers2')
        print('hello')
        # new parameter reconfigure call
        # self.add_on_set_parameters_callback(self.paramUpdateCallback)
        self.create_subscription(Odometry, "{}odometry/filtered".format(self.get_namespace()), self.blob, qos_profile_system_default)
        
        # config_path = rclpy.get_param("~vehicle_config")
        # with open(config_path, 'r') as stream:
        #     config = yaml.safe_load(stream)

        # self.linearController = LinearCascadedPController()
        # self.angularController = AngularCascadedPController()
        # self.accelerationCalculator = AccelerationCalculator(config)
        # self.trajectory_reader = TrajectoryReader(self.linearController, self.angularController)

        # self.maxLinearVelocity = config["maximum_linear_velocity"]
        # self.maxLinearAcceleration = config["maximum_linear_acceleration"]
        # self.maxAngularVelocity = config["maximum_angular_velocity"]
        # self.maxAngularAcceleration = config["maximum_angular_acceleration"]

        # self.lastTorque = None
        # self.lastForce = None
        # self.off = True

        # self.forcePub = rclpy.Publisher("net_force", Twist, queue_size=1)
        # self.steadyPub = rclpy.Publisher("steady", Bool, queue_size=1)
        # self.accelPub = rclpy.Publisher("~requested_accel", Twist, queue_size=1)
        # rclpy.Subscriber("odometry/filtered", Odometry, self.updateState)
        # rclpy.Subscriber("orientation", Quaternion, self.angularController.setTargetPosition)
        # rclpy.Subscriber("angular_velocity", Vector3, self.angularController.setTargetVelocity)
        # rclpy.Subscriber("disable_angular", Empty, self.angularController.disable)
        # rclpy.Subscriber("position", Vector3, self.linearController.setTargetPosition)
        # rclpy.Subscriber("linear_velocity", Vector3, self.linearController.setTargetVelocity)
        # rclpy.Subscriber("disable_linear", Empty, self.linearController.disable)
        # rclpy.Subscriber("off", Empty, self.turnOff)
        # rclpy.Subscriber("state/kill_switch", Bool, self.switch_cb)
        

        

    #     self.reconfigure_server = Server(NewControllerConfig, self.dynamicReconfigureCb)      
    #     # set default values in dynamic reconfig  
    #     self.reconfigure_server.update_configuration({
    #         "linear_position_p_x": config["linear_position_p"][0],
    #         "linear_position_p_y": config["linear_position_p"][1],
    #         "linear_position_p_z": config["linear_position_p"][2],
    #         "linear_velocity_p_x": config["linear_velocity_p"][0],
    #         "linear_velocity_p_y": config["linear_velocity_p"][1],
    #         "linear_velocity_p_z": config["linear_velocity_p"][2],
    #         "angular_position_p_x": config["angular_position_p"][0],
    #         "angular_position_p_y": config["angular_position_p"][1],
    #         "angular_position_p_z": config["angular_position_p"][2],
    #         "angular_velocity_p_x": config["angular_velocity_p"][0],
    #         "angular_velocity_p_y": config["angular_velocity_p"][1],
    #         "angular_velocity_p_z": config["angular_velocity_p"][2], 
    #         "linear_x": config["linear_damping"][0],
    #         "linear_y": config["linear_damping"][1],
    #         "linear_z": config["linear_damping"][2],
    #         "linear_rot_x": config["linear_damping"][3],
    #         "linear_rot_y": config["linear_damping"][4],
    #         "linear_rot_z": config["linear_damping"][5],
    #         "quadratic_x": config["quadratic_damping"][0],
    #         "quadratic_y": config["quadratic_damping"][1],
    #         "quadratic_z": config["quadratic_damping"][2],
    #         "quadratic_rot_x": config["quadratic_damping"][3],
    #         "quadratic_rot_y": config["quadratic_damping"][4],
    #         "quadratic_rot_z": config["quadratic_damping"][5],
    #         "volume": config["volume"],
    #         "center_x": config['cob'][0],
    #         "center_y": config['cob'][1],
    #         "center_z": config['cob'][2],            
    #         "max_linear_velocity_x": config["maximum_linear_velocity"][0],
    #         "max_linear_velocity_y": config["maximum_linear_velocity"][1],
    #         "max_linear_velocity_z": config["maximum_linear_velocity"][2],
    #         "max_linear_accel_x": config["maximum_linear_acceleration"][0],
    #         "max_linear_accel_y": config["maximum_linear_acceleration"][1],
    #         "max_linear_accel_z": config["maximum_linear_acceleration"][2],
    #         "max_angular_velocity_x": config["maximum_angular_velocity"][0],
    #         "max_angular_velocity_y": config["maximum_angular_velocity"][1],
    #         "max_angular_velocity_z": config["maximum_angular_velocity"][2],
    #         "max_angular_accel_x": config["maximum_angular_acceleration"][0],
    #         "max_angular_accel_y": config["maximum_angular_acceleration"][1],
    #         "max_angular_accel_z": config["maximum_angular_acceleration"][2]               
    #     })

    # def updateState(self, odomMsg):        
    #     linearAccel = self.linearController.update(odomMsg)
    #     angularAccel = self.angularController.update(odomMsg)

    #     self.accelPub.publish(Twist(Vector3(*linearAccel), Vector3(*angularAccel)))

    #     if np.linalg.norm(linearAccel) > 0 or np.linalg.norm(angularAccel) > 0:
    #         self.off = False

    #     if not self.off:
    #         netForce, netTorque = self.accelerationCalculator.accelToNetForce(odomMsg, linearAccel, angularAccel)
    #     else:
    #         netForce, netTorque = np.zeros(3), np.zeros(3)

    #     self.steadyPub.publish(self.linearController.steady and self.angularController.steady)

    #     if not np.array_equal(self.lastTorque, netTorque) or \
    #        not np.array_equal(self.lastForce, netForce):

    #         self.forcePub.publish(Twist(Vector3(*netForce), Vector3(*netTorque)))
    #         self.lastForce = netForce
    #         self.lastTorque = netTorque
    
    # def dynamicReconfigureCb(self, config, level):
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

    # def turnOff(self, msg=None):
    #     self.angularController.disable()
    #     self.linearController.disable()
    #     self.off = True

    # def switch_cb(self, msg):
    #     if not msg.data:
    #         self.turnOff()
    #     pass

def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()