#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3, Twist
from std_msgs.msg import Empty
from tf.transformations import quaternion_multiply, quaternion_inverse, quaternion_slerp
import numpy as np
from abc import ABCMeta, abstractmethod
import yaml

def msgToNumpy(msg):
    if hasattr(msg, "w"):
        return np.array([msg.x, msg.y, msg.z, msg.w])
    return np.array([msg.x, msg.y, msg.z])

def worldToBody(orientation, vector):
    """ 
    Rotates world-frame vector to body-frame

    Rotates vector to body frame

    Parameters:
    orientation (np.array): The current orientation of the robot as a quaternion
    vector (np.array): The 3D vector to rotate

    Returns: 
    np.array: 3 dimensional rotated vector

    """

    vector = np.append(vector, 0)
    orientationInv = quaternion_inverse(orientation)
    newVector = quaternion_multiply(orientationInv, quaternion_multiply(vector, orientation))
    return newVector[:3]


class CascadedPController:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.targetPosition = None
        self.targetVelocity = None
        self.targetAcceleration = None
        self.positionP = np.array([1, 1, 1])
        self.velocityP = np.array([5, 5, 5])

    @abstractmethod
    def computeCorrectiveVelocity(self, odom):
        """ 
        Computes a corrective velocity.
    
        If self.targetPosition is not None, will return a corrective body-frame velocity that moves the robot in the direction of self.targetPosiiton. Otherwise returns 0 vector.
    
        Parameters:
        odom (Odometry): The latest odometry message

        Returns: 
        np.array: 3 dimensional vector representing corrective body-frame velocity.
    
        """
        pass

    @abstractmethod
    def computeCorrectiveAcceleration(self, odom, correctiveVelocity):
        """ 
        Computes a corrective acceleration.
    
        If self.targetVelocity is not None, will return a corrective body-frame acceleration that transitions the robot twoards the desired self.targetVelocity. Otherwise returns 0 vector.
    
        Parameters:
        correctiveVelocity (np.array): Body-frame velocity vector that adds on to the self.targetVelocity. Is used in position correction.
        odom (Odometry): The latest odometry message

        Returns: 
        np.array: 3 dimensional vector representing corrective body-frame acceleration.
    
        """
        pass

    def setTargetPosition(self, targetPosition):
        """ 
        Sets target position
    
        Puts the controller in the Position state and sets self.targetPosition to targetPosition
    
        Parameters:
        targetPosition (np.array or Vector3): World-frame vector or quaternion to be achieved by the controller

        """

        self.targetPosition = msgToNumpy(targetPosition)
        # Zeros here because we want velocity correction to 
        # still happen but don't want it to hold a velocity
        self.targetVelocity = np.zeros(3)
        self.targetAcceleration = None

    def setTargetVelocity(self, targetVelocity):
        """ 
        Sets target velocity
    
        Puts the controller in the Velocity state and sets self.targetVelocity to targetVelocity
    
        Parameters:
        targetVelocity (np.array): Body-frame vector to be achieved by the controller

        """

        self.targetPosition = None
        self.targetVelocity = msgToNumpy(targetVelocity)
        self.targetAcceleration = None
    
    def disable(self, msg=None):
        """ 
        Disables the controller
    
        Puts the controller in the Disabled state

        """

        self.targetPosition = None
        self.targetVelocity = None
        self.targetAcceleration = None

    def update(self, odom):
        """ 
        Updates the controller
    
        Will compute an output acceleration to achieve the desired state
    
        Parameters:
        odom (Odometry): The latest odometry message

        Returns: 
        np.array: 3 dimensional vector representing net body-frame acceleration.

        """

        netAccel = np.zeros(3)

        if self.targetPosition is not None or self.targetVelocity is not None:
            correctiveVelocity = self.computeCorrectiveVelocity(odom)
            netAccel += self.computeCorrectiveAcceleration(odom, correctiveVelocity)

        if self.targetAcceleration is not None:
            netAccel += self.targetAcceleration

        return netAccel

class LinearCascadedPController(CascadedPController):

    def __init__(self):
        super(LinearCascadedPController, self).__init__()

    def computeCorrectiveVelocity(self, odom):

        if self.targetPosition is not None:
            currentPosition = msgToNumpy(odom.pose.pose.position) # [1 0 0]
            outputVel = (self.targetPosition - currentPosition) * self.positionP # [-1 0 1]
            orientation = msgToNumpy(odom.pose.pose.orientation)
            return worldToBody(orientation, outputVel)   
        else:
            return np.zeros(3)

    def computeCorrectiveAcceleration(self, odom, correctiveVelocity):

        if self.targetVelocity is not None:            
            currentVelocity = msgToNumpy(odom.twist.twist.linear)
            outputAccel = ((self.targetVelocity + correctiveVelocity) - currentVelocity) * self.velocityP
            return outputAccel       
        else:
            return np.zeros(3)

class AngularCascadedPController(CascadedPController):

    def __init__(self):
        super(AngularCascadedPController, self).__init__()

    def computeCorrectiveVelocity(self, odom):

        if self.targetPosition is not None:
            currentOrientation = msgToNumpy(odom.pose.pose.orientation)

            # Below code only works for small angles so lets find an orientation in the right direction but with a small angle
            intermediateOrientation = quaternion_slerp(currentOrientation, self.targetPosition, 0.01)
            dq = (intermediateOrientation - currentOrientation)
            outputVel = quaternion_multiply(quaternion_inverse(currentOrientation), dq)[:3] * self.positionP
            return outputVel        
        else:
            return np.zeros(3)

    def computeCorrectiveAcceleration(self, odom, correctiveVelocity):

        if self.targetVelocity is not None:            
            currentVelocity = msgToNumpy(odom.twist.twist.angular)
            outputAccel = ((self.targetVelocity + correctiveVelocity) - currentVelocity) * self.velocityP
            return outputAccel       
        else:
            return np.zeros(3)

class AccelerationCalculator:
    def __init__(self, config):
        self.mass = np.array(config["mass"])
        self.com = np.array(config["com"])
        self.inertia = np.array(config["inertia"])
        self.linearDrag = np.array(config["linear_damping"])
        self.quadraticDrag = np.array(config["quadratic_damping"])
        self.volume = np.array(config["volume"])
        self.cob = np.array(config["cob"])
        self.gravity = 9.8
        self.density = 1000

    def accelToNetForce(self, odom, linearAccel, angularAccel):
        """ 
        Converts vehicle acceleration into required net force.
    
        Will take the required acceleration and consider mass, buoyance, drag, and precession to compute the required net force.
    
        Parameters:
        odom (Odometry): The latest odometry message.
        linearAccel (np.array): The linear body-frame acceleration.
        angularAccel (np.array): The angular body-frame acceleration.

        Returns: 
        np.array: 3 dimensional vector representing net body-frame force.
        np.array: 3 dimensional vector representing net body-frame torque.

        """

        linearVelo = msgToNumpy(odom.twist.twist.linear)
        angularVelo = msgToNumpy(odom.twist.twist.angular)
        orientation = msgToNumpy(odom.pose.pose.orientation)

        # Force & Torque Initialization
        netForce = linearAccel * self.mass
        netTorque = angularAccel * self.inertia
        
        # Forces and Torques Calculation
        bodyFrameBuoyancy = worldToBody(orientation, np.array([0, 0, self.volume * self.gravity * self.density]))
        buoyancyTorque = np.cross((self.cob-self.com), bodyFrameBuoyancy)
        precessionTorque = -np.cross(angularVelo, (self.inertia * angularVelo))
        dragForce = self.linearDrag[:3] * linearVelo + self.quadraticDrag[:3] * abs(linearVelo) * linearVelo
        dragTorque = self.linearDrag[3:] * angularVelo + self.quadraticDrag[3:] * abs(angularVelo) * angularVelo
        gravityForce = worldToBody(orientation, np.array([0, 0, - self.gravity * self.mass])) 
        # Net Calculation
        netForce = netForce - bodyFrameBuoyancy - dragForce - gravityForce
        netTorque = netTorque - buoyancyTorque - precessionTorque - dragTorque

        return netForce, netTorque

    

class ControllerNode:

    def __init__(self):
        config_path = rospy.get_param("~vehicle_config")
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)

        self.linearController = LinearCascadedPController()
        self.angularController = AngularCascadedPController()
        self.accelerationCalculator = AccelerationCalculator(config)

        self.angularController.positionP = np.array([100, 100, 100])

        rospy.Subscriber("odometry/filtered", Odometry, self.updateState)
        rospy.Subscriber("orientation", Quaternion, self.angularController.setTargetPosition)
        rospy.Subscriber("angular_velocity", Vector3, self.angularController.setTargetVelocity)
        rospy.Subscriber("disable_angular", Empty, self.angularController.disable)
        rospy.Subscriber("position", Vector3, self.linearController.setTargetPosition)
        rospy.Subscriber("linear_velocity", Vector3, self.linearController.setTargetVelocity)
        rospy.Subscriber("disable_linear", Empty, self.linearController.disable)
        rospy.Subscriber("off", Empty, self.turnOff)
        self.forcePub = rospy.Publisher("net_force", Twist, queue_size=5)

        self.lastTorque = None
        self.lastForce = None
        self.off = True

    def updateState(self, odomMsg):
        linearAccel = self.linearController.update(odomMsg)
        angularAccel = self.angularController.update(odomMsg)

        if np.linalg.norm(linearAccel) > 0 or np.linalg.norm(angularAccel) > 0:
            self.off = False

        if not self.off:
            netForce, netTorque = self.accelerationCalculator.accelToNetForce(odomMsg, linearAccel, angularAccel)
        else:
            netForce, netTorque = np.zeros(3), np.zeros(3)

        if not np.array_equal(self.lastTorque, netTorque) or \
           not np.array_equal(self.lastForce, netForce):

            self.forcePub.publish(Twist(Vector3(*netForce), Vector3(*netTorque)))
            self.lastForce = netForce
            self.lastTorque = netTorque

    def turnOff(self, msg=None):
        self.angularController.disable()
        self.linearController.disable()
        self.off = True


if __name__ == '__main__':
    rospy.init_node("controller")
    controller = ControllerNode()
    rospy.spin()