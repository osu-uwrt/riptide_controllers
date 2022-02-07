import numpy as np
import transforms3d as tf3d
from abc import ABC, abstractmethod

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
    orientationInv = tf3d.quaternions.qinverse(orientation)
    newVector = tf3d.quaternions.qmult(orientationInv, tf3d.quaternions.qmult(vector, orientation))
    return newVector[:3]

def applyMax(vector, max_vector):
    """ 
    Scales a vector to obey maximums

    Parameters:
    vector (np.array): The unscaled vector
    max_vector (np.array): The maximum values for each element

    Returns: 
    np.array: Vector that obeys the maximums

    """

    scale = 1
    for i in range(len(vector)):
        if abs(vector[i]) > max_vector[i]:
            element_scale = max_vector[i] / abs(vector[i])
            if element_scale < scale:
                scale = element_scale

    return vector * scale


class CascadedPController(ABC):

    def __init__(self):
        self.targetPosition = None
        self.targetVelocity = None
        self.targetAcceleration = None
        self.positionP = np.zeros(3)
        self.velocityP = np.zeros(3)
        self.maxVelocity = np.zeros(3)
        self.maxAccel = np.zeros(3)
        self.steadyVelThresh = 0.00001
        self.steadyAccelThresh = 0.00001
        self.steady = True

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

        netAccel = applyMax(netAccel, self.maxAccel)
        
        if self.targetAcceleration is not None:                         # Trajectory mode
            self.steady = False
        elif self.targetPosition is not None:                           # Position mode
            self.steady = np.allclose(correctiveVelocity, np.zeros(3), atol=self.steadyVelThresh)
        elif self.targetVelocity is not None:                           # Velocity mode
            self.steady = np.allclose(netAccel, np.zeros(3), atol=self.steadyAccelThresh)
        else:
            self.steady = True

        return netAccel

class LinearCascadedPController(CascadedPController):

    def __init__(self):
        super(LinearCascadedPController, self).__init__()
        self.steadyVelThresh = 0.02
        self.steadyAccelThresh = 0.02

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
            targetVelocity = self.targetVelocity + correctiveVelocity
            targetVelocity = applyMax(targetVelocity, self.maxVelocity)
            currentVelocity = msgToNumpy(odom.twist.twist.linear)
            outputAccel = (targetVelocity - currentVelocity) * self.velocityP
            return outputAccel       
        else:
            return np.zeros(3)

class AngularCascadedPController(CascadedPController):

    def __init__(self):
        super(AngularCascadedPController, self).__init__()
        self.steadyVelThresh = 0.1
        self.steadyAccelThresh = 0.1

    def slerp(one, two, t):
        """Spherical Linear intERPolation."""
        return tf3d.quaternions.qmult(tf3d.quaternions.qmult(two, tf3d.quaternions.qinverse(one))**t, one)



    def computeCorrectiveVelocity(self, odom):

        if self.targetPosition is not None:
            currentOrientation = msgToNumpy(odom.pose.pose.orientation)

            # Below code only works for small angles so lets find an orientation in the right direction but with a small angle
            intermediateOrientation = self.slerp(currentOrientation, self.targetPosition, 0.01)
            dq = (intermediateOrientation - currentOrientation)
            outputVel = tf3d.quaternions.qmult(tf3d.quaternions.qinverse(currentOrientation), dq)[:3] * self.positionP
            return outputVel        
        else:
            return np.zeros(3)

    def computeCorrectiveAcceleration(self, odom, correctiveVelocity):

        if self.targetVelocity is not None:   
            targetVelocity = self.targetVelocity + correctiveVelocity
            for i in range(3):
                if abs(targetVelocity[i]) > self.maxVelocity[i]:
                    targetVelocity[i] = self.maxVelocity[i] * targetVelocity[i] / abs(targetVelocity[i])           
            currentVelocity = msgToNumpy(odom.twist.twist.angular)
            outputAccel = (targetVelocity - currentVelocity) * self.velocityP
            return outputAccel       
        else:
            return np.zeros(3)

class AccelerationCalculator:
    def __init__(self, config):
        self.mass = np.array(config["mass"])
        self.com = np.array(config["com"])
        self.inertia = np.array(config["inertia"])
        self.linearDrag = np.zeros(6)
        self.quadraticDrag = np.zeros(6)
        self.volume = np.array(config["volume"])
        self.cob = np.zeros(3)
        self.gravity = 9.8 # (m/sec^2)
        self.density = 1000 # density of water (kg/m^3)
        self.buoyancy = np.zeros(3)

    def accelToNetForce(self, odom, linearAccel, angularAccel):
        """ 
        Converts vehicle acceleration into required net force.
    
        Will take the required acceleration and consider mass, buoyancy, drag, and precession to compute the required net force.
    
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
        bodyFrameBuoyancy = worldToBody(orientation, self.buoyancy)
        buoyancyTorque = np.cross((self.cob-self.com), bodyFrameBuoyancy)
        precessionTorque = -np.cross(angularVelo, (self.inertia * angularVelo))
        dragForce = self.linearDrag[:3] * linearVelo + self.quadraticDrag[:3] * abs(linearVelo) * linearVelo
        dragTorque = self.linearDrag[3:] * angularVelo + self.quadraticDrag[3:] * abs(angularVelo) * angularVelo
        gravityForce = worldToBody(orientation, np.array([0, 0, - self.gravity * self.mass]))
                
        # Net Calculation
        netForce = netForce - bodyFrameBuoyancy - dragForce - gravityForce
        netTorque = netTorque - buoyancyTorque - precessionTorque - dragTorque

        return netForce, netTorque
