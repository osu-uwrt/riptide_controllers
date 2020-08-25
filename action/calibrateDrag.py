#! /usr/bin/env python
import rospy
import actionlib
import dynamic_reconfigure.client
import riptide_controllers.msg

from riptide_msgs.msg import AttitudeCommand, DepthCommand, LinearCommand
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Header
from geometry_msgs.msg import Quaternion, Vector3Stamped, Vector3, Twist
from dynamic_reconfigure.server import Server
from riptide_controllers.cfg import AttitudeControllerConfig
from tf.transformations import euler_from_quaternion, quaternion_multiply, quaternion_inverse, quaternion_from_euler
from math import sin, cos, tan, acos, pi
import numpy as np


# def rollAction(angle):
#     client = actionlib.SimpleActionClient(
#         "go_to_roll", riptide_controllers.msg.GoToRollAction)
#     client.wait_for_server()

#     client.send_goal(riptide_controllers.msg.GoToRollGoal(angle))
#     return client

# def pitchAction(angle):
#     client = actionlib.SimpleActionClient(
#         "go_to_pitch", riptide_controllers.msg.GoToPitchAction)
#     client.wait_for_server()

#     client.send_goal(riptide_controllers.msg.GoToPitchGoal(angle))
#     return client

# def yawAction(angle):
#     client = actionlib.SimpleActionClient(
#         "go_to_yaw", riptide_controllers.msg.GoToYawAction)
#     client.wait_for_server()

#     client.send_goal(riptide_controllers.msg.GoToYawGoal(angle))
#     return client



class CalibrateDragAction(object):

    def __init__(self):
        self.posePub = rospy.Publisher("position", Vector3, queue_size=5)
        self.orientPub = rospy.Publisher("orientation", Quaternion, queue_size=5)
        self.linPub = rospy.Publisher("linear_velocity", Vector3, queue_size=5)
        self.angPub = rospy.Publisher("angular_velocity", Vector3, queue_size=5)
        self.mass = 39.825 #kg
        self.forces = [[] for i in range(6)]
        self.velocities = [[] for i in range(6)]
        self.avgl = [[] for i in range(6)]    #average linear
        self.avgq = [[] for i in range(6)]    #average quadratic
        self._as = actionlib.SimpleActionServer("calibrate_drag", riptide_controllers.msg.CalibrateDragAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()
    
    # Example use: r, p, y = getEuler(odom)
    # RPY in radians
    def getEuler(self, odomMsg):
        quat = odomMsg.pose.pose.orientation
        quat = [quat.x, quat.y, quat.z, quat.w]
        return euler_from_quaternion(quat)

    def getQuat(self, r, p, y):
        return quaternion_from_euler(r, p, y)

    def restrictAngle(self, angle):
        return ((angle + 180) % 360 ) - 180

    # Roll, Pitch, and Yaw configuration
    def toPose(self, r, p, y, hold=True):
        quat = self.getQuat(r, p, y)
        self.orientPub.publish(quat[0], quat[1], quat[2], quat[3])
        if not hold:
            self.orientPub.publish(0, 0, 0, 0)
            rospy.sleep(0.5)



    # Apply force on corresponding axes and record velocities
    def collectData(self, axis, velo):
        veloPublishers = [
            lambda x: self.linPub.publish(x, 0, 0),
            lambda x: self.linPub.publish(0, x, 0),
            lambda x: self.linPub.publish(0, 0, x),
            lambda x: self.angPub.publish(x, 0, 0),
            lambda x: self.angPub.publish(0, x, 0),
            lambda x: self.angPub.publish(0, 0, x)
        ]

        twistFunctions = [
            lambda odom: odom.twist.twist.linear.x,
            lambda odom: odom.twist.twist.linear.y,
            lambda odom: odom.twist.twist.linear.z,
            lambda odom: odom.twist.twist.angular.x,
            lambda odom: odom.twist.twist.angular.y,
            lambda odom: odom.twist.twist.angular.z
        ]

        accelRead = [
            lambda x: x.linear.x,
            lambda x: x.linear.y,
            lambda x: x.linear.z,
            lambda x: x.angular.x,
            lambda x: x.angular.y,
            lambda x: x.angular.z
        ]

        velocities = []
        veloPublishers[axis](velo)
        rospy.sleep(1)
        odomMsg = rospy.wait_for_message("odometry/filtered", Odometry)
        v0 = 0
        v1 = twistFunctions[axis](odomMsg)
        i = 0
        while i < 10:
            if abs((v1 - v0)/v1) >= 0.02:
                odomMsg = rospy.wait_for_message("odometry/filtered", Odometry)
                v0 = v1
                v1 = twistFunctions[axis](odomMsg)
                i = 0
            else:
                i += 1
            rospy.sleep(0.1)

        for _ in range (10):
            odomMsg = rospy.wait_for_message("odometry/filtered", Odometry)
            velocities.append(twistFunctions[axis](odomMsg))
            rospy.sleep(0.05)

        odomMsg = rospy.wait_for_message("odometry/filtered", Odometry)
        accelMsg = rospy.wait_for_message("controller/requested_accel", Twist)
        self.forces[axis].append(accelRead[axis](accelMsg)*self.mass)
        veloPublishers[axis](0)
        velocity = np.average(velocities)
        self.velocities[axis].append(velocity)
    
    # Calcualte the multivariable linear regression of linear and quadratic damping
    def calculateParameters(self, axis):
        y = np.array(self.forces[axis])
        X = np.array([self.velocities[axis], np.abs(self.velocities[axis]) * np.array(self.velocities[axis])])
        X = X.T # transpose so input vectors are along the rows
        beta_hat = np.linalg.lstsq(X,y)[0]
        self.avgl[axis].append(beta_hat[0])
        self.avgq[axis].append(beta_hat[1])

    # Velocity configuration for each axis
    def testData(self, axis):
        axesVelo = [
            [0.5, -0.5, 1, -1, 2, -2],
            [0.5, -0.5, 1, -1, 2, -2],
            [0.5, -0.5, 1, -1, 2, -2],
            [0.2, -0.2, 0.5, -0.5, 1.2, -1.2],
            [0.2, -0.2, 0.5, -0.5, 1.2, -1.2],
            [0.2, -0.2, 0.5, -0.5, 1.2, -1.2],
        ]
        for velo in axesVelo[axis]:
            self.collectData(axis, velo)

    # Set depth in rqt before running action
    def execute_cb(self, goal):
        # Initialize starting position of robot
        odomMsg = rospy.wait_for_message("odometry/filtered", Odometry)
        startY = self.getEuler(odomMsg)[2] * 180 / pi
        rospy.loginfo("Starting drag calibration")

        # X axis
        self.toPose(0, 0, startY)
        self.testData(0)
        self.calculateParameters(0)

        # Y axis
        newY = self.restrictAngle(startY - 90)
        self.toPose(0, 0, newY)
        self.testData(1)
        self.calculateParameters(1)

        # Z axis
        self.toPose(-90, 0, newY)
        self.testData(2)
        self.calculateParameters(2)

        # X axis
        self.toPose(0, -90, startY, hold=False)
        self.testData(3)
        self.calculateParameters(3)

        # Y axis
        self.toPose(90, 0, startY, hold=False)
        self.testData(4)
        self.calculateParameters(4)

        # Z axis
        self.toPose(0, 0, startY, hold=False)
        self.testData(5)
        self.calculateParameters(5)

        self.toPose(0, 0, startY)

        # Display forces, linear damping, and quadratic damping [[x],[y],[z],[x-rot],[y-rot],[z-rot]]
        rospy.loginfo('Forces: ' + str(self.forces))
        rospy.loginfo('Velocities: ' + str(self.velocities))
        rospy.loginfo('Linear Damping: ' + str(self.avgl))
        rospy.loginfo('Quadratic Damping: ' + str(self.avgq))
        rospy.loginfo("Drag calibration completed.")

        self._as.set_succeeded()
  
        
if __name__ == '__main__':
    rospy.init_node('calibrate_drag')
    server = CalibrateDragAction()
    rospy.spin()
