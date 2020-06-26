#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3Stamped, Vector3
from tf.transformations import quaternion_multiply, quaternion_inverse, quaternion_slerp
import numpy as np

class QuaternionController:
    # Default goal is identity quaternion
    goal = [0, 0, 0, 1]
    P = 50
    velP = 5

    def __init__(self):
        rospy.Subscriber("command/orientation", Quaternion, self.cmdCb)
        rospy.Subscriber("odometry/filtered", Odometry, self.odomCb)
        self.momentPub = rospy.Publisher("command/moment", Vector3Stamped, queue_size=5)

    def cmdCb(self, msg):
        goal = [msg.x, msg.y, msg.z, msg.w]
        self.goal = goal / np.linalg.norm(goal)
        
    def odomCb(self, msg):
        q = msg.pose.pose.orientation
        q = [q.x, q.y, q.z, q.w]
        qInv = quaternion_inverse(q)

        # Get a nearby quaternion that is in the direction of out goal using SLERP
        # We need this because next step only works for small angles
        intermediateOrientation = quaternion_slerp(q, self.goal, 0.05)

        # Compute dq of our error and convert to angular velocity
        # This uses the dq/dt = .5*q*w equation
        dq = intermediateOrientation - q
        angularVel = quaternion_multiply(qInv, dq)[:3]

        # Apply P constant
        angularVel *= self.P

        # Get current angular velocity
        currentAngVel = msg.twist.twist.angular
        currentAngVel = [currentAngVel.x, currentAngVel.y, currentAngVel.z]

        # Compute acceleration from error in velocity
        errorAngVel = np.array(angularVel) - currentAngVel

        # Apply P constant
        angularAccel = errorAngVel * self.velP

        # Publish acceleration
        outAccel = Vector3Stamped()
        outAccel.vector.x = angularAccel[0]
        outAccel.vector.y = angularAccel[1]
        outAccel.vector.z = angularAccel[2]
        self.momentPub.publish(outAccel)
 

if __name__ == '__main__':
    rospy.init_node("quaternion_controller")
    controller = QuaternionController()
    rospy.spin()