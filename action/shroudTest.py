#! /usr/bin/env python3
import rospy
import actionlib

from geometry_msgs.msg import Vector3, Quaternion, Twist
from std_msgs.msg import Float32, Empty
from nav_msgs.msg import Odometry
import riptide_controllers.msg

import time
import math
import yaml
import numpy as np

from tf.transformations import quaternion_from_euler, euler_from_quaternion

def msg_to_numpy(msg):
    if hasattr(msg, "w"):
        return np.array([msg.x, msg.y, msg.z, msg.w])
    return np.array([msg.x, msg.y, msg.z])

class ShroudTestAction(object):

    _result = riptide_controllers.msg.ShroudTestResult()

    def __init__(self):
        self.position_pub = rospy.Publisher("position", Vector3, queue_size=1)
        self.orientation_pub = rospy.Publisher("orientation", Quaternion, queue_size=1)
        self.velocity_pub = rospy.Publisher("linear_velocity", Vector3, queue_size=1)
        self.off_pub = rospy.Publisher("off", Empty, queue_size=1)

        # Get the mass and COM
        with open(rospy.get_param('~vehicle_config'), 'r') as stream:
            vehicle = yaml.safe_load(stream)
            self.mass = vehicle['mass']
            self.linear_drag = vehicle['linear_damping'][0]
            self.quadratic_drag = vehicle['quadratic_damping'][0]
        
        
        self._as = actionlib.SimpleActionServer("shroud_test", riptide_controllers.msg.ShroudTestAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

      
    def execute_cb(self, goal):
        rospy.loginfo("Starting shroud test")
   

        # Submerge
        odom_msg = rospy.wait_for_message("odometry/filtered", Odometry)
        current_position = msg_to_numpy(odom_msg.pose.pose.position)
        current_orientation = msg_to_numpy(odom_msg.pose.pose.orientation)
        self.position_pub.publish(Vector3(current_position[0], current_position[1], -1.5))
        _, _, y = euler_from_quaternion(current_orientation)
        self.orientation_pub.publish(Quaternion(*quaternion_from_euler(0, 0, y)))

        # Wait for equilibrium
        rospy.sleep(15)

        self.velocity_pub.publish(Vector3(0.5, 0, 0))

        rospy.sleep(5)

        accel = rospy.wait_for_message("controller/requested_accel", Twist).linear.x
        vel = rospy.wait_for_message("odometry/filtered", Odometry).twist.twist.linear.x

        # Real forld force on the robot: drag_force
        # Force exerted by robot: (1-p) * (drag_force + correction_force)
        # Solve for p when drag_force = (1-p) * (drag_force + correction_force)

        drag_force = abs(self.linear_drag * vel + self.quadratic_drag * vel ** 2)
        correction_force = self.mass * accel
        p = correction_force / (drag_force + correction_force)

        rospy.loginfo("Drag: %f" % drag_force)
        rospy.loginfo("Correction: %f" % correction_force)
        rospy.loginfo("Percent reduction in force: %0.2f%%" % (p * 100))

        self.off_pub.publish()
        
        self._as.set_succeeded(self._result)


        
        
if __name__ == '__main__':
    rospy.init_node('shroud_test')
    server = ShroudTestAction()
    rospy.spin()
