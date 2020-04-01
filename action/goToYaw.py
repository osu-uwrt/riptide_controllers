#! /usr/bin/env python
import rospy
import actionlib

from riptide_msgs.msg import AttitudeCommand
from nav_msgs.msg import Odometry
import riptide_controllers.msg
from tf.transformations import euler_from_quaternion
import math
import numpy as np

def angleDiff(a1, a2):
    return (a1 - a2 + 180) % 360 - 180

class GoToYawAction(object):

    def __init__(self):
        self.yawPub = rospy.Publisher("command/yaw", AttitudeCommand, queue_size=1)
        self._as = actionlib.SimpleActionServer("go_to_yaw", riptide_controllers.msg.GoToYawAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

    def quatToEuler(self, quat):
        quat = [quat.x, quat.y, quat.z, quat.w]
        return np.array(euler_from_quaternion(quat)) * 180 / math.pi
      
    def execute_cb(self, goal):
        rospy.loginfo("Going to Yaw " + str(goal.yaw)+ " deg")
        self.yawPub.publish(goal.yaw, AttitudeCommand.POSITION)

        while abs(angleDiff(self.quatToEuler(rospy.wait_for_message("odometry/filtered", Odometry).pose.pose.orientation)[2], goal.yaw)) > 3:
            rospy.sleep(0.05)

            if self._as.is_preempt_requested():
                rospy.loginfo('Preempted Go To Yaw')
                self._as.set_preempted()
                return

        rospy.loginfo("At Yaw")
        self._as.set_succeeded()
        
        
if __name__ == '__main__':
    rospy.init_node('go_to_yaw')
    server = GoToYawAction()
    rospy.spin()