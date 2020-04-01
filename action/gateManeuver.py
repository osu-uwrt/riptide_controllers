#! /usr/bin/env python
import rospy
import actionlib
import dynamic_reconfigure.client

from riptide_msgs.msg import AttitudeCommand, LinearCommand
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float64, Int32
import riptide_controllers.msg

from tf.transformations import euler_from_quaternion
import time
import math
import numpy as np


def angleDiff(a, b):
    return ((a-b+180) % 360)-180


class GateManeuver(object):

    ROLL_P = 2
    CRUISE_VELOCITY = 90
    DRIVE_FORCE = 30

    def __init__(self):
        self.rollPub = rospy.Publisher(
            "command/roll", AttitudeCommand, queue_size=5)
        self.yawPub = rospy.Publisher(
            "command/yaw", AttitudeCommand, queue_size=5)
        self.XPub = rospy.Publisher(
            "command/x", LinearCommand, queue_size=5)
        self.YPub = rospy.Publisher(
            "command/y", LinearCommand, queue_size=5)
        self.ZPub = rospy.Publisher(
            "command/force_z", Float64, queue_size=5)

        self._as = actionlib.SimpleActionServer(
            "gate_maneuver", riptide_controllers.msg.GateManeuverAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

    def quatToEuler(self, quat):
        quat = [quat.x, quat.y, quat.z, quat.w]
        return np.array(euler_from_quaternion(quat)) * 180 / math.pi
    
    def execute_cb(self, goal):
        rospy.loginfo("Starting gate maneuver")
        self.lastRoll = 0
        self.rolls = 0
        self.justRolled = False

        self.XPub.publish(self.DRIVE_FORCE, LinearCommand.FORCE)
        self.rollPub.publish(self.CRUISE_VELOCITY, AttitudeCommand.VELOCITY)

        self.odomSub = rospy.Subscriber("odometry/filtered", Odometry, self.odomCb)

        while self.rolls < 2:
            rospy.sleep(0.05)

            if self._as.is_preempt_requested():
                rospy.loginfo('Preempted Gate Maneuver')
                self.cleanup()
                self._as.set_preempted()
                return

        rospy.loginfo("Leveling")

        self.cleanup()

        while abs(self.quatToEuler(rospy.wait_for_message("odometry/filtered", Odometry).pose.pose.orientation)[0]) > 5 and not rospy.is_shutdown():
            rospy.sleep(0.05)

        rospy.loginfo("Done")

        self._as.set_succeeded()

    def cleanup(self):
        self.rollPub.publish(0, AttitudeCommand.POSITION)
        self.odomSub.unregister()
        self.XPub.publish(0, LinearCommand.FORCE)

    def odomCb(self, msg):
        euler = self.quatToEuler(msg.pose.pose.orientation)
        if self.lastRoll < -90 and euler[0] > -90 and not self.justRolled:
            self.rolls += 1
            self.justRolled = True
        if euler[0] > 90:
            self.justRolled = False
        self.lastRoll = euler[0]


if __name__ == '__main__':
    rospy.init_node('gate_maneuver')
    server = GateManeuver()
    rospy.spin()
