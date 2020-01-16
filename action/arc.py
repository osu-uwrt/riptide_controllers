#! /usr/bin/env python
import rospy
import actionlib

from riptide_msgs.msg import AttitudeCommand, LinearCommand
from sensor_msgs.msg import Imu
from nortek_dvl.msg import Dvl
import riptide_controllers.msg
import math
import numpy as np

from tf.transformations import euler_from_quaternion


def angleDiff(a, b):
    return (a - b + 180) % 360 - 180

class Arc(object):
    P = 1

    def __init__(self):
        self.yawPub = rospy.Publisher(
            "command/yaw", AttitudeCommand, queue_size=5)
        self.YPub = rospy.Publisher(
            "command/y", LinearCommand, queue_size=5)

        self._as = actionlib.SimpleActionServer(
            "arc", riptide_controllers.msg.ArcAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

    def imuToEuler(self, msg):
        quat = msg.orientation
        quat = [quat.x, quat.y, quat.z, quat.w]
        return np.array(euler_from_quaternion(quat)) * 180 / math.pi

    def execute_cb(self, goal):
        rospy.loginfo("Driving in %fm arc"%goal.radius)
        self.lastVel = 0
        self.linearPos = 0
        self.angleTraveled = 0
        self.radius = goal.radius
        self.linearVelocity = -math.pi * goal.velocity / 180 * goal.radius
        self.startAngle = self.imuToEuler(rospy.wait_for_message("imu/data", Imu))[2]

        self.yawPub.publish(goal.velocity, AttitudeCommand.VELOCITY)
        self.YPub.publish(self.linearVelocity, LinearCommand.VELOCITY)

        self.imuSub = rospy.Subscriber("imu/data", Imu, self.imuCb)
        self.dvlSub = rospy.Subscriber("state/dvl", Dvl, self.dvlCb)

        while (self.angleTraveled < goal.angle and goal.velocity > 0) or (self.angleTraveled > goal.angle and goal.velocity < 0):
            rospy.sleep(0.1)

            if self._as.is_preempt_requested():
                rospy.loginfo('Preempted Arc Action')
                self.cleanup()
                self._as.set_preempted()
                return

        self.cleanup()

        self._as.set_succeeded()

    def cleanup(self):
        self.imuSub.unregister()
        self.dvlSub.unregister()
        self.yawPub.publish(0, AttitudeCommand.VELOCITY)
        self.YPub.publish(0, LinearCommand.VELOCITY)


    def imuCb(self, msg):
        euler = self.imuToEuler(msg)
        self.angleTraveled = angleDiff(euler[2], self.startAngle)

    def dvlCb(self, msg):
        if not math.isnan(msg.velocity.x):
            curVel = msg.velocity.y
        else:
            curVel = self.lastVel
        self.linearPos += (self.lastVel + curVel) / 2 / 8 # / 8 because this message comes in at 8 Hz
        self.lastVel = curVel
        targetPos = -math.pi * self.angleTraveled / 180 * self.radius

        self.YPub.publish(self.linearVelocity + self.P * (targetPos - self.linearPos), LinearCommand.VELOCITY)


if __name__ == '__main__':
    rospy.init_node('arc')
    server = Arc()
    rospy.spin()
