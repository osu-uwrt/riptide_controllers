#!/usr/bin/env python3

import rospy
import actionlib

from std_msgs.msg import Header, Float32MultiArray, Int16MultiArray
from riptide_controllers.msg import ThrusterTestAction

import yaml
import numpy as np

NEUTRAL_PWM = 1500
POSITIVE_PWM = 1550
NEGATIVE_PWM = 1450

class ThrusterTest(object):
    THRUSTER_PERCENT = 0.05

    def __init__(self):
        self.pwm_pub = rospy.Publisher("command/pwm", Int16MultiArray, queue_size=5)

        # Get the mass and COM
        with open(rospy.get_param('~vehicle_config'), 'r') as stream:
            self.vehicle_file = yaml.safe_load(stream)
            self.num_thrusters = len(self.vehicle_file["thrusters"])
            self.max_force = self.vehicle_file["thruster"]["max_force"]

        self._as = actionlib.SimpleActionServer("thruster_test", ThrusterTestAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

    def publish_pwm(self, pwm):
        msg = Int16MultiArray()
        msg.data = list(pwm.astype(int))
        self.pwm_pub.publish(msg)

    def execute_cb(self, goal):
        rospy.loginfo("Starting ThrusterTest Action")
        pwm = np.zeros(self.num_thrusters) + NEUTRAL_PWM
        while True:
            for i in range(self.num_thrusters):
                if self._as.is_preempt_requested():
                    rospy.loginfo('Preempted ThrusterTest Action')
                    self.publish_pwm(np.zeros(self.num_thrusters) + NEUTRAL_PWM)
                    self._as.set_preempted()
                    return

                thruster_type = self.vehicle_file["thrusters"][i]["type"]

                if thruster_type == 0:
                    pwm[i] = POSITIVE_PWM
                else:
                    pwm[i] = NEGATIVE_PWM

                for _ in range(3000):
                    self.publish_pwm(pwm)
                    rospy.sleep(0.001)
                pwm[i] = NEUTRAL_PWM

            
        
        #should never reach this point in the code
        rospy.loginfo("ThrustTest succeeded")
        self._as.set_succeeded()

if __name__ == '__main__':
    rospy.init_node('thruster_test')
    server = ThrusterTest()
    rospy.spin()