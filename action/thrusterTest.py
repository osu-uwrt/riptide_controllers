#!/usr/bin/env python

import rospy
import actionlib

from std_msgs.msg import Header, Float32MultiArray
from riptide_controllers.msg import ThrusterTestAction

import yaml
import numpy as np

class ThrusterTest(object):
    THRUSTER_PERCENT = 0.05

    def __init__(self):
        self.thrust_pub = rospy.Publisher("thruster_forces", Float32MultiArray, queue_size=5)

        # Get the mass and COM
        with open(rospy.get_param('~vehicle_config'), 'r') as stream:
            vehicle = yaml.safe_load(stream)
            self.num_thrusters = len(vehicle["thrusters"])
            self.max_force = vehicle["thruster"]["max_force"]

        self._as = actionlib.SimpleActionServer("thruster_test", ThrusterTestAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

    def publish_forces(self, forces):
        msg = Float32MultiArray()
        msg.data = forces
        self.thrust_pub.publish(msg)

    def execute_cb(self, goal):
        rospy.loginfo("Starting ThrusterTest Action")
        forces = np.zeros(self.num_thrusters)
        while True:
            for i in range(self.num_thrusters):
                if self._as.is_preempt_requested():
                    rospy.loginfo('Preempted ThrusterTest Action')
                    self.publish_forces(np.zeros(self.num_thrusters))
                    self._as.set_preempted()
                    return

                forces[i] = self.max_force * self.THRUSTER_PERCENT
                for _ in range(3000):
                    self.publish_forces(forces)
                    rospy.sleep(0.001)
                forces[i] = 0

            
        
        #should never reach this point in the code
        rospy.loginfo("ThrustTest succeeded")
        self._as.set_succeeded()

if __name__ == '__main__':
    rospy.init_node('thruster_test')
    server = ThrusterTest()
    rospy.spin()