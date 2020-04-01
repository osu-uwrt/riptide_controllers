#!/usr/bin/env python

import rospy
import actionlib

from riptide_msgs.msg import ThrustStamped, Thrust
from std_msgs.msg import Header

import riptide_controllers.msg


class ThrusterTest(object):

    def __init__(self):
        self.thrustPub = rospy.Publisher("command/thrust", ThrustStamped, queue_size=1)
        self._as = actionlib.SimpleActionServer("thruster_test", riptide_controllers.msg.ThrusterTestAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

    def execute_cb(self, goal):
        header = Header()
        rospy.loginfo("Starting ThrusterTest Action")
        forces = [0,0,0,0,0,0,0,0]
        while True:
            for i in range(8):
                if self._as.is_preempt_requested():
                    rospy.loginfo('Preempted ThrusterTest Action')
                    self.thrustPub.publish(header, Thrust(0,0,0,0,0,0,0,0))
                    self._as.set_preempted()
                    return
                forces[i] = 7
                for a in range(3000):
                    self.thrustPub.publish(header, Thrust(*forces))
                    rospy.sleep(0.001)
                forces[i] = 0

            
        
            #should never reach this point in the code
        rospy.loginfo("ThrustTest succeeded")
        self._as.set_succeeded()

if __name__ == '__main__':
    rospy.init_node('thruster_test')
    server = ThrusterTest()
    rospy.spin()