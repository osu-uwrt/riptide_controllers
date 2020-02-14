#! /usr/bin/env python
import rospy
import actionlib
import dynamic_reconfigure.client

from riptide_msgs.msg import DepthCommand, AttitudeCommand, Constants
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32
import riptide_controllers.msg

import time
import math
import yaml

WATER_DENSITY = 1000
GRAVITY = 9.81

class CalibrateAction(object):

    def __init__(self):
        self.depthPub = rospy.Publisher("command/depth", DepthCommand, queue_size=1)
        self.rollPub = rospy.Publisher("command/roll", AttitudeCommand, queue_size=1)
        self.pitchPub = rospy.Publisher("command/pitch", AttitudeCommand, queue_size=1)
        
        self._as = actionlib.SimpleActionServer("calibrate", riptide_controllers.msg.CalibrateAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

    def depthAction(self, depth):
        client = actionlib.SimpleActionClient(
            "go_to_depth", riptide_controllers.msg.GoToDepthAction)
        client.wait_for_server()

        # Sends the goal to the action server.
        client.send_goal(riptide_controllers.msg.GoToDepthGoal(depth))
        return client
      
    def execute_cb(self, goal):
        client = dynamic_reconfigure.client.Client("thruster_controller", timeout=30)

        with open(rospy.get_param('vehicle_file'), 'r') as stream:
            vehicle = yaml.safe_load(stream)
            mass = vehicle['mass']
            com = vehicle['com']
            

        rospy.loginfo("Starting calibration")

        volume = mass / WATER_DENSITY
        cobX = com[0]
        cobY = com[1]
        cobZ = com[2]

        client.update_configuration({"Volume": volume, "Buoyancy_X_POS": cobX, "Buoyancy_Y_POS": cobY, "Buoyancy_Z_POS": cobZ})

        self.depthPub.publish(True, -1.5)
        self.rollPub.publish(0, AttitudeCommand.POSITION)
        self.pitchPub.publish(0, AttitudeCommand.POSITION)

        rospy.sleep(3)

        # Recalibrate 10 times
        for _ in range(8):
            rospy.sleep(3)

            # Average 10 samples
            volumeAverage = 0
            for _ in range(10):
                forceMsg = rospy.wait_for_message("command/force_depth", Vector3Stamped).vector
                force = math.sqrt(forceMsg.x**2 + forceMsg.y**2 + forceMsg.z**2)
                if forceMsg.z < 0:
                    force *= -1

                volumeAdjust = force / GRAVITY / WATER_DENSITY
                volumeAverage += volumeAdjust / 10

            # Adjust in the right direction
            volume -= volumeAverage * 0.8
            client.update_configuration({"Volume": volume, "Buoyancy_X_POS": cobX, "Buoyancy_Y_POS": cobY, "Buoyancy_Z_POS": cobZ})
            if self._as.is_preempt_requested():
                rospy.loginfo('Preempted Calibration')
                self.cleanup()
                self._as.set_preempted()
                return

        Fb = volume * GRAVITY * WATER_DENSITY
        rospy.loginfo("Buoyant force calibration complete")

        for _ in range(8):
            rospy.sleep(3)

            cobYAverage = 0
            cobXAverage = 0
            for _ in range(10):
                momentMsg = rospy.wait_for_message("command/moment", Vector3Stamped).vector
                cobYAverage += momentMsg.x / Fb * 0.1
                cobXAverage += momentMsg.y / Fb * 0.1

            cobY -= cobYAverage
            cobX += cobXAverage

            client.update_configuration({"Volume": volume, "Buoyancy_X_POS": cobX, "Buoyancy_Y_POS": cobY, "Buoyancy_Z_POS": cobZ})

            if self._as.is_preempt_requested():
                rospy.loginfo('Preempted Calibration')
                self.cleanup()
                self._as.set_preempted()
                return

        rospy.loginfo("Buoyancy XY calibration complete")

        self.rollPub.publish(45, AttitudeCommand.POSITION)

        rospy.sleep(3)

        for i in range(8):
            rospy.sleep(3)

            cobZAverage = 0
            for _ in range(10):
                momentMsg = rospy.wait_for_message("command/moment", Vector3Stamped).vector
                cobZAverage += momentMsg.x / Fb * 0.1

            cobZ += cobZAverage

            client.update_configuration({"Volume": volume, "Buoyancy_X_POS": cobX, "Buoyancy_Y_POS": cobY, "Buoyancy_Z_POS": cobZ})

            if self._as.is_preempt_requested():
                rospy.loginfo('Preempted Calibration')
                self.cleanup()
                self._as.set_preempted()
                return


            
        rospy.loginfo("Calibration complete")

        self.cleanup()
        
        self._as.set_succeeded()

    def cleanup(self):
        self.rollPub.publish(0, AttitudeCommand.POSITION)
        self.pitchPub.publish(0, AttitudeCommand.POSITION)
        self.depthAction(0).wait_for_result()
        self.depthPub.publish(False, 0)
        self.rollPub.publish(0, AttitudeCommand.MOMENT)
        self.pitchPub.publish(0, AttitudeCommand.MOMENT)


        
        
if __name__ == '__main__':
    rospy.init_node('calibrate')
    server = CalibrateAction()
    rospy.spin()
