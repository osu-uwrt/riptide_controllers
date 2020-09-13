#! /usr/bin/env python
import rospy
import actionlib
import dynamic_reconfigure.client

from geometry_msgs.msg import Vector3, Quaternion, Twist
from std_msgs.msg import Float32, Empty
from nav_msgs.msg import Odometry
import riptide_controllers.msg

from tf.transformations import quaternion_from_euler, euler_from_quaternion

import time
import math
import yaml
import numpy as np

WATER_DENSITY = 1000
GRAVITY = 9.81

def msg_to_numpy(msg):
    if hasattr(msg, "w"):
        return np.array([msg.x, msg.y, msg.z, msg.w])
    return np.array([msg.x, msg.y, msg.z])

class CalibrateBuoyancyAction(object):

    def __init__(self):
        self.position_pub = rospy.Publisher("position", Vector3, queue_size=1)
        self.orientation_pub = rospy.Publisher("orientation", Quaternion, queue_size=1)
        self.off_pub = rospy.Publisher("off", Empty, queue_size=1)

        # Get the mass and COM
        with open(rospy.get_param('vehicle_file'), 'r') as stream:
            vehicle = yaml.safe_load(stream)
            self.mass = vehicle['mass']
            self.com = np.array(vehicle['com'])
            self.inertia = np.array(vehicle['inertia'])
        
        self._as = actionlib.SimpleActionServer("calibrate_buoyancy", riptide_controllers.msg.CalibrateBuoyancyAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

      
    def execute_cb(self, goal):
        self.client = dynamic_reconfigure.client.Client("controller", timeout=30)


        self.initial_config = self.client.get_configuration()
            

        rospy.loginfo("Starting buoyancy calibration")
        buoyant_force = self.mass * GRAVITY
        cob = self.com

        # Reset parameters to default
        self.client.update_configuration({
            "force": buoyant_force, 
            "center_x": cob[0], 
            "center_y": cob[1], 
            "center_z": cob[2],
            "linear_x": 0,
            "linear_y": 0,
            "linear_z": 0,
            "linear_rot_x": 0,
            "linear_rot_y": 0,
            "linear_rot_z": 0,
            "quadratic_x": 0,
            "quadratic_y": 0,
            "quadratic_z": 0,
            "quadratic_rot_x": 0,
            "quadratic_rot_y": 0,
            "quadratic_rot_z": 0,
        })

        # Submerge
        odom_msg = rospy.wait_for_message("odometry/filtered", Odometry)
        current_position = odom_msg.pose.pose.position
        self.position_pub.publish(Vector3(current_position.x, current_position.y, -1.5))
        self.orientation_pub.publish(Quaternion(0, 0, 0, 1))

        # Wait for equilibrium
        rospy.sleep(15)

        # Average 10 samples
        force_measurements = []
        for _ in range(10):
            force_msg = mass * msg_to_numpy(rospy.wait_for_message("controller/requested_accel", Twist).linear)
            force = np.linalg.norm(force_msg)
            if force_msg[2] < 0:
                force *= -1

            rospy.sleep(0.2)
            force_measurements.append(force)

        # Adjust buoyant force
        buoyant_force -= np.average(force_measurements)
        self.client.update_configuration({"force": buoyant_force})

        if self.check_preempted():
            return

        rospy.loginfo("Buoyant force calibration complete")

        # Adjust COB until converged
        last_adjustment_x = 0
        last_adjustment_y = 0
        converged_x = False
        converged_y = False
        while not converged_x or not converged_y:
            rospy.sleep(3)

            accel = msg_to_numpy(rospy.wait_for_message("controller/requested_accel", Twist).angular)
            torque = self.inertia * accel

            adjustment_x = torque[1] / buoyant_force
            adjustment_y = torque[0] / buoyant_force
            cob[0] += adjustment_x
            cob[1] -= adjustment_y

            self.client.update_configuration({"center_x": cob[0], "center_y": cob[1]})

            # If the direction of adjustment changed, stop
            if last_adjustment_x * adjustment_x < 0:
                converged_x = True
            if last_adjustment_y * adjustment_y < 0:
                converged_y = True
            last_adjustment_x = adjustment_x
            last_adjustment_y = adjustment_y

            if self.check_preempted():
                return

        rospy.loginfo("Buoyancy XY calibration complete")

        # Adjust orientation
        current_orientation = msg_to_numpy(rospy.wait_for_message("odometry/filtered", Odometry).pose.pose.orientation)
        _, _, y = euler_from_quaternion(current_orientation)
        self.orientation_pub.publish(Quaternion(*quaternion_from_euler(0, -math.pi / 4, y)))

        rospy.sleep(3)

        # Adjust COB until converged
        last_adjustment = 0
        converged = False
        while not converged:
            rospy.sleep(3)

            accel = msg_to_numpy(rospy.wait_for_message("controller/requested_accel", Twist).angular)
            torque = inertia * accel

            adjustment_z = torque[1] / buoyant_force / 2**.5
            cob[2] -= adjustment_z

            self.client.update_configuration({"center_z": cob[2]})

            # If the direction of adjustment changed, stop
            if last_adjustment * adjustment_z < 0:
                converged = True
            last_adjustment = adjustment_z

            
            if self.check_preempted():
                return

        rospy.loginfo("Calibration complete")
        self.cleanup()
        self._as.set_succeeded(buoyant_force, cob)

    def check_preempted(self):
        if self._as.is_preempt_requested():
            rospy.loginfo('Preempted Calibration')
            self.cleanup()
            self._as.set_preempted()
            return True

    def cleanup(self):
        self.client.update_configuration({
            "linear_x": self.initial_config['linear_x'],
            "linear_y": self.initial_config['linear_y'],
            "linear_z": self.initial_config['linear_z'],
            "linear_rot_x": self.initial_config['linear_rot_x'],
            "linear_rot_y": self.initial_config['linear_rot_y'],
            "linear_rot_z": self.initial_config['linear_rot_z'],
            "quadratic_x": self.initial_config['quadratic_x'],
            "quadratic_y": self.initial_config['quadratic_y'],
            "quadratic_z": self.initial_config['quadratic_z'],
            "quadratic_rot_x": self.initial_config['quadratic_rot_x'],
            "quadratic_rot_y": self.initial_config['quadratic_rot_y'],
            "quadratic_rot_z": self.initial_config['quadratic_rot_z'],
        })
        self.off_pub.publish()


        
        
if __name__ == '__main__':
    rospy.init_node('calibrate_buoyancy')
    server = CalibrateBuoyancyAction()
    rospy.spin()
