#! /usr/bin/env python3

# Determines the buoyancy parameter of the robot.
# Assumes the robot is upright

import rospy
import actionlib
import dynamic_reconfigure.client

from geometry_msgs.msg import Vector3, Quaternion, Twist
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
import riptide_controllers.msg

from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_inverse, quaternion_multiply

import math
import yaml
import numpy as np

WATER_DENSITY = 1000
GRAVITY = 9.81

def msg_to_numpy(msg):
    """Converts a Vector3 or Quaternion message to its numpy counterpart"""
    if hasattr(msg, "w"):
        return np.array([msg.x, msg.y, msg.z, msg.w])
    return np.array([msg.x, msg.y, msg.z])

def changeFrame(orientation, vector, w2b = True):
    """Converts vector into other frame from orientation quaternion. The w2b parameter will
     determine if the vector is converting from world to body or body to world"""

    vector = np.append(vector, 0)
    if w2b:
        orientation = quaternion_inverse(orientation)
    orientationInv = quaternion_inverse(orientation)
    newVector = quaternion_multiply(orientation, quaternion_multiply(vector, orientationInv))
    return newVector[:3]


class CalibrateBuoyancyAction(object):

    _result = riptide_controllers.msg.CalibrateBuoyancyResult()

    def __init__(self):
        self.position_pub = rospy.Publisher("position", Vector3, queue_size=1)
        self.orientation_pub = rospy.Publisher("orientation", Quaternion, queue_size=1)
        self.off_pub = rospy.Publisher("off", Empty, queue_size=1)

        # Get the mass and COM
        with open(rospy.get_param('~vehicle_config'), 'r') as stream:
            vehicle = yaml.safe_load(stream)
            self.mass = vehicle['mass']
            self.com = np.array(vehicle['com'])
            self.inertia = np.array(vehicle['inertia'])
        
        self._as = actionlib.SimpleActionServer("calibrate_buoyancy", riptide_controllers.msg.CalibrateBuoyancyAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

    def tune(self, initial_value, get_adjustment, apply_change, num_samples=10, delay=4):
        """Tunes a parameter of the robot"""
        current_value = np.array(initial_value)
        last_adjustment = np.zeros_like(current_value)
        converged = np.zeros_like(current_value)

        while not np.all(converged):
            # Wait for equilibrium
            rospy.sleep(delay)

            # Average a few samples
            average_adjustment = 0
            for _ in range(num_samples):
                average_adjustment += get_adjustment() / num_samples
                rospy.sleep(0.2)

            # Apply change
            current_value += average_adjustment
            apply_change(current_value)

            # Check if the value has converged
            converged = np.logical_or(converged, average_adjustment * last_adjustment < 0)
            last_adjustment = average_adjustment

            if self.check_preempted():
                return

        return current_value

      
    def execute_cb(self, goal):
        # Start reconfigure server and get starting config
        self.client = dynamic_reconfigure.client.Client("controller", timeout=30)
        self.initial_config = self.client.get_configuration()
            
        # Set variables to defaults
        rospy.loginfo("Starting buoyancy calibration")
        volume = self.mass / WATER_DENSITY
        cob = np.copy(self.com)

        # Reset parameters to default
        self.client.update_configuration({
            "volume": volume, 
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
        current_position = msg_to_numpy(odom_msg.pose.pose.position)
        current_orientation = msg_to_numpy(odom_msg.pose.pose.orientation)
        self.position_pub.publish(Vector3(current_position[0], current_position[1], -1))
        _, _, y = euler_from_quaternion(current_orientation)
        self.orientation_pub.publish(Quaternion(*quaternion_from_euler(0, 0, y)))

        # Wait for equilibrium
        rospy.sleep(10)

        # Volume adjustment function
        def get_volume_adjustment():
            body_force = self.mass * msg_to_numpy(rospy.wait_for_message("controller/requested_accel", Twist).linear)
            orientation = msg_to_numpy(rospy.wait_for_message("odometry/filtered", Odometry).pose.pose.orientation)
            world_z_force = changeFrame(orientation, body_force, w2b=False)[2]

            return -world_z_force / WATER_DENSITY / GRAVITY

        # Tune volume
        volume = self.tune(
            volume, 
            get_volume_adjustment, 
            lambda v: self.client.update_configuration({"volume": v})
        )

        rospy.loginfo("Volume calibration complete")
        buoyant_force = volume * WATER_DENSITY * GRAVITY

        # COB adjustment function
        def get_cob_adjustment():
            accel = msg_to_numpy(rospy.wait_for_message("controller/requested_accel", Twist).angular)
            torque = self.inertia * accel
            orientation = msg_to_numpy(rospy.wait_for_message("odometry/filtered", Odometry).pose.pose.orientation)
            body_force_z = changeFrame(orientation, np.array([0, 0, buoyant_force]))[2]

            adjustment_x = torque[1] / body_force_z
            adjustment_y = -torque[0] / body_force_z

            return np.array([adjustment_x, adjustment_y])

        # Tune X and Y COB
        self.tune(
            cob[:2], 
            get_cob_adjustment, 
            lambda cob: self.client.update_configuration({"center_x": cob[0], "center_y": cob[1]}),
            num_samples = 2,
            delay = 1
        )

        rospy.loginfo("Buoyancy XY calibration complete")

        # Adjust orientation
        self.orientation_pub.publish(Quaternion(*quaternion_from_euler(0, -math.pi / 4, y)))
        rospy.sleep(3)

        # Z COB function
        def get_cob_z_adjustment():
            accel = msg_to_numpy(rospy.wait_for_message("controller/requested_accel", Twist).angular)
            torque = self.inertia * accel
            orientation = msg_to_numpy(rospy.wait_for_message("odometry/filtered", Odometry).pose.pose.orientation)
            body_force_x = changeFrame(orientation, np.array([0, 0, buoyant_force]))[0]

            adjustment = -torque[1] / body_force_x

            return adjustment

        # Tune Z COB
        self.tune(
            cob[2], 
            get_cob_z_adjustment, 
            lambda z: self.client.update_configuration({"center_z": z})
        )

        rospy.loginfo("Calibration complete")
        self.cleanup()
        self._result.buoyant_force = buoyant_force
        self._result.center_of_buoyancy = cob
        self._as.set_succeeded(self._result)

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
