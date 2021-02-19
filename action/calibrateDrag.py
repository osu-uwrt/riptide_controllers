#! /usr/bin/env python
import rospy
import actionlib
import yaml

import dynamic_reconfigure.client
import riptide_controllers.msg
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Header
from geometry_msgs.msg import Quaternion, Vector3Stamped, Vector3, Twist
from dynamic_reconfigure.server import Server
from dynamic_reconfigure.client import Client

from tf.transformations import euler_from_quaternion, quaternion_multiply, quaternion_inverse, quaternion_from_euler
from math import sin, cos, tan, acos, pi
import numpy as np

# Z axis is a little off, load interia from puddles.yaml, and update dynamic reconfigure


class CalibrateDragAction(object):

    _result = riptide_controllers.msg.CalibrateDragResult()

    def __init__(self):
        self.orientation_pub = rospy.Publisher("orientation", Quaternion, queue_size=5)
        self.lin_vel_pub = rospy.Publisher("linear_velocity", Vector3, queue_size=5)
        self.ang_vel_pub = rospy.Publisher("angular_velocity", Vector3, queue_size=5)

        # Get the mass and COM
        with open(rospy.get_param('~vehicle_config'), 'r') as stream:
            vehicle = yaml.safe_load(stream)
            mass = vehicle['mass']
            rotational_inertia = np.array(vehicle['inertia'])
            self.inertia = np.array([mass, mass, mass, rotational_inertia[0], rotational_inertia[1], rotational_inertia[2]])

        self._as = actionlib.SimpleActionServer("calibrate_drag", riptide_controllers.msg.CalibrateDragAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()
        self.client = Client("controller", timeout=30)
    
    # Example use: r, p, y = get_euler(odom)
    # RPY in radians
    def get_euler(self, odom_msg):
        quat = odom_msg.pose.pose.orientation
        quat = [quat.x, quat.y, quat.z, quat.w]
        return euler_from_quaternion(quat)

    def restrict_angle(self, angle):
        return ((angle + 180) % 360 ) - 180

    # Roll, Pitch, and Yaw configuration
    def to_orientation(self, r, p, y):
        r *= pi / 180
        p *= pi / 180
        y *= pi / 180
        quat = quaternion_from_euler(r, p, y)
        self.orientation_pub.publish(Quaternion(*quat))
        rospy.sleep(5)


    # Apply force on corresponding axes and record velocities
    def collect_data(self, axis, velocity):
        publish_velocity = [
            lambda x: self.lin_vel_pub.publish(x, 0, 0),
            lambda y: self.lin_vel_pub.publish(0, y, 0),
            lambda z: self.lin_vel_pub.publish(0, 0, z),
            lambda x: self.ang_vel_pub.publish(x, 0, 0),
            lambda y: self.ang_vel_pub.publish(0, y, 0),
            lambda z: self.ang_vel_pub.publish(0, 0, z)
        ]

        get_twist = [
            lambda odom: odom.twist.twist.linear.x,
            lambda odom: odom.twist.twist.linear.y,
            lambda odom: odom.twist.twist.linear.z,
            lambda odom: odom.twist.twist.angular.x,
            lambda odom: odom.twist.twist.angular.y,
            lambda odom: odom.twist.twist.angular.z
        ]

        get_accel = [
            lambda twist: twist.linear.x,
            lambda twist: twist.linear.y,
            lambda twist: twist.linear.z,
            lambda twist: twist.angular.x,
            lambda twist: twist.angular.y,
            lambda twist: twist.angular.z
        ]

        
        publish_velocity[axis](velocity)

        rospy.sleep(1)
        last_vel = 0
        stable_measurements = 0
        while stable_measurements < 10:
            odom_msg = rospy.wait_for_message("odometry/filtered", Odometry)
            cur_vel = get_twist[axis](odom_msg)

            if abs((cur_vel - last_vel) / cur_vel) >= 0.1:
                stable_measurements = 0
            else:
                stable_measurements += 1
            last_vel = cur_vel
            rospy.sleep(0.1)

        velocities = []
        for _ in range (10):
            odom_msg = rospy.wait_for_message("odometry/filtered", Odometry)
            velocities.append(get_twist[axis](odom_msg))
            rospy.sleep(0.1)

        forces = []
        for _ in range (10):
            accel_msg = rospy.wait_for_message("controller/requested_accel", Twist)
            forces.append(get_accel[axis](accel_msg) * self.inertia[axis])
            rospy.sleep(0.1)

        publish_velocity[axis](0)
        rospy.sleep(0.1)

        return np.average(velocities), -np.average(forces)
    
    # Calcualte the multivariable linear regression of linear and quadratic damping
    def calculate_parameters(self, velocities, forces):
        y = np.array(forces)
        X = np.array([velocities, np.abs(velocities) * np.array(velocities)])
        X = X.T # transpose so input vectors are along the rows
        return np.linalg.lstsq(X,y)[0]

    # Set depth in rqt before running action
    def execute_cb(self, goal):
        self.client.update_configuration({
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
            "quadratic_rot_z": 0
        })

        # Initialize starting position of robot
        odom_msg = rospy.wait_for_message("odometry/filtered", Odometry)
        startY = self.get_euler(odom_msg)[2] * 180 / pi
        rospy.loginfo("Starting drag calibration")

        linear_params = np.zeros(6)
        quadratic_params = np.zeros(6)

        # Euler for ease of use
        axes_test_orientations = [
            [0, 0, startY],
            [0, 0, self.restrict_angle(startY - 90)],
            [0, -85, startY],
            [0, -85, startY],
            [90, 0, startY],
            [0, 0, startY]
        ]

        axis_velocities = [
            [0.05, -0.05, .1, -.1, .2, -.2],
            [0.05, -0.05, .1, -.1, .2, -.2],
            [-0.05, 0.05, -.1, .1, -.2, .2],
            [0.2, -0.2, 0.5, -0.5, 1.2, -1.2],
            [0.2, -0.2, 0.5, -0.5, 1.2, -1.2],
            [0.2, -0.2, 0.5, -0.5, 1.2, -1.2],
        ]

        for axis in range(6):
            self.to_orientation(*axes_test_orientations[axis])

            forces = []
            velocities = []
            for requested_velocity in axis_velocities[axis]:
                velocity, force = self.collect_data(axis, requested_velocity)
                forces.append(force)
                velocities.append(velocity)

            linear_params[axis], quadratic_params[axis] = self.calculate_parameters(velocities, forces)
            rospy.loginfo("Linear: %f" % linear_params[axis])
            rospy.loginfo("Quadratic: %f" % quadratic_params[axis])

        self.to_orientation(0, 0, startY)

        rospy.loginfo("Drag calibration completed.")

        self.client.update_configuration({
            "linear_x": linear_params[0],
            "linear_y": linear_params[1],
            "linear_z": linear_params[2],
            "linear_rot_x": linear_params[3],
            "linear_rot_y": linear_params[4],
            "linear_rot_z": linear_params[5],
            "quadratic_x": quadratic_params[0],
            "quadratic_y": quadratic_params[1],
            "quadratic_z": quadratic_params[2],
            "quadratic_rot_x": quadratic_params[3],
            "quadratic_rot_y": quadratic_params[4],
            "quadratic_rot_z": quadratic_params[5]
        })

        self._result.linear_drag = linear_params
        self._result.quadratic_drag = quadratic_params
        self._as.set_succeeded(self._result)
  
        
if __name__ == '__main__':
    rospy.init_node('calibrate_drag')
    server = CalibrateDragAction()
    rospy.spin()
