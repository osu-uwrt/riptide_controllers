#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist, Quaternion, Vector3, TransformStamped, Point32
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import PointCloud
import numpy as np
import yaml
from tf.transformations import euler_matrix, projection_from_matrix, projection_matrix, is_same_transform, quaternion_from_euler
from tf import TransformerROS, TransformListener
from math import pi
from scipy.optimize import minimize


def msg_to_numpy(msg):
    if hasattr(msg, "w"):
        return np.array([msg.x, msg.y, msg.z, msg.w])
    return np.array([msg.x, msg.y, msg.z])

class ThrusterSolverNode:

    def __init__(self):
        rospy.Subscriber("net_force", Twist, self.force_cb)

        self.thruster_pub = rospy.Publisher("thruster_forces", Float32MultiArray, queue_size=5)

        config_path = rospy.get_param("~vehicle_config")
        with open(config_path, 'r') as stream:
            config_file = yaml.safe_load(stream)

        thruster_info = config_file['thrusters']
        self.thruster_coeffs = np.zeros((len(thruster_info), 6))
        com = np.array(config_file["com"])
        self.max_force = config_file["thruster"]["max_force"]

        self.point_cloud = PointCloud()        
        self.max_thruster_height = 0 # level of water (may need to change)

        for i, thruster in enumerate(thruster_info):
            pose = np.array(thruster["pose"])
            rot_mat = euler_matrix(*pose[3:])
            body_force = np.dot(rot_mat, np.array([1, 0, 0, 0]))[:3]
            body_torque = np.cross(pose[:3]- com, body_force)

            self.thruster_coeffs[i, :3] = body_force
            self.thruster_coeffs[i, 3:] = body_torque                   

        self.initial_condition = []
        self.bounds = []
        for i in range(len(thruster_info)):
            self.initial_condition.append(0)
            self.bounds.append((-self.max_force, self.max_force))
        self.initial_condition = tuple(self.initial_condition)
        self.bounds = tuple(self.bounds)

        self.power_priority = 0.001

        self.timer = rospy.Timer(rospy.Duration(0.1), self.check_thrusters)
        self.listener = TransformListener()

        self.WATER_LEVEL = 0

        self.current_thruster_coeffs = self.thruster_coeffs

    # takes array of thruster forces and returns the array with each thruster above the water set to zero
    def check_thrusters(self):
        self.current_thruster_coeffs = self.thruster_coeffs
        for i in self.thruster_coeffs.shape[1]:
            trans = self.listener.lookupTransform("/world", "thruster_" + str(i), rospy.Time(0))      
            if trans[2] > self.WATER_LEVEL:   
                self.current_thruster_coeffs[:, i] = 0

    # Cost function forcing the thruster to output desired net force
    def force_cost(self, thruster_forces, desired_state):
        residual = np.dot(self.current_thruster_coeffs.T, thruster_forces) - desired_state
        return np.sum(residual ** 2)
    
    def force_cost_jac(self, thruster_forces, desired_state):
        residual = np.dot(self.current_thruster_coeffs.T, thruster_forces) - desired_state
        return np.dot(self.current_thruster_coeffs, 2 * residual)

    # Cost function forcing thrusters to find a solution that is low-power
    def power_cost(self, thruster_forces):
        return np.sum(thruster_forces ** 2)

    def power_cost_jac(self, thruster_forces):
        return 2 * thruster_forces

    def total_cost(self, thruster_forces, desired_state):
        total_cost = self.force_cost(thruster_forces, desired_state)
        # We care about low power a whole lot less thus the lower priority
        total_cost += self.power_cost(thruster_forces) * self.power_priority
        return total_cost

    def total_cost_jac(self, thruster_forces, desired_state):
        total_cost_jac = self.force_cost_jac(thruster_forces, desired_state)
        total_cost_jac += self.power_cost_jac(thruster_forces) * self.power_priority
        return total_cost_jac

    def force_cb(self, msg):
        desired_state = np.zeros(6)
        desired_state[:3] = msg_to_numpy(msg.linear)
        desired_state[3:] = msg_to_numpy(msg.angular)

        res = minimize(self.total_cost, self.initial_condition, args=(desired_state), method='SLSQP', \
                        jac=self.total_cost_jac, bounds=self.bounds)

        if self.force_cost(res.x, desired_state) > 0.01:
            rospy.logwarn("Unable to exert requested force")

        msg = Float32MultiArray()
        msg.data = res.x        

        self.thruster_pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node("thruster_solver")
    controller = ThrusterSolverNode()
    rospy.spin()