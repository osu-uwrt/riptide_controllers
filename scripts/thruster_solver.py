#!/usr/bin/env python

# thruster_solver node
#
# Input topics:
#   net_force: The force the control system wants the robot to exert on the world to move
#
# Output topics:
#   thruster_forces: Array containing how hard each thruster will push. 
#
# This node works via optimization. A cost function is proposed that measures how optimal a set of thruster forces is.
# This function takes into account exerting the correct total forces and power consumption. This node will also ignore
# thrusters that are currently out of the water and solve with the thrusters that are in the water. On each new net_force
# message, the robot solves for the optimal thruster forces and publishes them

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Int16MultiArray
import numpy as np
import yaml
from tf.transformations import euler_matrix
from tf import TransformListener
from scipy.optimize import minimize


NEUTRAL_PWM = 1500
MIN_PWM = 1230
MAX_PWM = 1770

def msg_to_numpy(msg):
    return np.array([msg.x, msg.y, msg.z])

class ThrusterSolverNode:

    def __init__(self):
        rospy.Subscriber("net_force", Twist, self.force_cb)

        self.thruster_pub = rospy.Publisher("thruster_forces", Float32MultiArray, queue_size=5)
        self.pwm_pub = rospy.Publisher("command/pwm", Int16MultiArray, queue_size=5)
        self.tf_namespace = rospy.get_param("~robot")

        # Load thruster info
        config_path = rospy.get_param("~vehicle_config")
        with open(config_path, 'r') as stream:
            config_file = yaml.safe_load(stream)

        thruster_info = config_file['thrusters']
        self.thruster_coeffs = np.zeros((len(thruster_info), 6))
        self.thruster_types = np.zeros(len(thruster_info))
        com = np.array(config_file["com"])
        self.max_force = config_file["thruster"]["max_force"]
        self.pwm_file = config_file["thruster"]

        for i, thruster in enumerate(thruster_info):
            pose = np.array(thruster["pose"])
            rot_mat = euler_matrix(*pose[3:])
            body_force = np.dot(rot_mat, np.array([1, 0, 0, 0]))[:3]
            body_torque = np.cross(pose[:3]- com, body_force)

            self.thruster_coeffs[i, :3] = body_force
            self.thruster_coeffs[i, 3:] = body_torque   

            self.thruster_types[i] = config_file["thrusters"][i]["type"]   



        self.initial_condition = []
        self.bounds = []
        for i in range(len(thruster_info)):
            self.initial_condition.append(0)
            self.bounds.append((-self.max_force, self.max_force))
        self.initial_condition = tuple(self.initial_condition)
        self.bounds = tuple(self.bounds)

        self.power_priority = 0.001
        self.current_thruster_coeffs = np.copy(self.thruster_coeffs)

        self.start_time = None
        self.timer = rospy.Timer(rospy.Duration(0.1), self.check_thrusters)
        self.listener = TransformListener()
        self.WATER_LEVEL = 0

    def publish_pwm(self, forces):
        pwm_values = []

        for i in range(self.thruster_coeffs.shape[0]):
            pwm = NEUTRAL_PWM

            if (abs(forces[i]) < self.pwm_file["MIN_THRUST"]):
                pwm = NEUTRAL_PWM

            elif (forces[i] > 0 and forces[i] <= self.pwm_file["STARTUP_THRUST"]):
                if self.thruster_types[i] == 0:
                    pwm = (int)(self.pwm_file["SU_THRUST"]["POS_SLOPE"] * forces[i] + self.pwm_file["SU_THRUST"]["POS_YINT"])
                else:
                    pwm = (int)(-self.pwm_file["SU_THRUST"]["POS_SLOPE"] * forces[i] + self.pwm_file["SU_THRUST"]["NEG_YINT"])

            elif (forces[i] > 0 and forces[i] > self.pwm_file["STARTUP_THRUST"]):
                if self.thruster_types[i] == 0:
                    pwm = (int)(self.pwm_file["THRUST"]["POS_SLOPE"] * forces[i] + self.pwm_file["THRUST"]["POS_YINT"])
                else:
                    pwm = (int)(-self.pwm_file["THRUST"]["POS_SLOPE"] * forces[i] + self.pwm_file["THRUST"]["NEG_YINT"])

            elif (forces[i] < 0 and forces[i] >= -self.pwm_file["STARTUP_THRUST"]):
                if self.thruster_types[i] == 0:
                    pwm = (int)(self.pwm_file["SU_THRUST"]["NEG_SLOPE"] * forces[i] + self.pwm_file["SU_THRUST"]["NEG_YINT"])
                else:
                    pwm = (int)(-self.pwm_file["SU_THRUST"]["NEG_SLOPE"] * forces[i] + self.pwm_file["SU_THRUST"]["POS_YINT"])

            elif (forces[i] < 0 and forces[i] < -self.pwm_file["STARTUP_THRUST"]):
                if self.thruster_types[i] == 0:
                    pwm = (int)(self.pwm_file["THRUST"]["NEG_SLOPE"] * forces[i] + self.pwm_file["THRUST"]["NEG_YINT"])
                else:
                    pwm = (int)(-self.pwm_file["THRUST"]["NEG_SLOPE"] * forces[i] + self.pwm_file["THRUST"]["POS_YINT"])

            else:
                pwm = NEUTRAL_PWM

            pwm_values.append(pwm)

        msg = Int16MultiArray()
        msg.data = pwm_values
        self.pwm_pub.publish(msg)


    # Timer callback which disables thrusters that are out of the water
    def check_thrusters(self, timer_event):
        try:
            if self.start_time is None:
                self.start_time = rospy.get_rostime()
            self.current_thruster_coeffs = np.copy(self.thruster_coeffs)
            for i in range(self.thruster_coeffs.shape[0]):
                trans, _ = self.listener.lookupTransform("/world", "/%s/thruster_%d" % (self.tf_namespace, i), rospy.Time(0))      
                if trans[2] > self.WATER_LEVEL:
                    self.current_thruster_coeffs[i, :] = 0
        except Exception as ex:
            # Supress startup errors
            if (rospy.get_rostime() - self.start_time).secs > 1:
                rospy.logerr(str(ex))

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

    # Combination of other cost functions
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

        # Optimize cost function to find optimal thruster forces
        res = minimize(self.total_cost, self.initial_condition, args=(desired_state), method='SLSQP', \
                        jac=self.total_cost_jac, bounds=self.bounds)

        # Warn if we did not find valid thruster forces
        if self.force_cost(res.x, desired_state) > 0.05:
            rospy.logwarn_throttle(1, "Unable to exert requested force")

        msg = Float32MultiArray()
        msg.data = res.x        

        self.publish_pwm(res.x)

        self.thruster_pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node("thruster_solver")
    controller = ThrusterSolverNode()
    rospy.spin()