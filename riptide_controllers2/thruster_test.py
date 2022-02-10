#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from riptide_msgs2.msg import PwmStamped
from riptide_msgs2.action import ThrusterTest

import yaml

import time

NEUTRAL_PWM = 1500
POSITIVE_PWM = 1550
NEGATIVE_PWM = 1450

class ThrusterTestActionServer(Node):
    THRUSTER_PERCENT = 0.05

    def __init__(self):
        super().__init__('thruster_test')
        self.declare_parameter("vehicle_config", rclpy.Parameter.Type.STRING)

        self.pwm_pub = self.create_publisher(PwmStamped ,"command/pwm", qos_profile_sensor_data)

        # Get the mass and COM
        with open(self.get_parameter('vehicle_config').value, 'r') as stream:
            self.vehicle_file = yaml.safe_load(stream)
            self.num_thrusters = len(self.vehicle_file["thrusters"])
            self.max_force = self.vehicle_file["thruster"]["max_force"]

        self.running = False

        self._action_server = ActionServer(
            self,
            ThrusterTest,
            'thruster_test',
            self.execute_cb,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup())

    ##############################
    # Protections 
    
    def destroy(self):
        self.destroy_node()
        self._action_server.destroy()

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        if self.running:
            return GoalResponse.REJECT
        else:
            self.running = True
            return GoalResponse.ACCEPT

    def cancel_callback(self, goal):
        return CancelResponse.ACCEPT


    def publish_pwm(self, pwm):
        msg = PwmStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pwm = pwm
        self.pwm_pub.publish(msg)

    def execute_cb(self, goal_handle: ServerGoalHandle):
        self.get_logger().info("Starting ThrusterTest Action")
        pwm = [NEUTRAL_PWM] * self.num_thrusters
        while True:
            for i in range(self.num_thrusters):
                if goal_handle.is_cancel_requested:
                    self.get_logger().info('Preempted ThrusterTest Action')
                    self.publish_pwm([NEUTRAL_PWM] * self.num_thrusters)

                    self.running = False
                    goal_handle.canceled()
                    return ThrusterTest.Result()

                thruster_type = self.vehicle_file["thrusters"][i]["type"]

                self.get_logger().info(f'Testing Thruster {i+1}')
                if thruster_type == 0:
                    pwm[i] = POSITIVE_PWM
                else:
                    pwm[i] = NEGATIVE_PWM

                for _ in range(300):
                    self.publish_pwm(pwm)
                    time.sleep(0.01)
                
                pwm[i] = NEUTRAL_PWM

        #should never reach this point in the code
        self.get_logger().info("ThrustTest succeeded")
        goal_handle.succeed()
        self.running = False
        return ThrusterTest.Result()

def main(args=None):
    rclpy.init(args=args)

    thruster_test_action_server = ThrusterTestActionServer()

    executor = MultiThreadedExecutor()
    rclpy.spin(thruster_test_action_server, executor=executor)

    thruster_test_action_server.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()