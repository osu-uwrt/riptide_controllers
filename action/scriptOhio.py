#! /usr/bin/env python
import rospy
import riptide_controllers.msg
import actionlib
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Quaternion, Vector3, Twist, PoseStamped
from moveit_msgs.msg import ExecuteTrajectoryActionGoal, ExecuteTrajectoryGoal
from nav_msgs.msg import Path, Odometry
from actionlib_msgs.msg import GoalStatus
import numpy as np
import math
from tf.transformations import quaternion_multiply, quaternion_inverse, quaternion_from_euler

def msg_2_numpy(msg):
    if hasattr(msg, "w"):
        return np.array([msg.x, msg.y, msg.z, msg.w])
    return np.array([msg.x, msg.y, msg.z])

def body_2_world(orientation, vector):
    """ 
    Rotates world-frame vector to body-frame

    Rotates vector to body frame

    Parameters:
    orientation (np.array): The current orientation of the robot as a quaternion
    vector (np.array): The 3D vector to rotate

    Returns: 
    np.array: 3 dimensional rotated vector

    """

    vector = np.append(vector, 0)
    orientation_inv = quaternion_inverse(orientation)
    new_vector = quaternion_multiply(orientation, quaternion_multiply(vector, orientation_inv))
    return new_vector[:3]



class ScriptOhioAction(object):

    _result = riptide_controllers.msg.CalibrateBuoyancyResult()

    def __init__(self):
        self.trajectory_pub = rospy.Publisher("/execute_trajectory/goal/", ExecuteTrajectoryActionGoal, queue_size=1)
        self.position_pub = rospy.Publisher("position", Vector3, queue_size=1)
        self.orientation_pub = rospy.Publisher("orientation", Quaternion, queue_size=1)
        
        self._as = actionlib.SimpleActionServer("script_ohio", riptide_controllers.msg.ScriptOhioAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

    def input_2_world(self, input_point, position, orientation):
        body_frame_position = self.scale * np.array([0, input_point[0] - self.middle_x, input_point[1] - self.middle_y])
        world_frame_position = body_2_world(orientation, body_frame_position) + position

        return world_frame_position

      
    def execute_cb(self, goal):
        points = []

        for i in range(63):
            points.append(np.array([math.cos(i/10.0), math.sin(i/10.0)]))

        points = np.array(points)

        Z_HEIGHT = 2
        TOTAL_TIME = 30.0
        dt = TOTAL_TIME / points.shape[0]
        self.min_x = np.min(points[:,0])
        self.max_x = np.max(points[:,0])
        self.middle_x = (self.max_x + self.min_x) / 2
        self.min_y = np.min(points[:,1])
        self.max_y = np.max(points[:,1])
        self.middle_y = (self.max_y + self.min_y) / 2

        self.scale = Z_HEIGHT / (self.max_y - self.min_y)

        odom_msg = rospy.wait_for_message("odometry/filtered", Odometry)
        position = msg_2_numpy(odom_msg.pose.pose.position)
        orientation = msg_2_numpy(odom_msg.pose.pose.orientation)


        world_frame_position = self.input_2_world(points[0], position, orientation)
        self.position_pub.publish(Vector3(*world_frame_position))
        rospy.sleep(4)


        trajectory = MultiDOFJointTrajectory()

        for i in range(points.shape[0]):
            world_frame_position = self.input_2_world(points[i], position, orientation)


            point = MultiDOFJointTrajectoryPoint()
            point.transforms.append(Transform(Vector3(*world_frame_position), Quaternion(*orientation)))
            point.time_from_start = rospy.Duration(i * dt)
            trajectory.points.append(point)

        trajectory_action = ExecuteTrajectoryActionGoal()
        trajectory_action.goal.trajectory.multi_dof_joint_trajectory = trajectory

        self.trajectory_pub.publish(trajectory_action)

        self._as.set_succeeded(self._result)

        rospy.sleep(TOTAL_TIME)

        world_frame_end_position = self.input_2_world(np.array([0,0]), position, orientation)
        self.position_pub.publish(Vector3(*world_frame_end_position))

        rospy.sleep(5)

        bowed_orientation = quaternion_multiply(orientation, quaternion_from_euler(0, 0.4, 0))
        self.orientation_pub.publish(Quaternion(*bowed_orientation))
        rospy.sleep(1)
        self.orientation_pub.publish(Quaternion(*orientation))
        rospy.sleep(1)





if __name__ == '__main__':
    rospy.init_node('script_ohio')
    server = ScriptOhioAction()
    rospy.spin()





