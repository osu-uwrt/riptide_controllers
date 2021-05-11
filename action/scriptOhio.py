#! /usr/bin/env python3
import rospy
import riptide_controllers.msg
import actionlib
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Quaternion, Vector3, PoseStamped
from moveit_msgs.msg import ExecuteTrajectoryActionGoal
from nav_msgs.msg import Path, Odometry
import numpy as np
from tf.transformations import quaternion_multiply, quaternion_inverse, quaternion_from_euler
import os
import csv

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

    def __init__(self):
        self.trajectory_pub = rospy.Publisher("execute_trajectory/goal/", ExecuteTrajectoryActionGoal, queue_size=1)
        self.position_pub = rospy.Publisher("position", Vector3, queue_size=1)
        self.orientation_pub = rospy.Publisher("orientation", Quaternion, queue_size=1)
        self.path_pub = rospy.Publisher("script_ohio_path", Path, queue_size=1)
        
        self._as = actionlib.SimpleActionServer("script_ohio", riptide_controllers.msg.ScriptOhioAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()
        self._result = riptide_controllers.msg.CalibrateBuoyancyResult()

        PATH = "~/Downloads/script_ohio.csv"
        self.Z_HEIGHT = 2.0
        self.TOTAL_TIME = 120.0

        self.points = []
        
        with open(os.path.expanduser(PATH), "r") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                self.points.append(map(int, map(float, row)))

        self.points = np.array(self.points)
        self.points *= np.array([1, -1])

        self.i_dot = self.points[-1]
        self.points = self.points[:-1]

        

    def input_2_world(self, input_point, position, orientation):
        body_frame_position = self.scale * np.array([0, input_point[0] - self.middle_x, input_point[1] - self.middle_y])
        world_frame_position = body_2_world(orientation, body_frame_position) + position

        return world_frame_position

      
    def execute_cb(self, goal):


        
        dt = self.TOTAL_TIME / self.points.shape[0]
        self.min_x = np.min(self.points[:,0])
        self.max_x = np.max(self.points[:,0])
        self.middle_x = (self.max_x + self.min_x) / 2
        self.min_y = np.min(self.points[:,1])
        self.max_y = np.max(self.points[:,1])
        self.middle_y = (self.max_y + self.min_y) / 2

        self.scale = self.Z_HEIGHT / (self.max_y - self.min_y)

        odom_msg = rospy.wait_for_message("odometry/filtered", Odometry)
        position = msg_2_numpy(odom_msg.pose.pose.position)
        orientation = msg_2_numpy(odom_msg.pose.pose.orientation)


        world_frame_position = self.input_2_world(self.points[0], position, orientation)
        self.position_pub.publish(Vector3(*world_frame_position))
        rospy.sleep(10)


        trajectory = MultiDOFJointTrajectory()
        path = Path()
        path.header.frame_id = "world"	
        path.header.stamp = rospy.get_rostime()


        for i in range(self.points.shape[0]):
            world_frame_position = self.input_2_world(self.points[i], position, orientation)

            point = MultiDOFJointTrajectoryPoint()
            point.transforms.append(Transform(Vector3(*world_frame_position), Quaternion(*orientation)))
            point.time_from_start = rospy.Duration(i * dt)
            trajectory.points.append(point)

            path_point = PoseStamped()	
            path_point.pose.position = Vector3(*world_frame_position)
            path_point.header.stamp = rospy.get_rostime() + rospy.Duration(i * dt)	
            path_point.header.frame_id = "world"	
            path.poses.append(path_point)

        trajectory_action = ExecuteTrajectoryActionGoal()
        trajectory_action.goal.trajectory.multi_dof_joint_trajectory = trajectory

        self.trajectory_pub.publish(trajectory_action)

        self._as.set_succeeded(self._result)

        self.path_pub.publish(path)
        rospy.sleep(self.TOTAL_TIME)

        world_frame_end_position = self.input_2_world(np.array(self.i_dot), position, orientation)
        self.position_pub.publish(Vector3(*world_frame_end_position))

        rospy.sleep(7)

        bowed_orientation = quaternion_multiply(orientation, quaternion_from_euler(0, 0.4, 0))
        self.orientation_pub.publish(Quaternion(*bowed_orientation))
        rospy.sleep(2)
        self.orientation_pub.publish(Quaternion(*orientation))
        rospy.sleep(2)





if __name__ == '__main__':
    rospy.init_node('script_ohio')
    server = ScriptOhioAction()
    rospy.spin()





