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



class SickTrickAction(object):

    def __init__(self):
        self.trajectory_pub = rospy.Publisher("execute_trajectory/goal/", ExecuteTrajectoryActionGoal, queue_size=1)
        self.position_pub = rospy.Publisher("position", Vector3, queue_size=1)
        self.orientation_pub = rospy.Publisher("orientation", Quaternion, queue_size=1)
        self.path_pub = rospy.Publisher("sick_trick_path", Path, queue_size=1)
        
        self._as = actionlib.SimpleActionServer("sick_trick", riptide_controllers.msg.SickTrickAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

        self.DT = 0.1

        

    def input_2_world(self, input_point, input_orientation, position, orientation):
        body_frame_position = np.array([input_point[0] - self.middle_x, input_point[1] - self.middle_y, input_point[2] - self.middle_z])
        world_frame_position = body_2_world(orientation, body_frame_position) + position
        world_frame_orientation = quaternion_multiply(orientation, input_orientation)

        return world_frame_position, world_frame_orientation

      
    def execute_cb(self, goal):

        PATH = "~/osu-uwrt/riptide_software/src/riptide_controllers/cfg/%s.csv" % goal.trick

        self.points = []
        
        with open(os.path.expanduser(PATH), "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                self.points.append(list(map(float, row)))

        self.points = np.array(self.points)

        
        self.min_x = np.min(self.points[:,0])
        self.max_x = np.max(self.points[:,0])
        self.middle_x = (self.max_x + self.min_x) / 2
        self.min_y = np.min(self.points[:,1])
        self.max_y = np.max(self.points[:,1])
        self.middle_y = (self.max_y + self.min_y) / 2
        self.min_z = np.min(self.points[:,2])
        self.max_z = np.max(self.points[:,2])
        self.middle_z = (self.max_z + self.min_z) / 2

        odom_msg = rospy.wait_for_message("odometry/filtered", Odometry)
        initial_position = msg_2_numpy(odom_msg.pose.pose.position)
        initial_orientation = msg_2_numpy(odom_msg.pose.pose.orientation)


        world_frame_position, world_frame_orientation = self.input_2_world(self.points[0][:3], self.points[0][3:], initial_position, initial_orientation)
        self.position_pub.publish(Vector3(*world_frame_position))
        self.orientation_pub.publish(Quaternion(*world_frame_orientation))
        rospy.sleep(10)


        trajectory = MultiDOFJointTrajectory()
        path = Path()
        path.header.frame_id = "world"	
        path.header.stamp = rospy.get_rostime()


        for i in range(self.points.shape[0]):
            world_frame_position, world_frame_orientation = self.input_2_world(self.points[i][:3], self.points[i][3:], initial_position, initial_orientation)

            point = MultiDOFJointTrajectoryPoint()
            point.transforms.append(Transform(Vector3(*world_frame_position), Quaternion(*world_frame_orientation)))
            point.time_from_start = rospy.Duration(i * self.DT)
            trajectory.points.append(point)

            path_point = PoseStamped()	
            path_point.pose.position = Vector3(*world_frame_position)
            path_point.pose.orientation = Quaternion(*world_frame_orientation)
            path_point.header.stamp = rospy.get_rostime() + rospy.Duration(i * self.DT)	
            path_point.header.frame_id = "world"	
            path.poses.append(path_point)

        trajectory_action = ExecuteTrajectoryActionGoal()
        trajectory_action.goal.trajectory.multi_dof_joint_trajectory = trajectory

        self.trajectory_pub.publish(trajectory_action)

        self.path_pub.publish(path)
        rospy.sleep(len(self.points) * self.DT)

        self._as.set_succeeded()





if __name__ == '__main__':
    rospy.init_node('sick_trick')
    server = SickTrickAction()
    rospy.spin()





