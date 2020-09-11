import rospy
import riptide_controllers.msg
import actionlib
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Quaternion, Vector3, Twist, PoseStamped
from nav_msgs.msg import Path
from actionlib_msgs.msg import GoalStatus
import numpy as np

rospy.init_node("test")

client = actionlib.SimpleActionClient(
    "puddles/follow_trajectory", riptide_controllers.msg.FollowTrajectoryAction)
client.wait_for_server()

path_pub = rospy.Publisher("path", Path, queue_size=1)

duration = 20
dt = 0.1
points = int(duration / dt)
trajectory = MultiDOFJointTrajectory()
path = Path()
path.header.frame_id = "world"
path.header.stamp = rospy.get_rostime()
positions = np.zeros((points, 3))
velocities = np.zeros((points, 3))
positions[0] = [0, 0, -1]

zero_vector = Vector3(0, 0, 0)

for i in range(1, points):
    acceleration = (np.random.rand(3)-0.5) / 2
    if positions[i - 1][2] > -0.4 and acceleration[2] > 0:
        acceleration[2] *= -1
    for axis in range(3):
        if abs(velocities[i - 1][axis]) > .18 and velocities[i - 1][axis] * acceleration[axis] > 0:
            acceleration[axis] *= -1
    velocities[i] = velocities[i-1] + acceleration * dt
    positions[i] = positions[i-1] + velocities[i] * dt
    

    point = MultiDOFJointTrajectoryPoint()
    point.transforms.append(Transform(Vector3(*positions[i]), Quaternion(0, 0, 0, 1)))
    point.velocities.append(Twist(Vector3(*velocities[i]), zero_vector))
    point.accelerations.append(Twist(Vector3(*acceleration), zero_vector))
    point.time_from_start = rospy.Duration(i * dt)
    trajectory.points.append(point)

    path_point = PoseStamped()
    path_point.pose.position = Vector3(*positions[i])
    path_point.header.stamp = rospy.get_rostime() + rospy.Duration(i * dt)
    path_point.header.frame_id = "world"
    path.poses.append(path_point)

client.send_goal(riptide_controllers.msg.FollowTrajectoryGoal(trajectory))

while client.get_state() not in [GoalStatus.SUCCEEDED, GoalStatus.PREEMPTED]:
    path_pub.publish(path)


