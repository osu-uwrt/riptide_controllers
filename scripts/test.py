import rospy
import riptide_controllers.msg
import actionlib
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Quaternion, Vector3, Twist

rospy.init_node("test")

client = actionlib.SimpleActionClient(
    "puddles/follow_trajectory", riptide_controllers.msg.FollowTrajectoryAction)
client.wait_for_server()

trajectory = MultiDOFJointTrajectory()
speed = 0.5
duration = 20
dt = 0.1
zero_vector = Vector3(0, 0, 0)

for i in range(int(duration / dt)):
    point = MultiDOFJointTrajectoryPoint()
    point.transforms.append(Transform(Vector3(-speed * dt * i, 0, -1), Quaternion(0, 0, 0, 1)))
    point.velocities.append(Twist(Vector3(-speed, 0, 0), zero_vector))
    point.accelerations.append(Twist(zero_vector, zero_vector))
    point.time_from_start = rospy.Duration(i * dt)

    trajectory.points.append(point)


client.send_goal(riptide_controllers.msg.FollowTrajectoryGoal(trajectory))


