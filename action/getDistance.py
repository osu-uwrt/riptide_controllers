#! /usr/bin/env python
import rospy
import actionlib
import message_filters

from riptide_msgs.msg import Dvl, LinearCommand
from geometry_msgs.msg import Vector3
from darknet_ros_msgs.msg import BoundingBoxes
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge, CvBridgeError
import riptide_controllers.msg

import math
import numpy as np
import cv2

bridge = CvBridge()

class GetDistance(object):

    _result = riptide_controllers.msg.GetDistanceResult()

    def __init__(self):
        self._as = actionlib.SimpleActionServer("get_distance", riptide_controllers.msg.GetDistanceAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

      
    def execute_cb(self, goal):
        rospy.loginfo("Finding distance to " + goal.object)
        self.goal = goal
        stereoSub = message_filters.Subscriber("/stereo/disparity", DisparityImage)
        bboxSub = message_filters.Subscriber("/state/bboxes", BoundingBoxes)
        ts = message_filters.TimeSynchronizer([stereoSub, bboxSub], 20)
        ts.registerCallback(self.imgCB)

        self.readings = []
        while len(self.readings) < 5:
            if self._as.is_preempt_requested():
                rospy.loginfo('Preempted Get Distance')
                stereoSub.sub.unregister()
                bboxSub.sub.unregister()
                self._as.set_preempted()
                return
        
        stereoSub.sub.unregister()
        bboxSub.sub.unregister()
        self._result.distance = np.median(self.readings)
        rospy.loginfo("Distance: %f"%self._result.distance)
        self._as.set_succeeded(self._result)

    def imgCB(self, stereo, bboxMsg):
        bboxes = bboxMsg.bounding_boxes
        if len([x for x in bboxes if x.Class == self.goal.object]) != 0:
            bbox = [x for x in bboxes if x.Class == self.goal.object][0]
            
            try:
                image = bridge.imgmsg_to_cv2(stereo.image)
            except CvBridgeError as e:
                print(e)
            sample_region = image[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax].reshape((-1,1))
            
            # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

            # Set flags (Just to avoid line break in the code)
            flags = cv2.KMEANS_RANDOM_CENTERS

            # Apply KMeans
            _,labels,centers = cv2.kmeans(sample_region,4,None,criteria,10,flags)

            labels = [l[0] for l in labels]
            disparity = -1
            while disparity < 0 and len(labels) != 0:
                maxLabel = max(set(labels), key=labels.count)
                disparity = centers[maxLabel][0]
                labels = [l for l in labels if l != maxLabel]

            self.readings.append(stereo.f * stereo.T / disparity)
        
        
        
if __name__ == '__main__':
    rospy.init_node('get_distance')
    server = GetDistance()
    rospy.spin()