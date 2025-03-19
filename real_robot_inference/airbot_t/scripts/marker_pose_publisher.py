#!/usr/bin/env python3
import yaml
import time
import sys
import os
import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Point

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from real_sensor import RealSense

with open("/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

rs = RealSense(config['realsense'])
rs.start()  

config = config['realsense']
length = config['extrinsic_calibrator']['tag_length']  
instrinsics = np.array(rs._instrinsics).reshape(3, 3) 
distortion = np.array(rs._distortion)
print(instrinsics)

obj_points = np.array([
    [+length / 2, -length / 2, 0],
    [+length / 2, +length / 2, 0], 
    [-length / 2, +length / 2, 0],  
    [-length / 2, -length / 2, 0]   
])

base = np.array([
    [9.99999995e-01, -7.59005975e-07, -9.75572810e-05, 5.74032376e-02],
    [-7.58053908e-07, -1.00000000e+00, 9.75908905e-06, 5.73699780e-03],
    [-9.75572884e-05, -9.75901505e-06, -9.99999995e-01, 7.38194332e-01],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
params = cv2.aruco.DetectorParameters()  
detector = cv2.aruco.ArucoDetector(dict, params)  

rospy.init_node('qr_code_publisher', anonymous=True)

point0_pub = rospy.Publisher('point0', Point, queue_size=10)
point1_pub = rospy.Publisher('point1', Point, queue_size=10)
new_point_pub = rospy.Publisher('new_point', Point, queue_size=10)

while not rospy.is_shutdown():
    data = None
    while data is None:
        data = rs.get() 
    color = data['color']  
    img = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) 
    corners, ids, _ = detector.detectMarkers(img)  
    corners = np.array(corners) 
    ids = np.array(ids)

    point0 = None
    point1 = None
    point2 = np.array([0.0, 0.0, 0.0])
    flag0 = False
    flag1 = False
    if ids is not None and len(corners) > 0:
        id_0_detected = 0 in ids
        id_1_detected = 1 in ids
        for i in range(len(ids)):
            corner = corners[i].reshape(4, 2)
            success, rvec, tvec = cv2.solvePnP(obj_points, corner, instrinsics, distortion)
            tvec_ = np.append(tvec, 1)
            if success:
                point = base @ tvec_
                if ids[i] == 0:
                    point0 = point[:3]
                    point0[0] *= -1.0
                    point0[0] += 0.57
                    point0[1] *= -1.0
                    point0_msg = Point(x=point0[0], y=point0[1], z=point0[2])
                    point0_pub.publish(point0_msg)
                    flag0 = True
                elif ids[i] == 1:
                    point1 = point[:3]
                    point1[0] *= -1.0
                    point1[0] += 0.57
                    point1[1] *= -1.0
                    point1_msg = Point(x=point1[0], y=point1[1], z=point1[2])
                    point1_pub.publish(point1_msg)
                    flag1 = True

                color = cv2.aruco.drawDetectedMarkers(color, corners, ids)
                color = cv2.drawFrameAxes(color, instrinsics, distortion, rvec, tvec, length)
                cv2.imshow("Pose Estimation", color)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    else:
        cv2.imshow("Pose Estimation", color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if flag0 and flag1:
        
        direction = point0 - point1
        direction_normalized = direction / np.linalg.norm(direction)
        distance = 0.025
        new_point = point0 + direction_normalized * distance
        new_point_msg = Point(x=new_point[0], y=new_point[1], z=new_point[2])
        new_point_pub.publish(new_point_msg)

        flag0 = False
        flag1 = False

rs.stop()