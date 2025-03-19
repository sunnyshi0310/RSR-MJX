#!/usr/bin/env python3
import yaml
import time
import os
import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Point
import sys
from std_msgs.msg import Header
from threading import Event
from datetime import datetime

sys.path.append('/airbot_sim2real_sl/scripts')

from real_sensor import RealSense

with open("/airbot_sim2real_sl/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
rs = RealSense(config['realsense'])
rs.start() 

config = config['realsense']
length = config['extrinsic_calibrator']['tag_length'] 
instrinsics = np.array(rs._instrinsics).reshape(3, 3) 
print(instrinsics)
distortion = np.array(rs._distortion)

obj_points = np.array([
    [+length / 2, -length / 2, 0], 
    [+length / 2, +length / 2, 0],  
    [-length / 2, +length / 2, 0],  
    [-length / 2, -length / 2, 0]   
])

base = [[ 9.99999995e-01, -7.59005975e-07, -9.75572810e-05,  5.74032376e-02],
        [-7.58053908e-07, -1.00000000e+00,  9.75908905e-06,  5.73699780e-03],
        [-9.75572884e-05, -9.75901505e-06, -9.99999995e-01,  7.38194332e-01],
        [ 0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  1.00000000e+00]]

dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5) 
params = cv2.aruco.DetectorParameters() 
detector = cv2.aruco.ArucoDetector(dict, params) 

rospy.init_node('qr_code_publisher', anonymous=True)
pub = rospy.Publisher('qr_coordinates', Point, queue_size=10)
point_msg = Point()

frame_count = 0 
last_save_time = time.time() 
step_complete_event = Event()
jishu = 1

def step_callback(msg):
        
        try:
            frame = rs.get()
            global jishu
            if frame is not None:
                
                filename = f"id_{jishu}.jpg"
                filepath = os.path.join("/src/airbot_sim2real_sl/data", filename)
                
                cv2.imwrite(filepath, frame["color"])
                rospy.loginfo(f"Saved step {msg.seq} image: {filename}")
                jishu += 1
        except Exception as e:
            rospy.logerr(f"Image save failed: {str(e)}")

time.sleep(10)
rospy.Subscriber("/airbot_play/step_complete", Header, step_callback)

while True:
    data = None
    while data is None:
        data = rs.get()  
    color = data['color'] 
    img = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)  
    corners, ids, _ = detector.detectMarkers(img)  
    corners = np.array(corners)  
    ids = np.array(ids)

    if ids is not None and len(corners) > 0:
        
        obj_points = np.array([
            [-length / 2, +length / 2, 0],
            [+length / 2, +length / 2, 0],
            [+length / 2, -length / 2, 0],
            [-length / 2, -length / 2, 0]
        ])

        for i in range(len(ids)):
            corner = corners[i].reshape(4, 2)
            success, rvec, tvec = cv2.solvePnP(obj_points, corner, instrinsics, distortion)
            tvec_ = np.append(tvec, 1)
            if success:
                point = base @ tvec_
                point[0] *= -1.0
                point[0] += 0.57  
                point[1] *= -1.0

                point_msg.x = point[0]
                point_msg.y = point[1]
                point_msg.z = point[2]
                
                pub.publish(point_msg)
                

        color = cv2.aruco.drawDetectedMarkers(color, corners, ids)
        color = cv2.drawFrameAxes(color, instrinsics, distortion, rvec, tvec, length)

    
    cv2.imshow("Pose Estimation", color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

rs.stop()

cv2.destroyAllWindows()

