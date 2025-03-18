#!/home/wang/anaconda3/envs/airbot/bin/python
import time
import threading
import queue

import pyrealsense2 as rs
import cv2
import numpy as np

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"

class RealSense:
    def __init__(self, config):
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        h = config['height']
        w = config['width']
        fps = config['fps']
        self._max_queue_size = config['max_queue_size']
        self._config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self._config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        self._align = rs.align(rs.stream.color)

        self._extrinsics = None
        self._tag_length = config['extrinsic_calibrator']['tag_length']
        self._instrinsics = None
        
        self._queue = queue.Queue()
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        # Start the sensor in the thread
        print(GREEN + "[RealSense]: Starting RealSense sensor..." + RESET)
        self._thread = threading.Thread(target=self._run)
        self._thread.start()
        time.sleep(1)  # Wait for auto-exposure to stabilize
        print(GREEN + "[RealSense]: RealSense sensor started." + RESET)

    def get(self):
        '''
        Return a dict:
        {
            "timestamp": float,
            "color": np.ndarray, [H, W, 3], BGR
            "depth": np.ndarray, [H, W], z16
            "extrinsics": np.ndarray, [4, 4], c2w
        }
        '''
        try:
            return self._queue.get(timeout=1)
        except:
            print(YELLOW + "[RealSense]: Failed to get RealSense data within 1 second." + RESET)
            return None

    def stop(self):
        print(GREEN + "[RealSense]: Stopping RealSense sensor..." + RESET)
        self._stop_event.set()
        self._thread.join()
        print(GREEN + "[RealSense]: RealSense sensor stopped." + RESET)

    def _get_instrinsics(self):
        profile = self._pipeline.start(self._config)

        # Get intrinsics
        color_stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_stream_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_intrinsics = color_stream_profile.get_intrinsics()
        self._instrinsics = np.zeros((3, 3), dtype=np.float32)
        self._instrinsics[0, 0] = color_intrinsics.fx
        self._instrinsics[1, 1] = color_intrinsics.fy
        self._instrinsics[0, 2] = color_intrinsics.ppx
        self._instrinsics[1, 2] = color_intrinsics.ppy
        self._instrinsics[2, 2] = 1
        self._distortion = np.array(color_intrinsics.coeffs)


    def _run(self):
        try:
            self._get_instrinsics()

        except Exception as e:
            print(RED + "[RealSense]: Failed to start the pipeline." + RESET)
            print(e)
            return

        self._extrinsics = self._get_extrinsic()

        while not self._stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames()

                # Align the frames
                aligned_frames = self._align.process(frames)

                # Extract aligned color and depth frames
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                timestamp = time.time()

                # Convert frames to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                self._queue.put({
                    "timestamp": timestamp,
                    "color": color_image,
                    "depth": depth_image,
                    "extrinsics": self._extrinsics,
                    "instrinsics": self._instrinsics,
                })

                if self._queue.qsize() > self._max_queue_size:
                    self._queue.get()

            except Exception as e:
                print(e)
                continue

        self._pipeline.stop()

    def _get_extrinsic(self):
        try:
            frames = self._pipeline.wait_for_frames()
            aligned_frames = self._align.process(frames)
            color_frame = aligned_frames.get_color_frame()
        except Exception as e:
            print(RED + "[RealSense]: Failed to capture frames for extrinsic calibration." + RESET)
            return None

        color_image = np.asanyarray(color_frame.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(tag_dict, params)

        corners, ids, _ = detector.detectMarkers(gray_image)
        if ids is None or len(corners) == 0:
            print(YELLOW + "[RealSense]: No markers detected for extrinsic calibration." + RESET)
            return None

        obj_points = np.array([
            [-self._tag_length / 2, self._tag_length / 2, 0],
            [self._tag_length / 2, self._tag_length / 2, 0],
            [self._tag_length / 2, -self._tag_length / 2, 0],
            [-self._tag_length / 2, -self._tag_length / 2, 0]
        ], dtype=np.float32)

        corners = corners[0].reshape(4, 2)
        success, rvec, tvec = cv2.solvePnP(obj_points, corners,
                                             self._instrinsics,
                                             self._distortion)
        if not success:
            print(RED + "[RealSense]: Failed to estimate pose." + RESET)
            return None

        # Convert rvec to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Compute c2w transformation matrix
        c2w = np.eye(4)
        c2w[:3, :3] = rmat.T  # Transpose of rotation matrix
        c2w[:3, 3] = -rmat.T @ tvec.flatten()  # Inverse translation
        print(GREEN + "[RealSense]: Successfully estimated extrinsic matrix." + RESET)
        print(GREEN + "[RealSense]: Extrinsic matrix:" + RESET)
        print(GREEN + str(c2w) + RESET)

        return c2w