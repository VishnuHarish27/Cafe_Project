import cv2
import os

class ArucoDetector:
    def __init__(self):
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.parameters = cv2.aruco.DetectorParameters_create()

    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        aruco_ids = []
        if ids is not None and len(ids) > 0:
            aruco_ids = ids.flatten().tolist()
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        return frame, aruco_ids
