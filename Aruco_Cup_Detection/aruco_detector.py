import cv2
import numpy as np

class ArucoDetector:
    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.adaptiveThreshWinSizeMin = 5
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        self.parameters.minMarkerPerimeterRate = 0.03
        self.parameters.maxMarkerPerimeterRate = 4.0
        self.parameters.minMarkerDistanceRate = 0.05
 # Adjust based on your requirements

    def detect_aruco_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        if ids is not None:
            for i in range(len(ids)):
                if ids[i] and len(ids[i]) > 0:
                    confidence = np.mean(ids[i])
                    if confidence > 98:  # Only consider markers with confidence > 98%
                        cv2.aruco.drawDetectedMarkers(frame, corners)
        return frame, ids
            parameters = cv2.aruco.DetectorParameters()
            parameters.adaptiveThreshWinSizeMin = 5
            parameters.adaptiveThreshWinSizeMax = 23
            parameters.adaptiveThreshWinSizeStep = 10
            parameters.minMarkerPerimeterRate = 0.03
            parameters.maxMarkerPerimeterRate = 4.0
            parameters.minMarkerDistanceRate = 0.05
