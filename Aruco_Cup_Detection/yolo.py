import logging
import cv2
import os
from collections import defaultdict
import numpy as np
from ultralytics import YOLO as YOLOv8, solutions
from database import log_violation

class YOLO:
    def __init__(self, use_gpu=False):
        device = 'cuda:0' if use_gpu else 'cpu'
        self.model = YOLOv8('yolov8x.pt').to(device)
        self.classes = self.model.names
        self.region_points = [(279,1040), (308,369), (1114,381),(1180,1055)]

        self.classes_to_count = [41]  # Updated classes to count
        self.buffer_zone = 30  # Buffer zone around boundary line

        # Init Object Counter
        try:
            self.counter = solutions.ObjectCounter(
                view_img=True,
                reg_pts=self.region_points,
                classes_names=self.classes,
                draw_tracks=True,
                line_thickness=2,
            )
        except AttributeError as e:
            raise ImportError(f"Error initializing ObjectCounter: {e}")

        self.crossed_objects = defaultdict(list)
        self.last_class_wise_count = defaultdict(lambda: {"IN": 0, "OUT": 0})
        self.detected_ids = set()

    def detect_and_check(self, frame, frame_idx):
        try:
            tracks = self.model.track(frame, persist=True, show=False, classes=self.classes_to_count)
        except Exception as e:
            print(f"Error during model tracking: {e}")
            return frame, []

        try:
            # Detect ArUco markers
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
            parameters = cv2.aruco.DetectorParameters()
            parameters.adaptiveThreshWinSizeMin = 5
            parameters.adaptiveThreshWinSizeMax = 23
            parameters.adaptiveThreshWinSizeStep = 10
            parameters.minMarkerPerimeterRate = 0.03
            parameters.maxMarkerPerimeterRate = 4.0
            parameters.minMarkerDistanceRate = 0.05
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            aruco_ids = []
            if ids is not None:
                aruco_ids = ids.flatten().tolist()
                for i in range(len(ids)):
                    aruco_id = ids[i][0]
                    if ids[i] and len(ids[i]) > 0:
                        confidence = np.mean(ids[i])
                        if confidence >= 10:  # Only consider markers with confidence > 50%
                            if aruco_id not in self.detected_ids:
                                self.detected_ids.add(aruco_id)
                            else:
                                logging.info(f"ArUco ID {aruco_id} detected again.")
                            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                        logging.debug(f"Detected ArUco ID {aruco_id} with confidence {confidence}")
            else:
                aruco_ids = []

            logging.debug(f"Frame {frame_idx} - ArUco IDs: {aruco_ids}")

            counting_results = self.counter.start_counting(frame, tracks)

            if isinstance(counting_results, dict):
                frame = counting_results.get('frame', frame)
                counts = counting_results.get('counts', {})
                class_wise_count = counts.get('class_wise_count', {})

                cup_count = sum([direction_counts.get('IN', 0) + direction_counts.get('OUT', 0)
                                 for cls, direction_counts in class_wise_count.items()])

                if cup_count != len(aruco_ids):
                    log_violation(frame_idx, cup_count, len(aruco_ids), aruco_ids)
                    self.save_violation_frame(frame, frame_idx, aruco_ids)
                    self.crossed_objects["violation"].append((frame_idx, aruco_ids))

                for cls, direction_counts in class_wise_count.items():
                    for direction, count in direction_counts.items():
                        if count != self.last_class_wise_count[cls][direction]:
                            self.last_class_wise_count[cls][direction] = count
                            # log_movement(cls, count, direction)  # Remove this as it's not used anymore

        except AttributeError as e:
            print(f"Attribute error: {e}")

        return frame, aruco_ids


    def save_violation_frame(self, frame, frame_idx, aruco_ids):
        filename = f"violation_{frame_idx}.jpg"
        filepath = os.path.join('violations', filename)
        cv2.imwrite(filepath, frame)
        logging.info(f"Violation frame saved: {filepath}")

    def save_clips(self, video_path, clips_folder):
        for label, frames in self.crossed_objects.items():
            for frame_idx, aruco_ids in frames:
                clip_name = f"{label}_clip_{frame_idx}.mp4"
                clip_path = os.path.join(clips_folder, clip_name)
                self.extract_clip(video_path, clip_path, frame_idx, aruco_ids)

    def extract_clip(self, video_path, clip_path, frame_idx, aruco_ids, clip_length=2):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        start_frame = max(frame_idx - fps * clip_length // 2, 0)
        end_frame = frame_idx + fps * clip_length // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)
            current_frame += 1

        cap.release()
        out.release()
        logging.info(f"Clip saved: {clip_path}")
