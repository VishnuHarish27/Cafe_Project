import logging
import cv2
import os
from collections import defaultdict
from datetime import datetime, timedelta
from ultralytics import YOLO as YOLOv8, solutions
from database import log_movement, clear_table

class YOLO:
    def __init__(self, use_gpu=False):
        device = 'cuda:0' if use_gpu else 'cpu'
        self.model = YOLOv8('yolov8x.pt').to(device)
        self.classes = self.model.names
        self.region_points = [(333,354), (528,406), (706,-1),(501,-1), (282,261), (366,291)]  # Updated region points
        self.classes_to_count = [41]  # Updated classes to count
        # Parameters for counting and tracking
        self.buffer_zone = 30  # Buffer zone around boundary line
        self.merge_window = timedelta(seconds=2)  # Time window to merge events

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
        self.last_logged_event = defaultdict(lambda: {"timestamp": None, "frame_idx": None})

    def detect(self, frame, frame_idx):
        try:
            tracks = self.model.track(frame, persist=True, show=False, classes=self.classes_to_count)
        except Exception as e:
            print(f"Error during model tracking: {e}")
            return frame

        try:
            counting_results = self.counter.start_counting(frame, tracks)

            if isinstance(counting_results, dict):
                frame = counting_results.get('frame', frame)
                counts = counting_results.get('counts', {})
                class_wise_count = counts.get('class_wise_count', {})

                for cls, direction_counts in class_wise_count.items():
                    in_count = direction_counts.get('IN', 0)
                    out_count = direction_counts.get('OUT', 0)

                    # Log movements to the database if counts have changed
                    if in_count > self.last_class_wise_count[cls]['IN']:
                        self.log_event('IN', in_count, cls, frame_idx)
                        self.last_class_wise_count[cls]['IN'] = in_count

                    if out_count > self.last_class_wise_count[cls]['OUT']:
                        self.log_event('OUT', out_count, cls, frame_idx)
                        self.last_class_wise_count[cls]['OUT'] = out_count

            else:
                print("Error: 'start_counting' did not return the expected dictionary format")

        except Exception as e:
            print(f"Error during counting: {e}")
            return frame

        return frame

    def log_event(self, direction, count, cls, frame_idx):
        current_time = datetime.now()
        last_event_time = self.last_logged_event[cls]['timestamp']
        last_event_frame_idx = self.last_logged_event[cls]['frame_idx']

        if last_event_time and current_time - last_event_time <= self.merge_window:
            # Merge the event
            frame_idx = min(last_event_frame_idx, frame_idx)
            self.last_logged_event[cls]['frame_idx'] = frame_idx
        else:
            # Log a new event
            log_movement(direction, count, cls, frame_idx, current_time.strftime('%Y-%m-%d %H:%M:%S'))
            self.crossed_objects[cls].append((direction, frame_idx, current_time.strftime('%Y-%m-%d %H:%M:%S')))
            self.last_logged_event[cls] = {"timestamp": current_time, "frame_idx": frame_idx}

    def save_clips(self, video_path, clip_folder):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for cls, events in self.crossed_objects.items():
            events.sort(key=lambda x: x[1])
            merged_events = []
            last_event = None

            for direction, frame_idx, timestamp in events:
                if last_event is None or frame_idx - last_event[1] > fps * 2:
                    merged_events.append((direction, frame_idx, timestamp))
                last_event = (direction, frame_idx, timestamp)

            for direction, frame_idx, timestamp in merged_events:
                clip_start = frame_idx
                clip_end = min(frame_idx + 130, frame_count)

                cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)

                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > clip_end:
                        break
                    frames.append(frame)

                if frames:
                    filename = f"{cls}_exchange_{frame_idx}.mp4"
                    logging.debug(f"Preparing to save clip: {filename}")
                    self.save_clip(frames, filename, clip_folder)

        cap.release()

    @staticmethod
    def save_clip(frames, filename, clip_folder):
        clip_directory = os.path.abspath(clip_folder)
        if not os.path.exists(clip_directory):
            os.makedirs(clip_directory)
            logging.debug(f"Created directory: {clip_directory}")
        filepath = os.path.join(clip_directory, filename)
        logging.debug(f"Saving to {filepath}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(filepath, fourcc, 20.0, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()
        logging.debug(f"Clip saved: {filepath}")

logging.basicConfig(level=logging.DEBUG)
