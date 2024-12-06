import cv2
from database import log_movement, clear_table
from ultralytics import RTDETR, solutions
from collections import defaultdict

class YOLO:
    def __init__(self, use_gpu=False):
        device = 'cuda:0' if use_gpu else 'cpu'
        self.model = RTDETR("rtdetr-x.pt").to(device)
        self.classes = self.model.names
        self.classes_to_count = [41]

        self.regions = [
            [(100, 866), (101, 555), (398, 510), (427, 832)],
            [(762,336), (687,71), (867,14), (991,254)],
            [(897,316), (1207,226), (1356,462), (1017,591)]
        ]

        # Parameters for counting and tracking
        self.buffer_zone = 30

        # Init Object Counters for each region
        self.counters = []
        for region in self.regions:
            counter = solutions.ObjectCounter(
                view_img=True,
                reg_pts=region,
                classes_names=self.classes,
                draw_tracks=True,
                line_thickness=2,
            )
            self.counters.append(counter)

        self.crossed_in_objects = [set(), set(), set()]
        self.crossed_out_objects = [set(), set(), set()]
        self.last_class_wise_count = [defaultdict(lambda: {"IN": 0, "OUT": 0}) for _ in range(3)]

    def detect(self, frame):
        try:
            tracks = self.model.track(frame, persist=True, show=False, classes=self.classes_to_count)
        except Exception as e:
            print(f"Error during model tracking: {e}")
            return frame

        for idx, counter in enumerate(self.counters):
            counting_results = counter.start_counting(frame, tracks)
            frame = counting_results.get('frame', frame)
            counts = counting_results.get('counts', {})
            class_wise_count = counts.get('class_wise_count', {})

            for cls, direction_counts in class_wise_count.items():
                in_count = direction_counts.get('IN', 0)
                out_count = direction_counts.get('OUT', 0)

                # Log movements to the database if counts have changed and are displayed in the video feed
                if in_count > self.last_class_wise_count[idx][cls]['IN']:
                    detected_in_count = in_count - self.last_class_wise_count[idx][cls]['IN']
                    for _ in range(detected_in_count):
                        log_movement(idx + 1, 'IN', 1, cls)
                    self.last_class_wise_count[idx][cls]['IN'] = in_count
                if out_count > self.last_class_wise_count[idx][cls]['OUT']:
                    detected_out_count = out_count - self.last_class_wise_count[idx][cls]['OUT']
                    for _ in range(detected_out_count):
                        log_movement(idx + 1, 'OUT', 1, cls)
                    self.last_class_wise_count[idx][cls]['OUT'] = out_count

        return frame

    def process_new_video(self, video_path):
        clear_table()
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.detect(frame)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
