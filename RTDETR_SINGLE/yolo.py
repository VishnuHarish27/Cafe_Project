import cv2
from database import log_movement, clear_table  # Assuming log_movement function exists in the database module
from ultralytics import RTDETR, solutions
from collections import defaultdict

class YOLO:
    def __init__(self, use_gpu=False):
        device = 'cuda:0' if use_gpu else 'cpu'
        self.model = RTDETR("rtdetr-x.pt").to(device)
        self.classes = self.model.names
        self.boundary_line = [(49, 84), (640, 395), (739, 147), (330, 2)]  # Updated boundary line
        self.classes_to_count = [41]  # Updated classes to count

        # Parameters for counting and tracking
        self.buffer_zone = 30  # Buffer zone around boundary line

        # Init Object Counter
        try:
            self.counter = solutions.ObjectCounter(
                view_img=True,
                reg_pts=self.boundary_line,
                classes_names=self.classes,
                draw_tracks=True,
                line_thickness=2,
            )
        except AttributeError as e:
            raise ImportError(f"Error initializing ObjectCounter: {e}")

        self.crossed_in_objects = set()
        self.crossed_out_objects = set()
        self.last_class_wise_count = defaultdict(lambda: {"IN": 0, "OUT": 0})

    def detect(self, frame):
        try:
            tracks = self.model.track(frame, persist=True, show=False, classes=self.classes_to_count)
        except Exception as e:
            print(f"Error during model tracking: {e}")
            return frame

        # Debugging statement to check the type and content of the result
        print(f"Tracks: {tracks}")

        try:
            counting_results = self.counter.start_counting(frame, tracks)

            # Debugging statement to check the type and content of the result
            print(f"Counting Results Type: {type(counting_results)}")
            print(f"Counting Results: {counting_results}")

            if isinstance(counting_results, dict):
                frame = counting_results.get('frame', frame)
                counts = counting_results.get('counts', {})

                # Debugging statement to check the structure of counts
                print(f"Counts: {counts}")

                class_wise_count = counts.get('class_wise_count', {})
                for cls, direction_counts in class_wise_count.items():
                    in_count = direction_counts.get('OUT', 0)
                    out_count = direction_counts.get('IN', 0)

                    # Log movements to the database if counts have changed
                    if in_count > self.last_class_wise_count[cls]['IN']:
                        log_movement('IN', in_count - self.last_class_wise_count[cls]['IN'], cls)
                        self.last_class_wise_count[cls]['IN'] = in_count
                    if out_count > self.last_class_wise_count[cls]['OUT']:
                        log_movement('OUT', out_count - self.last_class_wise_count[cls]['OUT'], cls)
                        self.last_class_wise_count[cls]['OUT'] = out_count

                # Display the number of crossed objects
                num_crossed_in = counts.get('out_counts', 0)  # Switched from 'in_counts' to 'out_counts'
                num_crossed_out = counts.get('in_counts', 0)  # Switched from 'out_counts' to 'in_counts'
                #cv2.putText(frame, f'IN: {num_crossed_in}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                #cv2.putText(frame, f'OUT: {num_crossed_out}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                print("Error: 'start_counting' did not return the expected dictionary format")

        except Exception as e:
            print(f"Error during counting: {e}")
            return frame

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
