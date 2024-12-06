import cv2
import numpy as np
import torch
from ultralytics import YOLO as YOLOv8

class PoseEstimator:
    def __init__(self, yolo_model_path='yolov8x-pose.pt'):
        # Initialize YOLOv8 model for person detection
        self.yolo = YOLOv8(yolo_model_path).model

        # Confidence threshold for person detection
        self.confidence_threshold = 0.5

    def detect_people(self, frame):
        # Convert frame from NumPy array to torch.Tensor and normalize
        frame_tensor = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

        # Perform person detection using YOLOv8
        with torch.no_grad():
            results = self.yolo(frame_tensor)

        people_boxes = []

        # Filter out detections for 'person' class with confidence above threshold
        for detection in results.pred[0]:
            if detection[-1] >= self.confidence_threshold and detection[5] == 0:  # 0 corresponds to 'person' class
                xmin, ymin, xmax, ymax = map(int, detection[:4])
                people_boxes.append((xmin, ymin, xmax, ymax))

        return people_boxes

    def estimate_poses(self, frame, people_boxes):
        # Example placeholder for pose estimation using OpenCV (Replace with actual pose estimation logic)
        for xmin, ymin, xmax, ymax in people_boxes:
            # Draw a rectangle around the person
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Draw a placeholder pose estimation (example)
            # Replace this with your actual pose estimation logic
            pose_points = [(xmin, ymin), (xmax, ymax)]  # Dummy example, replace with actual pose estimation

            # Draw lines connecting keypoints to visualize pose
            for i in range(len(pose_points) - 1):
                if pose_points[i] and pose_points[i + 1]:
                    cv2.line(frame, pose_points[i], pose_points[i + 1], (0, 0, 255), 3)

        return frame

if __name__ == "__main__":
    # Video input and output paths
    video_path = 'cun4.mp4'
    output_video_path = 'output_video_with_poses.mp4'

    # Initialize pose estimator
    pose_estimator = PoseEstimator()

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get video details
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as well (e.g., *'XVID', *'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform person detection
        people_boxes = pose_estimator.detect_people(frame)

        # Perform pose estimation (example)
        frame_with_poses = pose_estimator.estimate_poses(frame, people_boxes)

        # Write the frame with annotations to the output video
        out.write(frame_with_poses)

        # Display the frame with annotations (optional)
        cv2.imshow('Pose Estimation', frame_with_poses)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()
