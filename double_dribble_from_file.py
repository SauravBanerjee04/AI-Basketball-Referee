#python3 double_dribble_from_file.py IMG_1134.mov --rotate 180


import cv2
import numpy as np
import time
from ultralytics import YOLO
import argparse

class DoubleDribbleDetector:
    def __init__(self, video_path: str, rotate_deg: int = 0):
        # Load YOLO models
        self.pose_model = YOLO("yolov8s-pose.pt")
        self.ball_model = YOLO("basketballModel.pt")

        # Open the video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")

        # Rotation correction
        self.rotate_deg = rotate_deg
        self.rotation_map = {
            90:  cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }

        # Wrist indices in keypoints array
        self.body_index = {"left_wrist": 10, "right_wrist": 9}

        # Holding detection state
        self.hold_start_time = None
        self.is_holding = False
        self.was_holding = False
        self.hold_duration = 0.85  # seconds to count as a hold
        self.hold_threshold = 300  # pixel distance threshold

        # Dribble detection state
        self.prev_y_center = None
        self.prev_delta_y = None
        self.dribble_count = 0
        self.dribble_threshold = 18

        # Double‑dribble flash timer
        self.double_dribble_time = None

        # For text positioning
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            # Rotate frame if requested
            if self.rotate_deg in self.rotation_map:
                frame = cv2.rotate(frame, self.rotation_map[self.rotate_deg])

            annotated, ball_detected = self.process_frame(frame)
            self.check_double_dribble()

            # Flash red overlay on double‑dribble
            if self.double_dribble_time and time.time() - self.double_dribble_time <= 3:
                red = np.full_like(annotated, (0,0,255), dtype=np.uint8)
                annotated = cv2.addWeighted(annotated, 0.7, red, 0.3, 0)
                cv2.putText(
                    annotated, "Double dribble!",
                    (self.frame_width - 600, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4, cv2.LINE_AA
                )

            cv2.imshow("Basketball Referee AI", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        # 1) Pose detection
        res = self.pose_model(frame, verbose=False, conf=0.5)[0]
        annotated = res.plot()

        # 2) Extract keypoints tensor
        kps_tensor = res.keypoints.data
        if kps_tensor is None or kps_tensor.numel() == 0:
            return annotated, False

        # 3) Convert to NumPy and round
        kps_np = kps_tensor.cpu().numpy()
        rounded = np.round(kps_np, 1)

        # 4) Extract wrist coordinates
        try:
            left_wrist  = rounded[0][self.body_index["left_wrist"]]
            right_wrist = rounded[0][self.body_index["right_wrist"]]
        except Exception:
            return annotated, False

        # 5) Ball detection & logic
        ball_detected = False
        for ball_res in self.ball_model(frame, verbose=False, conf=0.65):
            for bbox in ball_res.boxes.xyxy:
                x1, y1, x2, y2 = bbox[:4]
                bx, by = (x1 + x2) / 2, (y1 + y2) / 2

                self.update_dribble_count(bx, by)
                self.prev_y_center = by
                ball_detected = True

                ld = np.hypot(bx - left_wrist[0],  by - left_wrist[1])
                rd = np.hypot(bx - right_wrist[0], by - right_wrist[1])
                self.check_holding(ld, rd)

                # Draw bounding box
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

                # Overlay info text
                info = [
                    f"Ball: ({bx:.1f},{by:.1f})",
                    f"L Wrist: ({left_wrist[0]:.1f},{left_wrist[1]:.1f})",
                    f"R Wrist: ({right_wrist[0]:.1f},{right_wrist[1]:.1f})",
                    f"Dist: {min(ld,rd):.1f}",
                    f"Holding: {'Yes' if self.is_holding else 'No'}",
                    f"Dribbles: {self.dribble_count}"
                ]
                for i, txt in enumerate(info):
                    cv2.putText(
                        annotated, txt,
                        (10, 20 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA
                    )

                # Blue tint if holding
                if self.is_holding:
                    blue = np.full_like(annotated, (255,0,0), dtype=np.uint8)
                    annotated = cv2.addWeighted(annotated, 0.7, blue, 0.3, 0)

        # 6) Reset hold when ball disappears
        if not ball_detected:
            self.hold_start_time = None
            self.is_holding = False

        return annotated, ball_detected

    def check_holding(self, ld, rd):
        if min(ld, rd) < self.hold_threshold:
            if self.hold_start_time is None:
                self.hold_start_time = time.time()
            elif time.time() - self.hold_start_time > self.hold_duration:
                self.is_holding = True
                self.was_holding = True
                self.dribble_count = 0
        else:
            self.hold_start_time = None
            self.is_holding = False

    def update_dribble_count(self, x, y):
        if self.prev_y_center is not None:
            dy = y - self.prev_y_center
            if self.prev_delta_y is not None and dy < 0 and self.prev_delta_y > self.dribble_threshold:
                self.dribble_count += 1
            self.prev_delta_y = dy

    def check_double_dribble(self):
        if self.was_holding and self.dribble_count > 0:
            print("Double dribble!")
            self.double_dribble_time = time.time()
            self.was_holding = False
            self.dribble_count = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect double dribble violations in a video file."
    )
    parser.add_argument(
        "video_path",
        help="Path to the input .mov (or other) video file."
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=0,
        choices=[0, 90, 180, 270],
        help="Rotate each frame by this many degrees to correct orientation."
    )
    args = parser.parse_args()

    detector = DoubleDribbleDetector(args.video_path, rotate_deg=args.rotate)
    detector.run()
