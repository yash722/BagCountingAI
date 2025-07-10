import cv2
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import datetime
import os

# Load YOLOv8 model
model = YOLO('weights/best_sack_det.pt')

# Load video
video_path = "videos/video1_LxLcshS2.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video file '{video_path}'")

# Define rectangular counting zone
zone_top_left = (400, 200)
zone_bottom_right = (600, 400)

# Initialize counter and state tracking
count = 0
object_states = {}

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=60)

# Frame skipping setup
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps / 10)

# OpenCV window
cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RGB", 1020, 500)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval != 0:
        frame_count += 1
        continue
    frame_count += 1

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame, verbose=False)[0]

    detections = []
    if results.boxes.data.numel() > 0:
        det_data = results.boxes.data.cpu().numpy()
        for *bbox, conf, cls in det_data:
            x1, y1, x2, y2 = map(int, bbox)
            conf = float(conf)
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "bag"))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw tracking box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Check if object is inside the zone
        inside = (zone_top_left[0] < cx < zone_bottom_right[0]) and \
                 (zone_top_left[1] < cy < zone_bottom_right[1])

        if track_id not in object_states:
            object_states[track_id] = {"was_inside": False}

        prev_inside = object_states[track_id]["was_inside"]

        # Object enters the zone
        if inside and not prev_inside:
            object_states[track_id]["was_inside"] = True

        # Object exits the zone
        elif not inside and prev_inside:
            # Exited from top
            if cy < zone_top_left[1]:
                count += 1
                print(f"[EXIT ↑] Count: {count} | ID: {track_id} | {datetime.datetime.now().replace(microsecond=0)}")

            # Exited from bottom
            elif cy > zone_bottom_right[1]:
                count -= 1
                print(f"[EXIT ↓] Count: {count} | ID: {track_id} | {datetime.datetime.now().replace(microsecond=0)}")

            # Update state
            object_states[track_id]["was_inside"] = False

    # Draw the counting zone
    cv2.rectangle(frame, zone_top_left, zone_bottom_right, (0, 255, 255), 2)
    cv2.putText(frame, 'Entry/Exit Zone', (zone_top_left[0], zone_top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display count
    cv2.putText(frame, f'Count: {count}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
