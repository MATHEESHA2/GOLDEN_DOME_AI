import cv2
from ultralytics import YOLO
import torch
import shared_data


# --- Configuration ---
# Classes to detect (COCO IDs)
REQUIRED_CLASS_IDS = [49, 32, 54]  

# Your custom labels
CUSTOM_LABELS = {
    49: 'Missile',
    32: 'Fighter Jet',
    54: 'Bomb'
}

# Load YOLO model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = YOLO("yolov11m.pt").to(device)

# Apply custom names
model._names = model.names.copy()
model._names.update(CUSTOM_LABELS)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Golden Dome System Activated...")
print("Tracking only:", REQUIRED_CLASS_IDS)
print("Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    H, W, _ = frame.shape
    shared_data.screen_width = W
    shared_data.screen_height = H
    shared_data.screen_center_x = W // 2
    shared_data.screen_center_y = H // 2

    if not success:
        print("Error reading frame")
        break

    # Run tracker
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=REQUIRED_CLASS_IDS, conf=0.1,iou=0.5, imgsz=720)

    for r in results:
        boxes = r.boxes  # Bounding boxes

        # Draw YOLO boxes first
        frame = r.plot(conf=False, boxes=True, labels=False)


        if boxes is not None:
            for box in boxes:
                # Extract box data
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                track_id = int(box.id[0]) if box.id is not None else -1

                # Compute center of object
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                shared_data.object_center_x = cx
                shared_data.object_center_y = cy
                shared_data.object_class = model._names[cls_id]
                shared_data.object_track_id = track_id


                # Draw center point
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)  # Yellow dot

                # Draw a line or label for center tracking
                cv2.putText(frame,
                            f"ID:{track_id} {model._names[cls_id]}",
                            (cx + 10, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2)

    # Show output
    cv2.imshow("Golden Dome - Object Tracking", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
