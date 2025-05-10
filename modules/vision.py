# modules/vision.py
from ultralytics import YOLO
from .utils import capture_frame

# Load YOLOv8 model once
yolo_model = YOLO("yolov8n.pt")

# Indoor objects to detect
indoor_objects = [
    "person", "chair", "table", "sofa", "lamp", "bed", "shelf", "tv", "monitor",
    "laptop", "keyboard", "mouse", "door", "window", "fan", "plant", "cupboard",
    "mirror", "stool", "bookshelf", "microwave", "refrigerator"
]

def detect_objects_and_direction(camera_url):
    frame = capture_frame(camera_url)
    if frame is None:
        return {"error": "Could not capture frame from camera."}

    results = yolo_model(frame)
    detections = results[0].boxes.data.cpu().numpy()

    objects = []
    frame_width = frame.shape[1]

    left_blocked = right_blocked = False

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = yolo_model.names[int(cls)]

        if label in indoor_objects:
            objects.append({
                "label": label,
                "confidence": float(conf)
            })
            center_x = (x1 + x2) / 2
            if center_x < frame_width / 2:
                left_blocked = True
            else:
                right_blocked = True

    direction = "No clear path, proceed cautiously."
    if left_blocked and not right_blocked:
        direction = "Move right."
    elif right_blocked and not left_blocked:
        direction = "Move left."

    return {
        "detected_objects": objects,
        "count": len(objects),
        "direction_suggestion": direction
    }
