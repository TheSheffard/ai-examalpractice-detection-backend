"""
M1 — Prohibited object detection.

Uses YOLOv8n (the nano variant — smallest, fastest, CPU-friendly). It's
pretrained on the COCO dataset, which already includes "cell phone" and
"book" as classes, so no custom training or dataset collection is needed.
We also keep "person" detections, since those become the base bounding
boxes shown in the live classroom overlay (green = normal person).
"""
from ultralytics import YOLO
from app.config import settings

_model = YOLO("yolov8n.pt")

# Only these COCO classes matter for this system.
TARGET_CLASSES = {"person", "cell phone", "book"}
FLAGGED_CLASSES = {"cell phone", "book"}

# Friendlier text for the invigilator's screen than the raw COCO class name.
DISPLAY_LABEL = {
    "person": "Person",
    "cell phone": "Phone detected",
    "book": "Book detected",
}


def detect_objects(frame):
    """
    Runs YOLOv8n on a single BGR frame (as read by OpenCV).
    Returns a list of dicts: label, confidence, bbox (normalized 0-1, x/y/w/h), flagged (bool)
    """
    h, w = frame.shape[:2]
    results = _model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        raw_label = _model.names[int(box.cls)]
        if raw_label not in TARGET_CLASSES:
            continue

        conf = float(box.conf)
        if conf < settings.OBJECT_CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "label": DISPLAY_LABEL.get(raw_label, raw_label),
            "confidence": conf,
            "flagged": raw_label in FLAGGED_CLASSES,
            "bbox": {
                "x": x1 / w,
                "y": y1 / h,
                "w": (x2 - x1) / w,
                "h": (y2 - y1) / h,
            },
        })
    return detections