# """
# M1 — Prohibited object detection.

# Uses YOLOv8n (the nano variant — smallest, fastest, CPU-friendly). It's
# pretrained on the COCO dataset, which already includes "cell phone" and
# "book" as classes, so no custom training or dataset collection is needed.
# We also keep "person" detections, since those become the base bounding
# boxes shown in the live classroom overlay (green = normal person).
# """
# from ultralytics import YOLO
# from app.config import settings

# _model = YOLO("yolov8n.pt")

# # Only these COCO classes matter for this system.
# TARGET_CLASSES = {"person", "cell phone", "book"}
# FLAGGED_CLASSES = {"cell phone", "book"}

# # Friendlier text for the invigilator's screen than the raw COCO class name.
# DISPLAY_LABEL = {
#     "person": "Person",
#     "cell phone": "Phone detected",
#     "book": "Book detected",
# }


# def detect_objects(frame):
#     """
#     Runs YOLOv8n on a single BGR frame (as read by OpenCV).
#     Returns a list of dicts: label, confidence, bbox (normalized 0-1, x/y/w/h), flagged (bool)
#     """
#     h, w = frame.shape[:2]
#     results = _model(frame, verbose=False)[0]

#     detections = []
#     for box in results.boxes:
#         raw_label = _model.names[int(box.cls)]
#         if raw_label not in TARGET_CLASSES:
#             continue

#         conf = float(box.conf)
#         if conf < settings.OBJECT_CONFIDENCE_THRESHOLD:
#             continue

#         x1, y1, x2, y2 = box.xyxy[0].tolist()
#         detections.append({
#             "label": DISPLAY_LABEL.get(raw_label, raw_label),
#             "confidence": conf,
#             "flagged": raw_label in FLAGGED_CLASSES,
#             "bbox": {
#                 "x": x1 / w,
#                 "y": y1 / h,
#                 "w": (x2 - x1) / w,
#                 "h": (y2 - y1) / h,
#             },
#         })
#     return detections

"""
M1 — Multi-person and prohibited-object detection.

YOLO detects:
    - person
    - cell phone
    - book

This module also enables persistent person tracking using ByteTrack.

Important:
    Person detection and person tracking are different things.

    Detection answers:
        "What is visible in this frame?"

    Tracking answers:
        "Is this the same person that appeared in the previous frame?"

This module does not perform pose or head-direction analysis.
Those modules should receive one person crop at a time.
"""

from ultralytics import YOLO

from app.config import settings


_model = YOLO("yolov8n.pt")


TARGET_CLASSES = {
    "person",
    "cell phone",
    "book",
}


FLAGGED_CLASSES = {
    "cell phone",
    "book",
}


DISPLAY_LABEL = {
    "person": "Person",
    "cell phone": "Phone detected",
    "book": "Book detected",
}


# Different classes need different confidence thresholds.
#
# Person detection should be sensitive because missed persons
# cannot be analyzed later.
#
# Phone and book detections should be stricter to reduce false alerts.
CLASS_CONFIDENCE = {
    "person": 0.35,
    "cell phone": 0.55,
    "book": 0.60,
}


def _clip(value: float, minimum: float = 0.0, maximum: float = 1.0):
    return max(minimum, min(maximum, value))


def _normalise_bbox(x1, y1, x2, y2, frame_width, frame_height):
    """
    Converts pixel coordinates into normalized coordinates.
    """

    x1 = _clip(x1 / frame_width)
    y1 = _clip(y1 / frame_height)
    x2 = _clip(x2 / frame_width)
    y2 = _clip(y2 / frame_height)

    return {
        "x": x1,
        "y": y1,
        "w": max(0.0, x2 - x1),
        "h": max(0.0, y2 - y1),
    }


def detect_objects(frame):
    """
    Detects and tracks multiple people and prohibited objects.

    Returns a list of dictionaries containing:

        label
        module
        signal
        confidence
        flagged
        bbox
        person_id

    person_id is available for tracked people when ByteTrack
    successfully assigns an ID.
    """

    frame_height, frame_width = frame.shape[:2]

    if frame_width <= 0 or frame_height <= 0:
        return []

    configured_confidence = getattr(
        settings,
        "OBJECT_CONFIDENCE_THRESHOLD",
        0.35,
    )

    # Use tracking instead of plain model(frame).
    #
    # persist=True tells Ultralytics to keep track identities
    # across consecutive calls.
    results = _model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=960,
        conf=configured_confidence,
        iou=0.50,
        verbose=False,
    )

    if not results:
        return []

    result = results[0]
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        raw_label = _model.names[class_id]

        if raw_label not in TARGET_CLASSES:
            continue

        confidence = float(box.conf[0])
        required_confidence = CLASS_CONFIDENCE.get(
            raw_label,
            configured_confidence,
        )

        if confidence < required_confidence:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        bbox = _normalise_bbox(
            x1,
            y1,
            x2,
            y2,
            frame_width,
            frame_height,
        )

        # ByteTrack may not assign an ID immediately.
        track_id = None

        if box.id is not None:
            track_id = f"person_{int(box.id[0])}"

        is_person = raw_label == "person"

        if is_person:
            signal_name = "person"
        elif raw_label == "cell phone":
            signal_name = "prohibited_object"
        elif raw_label == "book":
            signal_name = "prohibited_object"
        else:
            signal_name = "object"

        detections.append({
            "label": DISPLAY_LABEL.get(raw_label, raw_label),
            "module": "object_detection",
            "signal": signal_name,
            "raw_label": raw_label,
            "confidence": confidence,
            "flagged": raw_label in FLAGGED_CLASSES,
            "bbox": bbox,

            # Present for persons when tracking succeeds.
            # For phones/books, this is currently the object track ID,
            # not yet the owner/person ID.
            "person_id": track_id if is_person else None,

            # Useful for later object-to-person association.
            "track_id": track_id,
        })

    return detections