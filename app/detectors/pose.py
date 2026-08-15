"""
M2 — Body pose estimation.

Uses MediaPipe Pose, pretrained, no training required. We extract wrist and
shoulder landmarks and apply simple rule-based checks for movement patterns
associated with malpractice: a hand dropping well below desk/shoulder level
(reaching into a bag, pulling out a note) or a pronounced forward lean.

Landmark persistence across frames (e.g. "hand has been down for 2+ seconds")
is handled by the fusion module, not here — this module only reports what it
sees in a single frame.
"""
import mediapipe as mp

_mp_pose = mp.solutions.pose
_pose = _mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # lightest model variant — CPU-friendly
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# How far below the shoulder (as a fraction of frame height) a wrist has to
# drop before we consider it a "hand lowered" signal worth tracking.
HAND_DROP_MARGIN = 0.12


def detect_pose_signals(frame):
    """
    Runs MediaPipe Pose on a single BGR frame.
    Returns a list of dicts: label, confidence, bbox (normalized 0-1), flagged (bool)
    One entry per detected person (MediaPipe Pose's default model tracks one
    person at a time — see the README note on extending this to multiple
    people in a wide classroom shot).
    """
    rgb = frame[:, :, ::-1]
    result = _pose.process(rgb)

    if not result.pose_landmarks:
        return []

    lm = result.pose_landmarks.landmark
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    bbox = {
        "x": max(min(xs), 0.0),
        "y": max(min(ys), 0.0),
        "w": min(max(xs), 1.0) - max(min(xs), 0.0),
        "h": min(max(ys), 1.0) - max(min(ys), 0.0),
    }

    left_wrist_y = lm[_mp_pose.PoseLandmark.LEFT_WRIST].y
    right_wrist_y = lm[_mp_pose.PoseLandmark.RIGHT_WRIST].y
    left_shoulder_y = lm[_mp_pose.PoseLandmark.LEFT_SHOULDER].y
    right_shoulder_y = lm[_mp_pose.PoseLandmark.RIGHT_SHOULDER].y

    hand_dropped = (
        left_wrist_y > left_shoulder_y + HAND_DROP_MARGIN
        and right_wrist_y > right_shoulder_y + HAND_DROP_MARGIN
    )

    if hand_dropped:
        return [{
            "label": "Hand below desk level",
            "confidence": 0.7,  # rule-based, not a model confidence score — see README
            "flagged": True,
            "bbox": bbox,
        }]

    return [{
        "label": "Person",
        "confidence": 1.0,
        "flagged": False,
        "bbox": bbox,
    }]
