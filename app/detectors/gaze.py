"""
M3 — Head and gaze direction tracking.

Uses MediaPipe Face Mesh (pretrained) to get facial landmarks, then a
standard OpenCV solvePnP calculation to estimate head yaw/pitch — this is
geometry, not a trained model, so there's nothing to train here either.

Sustained turning away from the candidate's own desk area is flagged; a
single frame of looking away is not enough on its own (see fusion.py for
how persistence across frames is handled).
"""
import cv2
import numpy as np
import mediapipe as mp

_mp_face = mp.solutions.face_mesh
_face_mesh = _mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Landmark indices used for the solvePnP head-pose estimate
_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]  # nose tip, chin, eye corners, mouth corners

# Generic 3D face model points corresponding to the landmarks above
_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -330.0, -65.0),     # chin
    (-225.0, 170.0, -135.0),  # left eye corner
    (225.0, 170.0, -135.0),   # right eye corner
    (-150.0, -150.0, -125.0), # left mouth corner
    (150.0, -150.0, -125.0),  # right mouth corner
], dtype=np.float64)

YAW_THRESHOLD_DEGREES = 25.0  # beyond this, we consider the head "turned away"


def detect_gaze_signal(frame):
    """
    Runs MediaPipe Face Mesh + solvePnP on a single BGR frame.
    Returns a list with zero or one entry: label, confidence, bbox (normalized 0-1), flagged (bool)
    """
    h, w = frame.shape[:2]
    rgb = frame[:, :, ::-1]
    result = _face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return []

    face_landmarks = result.multi_face_landmarks[0]
    lm = face_landmarks.landmark

    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    bbox = {
        "x": max(min(xs), 0.0),
        "y": max(min(ys), 0.0),
        "w": min(max(xs), 1.0) - max(min(xs), 0.0),
        "h": min(max(ys), 1.0) - max(min(ys), 0.0),
    }

    image_points = np.array(
        [(lm[i].x * w, lm[i].y * h) for i in _LANDMARK_IDS], dtype=np.float64
    )

    focal_length = w
    camera_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        _MODEL_POINTS, image_points, camera_matrix, dist_coeffs
    )
    if not success:
        return []

    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    sy = (rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2) ** 0.5
    yaw = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))

    turned_away = abs(yaw) > YAW_THRESHOLD_DEGREES

    return [{
        "label": "Head turned away" if turned_away else "Face forward",
        "confidence": min(abs(yaw) / 90.0, 1.0),
        "flagged": turned_away,
        "bbox": bbox,
    }]
