"""
M3 — Per-person head direction tracking.

This estimates head pose, specifically yaw and pitch.
It does not claim to know where the eyes are looking.

The function should be called once per detected person crop.
Temporal persistence and alert generation belong in fusion.py.
"""

import cv2
import numpy as np
import mediapipe as mp


_mp_face = mp.solutions.face_mesh

_face_mesh = _mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65,
)


_LANDMARK_IDS = [
    1,    # nose tip
    152,  # chin
    33,   # left eye corner
    263,  # right eye corner
    61,   # left mouth corner
    291,  # right mouth corner
]


_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


# Hysteresis prevents the status from constantly switching
# when yaw is near the threshold.
TURN_ENTER_THRESHOLD = 30.0
TURN_CLEAR_THRESHOLD = 20.0

MIN_FACE_WIDTH_RATIO = 0.08


def _default_bbox():
    return {
        "x": 0.0,
        "y": 0.0,
        "w": 1.0,
        "h": 1.0,
    }


def _clamp(value):
    return max(0.0, min(1.0, float(value)))


def detect_gaze_signal(person_crop, person_bbox=None):
    """
    Analyze one person's crop.

    person_bbox must be the person's bounding box in the original
    classroom frame when available.
    """

    bbox = person_bbox if person_bbox is not None else _default_bbox()

    if person_crop is None or person_crop.size == 0:
        return [{
            "label": "Person",
            "module": "head_pose",
            "signal": "invalid_crop",
            "confidence": 0.0,
            "flagged": False,
            "bbox": bbox,
        }]

    height, width = person_crop.shape[:2]

    if width <= 0 or height <= 0:
        return [{
            "label": "Person",
            "module": "head_pose",
            "signal": "invalid_dimensions",
            "confidence": 0.0,
            "flagged": False,
            "bbox": bbox,
        }]

    rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
    result = _face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return [{
            "label": "Person",
            "module": "head_pose",
            "signal": "face_not_detected",
            "confidence": 0.0,
            "flagged": False,
            "bbox": bbox,
        }]

    face_landmarks = result.multi_face_landmarks[0]
    landmarks = face_landmarks.landmark

    xs = [_clamp(point.x) for point in landmarks]
    ys = [_clamp(point.y) for point in landmarks]

    face_x_min = min(xs)
    face_x_max = max(xs)
    face_y_min = min(ys)
    face_y_max = max(ys)

    face_width = face_x_max - face_x_min
    face_height = face_y_max - face_y_min

    # Very small faces produce unstable solvePnP results.
    if face_width < MIN_FACE_WIDTH_RATIO:
        return [{
            "label": "Person",
            "module": "head_pose",
            "signal": "face_too_small",
            "confidence": 0.0,
            "flagged": False,
            "bbox": bbox,
        }]

    image_points = np.array(
        [
            (
                landmarks[index].x * width,
                landmarks[index].y * height,
            )
            for index in _LANDMARK_IDS
        ],
        dtype=np.float64,
    )

    focal_length = float(width)

    camera_matrix = np.array(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        _MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return [{
            "label": "Person",
            "module": "head_pose",
            "signal": "head_pose_unavailable",
            "confidence": 0.0,
            "flagged": False,
            "bbox": bbox,
        }]

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    sy = (
        rotation_matrix[0, 0] ** 2
        + rotation_matrix[1, 0] ** 2
    ) ** 0.5

    yaw = float(
        np.degrees(
            np.arctan2(
                -rotation_matrix[2, 0],
                sy,
            )
        )
    )

    pitch = float(
        np.degrees(
            np.arctan2(
                rotation_matrix[2, 1],
                rotation_matrix[2, 2],
            )
        )
    )

    # This is only a raw signal.
    # Fusion should decide whether it persists long enough.
    turned_away = abs(yaw) >= TURN_ENTER_THRESHOLD

    confidence = min(
        0.95,
        max(0.0, abs(yaw) / 90.0),
    )

    return [{
        "label": (
            "Possible sustained head turn"
            if turned_away
            else "Head direction normal"
        ),
        "module": "head_pose",
        "signal": (
            "head_turn"
            if turned_away
            else "normal"
        ),
        "confidence": confidence,
        "flagged": False,
        "yaw": yaw,
        "pitch": pitch,
        "bbox": bbox,
    }]