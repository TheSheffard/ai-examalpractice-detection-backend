# """
# M2 — Body pose estimation.

# Uses MediaPipe Pose, pretrained, no training required. We extract wrist and
# shoulder landmarks and apply simple rule-based checks for movement patterns
# associated with malpractice: a hand dropping well below desk/shoulder level
# (reaching into a bag, pulling out a note) or a pronounced forward lean.

# Landmark persistence across frames (e.g. "hand has been down for 2+ seconds")
# is handled by the fusion module, not here — this module only reports what it
# sees in a single frame.
# """
# import mediapipe as mp

# # Access Pose solution directly from mp.solutions
# _mp_pose = mp.solutions.pose
# _pose = _mp_pose.Pose(
#     static_image_mode=False,
#     model_complexity=0,  # lightest model variant — CPU-friendly
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5,
# )

# # How far below the shoulder (as a fraction of frame height) a wrist has to
# # drop before we consider it a "hand lowered" signal worth tracking.
# HAND_DROP_MARGIN = 0.12


# def detect_pose_signals(frame):
#     """
#     Runs MediaPipe Pose on a single BGR frame.
#     Returns a list of dicts: label, confidence, bbox (normalized 0-1), flagged (bool)
#     One entry per detected person.
#     """
#     rgb = frame[:, :, ::-1]
#     result = _pose.process(rgb)

#     if not result.pose_landmarks:
#         return []

#     lm = result.pose_landmarks.landmark
#     xs = [p.x for p in lm]
#     ys = [p.y for p in lm]
#     bbox = {
#         "x": max(min(xs), 0.0),
#         "y": max(min(ys), 0.0),
#         "w": min(max(xs), 1.0) - max(min(xs), 0.0),
#         "h": min(max(ys), 1.0) - max(min(ys), 0.0),
#     }

#     left_wrist_y = lm[_mp_pose.PoseLandmark.LEFT_WRIST].y
#     right_wrist_y = lm[_mp_pose.PoseLandmark.RIGHT_WRIST].y
#     left_shoulder_y = lm[_mp_pose.PoseLandmark.LEFT_SHOULDER].y
#     right_shoulder_y = lm[_mp_pose.PoseLandmark.RIGHT_SHOULDER].y

#     hand_dropped = (
#         left_wrist_y > left_shoulder_y + HAND_DROP_MARGIN
#         and right_wrist_y > right_shoulder_y + HAND_DROP_MARGIN
#     )

#     if hand_dropped:
#         return [{
#             "label": "Hand below desk level",
#             "confidence": 0.7,
#             "flagged": True,
#             "bbox": bbox,
#         }]

#     return [{
#         "label": "Person",
#         "confidence": 1.0,
#         "flagged": False,
#         "bbox": bbox,
#     }]

"""
M2 — Per-person body pose estimation.

This module evaluates one detected person crop at a time.
It returns pose signals only. It does not decide whether malpractice
occurred. The fusion layer must apply persistence across multiple frames.

For multi-person monitoring, the caller should pass the person's
bounding box from the original classroom frame through `person_bbox`.
"""

import cv2
import mediapipe as mp


_mp_pose = mp.solutions.pose

_pose = _mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65,
)


MIN_LANDMARK_VISIBILITY = 0.60

# These values are relative to the person's crop.
# They will need calibration using your actual camera view.
DESK_LINE_Y = 0.78
DESK_MARGIN = 0.06
ELBOW_DROP_MARGIN = 0.08


def _visible(landmark) -> bool:
    """
    Checks whether MediaPipe considers a landmark reliable enough.
    """
    return getattr(landmark, "visibility", 0.0) >= MIN_LANDMARK_VISIBILITY


def _default_bbox() -> dict:
    """
    Fallback bounding box for a crop when the caller does not provide
    the person's original classroom-frame bounding box.

    For correct multi-person overlays, always pass person_bbox.
    """
    return {
        "x": 0.0,
        "y": 0.0,
        "w": 1.0,
        "h": 1.0,
    }


def detect_pose_signals(person_crop, person_bbox=None):
    """
    Analyze one person's cropped image.

    Parameters:
        person_crop:
            OpenCV BGR image containing one detected person.

        person_bbox:
            That person's normalized bounding box in the original
            classroom frame. Example:

            {
                "x": 0.35,
                "y": 0.20,
                "w": 0.15,
                "h": 0.55
            }

    Returns:
        A list containing one pose signal.

    Important:
        This function does not apply time persistence. A possible
        signal must remain present for multiple frames before the
        fusion module creates an alert.
    """

    # Always guarantee that a bbox exists in every returned result.
    bbox = person_bbox if person_bbox is not None else _default_bbox()

    if person_crop is None or person_crop.size == 0:
        return [{
            "label": "Person",
            "module": "pose",
            "confidence": 0.0,
            "flagged": False,
            "signal": "invalid_crop",
            "bbox": bbox,
        }]

    # MediaPipe expects RGB images.
    rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
    result = _pose.process(rgb)

    if not result.pose_landmarks:
        return [{
            "label": "Person",
            "module": "pose",
            "confidence": 0.0,
            "flagged": False,
            "signal": "pose_not_detected",
            "bbox": bbox,
        }]

    landmarks = result.pose_landmarks.landmark

    left_wrist = landmarks[_mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[_mp_pose.PoseLandmark.RIGHT_WRIST]

    left_elbow = landmarks[_mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[_mp_pose.PoseLandmark.RIGHT_ELBOW]

    visible_wrists = []

    if _visible(left_wrist) and _visible(left_elbow):
        visible_wrists.append(
            ("left", left_wrist, left_elbow)
        )

    if _visible(right_wrist) and _visible(right_elbow):
        visible_wrists.append(
            ("right", right_wrist, right_elbow)
        )

    # If neither wrist is reliable, do not call it suspicious.
    if not visible_wrists:
        return [{
            "label": "Person",
            "module": "pose",
            "confidence": 0.0,
            "flagged": False,
            "signal": "wrists_not_visible",
            "bbox": bbox,
        }]

    suspicious_hands = []

    for side, wrist, elbow in visible_wrists:
        # MediaPipe y coordinates increase downwards.
        wrist_below_desk = (
            wrist.y > DESK_LINE_Y + DESK_MARGIN
        )

        # Require the wrist to be meaningfully below the elbow.
        # This avoids flagging every normal seated writing posture.
        arm_is_extended_downward = (
            wrist.y > elbow.y + ELBOW_DROP_MARGIN
        )

        if wrist_below_desk and arm_is_extended_downward:
            suspicious_hands.append(side)

    if suspicious_hands:
        confidence = min(
            0.95,
            0.60 + (len(suspicious_hands) * 0.10),
        )

        return [{
            "label": "Possible hand below desk boundary",
            "module": "pose",
            "confidence": confidence,

            # Keep this False here.
            # The fusion module should decide whether the signal
            # persists long enough to become a real alert.
            "flagged": False,

            "signal": "hand_below_desk",
            "hands": suspicious_hands,
            "bbox": bbox,
        }]

    return [{
        "label": "Person",
        "module": "pose",
        "confidence": 1.0,
        "flagged": False,
        "signal": "normal",
        "bbox": bbox,
    }]