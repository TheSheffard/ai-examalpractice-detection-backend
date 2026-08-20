"""
M4 — Fusion and persistence rule engine.

Combines:
    M1: prohibited-object detection
    M2: body-pose signals
    M3: head-direction signals

This module:
    1. Combines signals from all detectors.
    2. Displays all valid detections on the live camera overlay.
    3. Tracks suspicious signals across frames.
    4. Escalates only persistent signals into logged alerts.

This is not a trained model. It is a rule-based decision-support layer.
An alert is an indicator for human review, not automatic proof of malpractice.
"""

import time
import uuid

from app.config import settings


# In-memory tracking state.

# Example:
#
# {
#     "track-id": {
#         "label": "Possible hand below desk boundary",
#         "module": "pose",
#         "signal": "hand_below_desk",
#         "person_id": "person_03",
#         "zone": "B2",
#         "bbox": {...},
#         "confidence": 0.72,
#         "count": 3,
#         "last_seen": 1720000000.0,
#         "escalated": False
#     }
# }
_tracked = {}


MATCH_DISTANCE_THRESHOLD = 0.15
STALE_SECONDS = 3.0


# These are known non-suspicious states.
# They should be shown as normal detections but should not
# enter the alert persistence system.
NON_SUSPICIOUS_SIGNALS = {
    None,
    "",
    "normal",
    "pose_not_detected",
    "face_not_detected",
    "face_too_small",
    "face_pose_unavailable",
    "head_pose_unavailable",
    "wrists_not_visible",
    "invalid_crop",
    "invalid_dimensions",
}


def _bbox_center(bbox):
    """
    Returns the center point of a normalized bounding box.
    """

    return (
        float(bbox.get("x", 0.0)) + float(bbox.get("w", 0.0)) / 2.0,
        float(bbox.get("y", 0.0)) + float(bbox.get("h", 0.0)) / 2.0,
    )


def _distance(point_a, point_b):
    return (
        (point_a[0] - point_b[0]) ** 2
        + (point_a[1] - point_b[1]) ** 2
    ) ** 0.5


def _is_valid_bbox(bbox):
    """
    Checks that a bounding box exists and has the required fields.
    """

    if not isinstance(bbox, dict):
        return False

    required = ("x", "y", "w", "h")

    return all(key in bbox for key in required)


def _is_suspicious_signal(signal):
    """
    Determines whether a raw signal should enter persistence tracking.

    The detectors may return flagged=False because they only report
    a possible signal. Fusion is responsible for deciding whether
    that signal persists long enough to become an alert.
    """

    if signal.get("flagged") is True:
        return True

    signal_name = signal.get("signal")

    if signal_name in NON_SUSPICIOUS_SIGNALS:
        return False

    # Signals explicitly produced by the pose and head-pose modules.
    known_suspicious_signals = {
        "hand_below_desk",
        "head_turn",
        "prohibited_object",
        "phone_detected",
        "book_detected",
        "suspicious_movement",
    }

    return signal_name in known_suspicious_signals


def _same_context(tracked, signal):
    """
    Prevents signals from different people or different zones
    being incorrectly combined.

    If person_id is available, it is the strongest matching key.
    Otherwise, the system falls back to zone, module, and label.
    """

    tracked_person_id = tracked.get("person_id")
    signal_person_id = signal.get("person_id")

    if tracked_person_id and signal_person_id:
        return tracked_person_id == signal_person_id

    tracked_zone = tracked.get("zone")
    signal_zone = signal.get("zone")

    if tracked_zone and signal_zone:
        if tracked_zone != signal_zone:
            return False

    return (
        tracked.get("module", "") == signal.get("module", "")
        and tracked.get("label", "") == signal.get("label", "")
    )


def _find_match(signal):
    """
    Finds the closest existing track for the same suspicious signal.

    For multiple people, person_id or zone is used when available.
    Spatial distance is used as a fallback.
    """

    bbox = signal["bbox"]
    center = _bbox_center(bbox)

    best_id = None
    best_distance = MATCH_DISTANCE_THRESHOLD

    for track_id, tracked in _tracked.items():
        if not _same_context(tracked, signal):
            continue

        tracked_center = _bbox_center(tracked["bbox"])
        distance = _distance(center, tracked_center)

        if distance < best_distance:
            best_id = track_id
            best_distance = distance

    return best_id


def _prune_stale():
    """
    Removes tracks that have not appeared recently.
    """

    now = time.time()

    stale_ids = [
        track_id
        for track_id, tracked in _tracked.items()
        if now - tracked["last_seen"] > STALE_SECONDS
    ]

    for track_id in stale_ids:
        del _tracked[track_id]


def _normalise_signal(signal):
    """
    Ensures every detector returns a consistent structure.

    Invalid signals are ignored instead of crashing the WebSocket.
    """

    bbox = signal.get("bbox")

    if not _is_valid_bbox(bbox):
        print(
            "Skipping malformed signal without valid bbox:",
            signal,
        )
        return None

    return {
        "label": signal.get("label", "Unknown"),
        "module": signal.get("module", "unknown"),
        "signal": signal.get("signal"),
        "confidence": float(signal.get("confidence", 0.0)),
        "flagged": bool(signal.get("flagged", False)),
        "bbox": bbox,
        "person_id": signal.get("person_id"),
        "zone": signal.get("zone"),
    }


def process_frame(objects, pose_signals, gaze_signals):
    """
    Processes one classroom frame.

    Returns:

        detections:
            All valid current detections for the live frontend overlay.

        new_alerts:
            Signals that have persisted long enough to become alerts.
    """

    all_signals = (
        (objects or [])
        + (pose_signals or [])
        + (gaze_signals or [])
    )

    detections = []
    new_alerts = []
    now = time.time()

    for raw_signal in all_signals:
        signal = _normalise_signal(raw_signal)

        if signal is None:
            continue

        suspicious = _is_suspicious_signal(signal)

        # Use a stable track ID when possible.
        # Do not generate a completely random display ID for the same
        # person on every frame.
        display_id = (
            signal.get("person_id")
            or f'{signal["module"]}-{signal["label"]}'
        )

        detections.append({
            "id": display_id,
            "status": "flagged" if suspicious else "normal",
            "label": signal["label"],
            "module": signal["module"],
            "signal": signal["signal"],
            "confidence": signal["confidence"],
            "bbox": signal["bbox"],
            "person_id": signal.get("person_id"),
            "zone": signal.get("zone"),
        })

        # Normal detections are displayed but are not tracked
        # as suspicious events.
        if not suspicious:
            continue

        match_id = _find_match(signal)

        if match_id is not None:
            tracked = _tracked[match_id]

            tracked["bbox"] = signal["bbox"]
            tracked["confidence"] = max(
                tracked["confidence"],
                signal["confidence"],
            )
            tracked["last_seen"] = now
            tracked["count"] += 1

        else:
            match_id = str(uuid.uuid4())

            _tracked[match_id] = {
                "label": signal["label"],
                "module": signal["module"],
                "signal": signal["signal"],
                "person_id": signal.get("person_id"),
                "zone": signal.get("zone"),
                "bbox": signal["bbox"],
                "confidence": signal["confidence"],
                "count": 1,
                "last_seen": now,
                "escalated": False,
            }

            tracked = _tracked[match_id]

        required_frames = settings.ALERT_PERSISTENCE_FRAMES

        if (
            tracked["count"] >= required_frames
            and not tracked["escalated"]
        ):
            tracked["escalated"] = True

            new_alerts.append({
                "label": tracked["label"],
                "module": tracked["module"],
                "signal": tracked["signal"],
                "confidence": tracked["confidence"],
                "bbox": tracked["bbox"],
                "person_id": tracked.get("person_id"),
                "zone": tracked.get("zone"),
            })

    _prune_stale()

    return detections, new_alerts