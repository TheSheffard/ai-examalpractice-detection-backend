"""
Combines M1 (objects), M2 (pose), M3 (gaze) into one list of detections per
frame, and decides when a flagged signal has persisted long enough to become
a logged alert rather than a one-off blip.

This is deliberately a simple rule engine, not a trained model — see the
system design document for why that's the right call for this project.

Tracking approach: since we don't have per-person IDs from the detectors,
each flagged signal is matched frame-to-frame by proximity (its bounding
box center) to the nearest tracked signal of the same label. If a match
persists for ALERT_PERSISTENCE_FRAMES consecutive checks, it's escalated.
This is intentionally simple — good enough for a single-camera prototype,
and noted as a natural extension point in the README.
"""
import time
import uuid
from app.config import settings

# In-memory tracking state: { track_id: {"label", "bbox", "count", "last_seen"} }
_tracked = {}

MATCH_DISTANCE_THRESHOLD = 0.15  # normalized coordinate distance to count as "the same thing"
STALE_SECONDS = 3.0              # forget a tracked signal if it hasn't reappeared in this long


def _bbox_center(bbox):
    return (bbox["x"] + bbox["w"] / 2, bbox["y"] + bbox["h"] / 2)


def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _find_match(label, bbox):
    center = _bbox_center(bbox)
    best_id, best_dist = None, MATCH_DISTANCE_THRESHOLD
    for track_id, t in _tracked.items():
        if t["label"] != label:
            continue
        dist = _distance(center, _bbox_center(t["bbox"]))
        if dist < best_dist:
            best_id, best_dist = track_id, dist
    return best_id


def _prune_stale():
    now = time.time()
    stale = [tid for tid, t in _tracked.items() if now - t["last_seen"] > STALE_SECONDS]
    for tid in stale:
        del _tracked[tid]


def process_frame(objects, pose_signals, gaze_signals):
    """
    Takes the raw output of all three detectors for one frame and returns:
      - detections: everything currently in frame, for the live overlay
      - new_alerts: any signal that just crossed the persistence threshold this frame
    """
    all_signals = objects + pose_signals + gaze_signals
    detections = []
    new_alerts = []
    now = time.time()

    for signal in all_signals:
        detection_id = str(uuid.uuid4())
        detections.append({
            "id": detection_id,
            "status": "flagged" if signal["flagged"] else "normal",
            "label": signal["label"],
            "module": signal.get("module", ""),
            "confidence": signal["confidence"],
            "bbox": signal["bbox"],
        })

        if not signal["flagged"]:
            continue

        # Track this flagged signal across frames
        match_id = _find_match(signal["label"], signal["bbox"])
        if match_id:
            _tracked[match_id]["bbox"] = signal["bbox"]
            _tracked[match_id]["count"] += 1
            _tracked[match_id]["last_seen"] = now
        else:
            match_id = str(uuid.uuid4())
            _tracked[match_id] = {
                "label": signal["label"],
                "module": signal.get("module", ""),
                "bbox": signal["bbox"],
                "confidence": signal["confidence"],
                "count": 1,
                "last_seen": now,
                "escalated": False,
            }

        tracked = _tracked[match_id]
        if tracked["count"] >= settings.ALERT_PERSISTENCE_FRAMES and not tracked["escalated"]:
            tracked["escalated"] = True
            new_alerts.append({
                "label": tracked["label"],
                "module": tracked["module"],
                "confidence": tracked["confidence"],
                "bbox": tracked["bbox"],
            })

    _prune_stale()
    return detections, new_alerts
