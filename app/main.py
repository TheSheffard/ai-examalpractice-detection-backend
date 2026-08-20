"""
Entry point for the Exam Monitor backend.

Endpoints:

    WebSocket /ws/monitor
        Receives camera frames from the frontend and runs:
        - M1: object and person detection
        - M2: per-person pose estimation
        - M3: per-person head-direction estimation
        - Fusion and persistence

    GET /alerts
        Returns recent saved alerts.

    GET /health
        Health check for local development and Render.
"""

import asyncio
import base64
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import PyMongoError

from app.config import settings
from app.detectors.gaze import detect_gaze_signal
from app.detectors.objects import detect_objects
from app.detectors.pose import detect_pose_signals
from app.fusion import process_frame
from app.storage.cloudinary_client import upload_snapshot
from app.storage.mongo_client import get_recent_alerts, insert_alert


# -------------------------------------------------------------------
# Background keep-alive task
# -------------------------------------------------------------------

async def _keep_alive() -> None:
    """
    Periodically pings the deployed service to reduce Render idling.
    """

    if not settings.KEEP_ALIVE_URL:
        return

    url = settings.KEEP_ALIVE_URL.rstrip("/") + "/health"
    timeout = httpx.Timeout(10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            await asyncio.sleep(
                settings.KEEP_ALIVE_INTERVAL_SECONDS
            )

            try:
                response = await client.get(url)
                response.raise_for_status()

            except asyncio.CancelledError:
                raise

            except httpx.HTTPError as error:
                print(f"[warn] keep-alive ping failed: {error}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Starts and stops background application tasks.
    """

    keep_alive_task = None

    if settings.KEEP_ALIVE_URL:
        keep_alive_task = asyncio.create_task(_keep_alive())

        print(
            "[info] keep-alive enabled: "
            f"every {settings.KEEP_ALIVE_INTERVAL_SECONDS:g}s"
        )

    try:
        yield

    finally:
        if keep_alive_task:
            keep_alive_task.cancel()

            await asyncio.gather(
                keep_alive_task,
                return_exceptions=True,
            )


# -------------------------------------------------------------------
# FastAPI application
# -------------------------------------------------------------------

app = FastAPI(
    title="Exam Monitor Backend",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Image helpers
# -------------------------------------------------------------------

def _decode_frame(data_url: str):
    """
    Converts a base64 JPEG data URL from the frontend into
    an OpenCV BGR image.
    """

    if not data_url:
        return None

    try:
        if "," in data_url:
            _header, encoded = data_url.split(",", 1)
        else:
            encoded = data_url

        image_bytes = base64.b64decode(encoded)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)

        return cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

    except Exception as error:
        print(f"[warn] unable to decode incoming frame: {error}")
        return None


def _encode_jpeg(frame) -> bytes:
    """
    Encodes an OpenCV frame as JPEG bytes.
    """

    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            80,
        ],
    )

    return buffer.tobytes() if ok else b""


# -------------------------------------------------------------------
# Bounding-box and person helpers
# -------------------------------------------------------------------

def _clip(value: float, minimum: float, maximum: float) -> float:
    """
    Keeps a value within a safe range.
    """

    return max(minimum, min(value, maximum))


def _crop_from_bbox(frame, bbox):
    """
    Converts a normalized bounding box into a pixel crop.

    Expected bbox format:

    {
        "x": 0.25,
        "y": 0.20,
        "w": 0.15,
        "h": 0.50
    }

    Coordinates are normalized from 0.0 to 1.0.
    """

    if not isinstance(bbox, dict):
        return None

    required_keys = ("x", "y", "w", "h")

    if not all(key in bbox for key in required_keys):
        return None

    frame_height, frame_width = frame.shape[:2]

    if frame_width <= 1 or frame_height <= 1:
        return None

    x = float(bbox.get("x", 0.0))
    y = float(bbox.get("y", 0.0))
    width = float(bbox.get("w", 0.0))
    height = float(bbox.get("h", 0.0))

    x1 = int(
        _clip(
            x * frame_width,
            0,
            frame_width - 1,
        )
    )

    y1 = int(
        _clip(
            y * frame_height,
            0,
            frame_height - 1,
        )
    )

    x2 = int(
        _clip(
            (x + width) * frame_width,
            x1 + 1,
            frame_width,
        )
    )

    y2 = int(
        _clip(
            (y + height) * frame_height,
            y1 + 1,
            frame_height,
        )
    )

    crop = frame[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        return None

    return crop


def _is_person_detection(detection: dict) -> bool:
    """
    Supports both the old and new object detector response formats.

    Old format:

        {
            "label": "Person"
        }

    New format:

        {
            "raw_label": "person"
        }
    """

    return (
        detection.get("raw_label") == "person"
        or detection.get("label") == "Person"
    )


def _analyse_people(frame, object_detections):
    """
    Runs M2 and M3 separately for every detected person.

    M1 detects the people first. Each person is then cropped from
    the classroom frame. Pose and head-direction analysis are run
    on that individual crop.

    Returns:

        pose_signals
        gaze_signals
    """

    pose_signals = []
    gaze_signals = []

    person_detections = [
        detection
        for detection in object_detections
        if _is_person_detection(detection)
    ]

    for person in person_detections:
        person_bbox = person.get("bbox")

        if not person_bbox:
            print(
                "[warn] person detection has no bbox:",
                person,
            )
            continue

        person_crop = _crop_from_bbox(
            frame,
            person_bbox,
        )

        if person_crop is None:
            print(
                "[warn] unable to crop person:",
                person,
            )
            continue

        person_id = person.get("person_id")
        zone = person.get("zone")

        # -----------------------------------------------------------
        # M2 — Body pose for this person
        # -----------------------------------------------------------

        try:
            person_pose_signals = detect_pose_signals(
                person_crop,
                person_bbox=person_bbox,
            )

        except TypeError:
            # Temporary compatibility with an older pose function
            # that only accepts person_crop.
            person_pose_signals = detect_pose_signals(
                person_crop,
            )

            for signal in person_pose_signals:
                signal["bbox"] = person_bbox

        except Exception as error:
            print(
                f"[warn] pose detection failed for {person_id}: {error}"
            )
            person_pose_signals = []

        for signal in person_pose_signals:
            signal["module"] = "pose estimation"
            signal["person_id"] = person_id
            signal["zone"] = zone

            # Ensure the fusion module always receives the person's
            # bounding box in the full classroom frame.
            signal["bbox"] = person_bbox

        pose_signals.extend(person_pose_signals)

        # -----------------------------------------------------------
        # M3 — Head direction for this person
        # -----------------------------------------------------------

        try:
            # This supports your current gaze function:
            #
            #     detect_gaze_signal(frame)
            #
            # It analyzes the cropped person, not the full classroom.
            person_gaze_signals = detect_gaze_signal(
                person_crop,
            )

        except TypeError:
            # Compatibility with a future gaze function that accepts
            # person_bbox as an optional second argument.
            person_gaze_signals = detect_gaze_signal(
                person_crop,
                person_bbox=person_bbox,
            )

        except Exception as error:
            print(
                f"[warn] gaze detection failed for {person_id}: {error}"
            )
            person_gaze_signals = []

        for signal in person_gaze_signals:
            signal["module"] = "head and gaze tracking"
            signal["person_id"] = person_id
            signal["zone"] = zone

            # The current gaze detector calculates a bbox inside
            # the crop. Replace it with the full-frame person bbox
            # so that fusion and frontend overlays use correct
            # classroom coordinates.
            signal["bbox"] = person_bbox

        gaze_signals.extend(person_gaze_signals)

    return pose_signals, gaze_signals


# -------------------------------------------------------------------
# HTTP routes
# -------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Simple service health check.
    """

    return {
        "status": "ok",
        "service": "exam-monitor-backend",
    }


@app.get("/alerts")
def alerts(limit: int = 100):
    """
    Returns recent alerts stored in MongoDB.
    """

    try:
        safe_limit = max(1, min(limit, 500))

        return get_recent_alerts(safe_limit)

    except PyMongoError as error:
        print(
            "[error] unable to fetch alerts from MongoDB: "
            f"{error.__class__.__name__}"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Alert storage is unavailable. "
                "Check the MongoDB connection settings."
            ),
        ) from error


# -------------------------------------------------------------------
# WebSocket monitoring pipeline
# -------------------------------------------------------------------

@app.websocket("/ws/monitor")
async def monitor(websocket: WebSocket):
    """
    Live classroom monitoring WebSocket.

    The frontend sends:

    {
        "type": "frame",
        "data": "data:image/jpeg;base64,..."
    }

    The backend returns:

    {
        "type": "detections",
        "detections": [...]
    }

    and, when a signal persists:

    {
        "type": "alert",
        ...
    }
    """

    await websocket.accept()

    print("[info] monitoring WebSocket connected")

    try:
        while True:
            raw_message = await websocket.receive_text()

            try:
                message = json.loads(raw_message)

            except json.JSONDecodeError:
                print("[warn] received invalid JSON message")
                continue

            if message.get("type") != "frame":
                continue

            data_url = message.get("data")

            if not data_url:
                print("[warn] received frame message without data")
                continue

            frame = _decode_frame(data_url)

            if frame is None:
                continue

            # -------------------------------------------------------
            # M1 — Detect people and prohibited objects
            # -------------------------------------------------------

            try:
                objects = detect_objects(frame)

            except Exception as error:
                print(
                    f"[error] object detection failed: {error}"
                )
                objects = []

            for detection in objects:
                detection["module"] = "object detection"

            # -------------------------------------------------------
            # M2 and M3 — Analyze each person separately
            # -------------------------------------------------------

            pose_signals, gaze_signals = _analyse_people(
                frame,
                objects,
            )

            # -------------------------------------------------------
            # Fusion — combine and apply persistence
            # -------------------------------------------------------

            try:
                detections, new_alerts = process_frame(
                    objects,
                    pose_signals,
                    gaze_signals,
                )

            except Exception as error:
                print(
                    f"[error] fusion processing failed: {error}"
                )
                continue

            # -------------------------------------------------------
            # Send live detections to frontend
            # -------------------------------------------------------

            await websocket.send_json({
                "type": "detections",
                "detections": detections,
            })

            # -------------------------------------------------------
            # Save and send persistent alerts
            # -------------------------------------------------------

            for alert in new_alerts:
                timestamp = datetime.now(
                    timezone.utc
                ).isoformat()

                public_id = (
                    f"alert_{int(time.time() * 1000)}"
                )

                snapshot_url = ""

                # Upload a snapshot.
                try:
                    jpeg_bytes = _encode_jpeg(frame)

                    if jpeg_bytes:
                        snapshot_url = upload_snapshot(
                            jpeg_bytes,
                            public_id,
                        )

                except Exception as error:
                    print(
                        f"[warn] snapshot upload failed: {error}"
                    )

                # Store the alert.
                try:
                    insert_alert(
                        label=alert.get(
                            "label",
                            "Unknown alert",
                        ),
                        module=alert.get(
                            "module",
                            "unknown",
                        ),
                        confidence=float(
                            alert.get(
                                "confidence",
                                0.0,
                            )
                        ),
                        snapshot_url=snapshot_url,
                        bbox=alert.get(
                            "bbox",
                            {},
                        ),
                    )

                except Exception as error:
                    print(
                        f"[warn] MongoDB insert failed: {error}"
                    )

                # Send the alert to the frontend.
                await websocket.send_json({
                    "type": "alert",
                    "label": alert.get(
                        "label",
                        "Unknown alert",
                    ),
                    "module": alert.get(
                        "module",
                        "unknown",
                    ),
                    "confidence": float(
                        alert.get(
                            "confidence",
                            0.0,
                        )
                    ),
                    "timestamp": timestamp,
                    "bbox": alert.get(
                        "bbox",
                        {},
                    ),
                    "personId": alert.get(
                        "person_id"
                    ),
                    "zone": alert.get(
                        "zone"
                    ),
                    "snapshotUrl": snapshot_url,
                })

    except WebSocketDisconnect:
        print("[info] monitoring WebSocket disconnected")

    except Exception as error:
        print(
            f"[error] monitoring WebSocket failed: {error}"
        )

        try:
            await websocket.close(
                code=1011,
                reason="Internal monitoring error",
            )

        except Exception:
            pass