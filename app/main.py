"""
Entry point. Exposes:
  - WS  /ws/monitor    — the live camera pipeline the frontend connects to
  - GET /alerts         — recent alert history (used if you want a page that
                           reloads the log independent of a live socket)
  - GET /health          — simple check for Render / uptime monitoring
"""
import base64
import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.detectors.objects import detect_objects
from app.detectors.pose import detect_pose_signals
from app.detectors.gaze import detect_gaze_signal
from app.fusion import process_frame
from app.storage.cloudinary_client import upload_snapshot
from app.storage.mongo_client import insert_alert, get_recent_alerts
from app.config import settings


async def _keep_alive() -> None:
    """Ping the deployed service often enough to prevent Render idling it."""
    url = settings.KEEP_ALIVE_URL.rstrip("/") + "/health"
    timeout = httpx.Timeout(10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            await asyncio.sleep(settings.KEEP_ALIVE_INTERVAL_SECONDS)
            try:
                response = await client.get(url)
                response.raise_for_status()
            except asyncio.CancelledError:
                raise
            except httpx.HTTPError as error:
                print(f"[warn] keep-alive ping failed: {error}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    keep_alive_task = None
    if settings.KEEP_ALIVE_URL:
        keep_alive_task = asyncio.create_task(_keep_alive())
        print(f"[info] keep-alive enabled: every {settings.KEEP_ALIVE_INTERVAL_SECONDS:g}s")

    try:
        yield
    finally:
        if keep_alive_task:
            keep_alive_task.cancel()
            await asyncio.gather(keep_alive_task, return_exceptions=True)


app = FastAPI(title="Exam Monitor Backend", lifespan=lifespan)

# Loosen this to your actual Vercel domain once deployed, e.g.
# allow_origins=["https://your-frontend.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _decode_frame(data_url: str):
    """Turns the base64 JPEG data URL sent by the frontend into an OpenCV BGR frame."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode_jpeg(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/alerts")
def alerts(limit: int = 100):
    return get_recent_alerts(limit)


@app.websocket("/ws/monitor")
async def monitor(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            import json
            msg = json.loads(raw)
            if msg.get("type") != "frame":
                continue

            frame = _decode_frame(msg["data"])
            if frame is None:
                continue

            # Run all three detectors on this frame
            objects = detect_objects(frame)
            for o in objects:
                o["module"] = "object detection"

            pose_signals = detect_pose_signals(frame)
            for p in pose_signals:
                p["module"] = "pose estimation"

            gaze_signals = detect_gaze_signal(frame)
            for g in gaze_signals:
                g["module"] = "head and gaze tracking"

            detections, new_alerts = process_frame(objects, pose_signals, gaze_signals)

            # 1. Always send the live overlay update
            await websocket.send_json({"type": "detections", "detections": detections})

            # 2. For anything that just crossed the persistence threshold:
            #    save a snapshot, log it, and push the alert
            for alert in new_alerts:
                timestamp = datetime.now(timezone.utc).isoformat()
                public_id = f"alert_{int(time.time() * 1000)}"

                snapshot_url = ""
                try:
                    jpeg_bytes = _encode_jpeg(frame)
                    snapshot_url = upload_snapshot(jpeg_bytes, public_id)
                except Exception as e:
                    # Don't let a Cloudinary hiccup take down the whole alert —
                    # log without a snapshot rather than losing the alert entirely.
                    print(f"[warn] snapshot upload failed: {e}")

                try:
                    insert_alert(
                        label=alert["label"],
                        module=alert["module"],
                        confidence=alert["confidence"],
                        snapshot_url=snapshot_url,
                        bbox=alert["bbox"],
                    )
                except Exception as e:
                    print(f"[warn] mongo insert failed: {e}")

                await websocket.send_json({
                    "type": "alert",
                    "label": alert["label"],
                    "module": alert["module"],
                    "confidence": alert["confidence"],
                    "timestamp": timestamp,
                    "bbox": alert["bbox"],
                    "snapshotUrl": snapshot_url,
                })

    except WebSocketDisconnect:
        pass
