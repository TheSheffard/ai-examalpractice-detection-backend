"""
Stores one document per confirmed alert in MongoDB (Atlas free tier in
production). Each document is small — the actual image lives in Cloudinary,
this just keeps a pointer to it plus the alert metadata.
"""
from datetime import datetime, timezone
from pymongo import MongoClient
from app.config import settings

_client = MongoClient(settings.MONGODB_URI)
_db = _client[settings.MONGODB_DB]
_alerts = _db["alerts"]


def insert_alert(label: str, module: str, confidence: float, snapshot_url: str, bbox: dict) -> dict:
    doc = {
        "label": label,
        "module": module,
        "confidence": confidence,
        "snapshot_url": snapshot_url,
        "bbox": bbox,
        "timestamp": datetime.now(timezone.utc),
    }
    result = _alerts.insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    return doc


def get_recent_alerts(limit: int = 100):
    docs = _alerts.find().sort("timestamp", -1).limit(limit)
    out = []
    for d in docs:
        d["_id"] = str(d["_id"])
        out.append(d)
    return out
