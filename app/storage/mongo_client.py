from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from app.config import settings

# Initialize client with connection timeouts to prevent long hangs
_client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=5000)
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
    try:
        # Verify connection on request
        _client.admin.command('ping')
        docs = _alerts.find().sort("timestamp", -1).limit(limit)
        out = []
        for d in docs:
            d["_id"] = str(d["_id"])
            out.append(d)
        return out
    except PyMongoError as e:
        print(f"[error] MongoDB connection failed: {e}")
        raise