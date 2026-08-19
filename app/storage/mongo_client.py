from datetime import datetime, timezone
import dns.resolver
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from app.config import settings

# Override dnspython resolver to fix Render DNS SRV lookup failures
try:
    resolver = dns.resolver.Resolver()
    resolver.nameservers = ["8.8.8.8", "1.1.1.1"]
    dns.resolver.default_resolver = resolver
except Exception as e:
    print(f"[warn] Failed to override DNS resolver: {e}")

_client = None

def _get_alerts_collection():
    """Lazily initialize MongoClient so boot crashes don't occur before port binding."""
    global _client
    if _client is None:
        _client = MongoClient(
            settings.MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
    return _client[settings.MONGODB_DB]["alerts"]


def insert_alert(label: str, module: str, confidence: float, snapshot_url: str, bbox: dict) -> dict:
    alerts = _get_alerts_collection()
    doc = {
        "label": label,
        "module": module,
        "confidence": confidence,
        "snapshot_url": snapshot_url,
        "bbox": bbox,
        "timestamp": datetime.now(timezone.utc),
    }
    result = alerts.insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    return doc


def get_recent_alerts(limit: int = 100):
    alerts = _get_alerts_collection()
    try:
        # Verify connection on request
        _client.admin.command('ping')
        docs = alerts.find().sort("timestamp", -1).limit(limit)
        out = []
        for d in docs:
            d["_id"] = str(d["_id"])
            out.append(d)
        return out
    except PyMongoError as e:
        print(f"[error] MongoDB connection failed: {e}")
        raise