"""
Central place for every setting the backend needs, loaded from environment
variables (.env locally, or Render's dashboard in production). Nothing here
should be hardcoded elsewhere in the app.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- MongoDB (event log storage) ---
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB: str = os.getenv("MONGODB_DB", "exam_monitor")

    # --- Cloudinary (snapshot image storage) ---
    CLOUDINARY_CLOUD_NAME: str = os.getenv("CLOUDINARY_CLOUD_NAME", "")
    CLOUDINARY_API_KEY: str = os.getenv("CLOUDINARY_API_KEY", "")
    CLOUDINARY_API_SECRET: str = os.getenv("CLOUDINARY_API_SECRET", "")

    # --- Detection thresholds (tune these once testing with a real camera) ---
    OBJECT_CONFIDENCE_THRESHOLD: float = float(os.getenv("OBJECT_CONFIDENCE_THRESHOLD", "0.5"))
    POSE_HAND_DROP_SECONDS: float = float(os.getenv("POSE_HAND_DROP_SECONDS", "2.0"))
    GAZE_AWAY_SECONDS: float = float(os.getenv("GAZE_AWAY_SECONDS", "4.0"))
    ALERT_PERSISTENCE_FRAMES: int = int(os.getenv("ALERT_PERSISTENCE_FRAMES", "3"))

    # --- Frame handling ---
    FRAME_CHECK_INTERVAL_SECONDS: float = float(os.getenv("FRAME_CHECK_INTERVAL_SECONDS", "1.0"))

    # --- Render keep-alive ---
    KEEP_ALIVE_URL: str = os.getenv("KEEP_ALIVE_URL", os.getenv("RENDER_EXTERNAL_URL", ""))
    KEEP_ALIVE_INTERVAL_SECONDS: float = float(os.getenv("KEEP_ALIVE_INTERVAL_SECONDS", "15"))


settings = Settings()
