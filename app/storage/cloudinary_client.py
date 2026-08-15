"""
Uploads alert snapshots to Cloudinary and returns the hosted URL to store in
MongoDB. Used instead of local disk because Render's filesystem is not a
durable place to keep files across restarts/redeploys.
"""
import cloudinary
import cloudinary.uploader
from app.config import settings

cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
    secure=True,
)


def upload_snapshot(image_bytes: bytes, public_id: str) -> str:
    """
    Uploads a JPEG snapshot and returns its permanent hosted URL.
    """
    result = cloudinary.uploader.upload(
        image_bytes,
        public_id=public_id,
        folder="exam_monitor_snapshots",
        resource_type="image",
    )
    return result["secure_url"]
