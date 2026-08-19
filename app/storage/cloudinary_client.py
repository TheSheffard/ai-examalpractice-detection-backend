import cloudinary
import cloudinary.uploader
from app.config import settings


def upload_snapshot(image_bytes: bytes, public_id: str) -> str:
    # Configure lazily at runtime to guarantee settings are loaded
    cloudinary.config(
        cloud_name=settings.CLOUDINARY_CLOUD_NAME.strip(),
        api_key=settings.CLOUDINARY_API_KEY.strip(),
        api_secret=settings.CLOUDINARY_API_SECRET.strip(),
        secure=True,
    )

    result = cloudinary.uploader.upload(
        image_bytes,
        public_id=public_id,
        folder="exam_monitor_snapshots",
        resource_type="image",
    )
    return result["secure_url"]