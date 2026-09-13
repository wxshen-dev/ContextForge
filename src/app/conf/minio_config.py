# Core dependencies: dataclasses, environment variables, and path handling.
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Load .env before reading MinIO-related configuration.
load_dotenv()


# MinIO object-storage configuration.
@dataclass
class MinIOConfig:
    endpoint: str    # MinIO endpoint, including protocol and port.
    access_key: str  # MinIO access key from MINIO_ACCESS_KEY.
    secret_key: str  # MinIO secret key from MINIO_SECRET_KEY.
    bucket_name: str # Default MinIO bucket for knowledge-base files.
    minio_img_dir: str # MinIO directory for storing images.
    minio_secure: bool # Whether to use SSL, choosing HTTPS instead of HTTP.


# Instantiate the MinIO config object from .env values.
minio_config = MinIOConfig(
    endpoint=os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    bucket_name=os.getenv("MINIO_BUCKET_NAME"),
    minio_img_dir=os.getenv("MINIO_IMG_DIR"),
    minio_secure=os.getenv("MINIO_SECURE") == "True"
)
