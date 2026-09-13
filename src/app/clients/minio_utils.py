# Python standard-library imports.
import os
import json

# MinIO official Python SDK client.
from minio import Minio

# Project configuration and logging utilities.
from app.conf.minio_config import minio_config  # MinIO settings.
from app.core.logger import logger            # Project-wide logger.

# Global MinIO client singleton.
_minio_client = None


# 1. Create a MinIO client connection.
def _create_minio_client() -> Minio:
    """
    Create and return a MinIO client connection.
    :return: Initialized MinIO client.
    """
    return Minio(
        endpoint=minio_config.endpoint,        # MinIO endpoint.
        access_key=minio_config.access_key,    # MinIO access key.
        secret_key=minio_config.secret_key,    # MinIO secret key.
        secure=minio_config.minio_secure       # Whether to enable HTTPS.
    )


# 2. Build the bucket access policy.
def _set_bucket_policy(bucket_name: str) -> str:
    """
    Generate a MinIO bucket access policy as a JSON string.
    The policy allows public read access to all objects in the bucket.
    :param bucket_name: Target bucket name.
    :return: Serialized JSON access policy.
    """
    # Policy template following the AWS S3 policy format supported by MinIO.
    policy = {
        "Version": "2012-10-17",  # Fixed policy version compatible with S3.
        "Statement": [
            {
                "Effect": "Allow",  # Allow access.
                "Principal": {"AWS": ["*"]},  # All users.
                "Action": ["s3:GetObject"],  # Read objects in the bucket.
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"],  # All objects in the bucket.
            }
        ],
    }
    # Serialize the policy dict for MinIO.
    return json.dumps(policy)


# 3. Ensure the bucket exists and is ready.
def _create_bucket_ready(client: Minio):
    """
    Check whether the MinIO bucket exists, create it if needed, and set its policy.
    :param client: Initialized MinIO client.
    """
    bucket_name = minio_config.bucket_name  # Target bucket from configuration.
    # Check whether the bucket exists.
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)  # Create the bucket when missing.
        # Allow public read access for uploaded image URLs.
        client.set_bucket_policy(bucket_name, _set_bucket_policy(bucket_name))
        logger.info(f"MinIO bucket {bucket_name} was created and its access policy was set.")
    else:
        # The bucket already exists; do not repeat setup.
        logger.info(f"MinIO bucket {bucket_name} already exists; setup skipped.")


def get_minio_client() -> Minio:
    """
    Get the global MinIO client using lazy loading.
    The first call initializes the client and bucket. Later calls reuse the singleton.
    :return: Global MinIO client instance.
    """
    # Explicitly use the global singleton.
    global _minio_client

    # Lazy-load only when the client has not been initialized.
    if _minio_client is None:
        logger.info("Initializing MinIO client for the first time.")
        client = _create_minio_client()          # Create the client connection.
        _create_bucket_ready(client)             # Check and initialize the bucket.
        _minio_client = client                   # Store the singleton for later reuse.
        logger.info("MinIO client initialized and ready.")

    # Return the reusable global client instance.
    return _minio_client
