"""S3/R2 client for artifact storage with tenant isolation.

Supports both AWS S3 and Cloudflare R2 (zero egress costs).
R2 credentials take precedence if set, otherwise falls back to S3.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def get_s3_client():
    """Initialize boto3 S3-compatible client with credentials from environment. Prefers R2 over S3."""
    import boto3
        
    r2_access_key = os.environ.get('R2_ACCESS_KEY_ID')
    r2_secret_key = os.environ.get('R2_SECRET_ACCESS_KEY')
    r2_endpoint = os.environ.get('R2_ENDPOINT_URL')
    
    if r2_access_key and r2_secret_key and r2_endpoint:
        logger.debug("Using Cloudflare R2 for storage")
        return boto3.client(
            's3',
            endpoint_url=r2_endpoint,
            aws_access_key_id=r2_access_key,
            aws_secret_access_key=r2_secret_key,
            region_name='auto',
        )
    
    logger.debug("Using AWS S3 for storage")
    return boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_REGION', 'us-east-1')
    )


def get_s3_bucket_name() -> str:
    """Get S3/R2 bucket name from environment"""
    bucket = os.environ.get('R2_BUCKET_NAME')
    if bucket:
        return bucket
    
    # Fall back to S3 bucket name
    bucket = os.environ.get('S3_BUCKET_NAME')
    if not bucket:
        raise ValueError("Neither R2_BUCKET_NAME nor S3_BUCKET_NAME environment variable is set")
    return bucket


def sanitize_path_component(component: str) -> str:
    """Sanitize path component to prevent directory traversal attacks."""
    # Remove any path traversal attempts
    sanitized = component.replace('..', '').replace('/', '_').replace('\\', '_')
    # Remove any null bytes
    sanitized = sanitized.replace('\x00', '')
    return sanitized


def get_tenant_prefix(owner_id: str, run_id: str) -> str:
    """Generate safe S3 path with tenant isolation."""
    # Sanitize inputs to prevent path traversal
    safe_owner_id = sanitize_path_component(owner_id)
    safe_run_id = sanitize_path_component(run_id)
    
    return f"tenants/{safe_owner_id}/runs/{safe_run_id}/"


def get_artifact_path(owner_id: str, run_id: str, artifact_type: str, step: int) -> str:
    """Generate S3 path for artifact."""
    prefix = get_tenant_prefix(owner_id, run_id)
    
    # Map artifact types to S3 subdirectories
    if artifact_type == "state":
        subdir = f"checkpoints/state/step-{step}/"
    elif artifact_type == "sampler" or artifact_type == "adapter":
        subdir = f"artifacts/sampler/step-{step}/"
    elif artifact_type == "merged":
        subdir = f"checkpoints/merged/step-{step}/"
    else:
        raise ValueError(f"Unknown artifact type: {artifact_type}")
    
    return prefix + subdir


def upload_file(local_path: str, s3_path: str, bucket: Optional[str] = None) -> Dict[str, Any]:
    """Upload a single file to S3."""
    if bucket is None:
        bucket = get_s3_bucket_name()
    
    s3_client = get_s3_client()
    local_file = Path(local_path)
    
    if not local_file.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")
    
    file_size = local_file.stat().st_size
    
    logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_path}")
    s3_client.upload_file(str(local_file), bucket, s3_path)
    
    return {
        "s3_uri": f"s3://{bucket}/{s3_path}",
        "file_size": file_size,
        "status": "success"
    }


def upload_directory(local_path: str, s3_prefix: str, bucket: Optional[str] = None) -> Dict[str, Any]:
    """Upload directory recursively to S3."""
    if bucket is None:
        bucket = get_s3_bucket_name()
    
    local_dir = Path(local_path)
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory not found: {local_path}")
    
    if not local_dir.is_dir():
        raise ValueError(f"Path is not a directory: {local_path}")
    
    # Ensure s3_prefix ends with /
    if not s3_prefix.endswith('/'):
        s3_prefix += '/'
    
    uploaded_files = []
    total_bytes = 0
    
    # Walk directory and upload all files
    for file_path in local_dir.rglob('*'):
        if file_path.is_file():
            # Calculate relative path from local_dir
            relative_path = file_path.relative_to(local_dir)
            s3_key = s3_prefix + str(relative_path).replace('\\', '/')
            
            result = upload_file(str(file_path), s3_key, bucket)
            uploaded_files.append({
                "local_path": str(file_path),
                "s3_key": s3_key,
                "size": result["file_size"]
            })
            total_bytes += result["file_size"]
    
    logger.info(f"Uploaded {len(uploaded_files)} files ({total_bytes} bytes) to s3://{bucket}/{s3_prefix}")
    
    return {
        "s3_uri": f"s3://{bucket}/{s3_prefix}",
        "files_uploaded": len(uploaded_files),
        "total_bytes": total_bytes,
        "files": uploaded_files,
        "status": "success"
    }


def upload_manifest(manifest_dict: Dict[str, Any], s3_prefix: str, bucket: Optional[str] = None) -> Dict[str, Any]:
    """Upload manifest.json to S3."""
    if bucket is None:
        bucket = get_s3_bucket_name()
    
    s3_client = get_s3_client()
    
    # Ensure s3_prefix ends with /
    if not s3_prefix.endswith('/'):
        s3_prefix += '/'
    
    manifest_key = s3_prefix + "manifest.json"
    manifest_json = json.dumps(manifest_dict, indent=2)
    
    logger.info(f"Uploading manifest to s3://{bucket}/{manifest_key}")
    s3_client.put_object(
        Bucket=bucket,
        Key=manifest_key,
        Body=manifest_json.encode('utf-8'),
        ContentType='application/json'
    )
    
    return {
        "s3_uri": f"s3://{bucket}/{manifest_key}",
        "size": len(manifest_json),
        "status": "success"
    }


def generate_signed_url(s3_uri: str, expiration: int = 3600) -> str:
    """Generate pre-signed URL for S3 object download."""
    # Parse S3 URI
    if not s3_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    
    parts = s3_uri[5:].split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''
    
    s3_client = get_s3_client()
    
    url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=expiration
    )
    return url


def generate_signed_url_for_prefix(s3_prefix: str, expiration: int = 3600, bucket: Optional[str] = None) -> str:
    """Generate pre-signed URL for downloading entire artifact directory.
    
    Note: S3 doesn't support signed URLs for prefixes directly.
    This returns a signed URL for a zip archive (if created) or the first file in the directory.
    For production, consider creating a zip on-the-fly or using CloudFront.
    """
    if bucket is None:
        bucket = get_s3_bucket_name()
    
    s3_uri = f"s3://{bucket}/{s3_prefix}"
    
    # For now, we'll return the URI - in production you might want to:
    # 1. Create a zip file and upload it
    # 2. Use AWS Lambda to create zips on-demand
    # 3. Use CloudFront for directory listings
    
    # Return the prefix URI (caller should handle individual file downloads)
    return s3_uri


def list_artifacts(owner_id: str, run_id: str, bucket: Optional[str] = None) -> list:
    """List all artifacts for a run."""
    if bucket is None:
        bucket = get_s3_bucket_name()
    
    prefix = get_tenant_prefix(owner_id, run_id)
    s3_client = get_s3_client()
    
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    
    if 'Contents' not in response:
        return []
    
    return [
        {
            "key": obj["Key"],
            "size": obj["Size"],
            "last_modified": obj["LastModified"].isoformat(),
            "s3_uri": f"s3://{bucket}/{obj['Key']}"
        }
        for obj in response["Contents"]
    ]


def delete_run_artifacts(owner_id: str, run_id: str, bucket: Optional[str] = None) -> Dict[str, Any]:
    """Delete all S3 artifacts for a run (use with caution)."""
    if bucket is None:
        bucket = get_s3_bucket_name()
    
    prefix = get_tenant_prefix(owner_id, run_id)
    s3_client = get_s3_client()
    
    # List all objects with prefix
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    
    if 'Contents' not in response:
        return {"deleted": 0, "status": "success"}
    
    # Delete all objects
    objects_to_delete = [{"Key": obj["Key"]} for obj in response["Contents"]]
    
    if objects_to_delete:
        delete_response = s3_client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": objects_to_delete}
        )
        
        deleted_count = len(delete_response.get("Deleted", []))
        logger.info(f"Deleted {deleted_count} artifacts for {owner_id}/{run_id}")
        
        return {
            "deleted": deleted_count,
            "status": "success"
        }
    
    return {"deleted": 0, "status": "success"}

