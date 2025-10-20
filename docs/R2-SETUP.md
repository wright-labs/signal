# Cloudflare R2 Setup Guide

This guide walks you through setting up Cloudflare R2 for artifact storage with zero egress fees (compared to AWS S3).

## Why R2?

- **Zero egress fees**: Download artifacts without paying for bandwidth
- **S3-compatible API**: Drop-in replacement for S3
- **Lower storage costs**: ~$0.015/GB/month vs S3's ~$0.023/GB/month
- **Faster in some regions**: Cloudflare's global network

## Setup Steps

### 1. Create Cloudflare Account

1. Go to [cloudflare.com](https://www.cloudflare.com/)
2. Sign up for a free account or log in
3. Navigate to **R2** from the left sidebar

### 2. Create R2 Bucket

1. Click **Create bucket**
2. Enter bucket name: `signal-training-artifacts`
3. Choose location hint (optional): `Automatic` or select region close to your Modal deployment
4. Click **Create bucket**

### 3. Generate API Credentials

1. In R2 dashboard, click **Manage R2 API Tokens**
2. Click **Create API token**
3. Configure token:
   - **Token name**: `signal-production`
   - **Permissions**: Select `Object Read & Write`
   - **Bucket**: Select `signal-training-artifacts` (or leave as "All buckets")
   - **TTL**: Leave blank for no expiration
4. Click **Create API Token**
5. **Important**: Copy the credentials immediately (they won't be shown again):
   - Access Key ID
   - Secret Access Key
   - Endpoint URL (format: `https://<account_id>.r2.cloudflarestorage.com`)

### 4. Configure Environment Variables

Add these environment variables to your deployment:

#### For Railway:
```bash
R2_ACCOUNT_ID=<your_account_id>
R2_ACCESS_KEY_ID=<your_access_key>
R2_SECRET_ACCESS_KEY=<your_secret_key>
R2_BUCKET_NAME=signal-training-artifacts
R2_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
```

#### For Modal:
Update your Modal secrets with:
```bash
modal secret create r2-secret \
  R2_ACCOUNT_ID=<your_account_id> \
  R2_ACCESS_KEY_ID=<your_access_key> \
  R2_SECRET_ACCESS_KEY=<your_secret_key> \
  R2_BUCKET_NAME=signal-training-artifacts \
  R2_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
```

Or update the existing `s3-secret` to include R2 variables.

### 5. Optional: Configure Public Access

If you want to serve artifacts publicly (not recommended for training checkpoints):

1. Go to your bucket settings
2. Under **Public Access**, configure CORS if needed
3. Generate presigned URLs for secure temporary access (already implemented in code)

### 6. Optional: Custom Domain

For faster access and better caching:

1. In R2 bucket settings, click **Custom Domains**
2. Click **Connect Domain**
3. Enter your domain (e.g., `artifacts.yourdomain.com`)
4. Follow DNS setup instructions
5. Update `R2_ENDPOINT_URL` to use your custom domain

## Migration from S3

The code automatically detects R2 credentials and uses them preferentially over S3. To migrate:

1. Set up R2 credentials as above
2. Deploy with new environment variables
3. New artifacts will be stored in R2
4. Old S3 artifacts remain accessible
5. Optional: Manually migrate old artifacts using AWS CLI + rclone

## Verification

Test your setup:

```python
from modal_runtime.s3_client import get_s3_client, get_s3_bucket_name

# This will use R2 if credentials are set
client = get_s3_client()
bucket = get_s3_bucket_name()

# Upload test file
client.put_object(
    Bucket=bucket,
    Key='test/hello.txt',
    Body=b'Hello R2!'
)

# Download test file
obj = client.get_object(Bucket=bucket, Key='test/hello.txt')
print(obj['Body'].read())  # Should print: b'Hello R2!'

# Clean up
client.delete_object(Bucket=bucket, Key='test/hello.txt')
```

## Cost Comparison

Example for 100GB storage, 500GB downloads/month:

**AWS S3:**

- Storage: 100GB × $0.023 = $2.30
- Egress: 500GB × $0.09 = $45.00
- **Total: $47.30/month**

**Cloudflare R2:**

- Storage: 100GB × $0.015 = $1.50
- Egress: 500GB × $0.00 = $0.00
- **Total: $1.50/month**

**Savings: $45.80/month (97% reduction!)**

## Troubleshooting

### Authentication Errors

- Verify credentials are correct
- Check that token has `Object Read & Write` permissions
- Ensure bucket name matches exactly

### Connection Errors

- Verify endpoint URL format: `https://<account_id>.r2.cloudflarestorage.com`
- Check that R2 is enabled for your Cloudflare account
- Ensure no firewall blocking Cloudflare IPs

### Upload Failures

- Check bucket permissions
- Verify file size limits (R2 supports up to 5TB per object)
- Check your Cloudflare R2 storage quota

## Support

- [Cloudflare R2 Documentation](https://developers.cloudflare.com/r2/)
- [R2 API Reference](https://developers.cloudflare.com/r2/api/s3/)
- [R2 Pricing](https://developers.cloudflare.com/r2/pricing/)

