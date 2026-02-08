# Banner Resizing API - Documentation

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API does not require authentication. In production, add API key authentication.

---

## Endpoints

### Health Check

#### `GET /health`

Check if the API is running.

**Response**:
```json
{
  "status": "ok"
}
```

---

### Create Job

#### `POST /api/v1/jobs`

Create a new banner resizing job.

**Request**:
```
Content-Type: multipart/form-data

Parameters:
- master_banner (file, required): Master banner image (JPEG, PNG, WebP)
- outputs (string, required): JSON array of output sizes
- assets (file, optional): Additional assets (logos, overlays)
```

**Example**:
```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -F "master_banner=@banner.jpg" \
  -F "outputs=[{\"width\": 300, \"height\": 250}, {\"width\": 728, \"height\": 90}]"
```

**Response** (201 Created):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "created_at": "2024-02-07T10:30:00Z",
  "outputs": [
    {
      "width": 300,
      "height": 250
    },
    {
      "width": 728,
      "height": 90
    }
  ]
}
```

**Error Responses**:
- `400 Bad Request`: Invalid input (missing required fields, invalid JSON)
- `413 Payload Too Large`: File exceeds maximum size (50MB)
- `415 Unsupported Media Type`: Invalid file format

---

### Get Job Status

#### `GET /api/v1/jobs/{job_id}`

Get the status and details of a job.

**Parameters**:
- `job_id` (path, required): Job ID returned from job creation

**Response** (200 OK):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "created_at": "2024-02-07T10:30:00Z",
  "updated_at": "2024-02-07T10:35:00Z",
  "outputs": [
    {
      "width": 300,
      "height": 250
    },
    {
      "width": 728,
      "height": 90
    }
  ]
}
```

**Status Values**:
- `pending`: Job queued, waiting to start
- `analyzing`: Analyzing banner content
- `planning`: Planning layout strategies
- `generating`: Generating output images
- `validating`: Validating output quality
- `completed`: Job finished successfully
- `failed`: Job failed with error

**Error Responses**:
- `404 Not Found`: Job ID does not exist

---

### List Jobs

#### `GET /api/v1/jobs`

List all jobs (development/debugging only).

**Response** (200 OK):
```json
{
  "jobs": [
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "completed",
      "created_at": "2024-02-07T10:30:00Z",
      "outputs": [
        {
          "width": 300,
          "height": 250
        }
      ]
    }
  ]
}
```

---

### Get Job Outputs

#### `GET /api/v1/jobs/{job_id}/outputs`

Get generated output images for a job.

**Parameters**:
- `job_id` (path, required): Job ID

**Response** (200 OK):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "outputs": [
    {
      "width": 300,
      "height": 250,
      "url": "/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000/outputs/300x250.webp",
      "quality_score": 0.95,
      "needs_manual_review": false
    },
    {
      "width": 728,
      "height": 90,
      "url": "/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000/outputs/728x90.webp",
      "quality_score": 0.87,
      "needs_manual_review": true,
      "warnings": ["Extreme aspect ratio detected", "Text may be too small"]
    }
  ]
}
```

**Error Responses**:
- `404 Not Found`: Job ID does not exist
- `400 Bad Request`: Job not yet completed

---

### Download Output Image

#### `GET /api/v1/jobs/{job_id}/outputs/{size}`

Download a specific output image.

**Parameters**:
- `job_id` (path, required): Job ID
- `size` (path, required): Output size in format `{width}x{height}` (e.g., `300x250`)

**Response** (200 OK):
- Content-Type: `image/webp`
- Binary image data

**Example**:
```bash
curl -O http://localhost:8000/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000/outputs/300x250.webp
```

**Error Responses**:
- `404 Not Found`: Job or output size does not exist
- `400 Bad Request`: Invalid size format

---

## Data Models

### OutputSize

```json
{
  "width": 300,
  "height": 250
}
```

### JobResponse

```json
{
  "job_id": "string (UUID)",
  "status": "pending|analyzing|planning|generating|validating|completed|failed",
  "created_at": "string (ISO 8601 timestamp)",
  "updated_at": "string (ISO 8601 timestamp)",
  "outputs": [
    {
      "width": 300,
      "height": 250
    }
  ]
}
```

### OutputMetadata

```json
{
  "width": 300,
  "height": 250,
  "url": "string (relative URL to download)",
  "quality_score": 0.95,
  "content_preservation_score": 1.0,
  "aspect_ratio_accuracy": 1.0,
  "visual_quality_score": 0.9,
  "confidence": 1.0,
  "needs_manual_review": false,
  "warnings": ["string"],
  "strategy_used": "safe-center-crop|focus-preserving-resize|content-aware-crop|adaptive-padding|smart-crop-with-protection|manual-review-recommended"
}
```

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Errors

| Status | Error | Solution |
|--------|-------|----------|
| 400 | Invalid JSON in outputs | Ensure outputs is valid JSON array |
| 400 | Missing required field | Check all required fields are provided |
| 413 | File too large | Reduce file size or increase limit |
| 415 | Unsupported file type | Use JPEG, PNG, or WebP |
| 404 | Job not found | Check job ID is correct |
| 500 | Internal server error | Check server logs |

---

## Rate Limiting

Currently, there is no rate limiting. In production, implement:

- Per-IP rate limiting: 100 requests/minute
- Per-API-key rate limiting: 1000 requests/hour
- Concurrent job limit: 10 jobs per user

---

## Pagination

List endpoints support pagination:

```
GET /api/v1/jobs?page=1&limit=20
```

---

## Webhooks (Future)

Future versions will support webhooks for job completion:

```json
{
  "event": "job.completed",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "timestamp": "2024-02-07T10:35:00Z"
}
```

---

## Examples

### Example 1: Create Job and Download Output

```bash
#!/bin/bash

# Create job
RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/jobs \
  -F "master_banner=@banner.jpg" \
  -F "outputs=[{\"width\": 300, \"height\": 250}]")

JOB_ID=$(echo $RESPONSE | jq -r '.job_id')
echo "Created job: $JOB_ID"

# Wait for completion
while true; do
  STATUS=$(curl -s http://localhost:8000/api/v1/jobs/$JOB_ID | jq -r '.status')
  echo "Status: $STATUS"
  
  if [ "$STATUS" = "completed" ]; then
    break
  fi
  
  sleep 2
done

# Download output
curl -O http://localhost:8000/api/v1/jobs/$JOB_ID/outputs/300x250.webp
echo "Downloaded output to 300x250.webp"
```

### Example 2: Create Job with Multiple Sizes

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -F "master_banner=@banner.jpg" \
  -F "outputs=[
    {\"width\": 300, \"height\": 250},
    {\"width\": 728, \"height\": 90},
    {\"width\": 160, \"height\": 600},
    {\"width\": 1200, \"height\": 628}
  ]"
```

### Example 3: Python Client

```python
import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"

# Create job
with open("banner.jpg", "rb") as f:
    files = {"master_banner": f}
    data = {"outputs": json.dumps([
        {"width": 300, "height": 250},
        {"width": 728, "height": 90}
    ])}
    
    response = requests.post(f"{BASE_URL}/jobs", files=files, data=data)
    job_id = response.json()["job_id"]
    print(f"Created job: {job_id}")

# Wait for completion
while True:
    response = requests.get(f"{BASE_URL}/jobs/{job_id}")
    status = response.json()["status"]
    print(f"Status: {status}")
    
    if status == "completed":
        break
    
    time.sleep(2)

# Get outputs
response = requests.get(f"{BASE_URL}/jobs/{job_id}/outputs")
outputs = response.json()["outputs"]

for output in outputs:
    print(f"Quality: {output['quality_score']}")
    print(f"Warnings: {output['warnings']}")
    
    # Download image
    img_response = requests.get(output["url"])
    with open(f"{output['width']}x{output['height']}.webp", "wb") as f:
        f.write(img_response.content)
```

---

## Performance Considerations

### Typical Response Times

- Job creation: < 100ms
- Content analysis: 1-5 seconds
- Layout planning: < 500ms
- Output generation: 5-30 seconds (depends on Replicate API)
- Quality validation: 1-2 seconds
- **Total per job**: 10-40 seconds

### Optimization Tips

1. **Batch jobs**: Create multiple jobs in parallel
2. **Cache results**: Store outputs for identical inputs
3. **Async processing**: Use webhooks instead of polling
4. **Reduce output sizes**: Fewer sizes = faster processing

---

## API Versioning

Current version: `v1`

Future versions will be available at `/api/v2`, `/api/v3`, etc.

Backward compatibility is maintained within major versions.

---

## Support

For issues or questions:
1. Check [Setup Guide](./SETUP.md)
2. Review [Replicate Integration Guide](./REPLICATE_INTEGRATION.md)
3. Check server logs: `tail -f logs/app.log`
4. Open an issue on GitHub

