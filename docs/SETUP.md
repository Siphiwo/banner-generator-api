# Banner Resizing API - Setup Guide

## Quick Start

### 1. Prerequisites

- Python 3.11+
- pip or conda
- Git

### 2. Clone & Install

```bash
# Clone the repository
git clone <repository-url>
cd banner-generator-api

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 3. Configure Environment

Create a `.env` file in the project root:

```bash
# Replicate AI API token (get from https://replicate.com)
REPLICATE_API_TOKEN=your_token_here

# Optional: Tesseract path (if not in system PATH)
# TESSERACT_PATH=/usr/bin/tesseract

# Optional: Storage path for job files
# STORAGE_PATH=./storage
```

### 4. Run the Server

```bash
# Development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# API health check
curl http://localhost:8000/api/v1/health

# Create a job (see API documentation for details)
curl -X POST http://localhost:8000/api/v1/jobs \
  -F "master_banner=@banner.jpg" \
  -F "outputs=[{\"width\": 300, \"height\": 250}]"
```

---

## System Requirements

### Minimum

- CPU: 2 cores
- RAM: 4GB
- Disk: 10GB (for models and job storage)

### Recommended

- CPU: 4+ cores
- RAM: 8GB+
- Disk: 50GB+ (for caching and job history)
- GPU: Optional (for local model inference)

---

## Dependencies

### Core

- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### Image Processing

- **OpenCV**: Computer vision (face detection, saliency)
- **Pillow**: Image manipulation
- **NumPy**: Numerical computing

### AI/ML

- **PyTorch**: Deep learning framework
- **TorchVision**: Computer vision models
- **Replicate**: API for running AI models

### Optional

- **Tesseract**: OCR for text detection
- **Pytest**: Testing framework

---

## Installation Troubleshooting

### OpenCV Installation Issues

If you encounter issues with `opencv-contrib-python`:

```bash
# Try the standard version instead
pip uninstall opencv-contrib-python
pip install opencv-python
```

### Tesseract Not Found

If text detection fails:

**Ubuntu/Debian**:
```bash
sudo apt-get install tesseract-ocr
```

**macOS**:
```bash
brew install tesseract
```

**Windows**:
Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### PyTorch Installation

If PyTorch installation is slow or fails:

```bash
# CPU-only version (smaller)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU version (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REPLICATE_API_TOKEN` | (required) | API token for Replicate AI |
| `STORAGE_PATH` | `./storage` | Directory for job files |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `TESSERACT_PATH` | (auto-detect) | Path to Tesseract executable |
| `MAX_UPLOAD_SIZE` | 50MB | Maximum file upload size |

### Application Settings

Edit `app/main.py` to customize:

```python
# Maximum concurrent jobs
MAX_CONCURRENT_JOBS = 10

# Job timeout (seconds)
JOB_TIMEOUT = 300

# Output quality (1-100)
OUTPUT_QUALITY = 90

# Supported output formats
SUPPORTED_FORMATS = ["webp", "jpg", "png"]
```

---

## Development Setup

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
# All tests
pytest

# Specific test file
pytest app/services/test_replicate_integration.py

# With coverage
pytest --cov=app
```

### Code Quality

```bash
# Format code
black app/

# Lint
flake8 app/

# Type checking
mypy app/
```

---

## Docker Setup (Optional)

### Build Image

```bash
docker build -t banner-resizer:latest .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -e REPLICATE_API_TOKEN=your_token \
  -v $(pwd)/storage:/app/storage \
  banner-resizer:latest
```

---

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn

gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Using Systemd

Create `/etc/systemd/system/banner-resizer.service`:

```ini
[Unit]
Description=Banner Resizer API
After=network.target

[Service]
Type=notify
User=www-data
WorkingDirectory=/opt/banner-resizer
Environment="REPLICATE_API_TOKEN=your_token"
ExecStart=/opt/banner-resizer/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable banner-resizer
sudo systemctl start banner-resizer
```

### Using Nginx Reverse Proxy

```nginx
upstream banner_resizer {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://banner_resizer;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Monitoring & Logging

### View Logs

```bash
# Real-time logs
tail -f logs/app.log

# Filter by level
grep ERROR logs/app.log
```

### Health Checks

```bash
# Basic health
curl http://localhost:8000/health

# Detailed health
curl http://localhost:8000/api/v1/health
```

### Metrics

The API exposes Prometheus metrics at `/metrics` (if enabled).

---

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Out of Memory

Reduce concurrent jobs or increase system RAM:

```python
MAX_CONCURRENT_JOBS = 5  # Reduce from 10
```

### Slow API Responses

1. Check system resources: `top`, `free -h`
2. Check Replicate API status: https://status.replicate.com
3. Increase timeout: `JOB_TIMEOUT = 600`

### Jobs Stuck in "PENDING"

1. Check logs for errors
2. Restart the application
3. Check Replicate API connectivity

---

## Next Steps

1. Read [API Documentation](./API.md)
2. Review [Replicate Integration Guide](./REPLICATE_INTEGRATION.md)
3. Check [Designer Quality Roadmap](./DESIGNER_QUALITY_ROADMAP.md)
4. Explore [Completion Tracker](./COMPLETION_TRACKER.md)

