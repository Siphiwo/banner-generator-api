# Banner Resizing API - Quick Start Guide

## 🚀 What Is This?

An AI-powered backend service that intelligently resizes banner images while preserving important content (faces, text, logos). Uses Replicate AI for seamless background extension.

**Status**: Production-ready foundation with Steps A-F complete. Steps G-N (designer quality features) ready for implementation.

---

## ⚡ Quick Start (5 Minutes)

### 1. Install

```bash
# Clone repository
git clone <repo-url>
cd banner-generator-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 2. Configure

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Replicate API token
# Get token from: https://replicate.com/account/api-tokens
REPLICATE_API_TOKEN=your_token_here
```

### 3. Run

```bash
# Start API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Test API
curl http://localhost:8000/health
```

### 4. Create Your First Job

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -F "master_banner=@banner.jpg" \
  -F "outputs=[{\"width\": 300, \"height\": 250}]"
```

---

## 📚 Documentation

All documentation is in the `docs/` folder:

| Document | Purpose |
|----------|---------|
| **[docs/README.md](docs/README.md)** | Documentation index & navigation |
| **[docs/SETUP.md](docs/SETUP.md)** | Complete setup guide |
| **[docs/API.md](docs/API.md)** | API endpoints & examples |
| **[docs/REPLICATE_INTEGRATION.md](docs/REPLICATE_INTEGRATION.md)** | Replicate AI integration |
| **[docs/DESIGNER_QUALITY_ROADMAP.md](docs/DESIGNER_QUALITY_ROADMAP.md)** | Implementation roadmap |
| **[docs/COMPLETION_TRACKER.md](docs/COMPLETION_TRACKER.md)** | Project status |
| **[docs/INTEGRATION_SUMMARY.md](docs/INTEGRATION_SUMMARY.md)** | Replicate integration summary |

**Start here**: [docs/README.md](docs/README.md)

---

## 🏗️ Project Structure

```
banner-generator-api/
├── docs/                          # 📖 All documentation
│   ├── README.md                  # Start here
│   ├── SETUP.md                   # Setup guide
│   ├── API.md                     # API reference
│   ├── REPLICATE_INTEGRATION.md   # AI integration
│   ├── DESIGNER_QUALITY_ROADMAP.md # Implementation plan
│   ├── COMPLETION_TRACKER.md      # Project status
│   └── INTEGRATION_SUMMARY.md     # Integration summary
│
├── app/                           # 🔧 Application code
│   ├── main.py                    # FastAPI app
│   ├── api/v1/                    # API routes
│   ├── models/                    # Data models
│   └── services/                  # Business logic
│       ├── analysis.py            # Banner analysis
│       ├── jobs.py                # Job management
│       ├── replicate_client.py    # Replicate AI client
│       └── test_replicate_integration.py
│
├── storage/                       # 💾 Job storage
│   └── jobs/                      # Job files
│
├── pyproject.toml                 # Dependencies
├── .env.example                   # Environment template
└── README_SETUP.md                # This file
```

---

## 🎯 What's Implemented

### ✅ Complete (Steps A-F)

- **Step A**: Banner content analysis (faces, text, saliency)
- **Step B**: Optional asset alignment
- **Step C**: Aspect ratio risk scoring
- **Step D**: Layout strategy generation
- **Step E**: Image generation and output rendering
- **Step F**: Validation and quality gates

### ⏭️ Next (Steps G-N)

- **Step G**: AI-powered background extension (Replicate inpainting)
- **Step H**: Eliminate letterbox strategy
- **Step I**: Asset compositing
- **Step J**: Asset quality validation
- **Step K**: Text readability validation
- **Step L**: Perceptual quality metrics
- **Step M**: Expose quality metadata through API
- **Step N**: Designer preference customization

---

## 🤖 AI Integration

### Replicate AI Setup

1. **Get API Token**
   - Go to [replicate.com](https://replicate.com)
   - Sign up for free account
   - Get API token from account settings

2. **Set Environment Variable**
   ```bash
   export REPLICATE_API_TOKEN=your_token_here
   ```

3. **Recommended Models**
   - **LaMa** (`twn39/lama`) - Fast, cheap, deterministic ✅ Recommended
   - **SDXL** (`lucataco/sdxl-inpainting`) - Higher quality
   - **Stable Diffusion** (`stability-ai/stable-diffusion-inpainting`) - Very high quality

### Cost Estimation

- **Per job** (6 output sizes): ~$0.05
- **Per month** (1000 jobs): ~$50
- **Replicate credit**: $1 = 55 runs of LaMa

---

## 🧪 Testing

### Run Integration Tests

```bash
python app/services/test_replicate_integration.py
```

Expected output:
```
✓ Client initializes without API token
✓ Client initializes with API token
✓ Client correctly reports unavailable without token
✓ Client availability: False (or True if token is set)
✓ Image converted to base64 (384 chars)
✓ Inpainting gracefully returns None without API token
✓ Inpainting API call successful with mocked API
✓ Global client is a singleton
✓ Model recommendations documented
✓ Error handling works correctly
```

### Test API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# API health
curl http://localhost:8000/api/v1/health

# Create job
curl -X POST http://localhost:8000/api/v1/jobs \
  -F "master_banner=@banner.jpg" \
  -F "outputs=[{\"width\": 300, \"height\": 250}]"

# Get job status
curl http://localhost:8000/api/v1/jobs/{job_id}

# List jobs
curl http://localhost:8000/api/v1/jobs

# Get outputs
curl http://localhost:8000/api/v1/jobs/{job_id}/outputs

# Download output
curl http://localhost:8000/api/v1/jobs/{job_id}/outputs/300x250.webp -O
```

---

## 🔧 Development

### Install Development Dependencies

```bash
pip install -e ".[dev]"
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

### Run Tests

```bash
pytest
```

---

## 📋 API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/health` | API health check |
| `POST` | `/api/v1/jobs` | Create job |
| `GET` | `/api/v1/jobs` | List jobs |
| `GET` | `/api/v1/jobs/{job_id}` | Get job status |
| `GET` | `/api/v1/jobs/{job_id}/outputs` | Get outputs |
| `GET` | `/api/v1/jobs/{job_id}/outputs/{size}` | Download output |

See [docs/API.md](docs/API.md) for full documentation.

---

## 🚀 Production Deployment

### Using Gunicorn

```bash
pip install gunicorn

gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Using Docker

```bash
docker build -t banner-resizer:latest .

docker run -p 8000:8000 \
  -e REPLICATE_API_TOKEN=your_token \
  -v $(pwd)/storage:/app/storage \
  banner-resizer:latest
```

### Using Systemd

See [docs/SETUP.md](docs/SETUP.md) for systemd configuration.

---

## 🐛 Troubleshooting

### "REPLICATE_API_TOKEN not set"
```bash
export REPLICATE_API_TOKEN=your_token_here
```

### "replicate package not installed"
```bash
pip install replicate
```

### "Port 8000 already in use"
```bash
# Find process
lsof -i :8000

# Kill process
kill -9 <PID>
```

### "Tesseract not found"
**Ubuntu/Debian**:
```bash
sudo apt-get install tesseract-ocr
```

**macOS**:
```bash
brew install tesseract
```

**Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

See [docs/SETUP.md](docs/SETUP.md) for more troubleshooting.

---

## 📖 Next Steps

1. **Read Documentation**: Start with [docs/README.md](docs/README.md)
2. **Setup Environment**: Follow [docs/SETUP.md](docs/SETUP.md)
3. **Understand API**: Review [docs/API.md](docs/API.md)
4. **Learn Integration**: Read [docs/REPLICATE_INTEGRATION.md](docs/REPLICATE_INTEGRATION.md)
5. **Implement Features**: Follow [docs/DESIGNER_QUALITY_ROADMAP.md](docs/DESIGNER_QUALITY_ROADMAP.md)

---

## 📞 Support

- **Documentation**: [docs/README.md](docs/README.md)
- **Setup Issues**: [docs/SETUP.md](docs/SETUP.md)
- **API Questions**: [docs/API.md](docs/API.md)
- **AI Integration**: [docs/REPLICATE_INTEGRATION.md](docs/REPLICATE_INTEGRATION.md)
- **Development**: [docs/DESIGNER_QUALITY_ROADMAP.md](docs/DESIGNER_QUALITY_ROADMAP.md)

---

## 📊 Project Status

**Current Phase**: Foundation Complete (Steps A-F)

**Next Phase**: Designer Quality Features (Steps G-N)

**Timeline**: 
- Steps G-H (Background Extension): 1-2 weeks
- Steps I-J (Asset Compositing): 1-2 weeks
- Steps K-L (Advanced Validation): 1 week
- Steps M-N (API Enhancements): 1 week

**Total**: ~4-6 weeks to full designer-quality implementation

---

## 🎓 Learning Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Replicate Documentation](https://replicate.com/docs)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## 📝 License

[Add your license here]

---

## 🙏 Acknowledgments

- Built with FastAPI, OpenCV, PyTorch, and Replicate AI
- Inspired by professional banner design workflows
- Designed for production use

---

**Last Updated**: February 7, 2026

**Version**: 0.1.0 (Foundation Complete)

