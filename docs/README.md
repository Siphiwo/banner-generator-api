# Documentation

This directory contains all documentation for the Banner Resizing API.

## Quick Navigation

### Getting Started
- **[SETUP.md](./SETUP.md)** - Installation and configuration guide
- **[API.md](./API.md)** - API endpoints and usage examples

### Integration & Development
- **[REPLICATE_INTEGRATION.md](./REPLICATE_INTEGRATION.md)** - Replicate AI integration guide
- **[DESIGNER_QUALITY_ROADMAP.md](./DESIGNER_QUALITY_ROADMAP.md)** - Implementation roadmap for designer-quality features
- **[COMPLETION_TRACKER.md](./COMPLETION_TRACKER.md)** - Project status and completed steps

### Architecture
- **[AI_INTEGRATION.md](./AI_INTEGRATION.md)** - AI integration master prompt
- **[INSTRUCTIONS.md](./INSTRUCTIONS.md)** - Backend development instructions

---

## Document Overview

### SETUP.md
Complete setup guide including:
- Quick start (5 minutes)
- System requirements
- Dependency installation
- Environment configuration
- Troubleshooting
- Production deployment

**Read this first if you're new to the project.**

### API.md
API reference including:
- All endpoints with examples
- Request/response formats
- Error handling
- Data models
- Python client examples
- Performance considerations

**Read this to understand how to use the API.**

### REPLICATE_INTEGRATION.md
Replicate AI integration guide including:
- Setup and configuration
- Recommended models
- Integration architecture
- Testing procedures
- Pricing and cost optimization
- Troubleshooting

**Read this to understand AI model integration.**

### DESIGNER_QUALITY_ROADMAP.md
Implementation roadmap including:
- 14 sequential implementation steps
- Phase 1: AI-powered background extension
- Phase 2: Asset compositing
- Phase 3: Text readability validation
- Phase 4: API enhancements
- Success criteria and testing strategy

**Read this to understand the development plan.**

### COMPLETION_TRACKER.md
Project status including:
- Completed steps (A-F)
- Current implementation status
- Known limitations
- Next steps and future enhancements

**Read this to understand what's been done and what's next.**

### AI_INTEGRATION.md
AI integration master prompt including:
- Role and context
- Core product requirements
- Technology constraints
- AI integration order (Steps A-F)
- Model selection rules
- Performance and cost discipline

**Read this to understand the AI integration philosophy.**

### INSTRUCTIONS.md
Backend development instructions including:
- Role and behavior expectations
- Development flow
- Expected high-level steps
- Quality rules
- Failure and recovery expectations

**Read this to understand development standards.**

---

## Project Structure

```
banner-generator-api/
├── docs/                          # Documentation (this directory)
│   ├── README.md                  # This file
│   ├── SETUP.md                   # Setup guide
│   ├── API.md                     # API reference
│   ├── REPLICATE_INTEGRATION.md   # Replicate integration
│   ├── DESIGNER_QUALITY_ROADMAP.md # Implementation roadmap
│   ├── COMPLETION_TRACKER.md      # Project status
│   ├── AI_INTEGRATION.md          # AI integration master prompt
│   └── INSTRUCTIONS.md            # Development instructions
│
├── app/                           # Application code
│   ├── main.py                    # FastAPI application
│   ├── api/                       # API routes
│   │   └── v1/
│   │       ├── routes.py          # Endpoint definitions
│   │       └── schemas.py         # Request/response models
│   ├── models/                    # Data models
│   │   └── jobs.py                # Job and analysis models
│   ├── services/                  # Business logic
│   │   ├── analysis.py            # Banner analysis pipeline
│   │   ├── jobs.py                # Job management
│   │   ├── replicate_client.py    # Replicate AI integration
│   │   └── test_replicate_integration.py # Integration tests
│   └── __init__.py
│
├── storage/                       # Job storage
│   └── jobs/                      # Job files and outputs
│
├── pyproject.toml                 # Project metadata and dependencies
├── README.md                       # Project README
└── .env.example                   # Environment variables template
```

---

## Development Workflow

### 1. Setup
```bash
# Follow SETUP.md
pip install -e .
export REPLICATE_API_TOKEN=your_token
```

### 2. Understand Current State
```bash
# Read COMPLETION_TRACKER.md to see what's done
# Read DESIGNER_QUALITY_ROADMAP.md to see what's next
```

### 3. Implement Feature
```bash
# Follow DESIGNER_QUALITY_ROADMAP.md step by step
# Update COMPLETION_TRACKER.md after each step
```

### 4. Test
```bash
# Run tests
python app/services/test_replicate_integration.py

# Test API
curl http://localhost:8000/api/v1/health
```

### 5. Deploy
```bash
# Follow SETUP.md production deployment section
```

---

## Key Concepts

### Content-Aware Resizing
The system analyzes banner content (faces, text, logos) and adapts the resize strategy to preserve important elements.

### Layout Strategies
Six different strategies for handling different aspect ratio changes:
1. **safe-center-crop**: Simple crop for low-risk cases
2. **focus-preserving-resize**: Letterbox (fit) mode
3. **content-aware-crop**: Crop around important content
4. **adaptive-padding**: Resize with background extension
5. **smart-crop-with-protection**: Strict content protection
6. **manual-review-recommended**: Conservative fallback

### Replicate AI Integration
Uses Replicate API to run AI models for:
- Background inpainting (LaMa model)
- Face detection and enhancement
- Text detection and validation

### Quality Validation
Automated checks for:
- Content preservation (faces/text/logos not cropped)
- Aspect ratio accuracy
- Visual quality (blur, artifacts, color shifts)
- Text readability
- Asset compositing quality

---

## Common Tasks

### Add a New API Endpoint
1. Define schema in `app/api/v1/schemas.py`
2. Implement route in `app/api/v1/routes.py`
3. Update `docs/API.md` with endpoint documentation

### Add a New Analysis Feature
1. Implement in `app/services/analysis.py`
2. Add data model in `app/models/jobs.py`
3. Integrate into `run_initial_pipeline()`
4. Update `docs/COMPLETION_TRACKER.md`

### Integrate a New AI Model
1. Add to `app/services/replicate_client.py`
2. Test with `app/services/test_replicate_integration.py`
3. Document in `docs/REPLICATE_INTEGRATION.md`
4. Update `docs/COMPLETION_TRACKER.md`

### Deploy to Production
1. Follow `docs/SETUP.md` production section
2. Set environment variables
3. Configure reverse proxy (Nginx)
4. Set up monitoring and logging

---

## Troubleshooting

### API not responding
- Check server is running: `curl http://localhost:8000/health`
- Check logs: `tail -f logs/app.log`
- Check port is not in use: `lsof -i :8000`

### Jobs stuck in pending
- Check Replicate API status: https://status.replicate.com
- Check API token is set: `echo $REPLICATE_API_TOKEN`
- Check server logs for errors

### Inpainting not working
- Verify Replicate API token is set
- Check replicate package is installed: `pip list | grep replicate`
- Run integration tests: `python app/services/test_replicate_integration.py`

### Text detection not working
- Verify Tesseract is installed
- Check Tesseract path is correct
- Run with debug logging: `LOG_LEVEL=DEBUG`

---

## Contributing

When contributing to the project:

1. **Read the documentation** - Understand the current state and design
2. **Follow the roadmap** - Implement features in the specified order
3. **Update the tracker** - Document what you've done
4. **Test thoroughly** - Run all tests before committing
5. **Update docs** - Keep documentation in sync with code

---

## Resources

### External Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Replicate Documentation](https://replicate.com/docs)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Models
- [LaMa Inpainting](https://replicate.com/twn39/lama)
- [SDXL Inpainting](https://replicate.com/lucataco/sdxl-inpainting)
- [CodeFormer Face Restoration](https://replicate.com/sczhou/codeformer)

### Related Projects
- [Replicate Python SDK](https://github.com/replicate/replicate-python)
- [OpenCV](https://github.com/opencv/opencv)
- [FastAPI](https://github.com/tiangolo/fastapi)

---

## License

[Add your license here]

---

## Support

For questions or issues:
1. Check the relevant documentation file
2. Review the troubleshooting section
3. Check server logs
4. Open an issue on GitHub

Last updated: February 7, 2026

