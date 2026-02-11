# Banner Builder API

Backend service for an AI-powered SaaS that performs content-aware banner resizing.

## Features

- **Content-Aware Resizing**: Automatically detects and protects faces, text, and logos
- **PSD Support**: Upload Photoshop files with semantic layer naming for precise control
- **AI-Powered Extension**: Intelligently extends backgrounds to fit new aspect ratios
- **Multiple Output Formats**: Generate multiple banner sizes from a single source
- **Rate Limiting**: Built-in credit-based rate limiting with tier support

## Tech Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI (async-first)
- **Server**: Uvicorn
- **Computer Vision**: OpenCV, Tesseract OCR
- **PSD Parsing**: psd-tools
- **AI Integration**: Replicate API

## Supported Input Formats

- **Standard Images**: PNG, JPG, JPEG
- **Photoshop Files**: PSD with semantic layer naming
  - See [PSD Usage Guide](docs/PSD_USAGE_GUIDE.md) for layer naming conventions
  - See [Designer Quick Reference](docs/PSD_DESIGNER_QUICK_REFERENCE.md) for quick tips

## Local Development

Create and activate a virtual environment, then install dependencies using `pip`:

```bash
pip install -e ".[dev]"
```

Run the development server:

```bash
uvicorn app.main:app --reload
```

## Quick Start with PSD Files

1. Prepare your PSD with semantic layer names:
   ```
   bg:gradient          (background)
   text:headline [lock] (protected text)
   logo:brand [lock]    (protected logo)
   product:shoe         (product image)
   ```

2. Upload via API:
   ```bash
   curl -X POST http://localhost:8000/api/v1/jobs \
     -F "master_banner=@banner.psd" \
     -F 'outputs=[{"width":1200,"height":628}]'
   ```

3. Check job status and get results

See [PSD Usage Guide](docs/PSD_USAGE_GUIDE.md) for complete documentation.

