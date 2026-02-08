# Quick Start Guide

## Option 1: Automated Setup (Recommended)

Run the setup script in PowerShell:

```powershell
.\setup_and_run.ps1
```

This will:
1. Create a virtual environment
2. Install all dependencies
3. Check your configuration
4. Start the server

## Option 2: Manual Setup

### Step 1: Create Virtual Environment

```powershell
python -m venv venv
```

### Step 2: Activate Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` in your prompt.

### Step 3: Install Dependencies

```powershell
pip install -e .
```

This will install:
- FastAPI (web framework)
- Uvicorn (web server)
- OpenCV (image processing)
- Pillow (image handling)
- PyTesseract (text detection)
- PyTorch (ML framework)
- Replicate (AI API client)
- And other dependencies

### Step 4: Set Replicate Token (Optional but Recommended)

```powershell
# Get token from: https://replicate.com/account/api-tokens
$env:REPLICATE_API_TOKEN = "your_token_here"
```

Or create a `.env` file:
```
REPLICATE_API_TOKEN=your_token_here
```

### Step 5: Start Server

```powershell
uvicorn app.main:app --reload
```

## Verify Installation

Once the server starts, open your browser:

- **API Health**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs
- **List Jobs**: http://localhost:8000/api/v1/jobs

## Test with Existing Data

You already have 4 test jobs. View them:

```powershell
# In browser
http://localhost:8000/api/v1/jobs

# Or with PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/jobs"
```

Check generated outputs in:
```
storage\jobs\34ccb744-7022-4bbf-bfd1-2cb7566a8947\
storage\jobs\861e7e4c-cba3-43c1-ab57-e51585a36b8d\
storage\jobs\c25f698e-b6ff-493e-90be-720e7b658800\
```

## Create New Job

```powershell
# Prepare test image
$testImage = "path\to\your\banner.jpg"

# Submit job
$response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/jobs" `
  -Method POST `
  -Form @{
    master_banner = Get-Item $testImage
    outputs = '[{"width":300,"height":250},{"width":728,"height":90},{"width":160,"height":600}]'
  }

# Get job ID
$job = $response.Content | ConvertFrom-Json
Write-Host "Job ID: $($job.job_id)"

# Check status
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/jobs/$($job.job_id)"
```

## Troubleshooting

### "python: command not found"
- Install Python 3.11+ from https://www.python.org/downloads/
- Make sure "Add to PATH" is checked during installation

### "Activate.ps1 cannot be loaded"
Run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "pip: command not found"
```powershell
python -m ensurepip --upgrade
```

### Dependencies fail to install
Try installing PyTorch separately first:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### Server starts but no AI inpainting
- Set `REPLICATE_API_TOKEN` environment variable
- System will fall back to edge replication (acceptable but not ideal)

## What's Next?

See `TEST_GUIDE.md` for comprehensive testing instructions.

## Quick Reference

| Command | Purpose |
|---------|---------|
| `.\setup_and_run.ps1` | Automated setup and run |
| `.\venv\Scripts\Activate.ps1` | Activate virtual environment |
| `pip install -e .` | Install dependencies |
| `uvicorn app.main:app --reload` | Start server |
| `deactivate` | Deactivate virtual environment |

## URLs

- Server: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Jobs: http://localhost:8000/api/v1/jobs
