# Testing Guide - Current Implementation

## What's Working Now

✅ Steps A-J Complete:
- Banner content analysis (faces, text, saliency)
- Aspect ratio risk scoring
- Layout strategy generation
- AI-powered background extension (LaMa inpainting)
- Asset compositing with transparency
- Quality validation with detailed scoring

## Prerequisites

1. **Python Environment**
   ```bash
   pip install -e .
   ```

2. **Replicate API Token** (Required for AI inpainting)
   ```bash
   # Get token from https://replicate.com/account/api-tokens
   # Then set it:
   set REPLICATE_API_TOKEN=your_token_here
   
   # Or create .env file:
   echo REPLICATE_API_TOKEN=your_token_here > .env
   ```

3. **Tesseract OCR** (Optional, for text detection)
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Add to PATH or set TESSERACT_CMD in .env

## Test Method 1: Use Existing Test Data (Fastest)

You already have 4 completed jobs in `storage/jobs/`. Test the API:

```bash
# Start the server
uvicorn app.main:app --reload

# In another terminal, test endpoints:
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/jobs
```

View results in browser:
- http://localhost:8000/api/v1/jobs
- http://localhost:8000/api/v1/jobs/34ccb744-7022-4bbf-bfd1-2cb7566a8947

## Test Method 2: Create New Job (Full Pipeline)

### Step 1: Prepare Test Images

Create a test banner (or use any image):
- Recommended: 1200x628 (Facebook banner size)
- Include faces, text, or logos for best results
- Optional: Add logo PNG with transparency

### Step 2: Start Server

```bash
uvicorn app.main:app --reload
```

### Step 3: Submit Job via PowerShell

```powershell
# Basic job (banner only)
$response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/jobs" `
  -Method POST `
  -Form @{
    master_banner = Get-Item "path\to\your\banner.jpg"
    outputs = '[{"width":300,"height":250},{"width":728,"height":90},{"width":160,"height":600}]'
  }

$jobId = ($response.Content | ConvertFrom-Json).job_id
Write-Host "Job ID: $jobId"

# With optional assets (logo)
$response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/jobs" `
  -Method POST `
  -Form @{
    master_banner = Get-Item "path\to\banner.jpg"
    additional_assets = Get-Item "path\to\logo.png"
    outputs = '[{"width":300,"height":250},{"width":728,"height":90}]'
  }
```

### Step 4: Check Job Status

```powershell
# Get job details
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/jobs/$jobId"

# List all jobs
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/jobs"
```

### Step 5: View Generated Outputs

Check `storage/jobs/{job_id}/` folder:
- `master.webp` - Original banner
- `master_foreground.png` - Detected foreground mask
- `master_protection.png` - Protected regions (faces/text/logos)
- `master_saliency.png` - Saliency heatmap
- `output_300x250.webp` - Generated banner
- `output_728x90.webp` - Generated banner
- `output_160x600.webp` - Generated banner

## Test Method 3: Use Test Script

```bash
python test_api.py
```

Or use the PowerShell script:

```powershell
.\test_api.ps1
```

## What to Look For

### 1. Content Analysis (Step A)
- Check `master_foreground.png` - Should show detected foreground
- Check `master_protection.png` - Should highlight faces/text
- Check `master_saliency.png` - Should show visual interest areas

### 2. Layout Strategies (Steps C-D)
Look at API response `layout_plans`:
- `safe-center-crop` - Low risk, simple crop
- `content-aware-crop` - Crops around important content
- `adaptive-padding` - Uses AI inpainting for backgrounds
- `manual-review-recommended` - High risk, needs review

### 3. AI Inpainting (Step G)
For extreme aspect ratios (160x600, 1200x300):
- Should use adaptive padding strategy
- Background should be seamlessly extended (no black bars)
- If Replicate token is missing, falls back to edge replication

### 4. Asset Compositing (Step I)
If you uploaded a logo:
- Should appear on all output sizes
- Should maintain aspect ratio
- Should avoid faces/text regions
- Should have transparency preserved

### 5. Quality Validation (Steps F, J)
Check API response `quality_checks`:
- `quality_score` - Overall score (0-1)
- `content_preservation_score` - Protected content preserved?
- `aspect_ratio_accuracy` - Dimensions correct?
- `visual_quality_score` - No blur/artifacts?
- `asset_compositing_score` - Assets placed well?
- `needs_manual_review` - Human review needed?
- `warnings` - List of issues detected

## Expected Results

### Good Quality Output
```json
{
  "quality_score": 1.0,
  "content_preservation_score": 1.0,
  "aspect_ratio_accuracy": 1.0,
  "visual_quality_score": 1.0,
  "asset_compositing_score": 1.0,
  "confidence": 1.0,
  "needs_manual_review": false,
  "warnings": []
}
```

### High-Risk Output (Needs Review)
```json
{
  "quality_score": 0.85,
  "confidence": 0.5,
  "needs_manual_review": true,
  "warnings": [
    "Extreme aspect ratio transformation (source 1.91:1 → target 0.27:1)",
    "Strategy flagged for manual review"
  ]
}
```

## Troubleshooting

### No AI Inpainting (Black Bars or Edge Replication)
- Check: `REPLICATE_API_TOKEN` is set
- Check: Server logs for "Replicate client initialized successfully"
- Fallback: System uses edge replication (acceptable but not ideal)

### No Text Detection
- Install Tesseract OCR
- Set `TESSERACT_CMD` in .env if not in PATH

### No Face Detection
- Should work out of box (uses OpenCV Haar cascades)
- Check server logs for errors

### Job Fails
- Check server logs: `uvicorn app.main:app --reload --log-level debug`
- Check file permissions on `storage/` folder
- Verify image format (JPG, PNG, WEBP supported)

## Performance Expectations

### Without AI Inpainting (No Replicate Token)
- Job creation: <1 second
- Processing: 2-5 seconds per job
- Cost: $0

### With AI Inpainting (Replicate Token Set)
- Job creation: <1 second
- Processing: 5-15 seconds per job (depends on output count)
- Cost: ~$0.05 per job (6 outputs)

## Next Steps After Testing

Once you verify everything works:

1. **Test with different banner types**:
   - Banners with faces
   - Banners with text
   - Banners with logos
   - Banners with complex backgrounds

2. **Test extreme aspect ratios**:
   - 160x600 (skyscraper)
   - 1200x300 (leaderboard)
   - 300x250 (medium rectangle)

3. **Test asset compositing**:
   - Upload PNG logos with transparency
   - Upload multiple assets
   - Check positioning and scaling

4. **Review quality checks**:
   - Look for false positives (good outputs flagged for review)
   - Look for false negatives (bad outputs not flagged)

## API Endpoints Reference

- `GET /health` - Server health
- `GET /api/v1/health` - API health
- `POST /api/v1/jobs` - Create job
- `GET /api/v1/jobs` - List all jobs
- `GET /api/v1/jobs/{job_id}` - Get job details

## Files to Inspect

After running a job, check these files:
```
storage/jobs/{job_id}/
├── master.webp                  # Original banner
├── master_foreground.png        # Foreground mask
├── master_protection.png        # Protected regions
├── master_saliency.png          # Saliency map
├── output_300x250.webp          # Generated output
├── output_728x90.webp           # Generated output
└── output_160x600.webp          # Generated output
```

## Success Criteria

✅ Server starts without errors
✅ Job creation returns job_id
✅ Analysis masks are generated
✅ Output images are created
✅ Quality checks are populated
✅ No crashes or exceptions
✅ Outputs look professional (no black bars, content preserved)

---

**Ready to test!** Start with Method 1 (existing data) for quickest verification, then try Method 2 (new job) for full pipeline testing.
