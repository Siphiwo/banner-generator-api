# Replicate AI Integration Guide

## Overview

This document describes how the banner resizing system integrates with Replicate AI for advanced image processing tasks.

**Replicate** is a platform for running open-source AI models via API. It provides:
- No GPU management required
- Pay-per-use pricing
- Reliable, scalable infrastructure
- Access to state-of-the-art models

---

## Setup & Configuration

### 1. Install Dependencies

```bash
pip install replicate requests
```

Or install from `pyproject.toml`:

```bash
pip install -e .
```

### 2. Get Replicate API Token

1. Go to [replicate.com](https://replicate.com)
2. Sign up for a free account
3. Navigate to your account settings
4. Copy your API token

### 3. Set Environment Variable

Add to your `.env` file:

```bash
REPLICATE_API_TOKEN=your_api_token_here
```

Or set it in your shell:

```bash
export REPLICATE_API_TOKEN=your_api_token_here
```

---

## Recommended Models

### Image Inpainting (Background Extension)

**Primary: LaMa (Large Mask Inpainting)**
- Model ID: `twn39/lama`
- Cost: ~$0.018 per run
- Speed: Fast (5-10 seconds)
- Quality: Excellent for backgrounds
- Deterministic: Yes
- **Recommended for production**

```python
from app.services.replicate_client import inpaint_background

result = inpaint_background(
    image=banner_image,
    mask=expansion_mask,
    prompt="seamless background extension"
)
```

**Alternative: SDXL Inpainting**
- Model ID: `lucataco/sdxl-inpainting`
- Cost: ~$0.0024 per run
- Speed: Faster than Stable Diffusion
- Quality: High quality, good for complex backgrounds
- Deterministic: No (uses sampling)

**Alternative: Stable Diffusion Inpainting**
- Model ID: `stability-ai/stable-diffusion-inpainting`
- Cost: ~$0.005 per run
- Speed: Moderate (10-20 seconds)
- Quality: Very high quality
- Deterministic: No (uses sampling)

### Face Detection & Enhancement

**CodeFormer (Face Restoration)**
- Model ID: `sczhou/codeformer`
- Cost: ~$0.0045 per run
- Use case: Enhance faces after resizing
- Note: Currently using OpenCV Haar cascades locally for speed

---

## Integration Architecture

### Client Module: `app/services/replicate_client.py`

The `ReplicateClient` class provides:

```python
class ReplicateClient:
    def __init__(self, api_token: Optional[str] = None)
    def is_available(self) -> bool
    def inpaint_background(
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str = "seamless background extension",
        model: str = "twn39/lama",
    ) -> Optional[np.ndarray]
    def detect_faces(image: np.ndarray, model: str) -> Optional[np.ndarray]
```

### Usage Pattern

```python
from app.services.replicate_client import get_replicate_client

client = get_replicate_client()

if client.is_available():
    result = client.inpaint_background(image, mask)
else:
    # Fallback to local processing
    result = fallback_inpainting(image, mask)
```

### Error Handling

All Replicate operations include graceful fallback:

1. **No API Token**: Operations return `None`, system falls back to local processing
2. **API Error**: Logged and returns `None`, fallback triggered
3. **Network Error**: Logged and returns `None`, fallback triggered
4. **Invalid Input**: Logged and returns `None`, fallback triggered

---

## Integration Points

### Step G: Background Extension (Adaptive Padding)

**File**: `app/services/analysis.py`

**Function**: `_generate_adaptive_padding_output()`

**Current Implementation**:
```python
# Uses cv2.BORDER_REPLICATE (edge replication)
output = cv2.copyMakeBorder(
    resized,
    top=pad_top,
    bottom=pad_bottom,
    left=pad_left,
    right=pad_right,
    borderType=cv2.BORDER_REPLICATE,
)
```

**Future Implementation** (with Replicate):
```python
from app.services.replicate_client import inpaint_background

# Create mask for expansion zones
mask = create_expansion_mask(resized, expansion_zones)

# Inpaint using Replicate
inpainted = inpaint_background(resized, mask)

if inpainted is not None:
    output = inpainted
else:
    # Fallback to edge replication
    output = cv2.copyMakeBorder(...)
```

---

## Testing

### Unit Tests

Run the integration tests:

```bash
python app/services/test_replicate_integration.py
```

Expected output:
```
============================================================
Running Replicate Integration Tests
============================================================
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
============================================================
Tests passed: 8/8
```

### Integration Testing

To test with real Replicate API:

1. Set `REPLICATE_API_TOKEN` environment variable
2. Run the test suite
3. Verify API calls are made and results are returned

---

## Pricing & Cost Optimization

### Cost Breakdown

| Model | Cost per Run | Runs per $1 | Use Case |
|-------|-------------|-----------|----------|
| LaMa | $0.018 | 55 | Background extension (recommended) |
| SDXL Inpainting | $0.0024 | 416 | High-quality backgrounds |
| Stable Diffusion | $0.005 | 200 | Complex backgrounds |
| CodeFormer | $0.0045 | 222 | Face enhancement |

### Cost Optimization Strategies

1. **Cache Results**: Store inpainted backgrounds for identical inputs
2. **Batch Processing**: Process multiple banners in parallel
3. **Fallback Strategy**: Use local processing for simple cases
4. **Model Selection**: Use LaMa for most cases (fastest, cheapest)

### Example Cost Calculation

For a typical job with 6 output sizes:
- 3 sizes use adaptive padding (inpainting): 3 × $0.018 = $0.054
- 3 sizes use crop (no inpainting): $0.00
- **Total per job: ~$0.05**

---

## Troubleshooting

### "REPLICATE_API_TOKEN not set"

**Solution**: Set the environment variable:
```bash
export REPLICATE_API_TOKEN=your_token_here
```

### "replicate package not installed"

**Solution**: Install the package:
```bash
pip install replicate
```

### "API Error: Invalid image format"

**Solution**: Ensure image is valid:
- Format: PNG, JPEG, or WebP
- Size: Less than 10MB
- Dimensions: Less than 4096×4096

### "Timeout waiting for model"

**Solution**: Replicate models can take 10-30 seconds. Increase timeout:
```python
client.replicate.run(model, input=input, timeout=60)
```

### "Rate limit exceeded"

**Solution**: Replicate has rate limits. Implement backoff:
```python
import time
for attempt in range(3):
    try:
        result = client.inpaint_background(image, mask)
        break
    except Exception as e:
        if attempt < 2:
            time.sleep(2 ** attempt)
        else:
            raise
```

---

## API Reference

### ReplicateClient.inpaint_background()

```python
def inpaint_background(
    image: np.ndarray,
    mask: np.ndarray,
    prompt: str = "seamless background extension",
    model: str = "twn39/lama",
) -> Optional[np.ndarray]:
    """
    Inpaint masked regions using Replicate's inpainting model.

    Args:
        image: Input image as numpy array (BGR or RGB)
        mask: Binary mask where 255 = region to inpaint, 0 = preserve
        prompt: Text prompt for inpainting (used by some models)
        model: Model identifier on Replicate

    Returns:
        Inpainted image as numpy array, or None if operation fails

    Example:
        >>> image = cv2.imread("banner.jpg")
        >>> mask = np.zeros_like(image[:, :, 0])
        >>> mask[100:200, 100:200] = 255  # Mark region to inpaint
        >>> result = client.inpaint_background(image, mask)
    """
```

### ReplicateClient.is_available()

```python
def is_available(self) -> bool:
    """
    Check if Replicate client is properly configured.

    Returns:
        True if API token is set and replicate package is installed
    """
```

---

## Future Enhancements

1. **Async Support**: Use `AsyncReplicate` for non-blocking API calls
2. **Webhook Support**: Use Replicate webhooks for long-running jobs
3. **Model Caching**: Cache model outputs for identical inputs
4. **A/B Testing**: Compare different inpainting models
5. **Custom Models**: Fine-tune models on banner-specific data

---

## References

- [Replicate Documentation](https://replicate.com/docs)
- [Replicate Python SDK](https://sdks.replicate.com/python)
- [LaMa Model](https://replicate.com/twn39/lama)
- [SDXL Inpainting](https://replicate.com/lucataco/sdxl-inpainting)
- [Stable Diffusion Inpainting](https://replicate.com/stability-ai/stable-diffusion-inpainting)

