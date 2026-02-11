"""
Direct HTTP client for Replicate API.

This is a workaround for Python 3.14 compatibility issues with the official replicate package.
Uses the Replicate HTTP API directly via requests.

IMPORTANT: All Replicate API calls go through centralized rate limiting to respect
Replicate's infrastructure and usage policies (max 600 req/min, exponential backoff on 429).
"""

import logging
import os
import time
from typing import Optional
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
import requests

from app.services.rate_limiter import (
    acquire_replicate_token,
    report_replicate_429,
    report_replicate_success,
)

logger = logging.getLogger(__name__)

# Output log file
OUTPUT_LOG = Path("output.txt")

# Retry configuration
# Note: Rate limiting is handled by centralized rate limiter
# These retries are only for transient server errors (5xx), not rate limits (429)
MAX_RETRIES = 2
RETRY_DELAY = 2  # seconds
RETRY_BACKOFF = 2  # exponential backoff multiplier


def log_to_file(message: str):
    """Append message to output.txt with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(OUTPUT_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


class ReplicateHTTPClient:
    """
    Direct HTTP client for Replicate API.
    
    Bypasses the official replicate package to avoid Python 3.14 compatibility issues.
    """

    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize Replicate HTTP client.

        Args:
            api_token: Replicate API token. If not provided, reads from REPLICATE_API_TOKEN env var.
        """
        self.api_token = api_token or os.environ.get("REPLICATE_API_TOKEN")
        self.base_url = "https://api.replicate.com/v1"
        
        if not self.api_token:
            message = "⚠️  REPLICATE API TOKEN NOT FOUND - AI inpainting DISABLED"
            print("\n" + "="*60)
            print(message)
            print("="*60)
            print("AI inpainting will be DISABLED")
            print("System will use edge replication fallback")
            print("\nTo enable AI inpainting:")
            print("  1. Get token: https://replicate.com/account/api-tokens")
            print("  2. Add to .env file: REPLICATE_API_TOKEN=your_token")
            print("  3. Restart server")
            print("="*60 + "\n")
            logger.warning("REPLICATE_API_TOKEN not set. Replicate features will be unavailable.")
            log_to_file(message)
            log_to_file("System will use FALLBACK (edge replication) for all jobs")
            self.available = False
            return
        
        # Token found
        message = "✅ REPLICATE API TOKEN FOUND - AI inpainting ENABLED"
        print("\n" + "="*60)
        print(message)
        print("="*60)
        print(f"Token: {self.api_token[:15]}...")
        
        # Validate token format
        if not self.api_token.startswith("r8_"):
            warning = "⚠️  WARNING: Token doesn't start with 'r8_' - might be invalid"
            print(warning)
            print("   This might not be a valid Replicate token")
            print("   Get a valid token from: https://replicate.com/account/api-tokens")
            log_to_file(warning)
        
        print("✓ Using HTTP API client (Python 3.14 compatible)")
        print("✓ AI inpainting is ENABLED")
        print("✓ Using model: twn39/lama")
        print("="*60 + "\n")
        
        self.available = True
        logger.info("Replicate HTTP client initialized successfully")
        log_to_file(message)
        log_to_file("Using model: twn39/lama")
        log_to_file("Ready to process jobs with AI inpainting")

    def is_available(self) -> bool:
        """Check if Replicate client is properly configured."""
        return self.available and self.api_token is not None

    def inpaint_background(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str = "seamless background extension",
        model: str = "twn39/lama:2b91ca2340801c2a5be745612356fac36a17f698354a07f48a62d564d3b3a7a0",
    ) -> Optional[np.ndarray]:
        """
        Inpaint masked regions using Replicate's LaMa model via HTTP API.

        This method respects Replicate's rate limits through centralized rate limiting:
        - Acquires token before making request
        - Reports 429 errors for exponential backoff
        - Reports success to gradually restore capacity

        Args:
            image: Input image as numpy array (BGR or RGB)
            mask: Binary mask where 255 = region to inpaint, 0 = preserve
            prompt: Text prompt for inpainting (used by some models)
            model: Model identifier on Replicate (default: LaMa)

        Returns:
            Inpainted image as numpy array, or None if operation fails
        """
        if not self.is_available():
            message = "🔄 BYPASSING REPLICATE - Using FALLBACK (edge replication)"
            print(f"\n{message}")
            logger.warning("Replicate not available. Inpainting skipped.")
            log_to_file(message)
            return None

        # Acquire rate limit token before making request
        logger.info("Acquiring rate limit token for Replicate API call...")
        if not acquire_replicate_token(timeout=30.0):
            error_msg = "❌ RATE LIMITER TIMEOUT - Using FALLBACK (edge replication)"
            print(f"\n{error_msg}")
            logger.error("Failed to acquire rate limit token within 30s")
            log_to_file(error_msg)
            return None

        # Retry logic for transient errors (5xx only, not rate limits)
        for attempt in range(MAX_RETRIES + 1):
            try:
                result = self._attempt_inpainting(image, mask, prompt, model, attempt)
                
                # Report success to rate limiter
                report_replicate_success()
                
                return result
                
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else 0
                
                # Handle 429 (rate limit) - NO RETRY, let rate limiter handle backoff
                if status_code == 429:
                    # Report to rate limiter for exponential backoff
                    report_replicate_429()
                    
                    error_msg = f"❌ REPLICATE RATE LIMITED (429) - Backoff applied, next request will wait"
                    print(f"\n{error_msg}")
                    logger.error("Replicate API rate limited (429) - exponential backoff triggered")
                    log_to_file(error_msg)
                    log_to_file("Rate limiter will enforce minimum 30s wait before next request")
                    
                    # Return immediately - do NOT retry
                    # Rate limiter will handle backoff for subsequent requests
                    return None
                
                # Don't retry on other client errors (4xx)
                if 400 <= status_code < 500:
                    error_msg = f"❌ REPLICATE CLIENT ERROR ({status_code}): {str(e)} - Using FALLBACK"
                    print(f"\n{error_msg}")
                    logger.error(f"Replicate client error: {e}")
                    log_to_file(error_msg)
                    return None
                
                # Retry ONLY on server errors (5xx)
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY * (RETRY_BACKOFF ** attempt)
                    error_detail = str(e)
                    if e.response:
                        try:
                            error_body = e.response.json()
                            error_detail = f"{status_code}: {error_body.get('detail', error_body)}"
                        except:
                            error_detail = f"{status_code}: {e.response.text[:200]}"
                    
                    retry_msg = f"⏳ Server error ({error_detail}) - Retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})..."
                    print(f"{retry_msg}")
                    log_to_file(retry_msg)
                    log_to_file(f"Full error: {e}")
                    time.sleep(delay)
                else:
                    error_detail = str(e)
                    if e.response:
                        try:
                            error_body = e.response.json()
                            error_detail = f"{status_code}: {error_body}"
                        except:
                            error_detail = f"{status_code}: {e.response.text[:500]}"
                    
                    error_msg = f"❌ REPLICATE FAILED after {MAX_RETRIES + 1} attempts - Using FALLBACK"
                    print(f"\n{error_msg}")
                    print(f"   Error details: {error_detail}")
                    logger.error(f"Replicate failed after retries: {error_detail}")
                    log_to_file(error_msg)
                    log_to_file(f"Full error: {error_detail}")
                    return None
                    
            except Exception as e:
                error_msg = f"❌ REPLICATE ERROR: {str(e)} - Using FALLBACK"
                print(f"\n{error_msg}")
                logger.error(f"Inpainting error: {e}")
                log_to_file(error_msg)
                return None
        
        return None

    def _attempt_inpainting(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        model: str,
        attempt: int,
    ) -> Optional[np.ndarray]:
        """Single attempt at inpainting (used by retry logic)."""
        # Convert numpy arrays to PIL Images
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask)

        # Convert to data URIs for API
        image_uri = self._image_to_data_uri(pil_image)
        mask_uri = self._image_to_data_uri(pil_mask)

        attempt_suffix = f" (attempt {attempt + 1})" if attempt > 0 else ""
        info_msg = f"🚀 CALLING REPLICATE API{attempt_suffix} - Model: {model}, Size: {image.shape[1]}x{image.shape[0]}, Mask: {np.sum(mask > 0) / mask.size:.1%}"
        print(f"\n{info_msg}")
        print(f"   Model: {model}")
        print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
        print(f"   Mask coverage: {np.sum(mask > 0) / mask.size:.1%}")
        logger.info(f"Calling Replicate HTTP API: {model}")
        log_to_file(info_msg)

        # Create prediction
        headers = {
            "Authorization": f"Token {self.api_token}",
            "Content-Type": "application/json",
        }

        # Use the correct API format for predictions
        # Format: owner/model-name (without version)
        model_path = self._get_model_version(model)

        payload = {
            "version": model_path,  # Use 'version' field for model identifier
            "input": {
                "image": image_uri,
                "mask": mask_uri,
            }
        }

        # Create prediction using the unified endpoint
        log_to_file(f"Creating prediction for model: {model_path}")
        response = requests.post(
            f"{self.base_url}/predictions",  # Unified endpoint for all models
            headers=headers,
            json=payload,
            timeout=30,
        )
        
        # Handle specific error codes
        if response.status_code == 404:
            error_detail = "Model not found"
            try:
                error_body = response.json()
                error_detail = error_body.get("detail", error_detail)
            except:
                pass
            error_msg = f"❌ REPLICATE API ERROR (404): Model not found - {model_path}"
            print(f"\n{error_msg}")
            print(f"   Detail: {error_detail}")
            print(f"   Tip: Check if model exists at https://replicate.com/{model_path}")
            logger.error(f"Replicate API 404 error: {error_detail}")
            log_to_file(error_msg)
            log_to_file(f"Detail: {error_detail}")
            log_to_file("Tip: Verify model name and that it's publicly accessible")
            raise requests.exceptions.HTTPError(error_msg, response=response)
        
        elif response.status_code == 422:
            error_detail = response.json().get("detail", "Unknown error")
            error_msg = f"❌ REPLICATE API ERROR (422): Invalid request - {error_detail}"
            print(f"\n{error_msg}")
            logger.error(f"Replicate API 422 error: {error_detail}")
            log_to_file(error_msg)
            log_to_file("Tip: Check if model exists and input format is correct")
            raise requests.exceptions.HTTPError(error_msg, response=response)
        
        elif response.status_code == 429:
            error_msg = "❌ REPLICATE RATE LIMITED (429): Too many requests"
            print(f"\n{error_msg}")
            logger.warning("Replicate API rate limited")
            log_to_file(error_msg)
            log_to_file("Tip: Wait a few minutes or upgrade your Replicate plan")
            raise requests.exceptions.HTTPError(error_msg, response=response)
        
        elif response.status_code == 401:
            error_msg = "❌ REPLICATE AUTH ERROR (401): Invalid token"
            print(f"\n{error_msg}")
            logger.error("Replicate API authentication failed")
            log_to_file(error_msg)
            log_to_file("Tip: Check your token at https://replicate.com/account/api-tokens")
            raise requests.exceptions.HTTPError(error_msg, response=response)
        
        response.raise_for_status()
        prediction = response.json()

        # Poll for completion
        prediction_url = prediction["urls"]["get"]
        output_url = self._wait_for_prediction(prediction_url, headers)

        if output_url:
            # Download result
            inpainted = self._download_image(output_url)
            if inpainted is not None:
                success_msg = "✅ REPLICATE SUCCESS - AI inpainting completed"
                print(f"{success_msg}")
                logger.info("Inpainting completed successfully")
                log_to_file(success_msg)
                return inpainted

        fail_msg = "❌ REPLICATE FAILED: No output received"
        print(f"\n{fail_msg}")
        logger.error("Inpainting failed: No output received")
        log_to_file(fail_msg)
        return None

    def _get_model_version(self, model: str) -> str:
        """
        Get the correct model version for Replicate API.
        
        The API expects either:
        - owner/name (for official models)
        - owner/name:version_hash (for community models)
        - version_hash (64-character hash)
        
        This method extracts the version hash if present, otherwise returns the model path.
        """
        # If model contains a colon, extract the version hash
        if ":" in model:
            parts = model.split(":")
            if len(parts) == 2:
                # Return the full version hash (after the colon)
                return parts[1]
        
        # Return the model as-is (owner/name format)
        return model

    def _image_to_data_uri(self, pil_image: Image.Image) -> str:
        """Convert PIL Image to data URI."""
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        b64_data = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/png;base64,{b64_data}"

    def _wait_for_prediction(
        self,
        prediction_url: str,
        headers: dict,
        max_wait: int = 120,
        poll_interval: int = 2,
    ) -> Optional[str]:
        """Poll prediction until completion."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            response = requests.get(prediction_url, headers=headers, timeout=10)
            response.raise_for_status()
            prediction = response.json()

            status = prediction["status"]
            
            if status == "succeeded":
                output = prediction.get("output")
                if isinstance(output, str):
                    return output
                elif isinstance(output, list) and len(output) > 0:
                    return output[0]
                return None
            
            elif status == "failed":
                error = prediction.get("error", "Unknown error")
                logger.error(f"Prediction failed: {error}")
                return None
            
            elif status in ["starting", "processing"]:
                time.sleep(poll_interval)
                continue
            
            else:
                logger.warning(f"Unknown prediction status: {status}")
                return None

        logger.error(f"Prediction timed out after {max_wait}s")
        return None

    def _download_image(self, url: str) -> Optional[np.ndarray]:
        """Download image from URL and convert to numpy array."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            pil_image = Image.open(BytesIO(response.content))
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            return None


# Global client instance
_replicate_client: Optional[ReplicateHTTPClient] = None


def get_replicate_client() -> ReplicateHTTPClient:
    """Get or create global Replicate HTTP client instance."""
    global _replicate_client
    if _replicate_client is None:
        _replicate_client = ReplicateHTTPClient()
    return _replicate_client


def inpaint_background(
    image: np.ndarray,
    mask: np.ndarray,
    prompt: str = "seamless background extension",
) -> Optional[np.ndarray]:
    """
    Convenience function to inpaint background using Replicate HTTP API.

    Args:
        image: Input image as numpy array
        mask: Binary mask where 255 = region to inpaint
        prompt: Text prompt for inpainting

    Returns:
        Inpainted image or None if operation fails
    """
    client = get_replicate_client()
    return client.inpaint_background(image, mask, prompt)
