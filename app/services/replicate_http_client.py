"""
Direct HTTP client for Replicate API.

This is a workaround for Python 3.14 compatibility issues with the official replicate package.
Uses the Replicate HTTP API directly via requests.
"""

import logging
import os
import time
from typing import Optional
import base64
from io import BytesIO

import numpy as np
import cv2
from PIL import Image
import requests

logger = logging.getLogger(__name__)


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
            print("\n" + "="*60)
            print("⚠️  REPLICATE API TOKEN NOT FOUND")
            print("="*60)
            print("AI inpainting will be DISABLED")
            print("System will use edge replication fallback")
            print("\nTo enable AI inpainting:")
            print("  1. Get token: https://replicate.com/account/api-tokens")
            print("  2. Add to .env file: REPLICATE_API_TOKEN=your_token")
            print("  3. Restart server")
            print("="*60 + "\n")
            logger.warning("REPLICATE_API_TOKEN not set. Replicate features will be unavailable.")
            self.available = False
            return
        
        # Token found
        print("\n" + "="*60)
        print("✅ REPLICATE API TOKEN FOUND")
        print("="*60)
        print(f"Token: {self.api_token[:15]}...")
        
        # Validate token format
        if not self.api_token.startswith("r8_"):
            print("⚠️  WARNING: Token doesn't start with 'r8_'")
            print("   This might not be a valid Replicate token")
            print("   Get a valid token from: https://replicate.com/account/api-tokens")
        
        print("✓ Using HTTP API client (Python 3.14 compatible)")
        print("✓ AI inpainting is ENABLED")
        print("✓ Using model: twn39/lama")
        print("="*60 + "\n")
        
        self.available = True
        logger.info("Replicate HTTP client initialized successfully")

    def is_available(self) -> bool:
        """Check if Replicate client is properly configured."""
        return self.available and self.api_token is not None

    def inpaint_background(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str = "seamless background extension",
        model: str = "twn39/lama",
    ) -> Optional[np.ndarray]:
        """
        Inpaint masked regions using Replicate's LaMa model via HTTP API.

        Args:
            image: Input image as numpy array (BGR or RGB)
            mask: Binary mask where 255 = region to inpaint, 0 = preserve
            prompt: Text prompt for inpainting (used by some models)
            model: Model identifier on Replicate (default: LaMa)

        Returns:
            Inpainted image as numpy array, or None if operation fails
        """
        if not self.is_available():
            print("\n🔄 BYPASSING REPLICATE - Using edge replication fallback")
            logger.warning("Replicate not available. Inpainting skipped.")
            return None

        try:
            # Convert numpy arrays to PIL Images
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pil_mask = Image.fromarray(mask)

            # Convert to data URIs for API
            image_uri = self._image_to_data_uri(pil_image)
            mask_uri = self._image_to_data_uri(pil_mask)

            print(f"\n🚀 CALLING REPLICATE API (HTTP)")
            print(f"   Model: {model}")
            print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
            print(f"   Mask coverage: {np.sum(mask > 0) / mask.size:.1%}")
            logger.info(f"Calling Replicate HTTP API: {model}")

            # Create prediction
            headers = {
                "Authorization": f"Token {self.api_token}",
                "Content-Type": "application/json",
            }

            payload = {
                "version": self._get_model_version(model),
                "input": {
                    "image": image_uri,
                    "mask": mask_uri,
                }
            }

            # Create prediction
            response = requests.post(
                f"{self.base_url}/predictions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            prediction = response.json()

            # Poll for completion
            prediction_url = prediction["urls"]["get"]
            output_url = self._wait_for_prediction(prediction_url, headers)

            if output_url:
                # Download result
                inpainted = self._download_image(output_url)
                if inpainted is not None:
                    print(f"✅ REPLICATE SUCCESS - AI inpainting completed")
                    logger.info("Inpainting completed successfully")
                    return inpainted

            print(f"\n❌ REPLICATE FAILED: No output received")
            logger.error("Inpainting failed: No output received")
            return None

        except Exception as e:
            print(f"\n❌ REPLICATE FAILED: {e}")
            print("   Falling back to edge replication")
            logger.error(f"Inpainting failed: {e}")
            return None

    def _get_model_version(self, model: str) -> str:
        """Get the latest version hash for a model."""
        # For twn39/lama, use a known working version
        # In production, you'd query the API for the latest version
        if model == "twn39/lama":
            return "twn39/lama:latest"
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
