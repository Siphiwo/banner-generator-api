"""
Replicate AI integration for banner resizing operations.

This module provides a unified interface to Replicate's API for:
- Image inpainting (background extension)
- Vision models (face detection, text detection)
- Image processing tasks

All operations include graceful fallback and error handling.
"""

import logging
import os
from typing import Optional
import base64
from io import BytesIO

import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


class ReplicateClient:
    """
    Client for interacting with Replicate AI models.

    Handles authentication, model selection, and API calls with fallback strategies.
    """

    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize Replicate client.

        Args:
            api_token: Replicate API token. If not provided, reads from REPLICATE_API_TOKEN env var.
        """
        self.api_token = api_token or os.environ.get("REPLICATE_API_TOKEN")
        if not self.api_token:
            logger.warning(
                "REPLICATE_API_TOKEN not set. Replicate features will be unavailable."
            )
        
        # Import replicate here to avoid hard dependency
        try:
            import replicate
            self.replicate = replicate.Replicate(api_token=self.api_token)
            self.available = True
        except (ImportError, Exception) as e:
            logger.warning(f"replicate package not available: {e}")
            self.replicate = None
            self.available = False

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
        Inpaint masked regions using Replicate's LaMa model.

        Args:
            image: Input image as numpy array (BGR or RGB)
            mask: Binary mask where 255 = region to inpaint, 0 = preserve
            prompt: Text prompt for inpainting (used by some models)
            model: Model identifier on Replicate (default: LaMa)

        Returns:
            Inpainted image as numpy array, or None if operation fails
        """
        if not self.is_available():
            logger.warning("Replicate not available. Inpainting skipped.")
            return None

        try:
            # Convert numpy arrays to PIL Images for API
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pil_mask = Image.fromarray(mask)

            # Convert to base64 for API transmission
            image_b64 = self._image_to_base64(pil_image)
            mask_b64 = self._image_to_base64(pil_mask)

            logger.info(f"Calling Replicate inpainting model: {model}")

            # Call Replicate API
            output = self.replicate.run(
                model,
                input={
                    "image": image_b64,
                    "mask": mask_b64,
                    "prompt": prompt,
                },
            )

            # Handle different output formats
            if isinstance(output, str):
                # URL output - download and convert
                inpainted = self._download_image(output)
            elif isinstance(output, list) and len(output) > 0:
                # List of URLs
                inpainted = self._download_image(output[0])
            else:
                logger.error(f"Unexpected output format from inpainting model: {type(output)}")
                return None

            logger.info("Inpainting completed successfully")
            return inpainted

        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            return None

    def detect_faces(
        self,
        image: np.ndarray,
        model: str = "sczhou/codeformer",
    ) -> Optional[np.ndarray]:
        """
        Detect faces in image using Replicate model.

        Note: This is a placeholder for future face detection integration.
        Currently, we use OpenCV's Haar cascades locally for speed.

        Args:
            image: Input image as numpy array
            model: Model identifier on Replicate

        Returns:
            Annotated image with face bounding boxes, or None if operation fails
        """
        if not self.is_available():
            logger.warning("Replicate not available. Face detection skipped.")
            return None

        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_b64 = self._image_to_base64(pil_image)

            logger.info(f"Calling Replicate face detection model: {model}")

            output = self.replicate.run(
                model,
                input={"image": image_b64},
            )

            logger.info("Face detection completed")
            return output

        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return None

    def _image_to_base64(self, pil_image: Image.Image) -> str:
        """Convert PIL Image to base64 string for API transmission."""
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def _download_image(self, url: str) -> Optional[np.ndarray]:
        """Download image from URL and convert to numpy array."""
        try:
            import requests
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            pil_image = Image.open(BytesIO(response.content))
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            return None


# Global client instance
_replicate_client: Optional[ReplicateClient] = None


def get_replicate_client() -> ReplicateClient:
    """Get or create global Replicate client instance."""
    global _replicate_client
    if _replicate_client is None:
        _replicate_client = ReplicateClient()
    return _replicate_client


def inpaint_background(
    image: np.ndarray,
    mask: np.ndarray,
    prompt: str = "seamless background extension",
) -> Optional[np.ndarray]:
    """
    Convenience function to inpaint background using Replicate.

    Args:
        image: Input image as numpy array
        mask: Binary mask where 255 = region to inpaint
        prompt: Text prompt for inpainting

    Returns:
        Inpainted image or None if operation fails
    """
    client = get_replicate_client()
    return client.inpaint_background(image, mask, prompt)
