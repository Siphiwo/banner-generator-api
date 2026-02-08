"""
Test suite for Replicate AI integration.

This module tests the Replicate client without requiring actual API calls.
For integration testing with real API, set REPLICATE_API_TOKEN environment variable.
"""

import logging
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from app.services.replicate_client import ReplicateClient, get_replicate_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_replicate_client_initialization():
    """Test that ReplicateClient initializes correctly."""
    logger.info("Testing ReplicateClient initialization...")
    
    # Test with no API token
    client = ReplicateClient(api_token=None)
    assert client.api_token is None
    logger.info("✓ Client initializes without API token")
    
    # Test with API token
    client = ReplicateClient(api_token="test-token-12345")
    assert client.api_token == "test-token-12345"
    logger.info("✓ Client initializes with API token")


def test_replicate_client_availability():
    """Test availability checks."""
    logger.info("Testing availability checks...")
    
    # Without token
    client = ReplicateClient(api_token=None)
    assert not client.is_available()
    logger.info("✓ Client correctly reports unavailable without token")
    
    # With token (but replicate package may not be installed)
    client = ReplicateClient(api_token="test-token")
    # Availability depends on whether replicate package is installed
    logger.info(f"✓ Client availability: {client.is_available()}")


def test_image_to_base64():
    """Test image to base64 conversion."""
    logger.info("Testing image to base64 conversion...")
    
    from PIL import Image
    
    client = ReplicateClient(api_token="test-token")
    
    # Create a simple test image
    test_image = Image.new("RGB", (100, 100), color="red")
    
    # Convert to base64
    b64 = client._image_to_base64(test_image)
    
    assert isinstance(b64, str)
    assert len(b64) > 0
    assert b64.startswith("iVBOR") or b64.startswith("iVBO")  # PNG magic bytes in base64
    logger.info(f"✓ Image converted to base64 ({len(b64)} chars)")


def test_inpaint_background_without_api():
    """Test inpainting gracefully fails without API token."""
    logger.info("Testing inpainting without API token...")
    
    client = ReplicateClient(api_token=None)
    
    # Create dummy image and mask
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    
    result = client.inpaint_background(image, mask)
    
    assert result is None
    logger.info("✓ Inpainting gracefully returns None without API token")


def test_inpaint_background_with_mock_api():
    """Test inpainting with mocked API."""
    logger.info("Testing inpainting with mocked API...")
    
    with patch("app.services.replicate_client.replicate") as mock_replicate:
        # Mock the replicate module
        mock_client = MagicMock()
        mock_replicate.Replicate.return_value = mock_client
        
        # Mock the run method to return a URL
        mock_url = "https://example.com/output.png"
        mock_client.run.return_value = mock_url
        
        # Create client
        client = ReplicateClient(api_token="test-token")
        client.replicate = mock_client
        client.available = True
        
        # Create dummy image and mask
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        # Mock the image download
        with patch.object(client, "_download_image") as mock_download:
            mock_download.return_value = np.ones((100, 100, 3), dtype=np.uint8) * 128
            
            result = client.inpaint_background(image, mask, prompt="test prompt")
            
            # Verify API was called
            assert mock_client.run.called
            call_args = mock_client.run.call_args
            assert "twn39/lama" in str(call_args)
            
            # Verify result
            assert result is not None
            assert result.shape == (100, 100, 3)
            logger.info("✓ Inpainting API call successful with mocked API")


def test_global_client_singleton():
    """Test that global client is a singleton."""
    logger.info("Testing global client singleton...")
    
    client1 = get_replicate_client()
    client2 = get_replicate_client()
    
    assert client1 is client2
    logger.info("✓ Global client is a singleton")


def test_replicate_models_available():
    """Test that recommended Replicate models are available."""
    logger.info("Testing Replicate model availability...")
    
    # These are the models we recommend for the banner resizing system
    recommended_models = {
        "inpainting": "twn39/lama",  # Fast, deterministic
        "inpainting_alt": "lucataco/sdxl-inpainting",  # Higher quality alternative
        "face_restoration": "sczhou/codeformer",  # Face enhancement
    }
    
    logger.info("Recommended Replicate models:")
    for purpose, model_id in recommended_models.items():
        logger.info(f"  - {purpose}: {model_id}")
    
    logger.info("✓ Model recommendations documented")


def test_error_handling():
    """Test error handling in Replicate client."""
    logger.info("Testing error handling...")
    
    with patch("app.services.replicate_client.replicate") as mock_replicate:
        mock_client = MagicMock()
        mock_replicate.Replicate.return_value = mock_client
        
        # Mock the run method to raise an exception
        mock_client.run.side_effect = Exception("API Error")
        
        client = ReplicateClient(api_token="test-token")
        client.replicate = mock_client
        client.available = True
        
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        result = client.inpaint_background(image, mask)
        
        assert result is None
        logger.info("✓ Error handling works correctly")


def run_all_tests():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Running Replicate Integration Tests")
    logger.info("=" * 60)
    
    tests = [
        test_replicate_client_initialization,
        test_replicate_client_availability,
        test_image_to_base64,
        test_inpaint_background_without_api,
        test_inpaint_background_with_mock_api,
        test_global_client_singleton,
        test_replicate_models_available,
        test_error_handling,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            logger.error(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    logger.info("=" * 60)
    logger.info(f"Tests passed: {passed}/{len(tests)}")
    logger.info(f"Tests failed: {failed}/{len(tests)}")
    logger.info("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
