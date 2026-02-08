"""
Quick test to verify .env file is being loaded correctly.
Run this before starting the server to check configuration.
"""

import os
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
    print(f"✓ Loaded .env file from: {env_path}")
except ImportError:
    print("✗ python-dotenv not installed. Run: pip install python-dotenv")
    exit(1)

# Check for Replicate API token
token = os.environ.get("REPLICATE_API_TOKEN")

print("\n" + "="*50)
print("Environment Configuration Check")
print("="*50 + "\n")

if token:
    # Show first 10 characters for security
    masked_token = token[:10] + "..." if len(token) > 10 else token
    print(f"✓ REPLICATE_API_TOKEN is set: {masked_token}")
    print(f"  Token length: {len(token)} characters")
    
    # Validate token format
    if token.startswith("r8_"):
        print("  ✓ Token format looks correct (starts with r8_)")
    else:
        print("  ⚠ Warning: Token should start with 'r8_'")
    
    print("\n✓ AI inpainting will be ENABLED")
else:
    print("✗ REPLICATE_API_TOKEN is NOT set")
    print("\nTo enable AI inpainting:")
    print("  1. Get token from: https://replicate.com/account/api-tokens")
    print("  2. Edit .env file and replace 'your_token_here' with your token")
    print("  3. Run this test again to verify")
    print("\n⚠ AI inpainting will be DISABLED (fallback to edge replication)")

# Check other optional settings
print("\n" + "-"*50)
print("Optional Settings")
print("-"*50 + "\n")

tesseract_cmd = os.environ.get("TESSERACT_CMD")
if tesseract_cmd:
    print(f"✓ TESSERACT_CMD: {tesseract_cmd}")
else:
    print("  TESSERACT_CMD: Not set (will use system PATH)")

storage_path = os.environ.get("STORAGE_PATH", "./storage")
print(f"  STORAGE_PATH: {storage_path}")

log_level = os.environ.get("LOG_LEVEL", "INFO")
print(f"  LOG_LEVEL: {log_level}")

print("\n" + "="*50)
print("Configuration check complete!")
print("="*50 + "\n")

if token:
    print("✓ Ready to start server with AI inpainting enabled")
    print("  Run: uvicorn app.main:app --reload")
else:
    print("⚠ Ready to start server (AI inpainting disabled)")
    print("  Run: uvicorn app.main:app --reload")
    print("  Or set token first, then start server")
