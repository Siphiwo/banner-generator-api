"""
Diagnostic script to check .env file loading and Replicate configuration.
Run this to troubleshoot environment variable issues.
"""

import os
import sys
from pathlib import Path

print("\n" + "="*70)
print("🔍 ENVIRONMENT DIAGNOSTICS")
print("="*70 + "\n")

# 1. Check Python version
print(f"1. Python Version: {sys.version}")
print()

# 2. Check .env file
env_path = Path(__file__).parent / ".env"
print(f"2. .env File Location: {env_path}")
print(f"   Exists: {env_path.exists()}")

if env_path.exists():
    print(f"   Size: {env_path.stat().st_size} bytes")
    print("\n   Content preview:")
    with open(env_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:10], 1):
            line = line.rstrip()
            if line and not line.startswith('#'):
                # Mask token value
                if '=' in line:
                    key, value = line.split('=', 1)
                    if 'TOKEN' in key or 'KEY' in key:
                        value = value[:15] + '...' if len(value) > 15 else value
                    print(f"   Line {i}: {key}={value}")
                else:
                    print(f"   Line {i}: {line}")
print()

# 3. Check python-dotenv
print("3. python-dotenv Package:")
try:
    import dotenv
    try:
        version = dotenv.__version__
        print(f"   ✓ Installed (version: {version})")
    except AttributeError:
        print(f"   ✓ Installed")
except ImportError:
    print("   ✗ NOT installed")
    print("   Run: pip install python-dotenv")
print()

# 4. Try loading .env
print("4. Loading .env File:")
try:
    from dotenv import load_dotenv
    result = load_dotenv(dotenv_path=env_path, override=True, verbose=True)
    print(f"   Result: {result}")
except Exception as e:
    print(f"   ✗ Error: {e}")
print()

# 5. Check environment variables
print("5. Environment Variables:")
token = os.environ.get("REPLICATE_API_TOKEN")
if token:
    print(f"   ✓ REPLICATE_API_TOKEN is set")
    print(f"   Value: {token[:15]}..." if len(token) > 15 else f"   Value: {token}")
    print(f"   Length: {len(token)} characters")
    
    # Validate format
    if token.startswith("r8_"):
        print(f"   ✓ Format looks correct (starts with r8_)")
    else:
        print(f"   ⚠ WARNING: Token should start with 'r8_'")
        print(f"   Current prefix: {token[:3]}")
else:
    print("   ✗ REPLICATE_API_TOKEN is NOT set")
print()

# 6. Check replicate package
print("6. Replicate Package:")
try:
    import replicate
    try:
        version = replicate.__version__
        print(f"   ✓ Installed (version: {version})")
    except AttributeError:
        print(f"   ✓ Installed")
    
    # Try to initialize client
    if token:
        try:
            client = replicate.Client(api_token=token)
            print(f"   ✓ Client initialized successfully")
        except Exception as e:
            print(f"   ✗ Client initialization failed: {e}")
    else:
        print(f"   ⚠ Cannot test client (no token)")
        
except ImportError:
    print("   ✗ NOT installed")
    print("   Run: pip install replicate")
print()

# 7. Summary
print("="*70)
print("📋 SUMMARY")
print("="*70)

issues = []
if not env_path.exists():
    issues.append("❌ .env file not found")
try:
    import dotenv
except ImportError:
    issues.append("❌ python-dotenv not installed")
    
if not token:
    issues.append("❌ REPLICATE_API_TOKEN not set")
elif not token.startswith("r8_"):
    issues.append("⚠️  Token format may be invalid (should start with r8_)")

try:
    import replicate
except ImportError:
    issues.append("❌ replicate package not installed")

if not issues:
    print("✅ All checks passed! Configuration looks good.")
    print("\nYou can now start the server:")
    print("  uvicorn app.main:app --reload")
else:
    print("Issues found:\n")
    for issue in issues:
        print(f"  {issue}")
    
    print("\nRecommended actions:")
    if not env_path.exists():
        print("  1. Create .env file with: REPLICATE_API_TOKEN=your_token")
    try:
        import dotenv
    except ImportError:
        print("  1. Install python-dotenv: pip install python-dotenv")
    
    if not token or not token.startswith("r8_"):
        print("  2. Get valid token from: https://replicate.com/account/api-tokens")
        print("  3. Update .env file with correct token")
    
    try:
        import replicate
    except ImportError:
        print("  4. Install replicate: pip install replicate")

print("="*70 + "\n")
