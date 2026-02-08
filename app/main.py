import os
from pathlib import Path
from fastapi import FastAPI

from app.api.v1.routes import router as api_v1_router

# Load environment variables from .env file
print("\n" + "="*60)
print("🔧 LOADING ENVIRONMENT CONFIGURATION")
print("="*60)

env_path = Path(__file__).parent.parent / ".env"
print(f"Looking for .env file at: {env_path}")

if env_path.exists():
    print(f"✓ .env file found")
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=True)
        print(f"✓ .env file loaded successfully")
        
        # Check if token was loaded
        token = os.environ.get("REPLICATE_API_TOKEN")
        if token:
            print(f"✓ REPLICATE_API_TOKEN loaded: {token[:15]}...")
        else:
            print(f"⚠ REPLICATE_API_TOKEN not found in .env file")
            
    except ImportError:
        print("✗ python-dotenv not installed")
        print("  Run: pip install python-dotenv")
    except Exception as e:
        print(f"✗ Error loading .env: {e}")
else:
    print(f"⚠ .env file not found at: {env_path}")
    print(f"  Create it with: REPLICATE_API_TOKEN=your_token_here")

print("="*60 + "\n")


def create_app() -> FastAPI:
    """
    Application factory for the Banner Builder API.

    Keeping this as a separate function makes it easier to extend
    configuration and testing later.
    """
    app = FastAPI(
        title="Banner Builder API",
        version="0.1.0",
        description="Backend for content-aware banner resizing.",
    )

    # Infrastructure-level health check (non-versioned) primarily for ops.
    @app.get("/health", tags=["health"])
    async def root_health_check() -> dict:
        """Simple root health check endpoint."""
        return {"status": "ok"}

    # Public, versioned API routes.
    app.include_router(api_v1_router)

    return app


app = create_app()

