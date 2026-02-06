from fastapi import FastAPI

from app.api.v1.routes import router as api_v1_router


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

