"""Uvicorn entry point for the API."""

import uvicorn
from src.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.serving.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )