"""
Launch script for running the FastAPI application locally with Uvicorn.
Can be executed from any working directory.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from backend.app.core.config import settings

if __name__ == "__main__":
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Project root: {settings.PROJECT_ROOT}")
    print(f"Docs URL: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"Health URL: http://{settings.HOST}:{settings.PORT}{settings.API_PREFIX}/health")
    print("Press Ctrl+C to stop.")

    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG or settings.ENVIRONMENT == "development",
    )
