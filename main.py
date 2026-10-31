"""
Root-level entrypoint for Render and Uvicorn deployment.
Supports both `uvicorn main:app` and `uvicorn app.main:app`.
"""
from app.main import app

__all__ = ["app"]

