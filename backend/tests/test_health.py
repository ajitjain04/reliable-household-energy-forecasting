"""
Test suite for FastAPI health endpoint and basic app verification.
Can be executed via:
    python -m unittest backend/tests/test_health.py
or
    python backend/tests/test_health.py
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings


class TestHealthEndpoint(unittest.TestCase):
    """Test cases for FastAPI health check and routing."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        """Test GET / returns 200 with service welcome metadata."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("docs_url", data)
        self.assertEqual(data["version"], settings.APP_VERSION)
        self.assertEqual(data["health_url"], f"{settings.API_PREFIX}/health")

    def test_health_endpoint_status(self):
        """Test GET /api/v1/health returns 200 and status == 'ok'."""
        response = self.client.get(f"{settings.API_PREFIX}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["app_name"], settings.APP_NAME)
        self.assertEqual(data["version"], settings.APP_VERSION)
        self.assertIn("timestamp", data)
        self.assertIn("project_root", data)

    def test_artifacts_presence(self):
        """Test that health check detects existing research artifacts."""
        response = self.client.get(f"{settings.API_PREFIX}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["artifacts_present"], "Required research artifacts should be present")
        self.assertIn("artifact_dirs", data)
        self.assertTrue(data["artifact_dirs"]["outputs"])
        self.assertTrue(data["artifact_dirs"]["tables"])
        self.assertTrue(data["artifact_dirs"]["models"])
        self.assertTrue(data["artifact_dirs"]["scalers"])

    def test_cors_headers(self):
        """Test that CORS headers allow Vite development server origin."""
        headers = {"Origin": "http://localhost:5173"}
        response = self.client.get(f"{settings.API_PREFIX}/health", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5173")


if __name__ == "__main__":
    unittest.main()
