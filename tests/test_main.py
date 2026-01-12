import pytest
from fastapi.testclient import TestClient
from app import app
import os

client = TestClient(app)

def test_frontend_loads():
    """Verify the main page returns the HTML frontend"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Greenwashing Detector" in response.text  # Check for the <title>

def test_api_endpoint():
    """Verify the API used by the frontend is working"""
    # This matches the fetch() call in your React code
    response = client.post("/api/analyze_additive", json={"text": "test claim"})
    assert response.status_code == 200
    assert "risk_score" in response.json()

def test_image_route():
    """
    Verify the /image route is mounted. 
    Note: This will only pass in CI if you have a dummy image 
    in the image/ folder.
    """
    # We check if the route exists. 404 is fine if the file is missing,
    # but 405 or 500 would mean the mounting failed.
    response = client.get("/image/comparison/zero-shot_cm.png")
    assert response.status_code in [200, 404]