"""Tests for TRACE-X FastAPI Backend."""

from __future__ import annotations

import io
from fastapi.testclient import TestClient
import pytest

from api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "TRACE-X" in data["service"]


def test_case_creation_and_polling(client):
    # Create synthetic mock MP4 bytes
    fake_video = io.BytesIO(b"fake-mp4-stream-bytes-for-unit-test")
    fake_video.name = "test_clip.mp4"

    response = client.post(
        "/cases",
        files={"clip": ("test_clip.mp4", fake_video, "video/mp4")},
        data={"colour": "white", "body_type": "suv"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "case_id" in data
    assert data["status"] == "queued"

    case_id = data["case_id"]

    # Check status endpoint
    status_resp = client.get(f"/cases/{case_id}/status")
    assert status_resp.status_code == 200
    s_data = status_resp.json()
    assert s_data["case_id"] == case_id


def test_unsupported_file_extension(client):
    fake_doc = io.BytesIO(b"text content")
    response = client.post(
        "/cases",
        files={"clip": ("notes.pdf", fake_doc, "application/pdf")}
    )
    assert response.status_code == 400
    assert "Unsupported video format" in response.json()["detail"]


def test_tracks_endpoint(client):
    response = client.get("/cases/case-12345/tracks")
    assert response.status_code == 200
    assert "tracks" in response.json()
