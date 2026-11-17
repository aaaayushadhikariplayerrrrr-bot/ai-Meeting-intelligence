def test_root_endpoint(client):
    """Test the root welcome endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "name" in data
    assert "version" in data


def test_health_endpoint(client):
    """Test the /health endpoint directly."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"
    assert data["app_name"] == "Multilingual Meeting Intelligence System"
    assert "timestamp" in data
    assert "version" in data


def test_api_v1_health_endpoint(client):
    """Test the versioned /api/v1/health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"


def test_cors_headers(client):
    """Test CORS headers for local frontend origins."""
    response = client.get(
        "/health",
        headers={"Origin": "http://localhost:5173"}
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
