import pytest

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    
def test_get_languages(client):
    response = client.get("/api/v1/analyze/languages")
    assert response.status_code == 200
    data = response.json()
    assert "languages" in data
    assert "python" in data["languages"]
