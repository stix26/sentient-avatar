import pytest
from fastapi.testclient import TestClient


def test_avatar_creation(client: TestClient) -> None:
    response = client.post(
        "/api/v1/avatar",
        json={
            "gender": "female",
            "age": 24,
            "hair_style": "long_straight",
            "hair_color": "black",
            "eye_color": "brown",
            "skin_tone": "fair",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["gender"] == "female"
    assert data["age"] == 24
    assert data["hair_style"] == "long_straight"
    assert data["hair_color"] == "black"
    assert data["eye_color"] == "brown"
    assert data["skin_tone"] == "fair"


def test_avatar_update(client: TestClient) -> None:
    # First create an avatar
    create_response = client.post(
        "/api/v1/avatar",
        json={
            "gender": "female",
            "age": 24,
            "hair_style": "long_straight",
            "hair_color": "black",
            "eye_color": "brown",
            "skin_tone": "fair",
        },
    )
    avatar_id = create_response.json()["id"]

    # Then update it
    update_response = client.put(
        f"/api/v1/avatar/{avatar_id}",
        json={"hair_style": "short_curly", "hair_color": "blonde"},
    )
    assert update_response.status_code == 200
    data = update_response.json()
    assert data["hair_style"] == "short_curly"
    assert data["hair_color"] == "blonde"
    assert data["eye_color"] == "brown"  # Unchanged
    assert data["skin_tone"] == "fair"  # Unchanged


def test_avatar_emotions(client: TestClient) -> None:
    # Create an avatar
    create_response = client.post(
        "/api/v1/avatar",
        json={
            "gender": "female",
            "age": 24,
            "hair_style": "long_straight",
            "hair_color": "black",
            "eye_color": "brown",
            "skin_tone": "fair",
        },
    )
    avatar_id = create_response.json()["id"]

    # Test emotional response
    emotion_response = client.post(
        f"/api/v1/avatar/{avatar_id}/emotion",
        json={"emotion": "happy", "intensity": 0.8},
    )
    assert emotion_response.status_code == 200
    data = emotion_response.json()
    assert data["current_emotion"] == "happy"
    assert data["emotion_intensity"] == 0.8


def test_avatar_cognitive(client: TestClient) -> None:
    # Create an avatar
    create_response = client.post(
        "/api/v1/avatar",
        json={
            "gender": "female",
            "age": 24,
            "hair_style": "long_straight",
            "hair_color": "black",
            "eye_color": "brown",
            "skin_tone": "fair",
        },
    )
    avatar_id = create_response.json()["id"]

    # Test cognitive response
    cognitive_response = client.post(
        f"/api/v1/avatar/{avatar_id}/cognitive",
        json={"thought": "What is the meaning of life?", "context": "philosophical"},
    )
    assert cognitive_response.status_code == 200
    data = cognitive_response.json()
    assert "response" in data
    assert "thought_process" in data
    assert "emotional_impact" in data


def test_avatar_physical(client: TestClient) -> None:
    # Create an avatar
    create_response = client.post(
        "/api/v1/avatar",
        json={
            "gender": "female",
            "age": 24,
            "hair_style": "long_straight",
            "hair_color": "black",
            "eye_color": "brown",
            "skin_tone": "fair",
        },
    )
    avatar_id = create_response.json()["id"]

    # Test physical response
    physical_response = client.post(
        f"/api/v1/avatar/{avatar_id}/physical",
        json={"action": "wave", "intensity": 0.7},
    )
    assert physical_response.status_code == 200
    data = physical_response.json()
    assert data["action"] == "wave"
    assert data["intensity"] == 0.7
    assert "animation_data" in data
    assert "physical_state" in data


@pytest.mark.asyncio
async def test_avatar_streaming(client: TestClient) -> None:
    # Create an avatar
    create_response = client.post(
        "/api/v1/avatar",
        json={
            "gender": "female",
            "age": 24,
            "hair_style": "long_straight",
            "hair_color": "black",
            "eye_color": "brown",
            "skin_tone": "fair",
        },
    )
    avatar_id = create_response.json()["id"]

    # Test streaming response
    with client.stream("GET", f"/api/v1/avatar/{avatar_id}/stream") as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line:
                data = response.json()
                assert "frame" in data
                assert "audio" in data
                assert "state" in data
