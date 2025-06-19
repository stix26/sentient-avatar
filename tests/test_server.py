from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from sentient_avatar.avatar_server import app

# Test data
TEST_AUDIO = b"test audio data"
TEST_IMAGE = b"test image data"
TEST_TEXT = "Hello, world!"
TEST_VECTOR = [0.1] * 1536


@pytest.fixture
def mock_llm_service():
    service = Mock()
    service.generate = AsyncMock(return_value={"text": TEST_TEXT})
    service.chat = AsyncMock(return_value={"text": TEST_TEXT})
    return service


@pytest.fixture
def mock_asr_service():
    service = Mock()
    service.transcribe = AsyncMock(return_value={"text": TEST_TEXT})
    service.transcribe_stream = AsyncMock(return_value={"text": TEST_TEXT})
    return service


@pytest.fixture
def mock_tts_service():
    service = Mock()
    service.synthesize = AsyncMock(return_value={"audio": TEST_AUDIO})
    service.clone_voice = AsyncMock(return_value={"audio": TEST_AUDIO})
    return service


@pytest.fixture
def mock_avatar_service():
    service = Mock()
    service.generate_video = AsyncMock(return_value={"video": b"test video"})
    service.generate_stream = AsyncMock(return_value={"video": b"test video"})
    return service


@pytest.fixture
def mock_vision_service():
    service = Mock()
    service.analyze_image = AsyncMock(return_value={"analysis": "test analysis"})
    service.describe_image = AsyncMock(return_value={"description": "test description"})
    service.detect_objects = AsyncMock(return_value={"objects": ["test object"]})
    return service


@pytest.fixture
def mock_vector_store_service():
    service = Mock()
    service.upsert_points = AsyncMock(return_value={"status": "success"})
    service.search_points = AsyncMock(
        return_value={"points": [{"payload": {"text": TEST_TEXT}}]}
    )
    return service


@pytest.fixture
def client(
    mock_llm_service,
    mock_asr_service,
    mock_tts_service,
    mock_avatar_service,
    mock_vision_service,
    mock_vector_store_service,
):
    with (
        patch(
            "sentient_avatar.avatar_server.LLMService", return_value=mock_llm_service
        ),
        patch(
            "sentient_avatar.avatar_server.ASRService", return_value=mock_asr_service
        ),
        patch(
            "sentient_avatar.avatar_server.TTSService", return_value=mock_tts_service
        ),
        patch(
            "sentient_avatar.avatar_server.AvatarService",
            return_value=mock_avatar_service,
        ),
        patch(
            "sentient_avatar.avatar_server.VisionService",
            return_value=mock_vision_service,
        ),
        patch(
            "sentient_avatar.avatar_server.VectorStoreService",
            return_value=mock_vector_store_service,
        ),
    ):
        with TestClient(app) as test_client:
            yield test_client


# Health Check Tests
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


# Chat Endpoint Tests
def test_chat_endpoint(client):
    response = client.post("/chat", json={"text": TEST_TEXT})
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "audio" in data
    assert "video" in data


def test_chat_endpoint_invalid_input(client):
    response = client.post("/chat", json={})
    assert response.status_code == 422


# Transcribe Endpoint Tests
def test_transcribe_endpoint(client):
    files = {"file": ("test.wav", TEST_AUDIO, "audio/wav")}
    response = client.post("/transcribe", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data


def test_transcribe_endpoint_invalid_file(client):
    response = client.post("/transcribe", files={})
    assert response.status_code == 422


# Analyze Image Endpoint Tests
def test_analyze_image_endpoint(client):
    files = {"file": ("test.jpg", TEST_IMAGE, "image/jpeg")}
    data = {"prompt": "What's in this image?"}
    response = client.post("/analyze-image", files=files, data=data)
    assert response.status_code == 200
    data = response.json()
    assert "analysis" in data


def test_analyze_image_endpoint_invalid_file(client):
    response = client.post("/analyze-image", files={})
    assert response.status_code == 422


# Voices Endpoint Tests
def test_voices_endpoint(client):
    response = client.get("/voices")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


# Styles Endpoint Tests
def test_styles_endpoint(client):
    response = client.get("/styles")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


# WebSocket Tests
def test_websocket_connection(client):
    with client.websocket_connect("/ws/test_client") as websocket:
        # Send a text message
        websocket.send_text(TEST_TEXT)
        response = websocket.receive_json()
        assert "text" in response
        assert "audio" in response
        assert "video" in response


def test_websocket_audio_stream(client):
    with client.websocket_connect("/ws/test_client") as websocket:
        # Send audio data
        websocket.send_bytes(TEST_AUDIO)
        response = websocket.receive_json()
        assert "text" in response
        assert "audio" in response
        assert "video" in response


def test_websocket_invalid_message(client):
    with client.websocket_connect("/ws/test_client") as websocket:
        # Send invalid message
        websocket.send_text("invalid")
        response = websocket.receive_json()
        assert "error" in response
