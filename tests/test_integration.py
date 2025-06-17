import pytest
import asyncio
import json
import base64
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from sentient_avatar.avatar_server import app
from sentient_avatar.services.factory import ServiceFactory
from sentient_avatar.config.config import Config, get_default_config

@pytest.fixture
def config():
    """Create test configuration."""
    return Config.parse_obj(get_default_config())

@pytest.fixture
def service_factory(config):
    """Create service factory."""
    return ServiceFactory(config)

@pytest.fixture
def client(service_factory):
    """Create test client."""
    with patch('sentient_avatar.avatar_server.service_factory', service_factory):
        with TestClient(app) as client:
            yield client

@pytest.mark.asyncio
async def test_chat_endpoint(client):
    """Test chat endpoint."""
    # Mock service responses
    with patch('sentient_avatar.services.llm.LLMService.generate') as mock_generate:
        with patch('sentient_avatar.services.tts.TTSService.synthesize') as mock_synthesize:
            with patch('sentient_avatar.services.avatar.AvatarService.generate_video') as mock_generate_video:
                mock_generate.return_value = "Test response"
                mock_synthesize.return_value = b"test audio"
                mock_generate_video.return_value = b"test video"
                
                # Test chat endpoint
                response = client.post(
                    "/chat",
                    data={"text": "Hello"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "text" in data
                assert "audio" in data
                assert "video" in data
                
                # Verify service calls
                mock_generate.assert_called_once_with("Hello")
                mock_synthesize.assert_called_once_with("Test response")
                mock_generate_video.assert_called_once_with(b"test audio")

@pytest.mark.asyncio
async def test_transcribe_endpoint(client):
    """Test transcribe endpoint."""
    # Mock service response
    with patch('sentient_avatar.services.asr.ASRService.transcribe') as mock_transcribe:
        mock_transcribe.return_value = "Test transcription"
        
        # Test transcribe endpoint
        with open("tests/data/test_audio.wav", "rb") as f:
            response = client.post(
                "/transcribe",
                files={"audio": ("test.wav", f, "audio/wav")}
            )
            
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert data["text"] == "Test transcription"
        
        # Verify service call
        mock_transcribe.assert_called_once()

@pytest.mark.asyncio
async def test_analyze_image_endpoint(client):
    """Test analyze image endpoint."""
    # Mock service response
    with patch('sentient_avatar.services.vision.VisionService.analyze_image') as mock_analyze:
        mock_analyze.return_value = {
            "description": "Test description",
            "objects": ["object1", "object2"]
        }
        
        # Test analyze image endpoint
        with open("tests/data/test_image.jpg", "rb") as f:
            response = client.post(
                "/analyze-image",
                files={"image": ("test.jpg", f, "image/jpeg")},
                data={"prompt": "Describe this image"}
            )
            
        assert response.status_code == 200
        data = response.json()
        assert "description" in data
        assert "objects" in data
        
        # Verify service call
        mock_analyze.assert_called_once()

@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Test health endpoint."""
    # Mock service responses
    with patch('sentient_avatar.services.llm.LLMService.health_check') as mock_llm_health:
        with patch('sentient_avatar.services.asr.ASRService.health_check') as mock_asr_health:
            with patch('sentient_avatar.services.tts.TTSService.health_check') as mock_tts_health:
                with patch('sentient_avatar.services.avatar.AvatarService.health_check') as mock_avatar_health:
                    with patch('sentient_avatar.services.vision.VisionService.health_check') as mock_vision_health:
                        with patch('sentient_avatar.services.vector_store.VectorStoreService.health_check') as mock_vector_store_health:
                            mock_llm_health.return_value = True
                            mock_asr_health.return_value = True
                            mock_tts_health.return_value = True
                            mock_avatar_health.return_value = True
                            mock_vision_health.return_value = True
                            mock_vector_store_health.return_value = True
                            
                            # Test health endpoint
                            response = client.get("/health")
                            
                            assert response.status_code == 200
                            data = response.json()
                            assert "llm" in data
                            assert "asr" in data
                            assert "tts" in data
                            assert "avatar" in data
                            assert "vision" in data
                            assert "vector_store" in data
                            
                            # Verify service calls
                            mock_llm_health.assert_called_once()
                            mock_asr_health.assert_called_once()
                            mock_tts_health.assert_called_once()
                            mock_avatar_health.assert_called_once()
                            mock_vision_health.assert_called_once()
                            mock_vector_store_health.assert_called_once()

@pytest.mark.asyncio
async def test_websocket_endpoint(client):
    """Test WebSocket endpoint."""
    # Mock service responses
    with patch('sentient_avatar.services.llm.LLMService.generate') as mock_generate:
        with patch('sentient_avatar.services.tts.TTSService.synthesize') as mock_synthesize:
            with patch('sentient_avatar.services.avatar.AvatarService.generate_video') as mock_generate_video:
                mock_generate.return_value = "Test response"
                mock_synthesize.return_value = b"test audio"
                mock_generate_video.return_value = b"test video"
                
                # Test WebSocket endpoint
                with client.websocket_connect("/ws/test_client") as websocket:
                    # Send text message
                    websocket.send_json({
                        "type": "text",
                        "text": "Hello"
                    })
                    
                    # Receive response
                    response = websocket.receive_json()
                    assert response["type"] == "response"
                    assert "text" in response
                    assert "audio" in response
                    assert "video" in response
                    
                    # Verify service calls
                    mock_generate.assert_called_once_with("Hello")
                    mock_synthesize.assert_called_once_with("Test response")
                    mock_generate_video.assert_called_once_with(b"test audio")

@pytest.mark.asyncio
async def test_rate_limiting(client):
    """Test rate limiting."""
    # Mock rate limiter
    with patch('sentient_avatar.avatar_server.rate_limiter') as mock_rate_limiter:
        mock_rate_limiter.is_rate_limited.return_value = True
        
        # Test rate limited request
        response = client.post(
            "/chat",
            data={"text": "Hello"}
        )
        
        assert response.status_code == 429
        assert response.json()["detail"] == "Rate limit exceeded"
        
        # Verify rate limiter call
        mock_rate_limiter.is_rate_limited.assert_called_once_with("chat")

@pytest.mark.asyncio
async def test_error_handling(client):
    """Test error handling."""
    # Mock service error
    with patch('sentient_avatar.services.llm.LLMService.generate', side_effect=Exception("Test error")):
        # Test error handling
        response = client.post(
            "/chat",
            data={"text": "Hello"}
        )
        
        assert response.status_code == 500
        assert "detail" in response.json()
        
    # Test invalid request
    response = client.post(
        "/chat",
        data={}  # Missing required field
    )
    
    assert response.status_code == 422  # Validation error

@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    """Test metrics endpoint."""
    # Mock metrics collector
    with patch('sentient_avatar.avatar_server.metrics') as mock_metrics:
        mock_metrics.get_metrics.return_value = {
            "requests": {"total": 10, "latency": 1.0},
            "services": {"health": {"llm": 1}, "latency": {"llm": 0.5}},
            "system": {"cpu": 50.0, "memory": 1024, "disk": 2048}
        }
        
        # Test metrics endpoint
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "requests" in data
        assert "services" in data
        assert "system" in data
        
        # Verify metrics call
        mock_metrics.get_metrics.assert_called_once() 