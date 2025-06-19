import asyncio
import base64
import json
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from sentient_avatar.config.config import Config, get_default_config
from sentient_avatar.services.asr import ASRService
from sentient_avatar.services.avatar import AvatarService
from sentient_avatar.services.factory import ServiceFactory
from sentient_avatar.services.llm import LLMService
from sentient_avatar.services.tts import TTSService
from sentient_avatar.services.vector_store import VectorStoreService
from sentient_avatar.services.vision import VisionService

# Test data
TEST_AUDIO = b"test audio data"
TEST_IMAGE = b"test image data"
TEST_TEXT = "Hello, world!"
TEST_VECTOR = [0.1] * 1536


@pytest.fixture
def mock_response():
    return {"status": "success", "data": "test data", "metadata": {"key": "value"}}


@pytest.fixture
def mock_health_response():
    return {"status": "healthy"}


@pytest.fixture
def config():
    """Create test configuration."""
    return Config.parse_obj(get_default_config())


@pytest.fixture
def service_factory(config):
    """Create service factory."""
    return ServiceFactory(config)


@pytest.mark.asyncio
async def test_llm_service(service_factory):
    """Test LLM service."""
    llm = service_factory.get_llm()

    # Test health check
    health = await llm.health_check()
    assert health is True

    # Test generate
    response = await llm.generate("Hello, how are you?")
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_asr_service(service_factory):
    """Test ASR service."""
    asr = service_factory.get_asr()

    # Test health check
    health = await asr.health_check()
    assert health is True

    # Test transcribe
    with open("tests/data/test_audio.wav", "rb") as f:
        audio_data = f.read()

    text = await asr.transcribe(audio_data)
    assert isinstance(text, str)
    assert len(text) > 0


@pytest.mark.asyncio
async def test_tts_service(service_factory):
    """Test TTS service."""
    tts = service_factory.get_tts()

    # Test health check
    health = await tts.health_check()
    assert health is True

    # Test synthesize
    audio = await tts.synthesize("Hello, this is a test.")
    assert isinstance(audio, bytes)
    assert len(audio) > 0

    # Test get voices
    voices = await tts.get_voices()
    assert isinstance(voices, list)
    assert len(voices) > 0


@pytest.mark.asyncio
async def test_avatar_service(service_factory):
    """Test avatar service."""
    avatar = service_factory.get_avatar()

    # Test health check
    health = await avatar.health_check()
    assert health is True

    # Test generate video
    with open("tests/data/test_audio.wav", "rb") as f:
        audio_data = f.read()

    video = await avatar.generate_video(audio_data)
    assert isinstance(video, bytes)
    assert len(video) > 0


@pytest.mark.asyncio
async def test_vision_service(service_factory):
    """Test vision service."""
    vision = service_factory.get_vision()

    # Test health check
    health = await vision.health_check()
    assert health is True

    # Test analyze image
    with open("tests/data/test_image.jpg", "rb") as f:
        image_data = f.read()

    analysis = await vision.analyze_image(image_data, "Describe this image")
    assert isinstance(analysis, dict)
    assert "description" in analysis

    # Test get styles
    styles = await vision.get_available_styles()
    assert isinstance(styles, list)
    assert len(styles) > 0


@pytest.mark.asyncio
async def test_vector_store_service(service_factory):
    """Test vector store service."""
    vector_store = service_factory.get_vector_store()

    # Test health check
    health = await vector_store.health_check()
    assert health is True

    # Test create collection
    collection = await vector_store.create_collection()
    assert isinstance(collection, dict)

    # Test upsert points
    points = [
        {"id": "1", "vector": [0.1] * 1536, "payload": {"text": "Test point 1"}},
        {"id": "2", "vector": [0.2] * 1536, "payload": {"text": "Test point 2"}},
    ]

    result = await vector_store.upsert_points(points)
    assert isinstance(result, dict)

    # Test search points
    query_vector = [0.1] * 1536
    results = await vector_store.search_points(query_vector, limit=2)
    assert isinstance(results, list)
    assert len(results) > 0

    # Test delete points
    result = await vector_store.delete_points(["1", "2"])
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_service_factory(service_factory):
    """Test service factory."""
    # Test get services
    assert isinstance(service_factory.get_llm(), LLMService)
    assert isinstance(service_factory.get_asr(), ASRService)
    assert isinstance(service_factory.get_tts(), TTSService)
    assert isinstance(service_factory.get_avatar(), AvatarService)
    assert isinstance(service_factory.get_vision(), VisionService)
    assert isinstance(service_factory.get_vector_store(), VectorStoreService)

    # Test initialize
    await service_factory.initialize()

    # Test cleanup
    await service_factory.cleanup()


@pytest.mark.asyncio
async def test_error_handling(service_factory):
    """Test error handling."""
    # Test invalid service type
    with pytest.raises(ValueError):
        service_factory.get_service("invalid")

    # Test service errors
    llm = service_factory.get_llm()
    with patch.object(llm, "generate", side_effect=Exception("Test error")):
        with pytest.raises(Exception):
            await llm.generate("Test")

    # Test rate limiting
    with patch.object(
        service_factory.rate_limiter, "is_rate_limited", return_value=True
    ):
        with pytest.raises(Exception):
            await llm.generate("Test")

    # Test caching
    with patch.object(service_factory.cache, "get", return_value=None):
        response = await llm.generate("Test")
        assert isinstance(response, str)


# LLM Service Tests
@pytest.mark.asyncio
async def test_llm_service_health_check(mock_health_response):
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_get.return_value.__aenter__.return_value.json.return_value = (
            mock_health_response
        )
        service = LLMService()
        result = await service.health_check()
        assert result is True


@pytest.mark.asyncio
async def test_llm_service_generate(mock_response):
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.json.return_value = mock_response
        service = LLMService()
        result = await service.generate(TEST_TEXT)
        assert result == mock_response


@pytest.mark.asyncio
async def test_llm_service_chat(mock_response):
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.json.return_value = mock_response
        service = LLMService()
        result = await service.chat([{"role": "user", "content": TEST_TEXT}])
        assert result == mock_response


# ASR Service Tests
@pytest.mark.asyncio
async def test_asr_service_health_check(mock_health_response):
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_get.return_value.__aenter__.return_value.json.return_value = (
            mock_health_response
        )
        service = ASRService()
        result = await service.health_check()
        assert result is True


@pytest.mark.asyncio
async def test_asr_service_transcribe(mock_response):
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.json.return_value = mock_response
        service = ASRService()
        result = await service.transcribe(TEST_AUDIO)
        assert result == mock_response


@pytest.mark.asyncio
async def test_asr_service_transcribe_stream(mock_response):
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.json.return_value = mock_response
        service = ASRService()
        result = await service.transcribe_stream(TEST_AUDIO)
        assert result == mock_response


# TTS Service Tests
@pytest.mark.asyncio
async def test_tts_service_health_check(mock_health_response):
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_get.return_value.__aenter__.return_value.json.return_value = (
            mock_health_response
        )
        service = TTSService()
        result = await service.health_check()
        assert result is True


@pytest.mark.asyncio
async def test_tts_service_synthesize(mock_response):
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.json.return_value = mock_response
        service = TTSService()
        result = await service.synthesize(TEST_TEXT)
        assert result == mock_response


@pytest.mark.asyncio
async def test_tts_service_clone_voice(mock_response):
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.json.return_value = mock_response
        service = TTSService()
        result = await service.clone_voice(TEST_AUDIO, TEST_TEXT)
        assert result == mock_response


# Avatar Service Tests
@pytest.mark.asyncio
async def test_avatar_service_health_check(mock_health_response):
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_get.return_value.__aenter__.return_value.json.return_value = (
            mock_health_response
        )
        service = AvatarService()
        result = await service.health_check()
        assert result is True


@pytest.mark.asyncio
async def test_avatar_service_generate_video(mock_response):
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.json.return_value = mock_response
        service = AvatarService()
        result = await service.generate_video(TEST_AUDIO)
        assert result == mock_response


@pytest.mark.asyncio
async def test_avatar_service_generate_stream(mock_response):
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.json.return_value = mock_response
        service = AvatarService()
        result = await service.generate_stream(TEST_AUDIO)
        assert result == mock_response


# Vision Service Tests
@pytest.mark.asyncio
async def test_vision_service_health_check(mock_health_response):
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_get.return_value.__aenter__.return_value.json.return_value = (
            mock_health_response
        )
        service = VisionService()
        result = await service.health_check()
        assert result is True


@pytest.mark.asyncio
async def test_vision_service_analyze_image(mock_response):
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.json.return_value = mock_response
        service = VisionService()
        result = await service.analyze_image(TEST_IMAGE)
        assert result == mock_response


@pytest.mark.asyncio
async def test_vision_service_detect_objects(mock_response):
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.json.return_value = mock_response
        service = VisionService()
        result = await service.detect_objects(TEST_IMAGE)
        assert result == mock_response


# Vector Store Service Tests
@pytest.mark.asyncio
async def test_vector_store_service_health_check(mock_health_response):
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_get.return_value.__aenter__.return_value.json.return_value = (
            mock_health_response
        )
        service = VectorStoreService()
        result = await service.health_check()
        assert result is True


@pytest.mark.asyncio
async def test_vector_store_service_create_collection(mock_response):
    with patch("aiohttp.ClientSession.put") as mock_put:
        mock_put.return_value.__aenter__.return_value.json.return_value = mock_response
        service = VectorStoreService()
        result = await service.create_collection("test_collection")
        assert result == mock_response


@pytest.mark.asyncio
async def test_vector_store_service_upsert_points(mock_response):
    with patch("aiohttp.ClientSession.put") as mock_put:
        mock_put.return_value.__aenter__.return_value.json.return_value = mock_response
        service = VectorStoreService()
        points = [{"id": "1", "vector": TEST_VECTOR, "payload": {"text": TEST_TEXT}}]
        result = await service.upsert_points("test_collection", points)
        assert result == mock_response


@pytest.mark.asyncio
async def test_vector_store_service_search_points(mock_response):
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value.json.return_value = mock_response
        service = VectorStoreService()
        result = await service.search_points("test_collection", TEST_VECTOR)
        assert result == mock_response
