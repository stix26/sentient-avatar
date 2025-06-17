import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import base64
import json
from datetime import datetime

from sentient_avatar.agent_core import SentientAgent
from sentient_avatar.audio_pipeline import AudioPipeline, AudioChunk
from sentient_avatar.memory import Memory
from sentient_avatar.vision import Vision

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
    service.search_points = AsyncMock(return_value={"points": [{"payload": {"text": TEST_TEXT}}]})
    return service

# Agent Core Tests
@pytest.mark.asyncio
async def test_agent_core_initialization(mock_llm_service):
    agent = SentientAgent(llm_service=mock_llm_service)
    assert agent.llm_service == mock_llm_service
    assert agent.conversation_history == []

@pytest.mark.asyncio
async def test_agent_core_process_input(mock_llm_service):
    agent = SentientAgent(llm_service=mock_llm_service)
    result = await agent.process_input(TEST_TEXT)
    assert result == TEST_TEXT
    assert len(agent.conversation_history) == 2  # User input and response

@pytest.mark.asyncio
async def test_agent_core_execute_task(mock_llm_service):
    agent = SentientAgent(llm_service=mock_llm_service)
    result = await agent.execute_task(TEST_TEXT)
    assert result == TEST_TEXT

# Audio Pipeline Tests
@pytest.mark.asyncio
async def test_audio_pipeline_initialization(mock_asr_service, mock_tts_service):
    pipeline = AudioPipeline(asr_service=mock_asr_service, tts_service=mock_tts_service)
    assert pipeline.asr_service == mock_asr_service
    assert pipeline.tts_service == mock_tts_service
    assert pipeline.audio_buffer == b""

@pytest.mark.asyncio
async def test_audio_pipeline_process_chunk(mock_asr_service, mock_tts_service):
    pipeline = AudioPipeline(asr_service=mock_asr_service, tts_service=mock_tts_service)
    chunk = AudioChunk(data=TEST_AUDIO, sample_rate=16000, timestamp=datetime.now(), is_final=True)
    result = await pipeline.process_audio_chunk(chunk)
    assert result["transcription"] == TEST_TEXT
    assert result["status"] == "complete"

@pytest.mark.asyncio
async def test_audio_pipeline_synthesize_speech(mock_asr_service, mock_tts_service):
    pipeline = AudioPipeline(asr_service=mock_asr_service, tts_service=mock_tts_service)
    result = await pipeline.synthesize_speech(TEST_TEXT)
    assert isinstance(result, AudioChunk)
    assert result.data == TEST_AUDIO

# Memory Tests
@pytest.mark.asyncio
async def test_memory_initialization(mock_vector_store_service):
    memory = Memory(vector_store_service=mock_vector_store_service)
    assert memory.vector_store_service == mock_vector_store_service
    assert memory.collection_name == "memories"

@pytest.mark.asyncio
async def test_memory_store_memory(mock_vector_store_service):
    memory = Memory(vector_store_service=mock_vector_store_service)
    result = await memory.store_memory(TEST_TEXT)
    assert result["status"] == "success"

@pytest.mark.asyncio
async def test_memory_search_memories(mock_vector_store_service):
    memory = Memory(vector_store_service=mock_vector_store_service)
    result = await memory.search_memories(TEST_TEXT)
    assert len(result) == 1
    assert result[0]["content"] == TEST_TEXT

@pytest.mark.asyncio
async def test_memory_get_memory_context(mock_vector_store_service):
    memory = Memory(vector_store_service=mock_vector_store_service)
    result = await memory.get_memory_context(TEST_TEXT)
    assert TEST_TEXT in result

# Vision Tests
@pytest.mark.asyncio
async def test_vision_initialization(mock_vision_service):
    vision = Vision(vision_service=mock_vision_service)
    assert vision.vision_service == mock_vision_service

@pytest.mark.asyncio
async def test_vision_analyze_image(mock_vision_service):
    vision = Vision(vision_service=mock_vision_service)
    result = await vision.analyze_image(TEST_IMAGE)
    assert result["analysis"] == "test analysis"

@pytest.mark.asyncio
async def test_vision_detect_objects(mock_vision_service):
    vision = Vision(vision_service=mock_vision_service)
    result = await vision.detect_objects(TEST_IMAGE)
    assert result["objects"] == ["test object"]

@pytest.mark.asyncio
async def test_vision_validate_image():
    vision = Vision(None)
    result = vision.validate_image(TEST_IMAGE)
    assert result is False  # Invalid image data should return False 