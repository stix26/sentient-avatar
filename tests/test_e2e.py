import pytest
import asyncio
import json
import base64
import os
import subprocess
import time
from pathlib import Path
import requests
import websockets
import wave
import numpy as np
from PIL import Image

from sentient_avatar.config.config import get_default_config
from sentient_avatar.services.factory import ServiceFactory

@pytest.fixture(scope="session")
def config():
    """Create test configuration."""
    return get_default_config()

@pytest.fixture(scope="session")
def service_factory(config):
    """Create service factory."""
    return ServiceFactory.create()

@pytest.fixture(scope="session", autouse=True)
def setup_services():
    """Set up services for testing."""
    # Start services
    subprocess.run(["docker-compose", "up", "-d"])
    
    # Wait for services to be ready
    time.sleep(30)
    
    yield
    
    # Stop services
    subprocess.run(["docker-compose", "down"])

@pytest.mark.asyncio
async def test_chat_flow(service_factory):
    """Test complete chat flow."""
    # Get services
    llm = service_factory.get_llm()
    tts = service_factory.get_tts()
    avatar = service_factory.get_avatar()
    
    # Test chat
    response = await llm.generate("Hello, how are you?")
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Test TTS
    audio = await tts.synthesize(response)
    assert isinstance(audio, bytes)
    assert len(audio) > 0
    
    # Test avatar
    video = await avatar.generate_video(audio)
    assert isinstance(video, bytes)
    assert len(video) > 0

@pytest.mark.asyncio
async def test_audio_flow(service_factory):
    """Test complete audio flow."""
    # Get services
    asr = service_factory.get_asr()
    llm = service_factory.get_llm()
    tts = service_factory.get_tts()
    avatar = service_factory.get_avatar()
    
    # Create test audio
    sample_rate = 16000
    duration = 3  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t)
    audio_data = (audio_data * 32767).astype(np.int16)
    
    with wave.open("test_audio.wav", "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    # Test ASR
    with open("test_audio.wav", "rb") as f:
        text = await asr.transcribe(f.read())
    assert isinstance(text, str)
    
    # Test LLM
    response = await llm.generate(text)
    assert isinstance(response, str)
    
    # Test TTS
    audio = await tts.synthesize(response)
    assert isinstance(audio, bytes)
    
    # Test avatar
    video = await avatar.generate_video(audio)
    assert isinstance(video, bytes)
    
    # Cleanup
    os.remove("test_audio.wav")

@pytest.mark.asyncio
async def test_vision_flow(service_factory):
    """Test complete vision flow."""
    # Get services
    vision = service_factory.get_vision()
    llm = service_factory.get_llm()
    tts = service_factory.get_tts()
    avatar = service_factory.get_avatar()
    
    # Create test image
    image = Image.new("RGB", (100, 100), color="red")
    image.save("test_image.jpg")
    
    # Test vision
    with open("test_image.jpg", "rb") as f:
        analysis = await vision.analyze_image(f.read(), "Describe this image")
    assert isinstance(analysis, dict)
    assert "description" in analysis
    
    # Test LLM
    response = await llm.generate(analysis["description"])
    assert isinstance(response, str)
    
    # Test TTS
    audio = await tts.synthesize(response)
    assert isinstance(audio, bytes)
    
    # Test avatar
    video = await avatar.generate_video(audio)
    assert isinstance(video, bytes)
    
    # Cleanup
    os.remove("test_image.jpg")

@pytest.mark.asyncio
async def test_memory_flow(service_factory):
    """Test complete memory flow."""
    # Get services
    vector_store = service_factory.get_vector_store()
    llm = service_factory.get_llm()
    
    # Test vector store
    points = [
        {
            "id": "1",
            "vector": [0.1] * 1536,
            "payload": {"text": "Test memory 1"}
        },
        {
            "id": "2",
            "vector": [0.2] * 1536,
            "payload": {"text": "Test memory 2"}
        }
    ]
    
    # Store memories
    result = await vector_store.upsert_points(points)
    assert isinstance(result, dict)
    
    # Search memories
    query_vector = [0.1] * 1536
    results = await vector_store.search_points(query_vector, limit=2)
    assert isinstance(results, list)
    assert len(results) > 0
    
    # Delete memories
    result = await vector_store.delete_points(["1", "2"])
    assert isinstance(result, dict)

@pytest.mark.asyncio
async def test_websocket_flow():
    """Test WebSocket flow."""
    # Connect to WebSocket
    async with websockets.connect("ws://localhost:8000/ws/test_client") as websocket:
        # Send text message
        await websocket.send(json.dumps({
            "type": "text",
            "text": "Hello"
        }))
        
        # Receive response
        response = json.loads(await websocket.recv())
        assert response["type"] == "response"
        assert "text" in response
        assert "audio" in response
        assert "video" in response
        
        # Send audio message
        with open("test_audio.wav", "rb") as f:
            audio_data = f.read()
        
        await websocket.send(json.dumps({
            "type": "audio",
            "audio": base64.b64encode(audio_data).decode()
        }))
        
        # Receive response
        response = json.loads(await websocket.recv())
        assert response["type"] == "response"
        assert "text" in response
        assert "response" in response
        assert "audio" in response
        assert "video" in response
        
        # Send image message
        with open("test_image.jpg", "rb") as f:
            image_data = f.read()
        
        await websocket.send(json.dumps({
            "type": "image",
            "image": base64.b64encode(image_data).decode(),
            "prompt": "Describe this image"
        }))
        
        # Receive response
        response = json.loads(await websocket.recv())
        assert response["type"] == "response"
        assert "analysis" in response
        assert "response" in response
        assert "audio" in response
        assert "video" in response

@pytest.mark.asyncio
async def test_http_flow():
    """Test HTTP flow."""
    # Test chat endpoint
    response = requests.post(
        "http://localhost:8000/chat",
        data={"text": "Hello"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "audio" in data
    assert "video" in data
    
    # Test transcribe endpoint
    with open("test_audio.wav", "rb") as f:
        response = requests.post(
            "http://localhost:8000/transcribe",
            files={"audio": ("test.wav", f, "audio/wav")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    
    # Test analyze image endpoint
    with open("test_image.jpg", "rb") as f:
        response = requests.post(
            "http://localhost:8000/analyze-image",
            files={"image": ("test.jpg", f, "image/jpeg")},
            data={"prompt": "Describe this image"}
        )
    assert response.status_code == 200
    data = response.json()
    assert "description" in data
    
    # Test health endpoint
    response = requests.get("http://localhost:8000/health")
    assert response.status_code == 200
    data = response.json()
    assert "llm" in data
    assert "asr" in data
    assert "tts" in data
    assert "avatar" in data
    assert "vision" in data
    assert "vector_store" in data
    
    # Test metrics endpoint
    response = requests.get("http://localhost:8000/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "requests" in data
    assert "services" in data
    assert "system" in data

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling."""
    # Test invalid request
    response = requests.post(
        "http://localhost:8000/chat",
        data={}  # Missing required field
    )
    assert response.status_code == 422
    
    # Test rate limiting
    for _ in range(101):  # Exceed rate limit
        requests.post(
            "http://localhost:8000/chat",
            data={"text": "Hello"}
        )
    
    response = requests.post(
        "http://localhost:8000/chat",
        data={"text": "Hello"}
    )
    assert response.status_code == 429
    
    # Test invalid WebSocket message
    async with websockets.connect("ws://localhost:8000/ws/test_client") as websocket:
        await websocket.send(json.dumps({
            "type": "invalid"
        }))
        
        response = json.loads(await websocket.recv())
        assert "error" in response 