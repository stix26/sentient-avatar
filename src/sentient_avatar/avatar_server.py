from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, List, Dict, Any
import asyncio
import json
import logging
from datetime import datetime
import base64
import os
from pathlib import Path

from .services.factory import ServiceFactory
from .services.llm import LLMService
from .services.asr import ASRService
from .services.tts import TTSService
from .services.avatar import AvatarService
from .services.vision import VisionService
from .services.vector_store import VectorStoreService
from .config.config import Config, load_config
from .logging.logger import get_logger
from .monitoring.metrics import MetricsCollector
from .rate_limit.rate_limiter import RateLimiter
from .cache.redis_cache import RedisCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sentient Avatar API",
    description="API for the Sentient Avatar system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
service_factory: Optional[ServiceFactory] = None
metrics: Optional[MetricsCollector] = None
rate_limiter: Optional[RateLimiter] = None
cache: Optional[RedisCache] = None

# WebSocket connections
connections: Dict[str, WebSocket] = {}

def get_service_factory() -> ServiceFactory:
    """Get service factory instance.
    
    Returns:
        Service factory instance
    """
    if service_factory is None:
        raise HTTPException(status_code=500, detail="Service factory not initialized")
    return service_factory

@app.on_event("startup")
async def startup():
    """Initialize services on startup."""
    global service_factory, metrics, rate_limiter, cache
    
    try:
        # Load configuration
        config = load_config()
        
        # Create service factory
        service_factory = ServiceFactory.create()
        
        # Initialize services
        await service_factory.initialize()
        
        # Get shared components
        metrics = service_factory.metrics
        rate_limiter = service_factory.rate_limiter
        cache = service_factory.cache
        
        logger.info("Services initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize services")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup services on shutdown."""
    if service_factory:
        await service_factory.cleanup()
        logger.info("Services cleaned up successfully")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time communication.
    
    Args:
        websocket: WebSocket connection
        client_id: Client ID
    """
    await websocket.accept()
    connections[client_id] = websocket
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process message
            response = await process_message(message)
            
            # Send response
            await websocket.send_json(response)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if client_id in connections:
            del connections[client_id]

async def process_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Process WebSocket message.
    
    Args:
        message: Message data
        
    Returns:
        Response data
    """
    message_type = message.get('type')
    
    if message_type == 'text':
        return await process_text_message(message)
    elif message_type == 'audio':
        return await process_audio_message(message)
    elif message_type == 'image':
        return await process_image_message(message)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown message type: {message_type}")

async def process_text_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Process text message.
    
    Args:
        message: Message data
        
    Returns:
        Response data
    """
    text = message.get('text')
    if not text:
        raise HTTPException(status_code=400, detail="Missing text")
        
    # Get services
    llm = get_service_factory().get_llm()
    tts = get_service_factory().get_tts()
    avatar = get_service_factory().get_avatar()
    
    # Generate response
    response = await llm.generate(text)
    
    # Synthesize speech
    audio = await tts.synthesize(response)
    
    # Generate avatar video
    video = await avatar.generate_video(audio)
    
    return {
        'type': 'response',
        'text': response,
        'audio': base64.b64encode(audio).decode(),
        'video': base64.b64encode(video).decode()
    }

async def process_audio_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Process audio message.
    
    Args:
        message: Message data
        
    Returns:
        Response data
    """
    audio = message.get('audio')
    if not audio:
        raise HTTPException(status_code=400, detail="Missing audio")
        
    # Get services
    asr = get_service_factory().get_asr()
    llm = get_service_factory().get_llm()
    tts = get_service_factory().get_tts()
    avatar = get_service_factory().get_avatar()
    
    # Transcribe audio
    text = await asr.transcribe(audio)
    
    # Generate response
    response = await llm.generate(text)
    
    # Synthesize speech
    response_audio = await tts.synthesize(response)
    
    # Generate avatar video
    video = await avatar.generate_video(response_audio)
    
    return {
        'type': 'response',
        'text': text,
        'response': response,
        'audio': base64.b64encode(response_audio).decode(),
        'video': base64.b64encode(video).decode()
    }

async def process_image_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Process image message.
    
    Args:
        message: Message data
        
    Returns:
        Response data
    """
    image = message.get('image')
    prompt = message.get('prompt')
    if not image:
        raise HTTPException(status_code=400, detail="Missing image")
        
    # Get services
    vision = get_service_factory().get_vision()
    llm = get_service_factory().get_llm()
    tts = get_service_factory().get_tts()
    avatar = get_service_factory().get_avatar()
    
    # Analyze image
    analysis = await vision.analyze_image(image, prompt)
    
    # Generate response
    response = await llm.generate(analysis)
    
    # Synthesize speech
    audio = await tts.synthesize(response)
    
    # Generate avatar video
    video = await avatar.generate_video(audio)
    
    return {
        'type': 'response',
        'analysis': analysis,
        'response': response,
        'audio': base64.b64encode(audio).decode(),
        'video': base64.b64encode(video).decode()
    }

@app.post("/chat")
async def chat(
    text: str = Form(...),
    service_factory: ServiceFactory = Depends(get_service_factory)
):
    """Chat endpoint.
    
    Args:
        text: Input text
        service_factory: Service factory
        
    Returns:
        Response data
    """
    # Check rate limit
    if rate_limiter and await rate_limiter.is_rate_limited("chat"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
    # Get services
    llm = service_factory.get_llm()
    tts = service_factory.get_tts()
    avatar = service_factory.get_avatar()
    
    # Generate response
    response = await llm.generate(text)
    
    # Synthesize speech
    audio = await tts.synthesize(response)
    
    # Generate avatar video
    video = await avatar.generate_video(audio)
    
    return {
        'text': response,
        'audio': base64.b64encode(audio).decode(),
        'video': base64.b64encode(video).decode()
    }

@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    service_factory: ServiceFactory = Depends(get_service_factory)
):
    """Transcribe endpoint.
    
    Args:
        audio: Audio file
        service_factory: Service factory
        
    Returns:
        Transcription
    """
    # Check rate limit
    if rate_limiter and await rate_limiter.is_rate_limited("transcribe"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
    # Get ASR service
    asr = service_factory.get_asr()
    
    # Read audio file
    audio_data = await audio.read()
    
    # Transcribe audio
    text = await asr.transcribe(audio_data)
    
    return {'text': text}

@app.post("/analyze-image")
async def analyze_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    service_factory: ServiceFactory = Depends(get_service_factory)
):
    """Analyze image endpoint.
    
    Args:
        image: Image file
        prompt: Analysis prompt
        service_factory: Service factory
        
    Returns:
        Analysis results
    """
    # Check rate limit
    if rate_limiter and await rate_limiter.is_rate_limited("analyze-image"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
    # Get vision service
    vision = service_factory.get_vision()
    
    # Read image file
    image_data = await image.read()
    
    # Analyze image
    analysis = await vision.analyze_image(image_data, prompt)
    
    return analysis

@app.get("/health")
async def health(service_factory: ServiceFactory = Depends(get_service_factory)):
    """Health check endpoint.
    
    Args:
        service_factory: Service factory
        
    Returns:
        Health status
    """
    health_status = {}
    
    # Check service health
    for service_type in ['llm', 'asr', 'tts', 'avatar', 'vision', 'vector_store']:
        service = getattr(service_factory, f"get_{service_type}")()
        health_status[service_type] = await service.health_check()
        
    return health_status

@app.get("/voices")
async def get_voices(service_factory: ServiceFactory = Depends(get_service_factory)):
    """Get available voices.
    
    Args:
        service_factory: Service factory
        
    Returns:
        List of voices
    """
    tts = service_factory.get_tts()
    return await tts.get_voices()

@app.get("/styles")
async def get_styles(service_factory: ServiceFactory = Depends(get_service_factory)):
    """Get available styles.
    
    Args:
        service_factory: Service factory
        
    Returns:
        List of styles
    """
    vision = service_factory.get_vision()
    return await vision.get_available_styles()

@app.get("/metrics")
async def get_metrics():
    """Get metrics.
    
    Returns:
        Metrics data
    """
    if not metrics:
        raise HTTPException(status_code=500, detail="Metrics not initialized")
    return metrics.get_metrics() 