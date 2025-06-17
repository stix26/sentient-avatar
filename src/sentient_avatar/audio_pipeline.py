from typing import AsyncGenerator, Dict, Any, Optional
import asyncio
import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class AudioChunk:
    """Represents a chunk of audio data"""
    data: bytes
    sample_rate: int
    timestamp: datetime
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None

class AudioPipeline:
    """Handles audio processing and streaming"""
    
    def __init__(
        self,
        asr_service,
        tts_service,
        chunk_size: int = 1024,
        sample_rate: int = 16000
    ):
        self.asr_service = asr_service
        self.tts_service = tts_service
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self._audio_buffer = bytearray()
        self._is_processing = False
    
    async def process_audio_chunk(
        self,
        chunk: AudioChunk,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single audio chunk
        
        Args:
            chunk: Audio chunk to process
            language: Optional language code
            
        Returns:
            Dict containing processing results
        """
        try:
            # Add chunk to buffer
            self._audio_buffer.extend(chunk.data)
            
            # Process if buffer is full or chunk is final
            if len(self._audio_buffer) >= self.chunk_size or chunk.is_final:
                # Get buffer contents
                buffer_data = bytes(self._audio_buffer)
                
                # Clear buffer
                self._audio_buffer.clear()
                
                # Transcribe audio
                transcription = await self.asr_service.transcribe(
                    buffer_data,
                    language=language
                )
                
                return {
                    "transcription": transcription,
                    "timestamp": datetime.utcnow(),
                    "is_final": chunk.is_final
                }
            
            return {
                "status": "buffering",
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            raise
    
    async def stream_audio_processing(
        self,
        audio_stream: AsyncGenerator[AudioChunk, None],
        language: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream audio processing results
        
        Args:
            audio_stream: Async generator of audio chunks
            language: Optional language code
            
        Yields:
            Dict containing processing results
        """
        try:
            self._is_processing = True
            
            async for chunk in audio_stream:
                if not self._is_processing:
                    break
                    
                result = await self.process_audio_chunk(chunk, language)
                yield result
                
        except Exception as e:
            logger.error(f"Error in audio stream processing: {e}")
            raise
        finally:
            self._is_processing = False
    
    async def synthesize_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        model: str = "bark",
        temperature: float = 0.7,
        speed: float = 1.0
    ) -> AudioChunk:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            voice: Optional voice to use
            model: TTS model to use
            temperature: Generation temperature
            speed: Speech speed
            
        Returns:
            AudioChunk containing synthesized speech
        """
        try:
            # Synthesize speech
            audio_data, metadata = await self.tts_service.synthesize(
                text=text,
                voice=voice,
                model=model,
                temperature=temperature,
                speed=speed
            )
            
            return AudioChunk(
                data=audio_data,
                sample_rate=self.sample_rate,
                timestamp=datetime.utcnow(),
                is_final=True,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            raise
    
    async def clone_voice(
        self,
        reference_audio: bytes,
        text: str,
        language: Optional[str] = None
    ) -> AudioChunk:
        """
        Clone voice from reference audio
        
        Args:
            reference_audio: Reference audio data
            text: Text to synthesize
            language: Optional language code
            
        Returns:
            AudioChunk containing cloned voice
        """
        try:
            # Clone voice
            audio_data, metadata = await self.tts_service.clone_voice(
                reference_audio=reference_audio,
                text=text,
                language=language
            )
            
            return AudioChunk(
                data=audio_data,
                sample_rate=self.sample_rate,
                timestamp=datetime.utcnow(),
                is_final=True,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error cloning voice: {e}")
            raise
    
    def stop_processing(self):
        """Stop audio processing"""
        self._is_processing = False
        self._audio_buffer.clear() 