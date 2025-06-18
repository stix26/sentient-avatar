from __future__ import annotations
from typing import Any, Dict, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    data: bytes
    sample_rate: int
    timestamp: datetime
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None


class AudioPipeline:
    """Simple audio pipeline for testing."""

    def __init__(self, asr_service, tts_service) -> None:
        self.asr_service = asr_service
        self.tts_service = tts_service
        self.audio_buffer = b""

    async def process_audio_chunk(
        self, chunk: AudioChunk, language: Optional[str] = None
    ) -> Dict[str, Any]:
        transcription = await self.asr_service.transcribe(chunk.data)
        self.audio_buffer += chunk.data
        return {
            "transcription": (
                transcription["text"]
                if isinstance(transcription, dict)
                else transcription
            ),
            "status": "complete",
        }

    async def synthesize_speech(self, text: str) -> AudioChunk:
        result = await self.tts_service.synthesize(text)
        audio = result.get("audio") if isinstance(result, dict) else result
        return AudioChunk(
            data=audio, sample_rate=16000, timestamp=datetime.utcnow(), is_final=True
        )
