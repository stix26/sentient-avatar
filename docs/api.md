# Sentient Avatar API Documentation

## Overview

The Sentient Avatar API provides endpoints for interacting with the avatar system, including text chat, audio transcription, image analysis, and real-time streaming. All endpoints are prefixed with `/api/v1`.

## Authentication

Currently, the API does not require authentication. However, it is recommended to implement authentication in production environments.

## Endpoints

### Health Check

```http
GET /health
```

Check the health status of all services.

**Response**
```json
{
    "status": "healthy"
}
```

### Chat

```http
POST /chat
```

Process text input and return a response with synthesized speech and avatar video.

**Request Body**
```json
{
    "text": "Hello, how are you?",
    "voice": "en_US-hfc_female-medium",
    "model": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "temperature": 0.7,
    "max_tokens": 1000,
    "animation_style": "natural",
    "motion_scale": 1.0,
    "stillness": 0.5
}
```

**Response**
```json
{
    "text": "I'm doing well, thank you for asking!",
    "audio": "base64_encoded_audio_data",
    "video": "base64_encoded_video_data"
}
```

### Transcribe

```http
POST /transcribe
```

Transcribe audio to text.

**Request**
- Content-Type: `multipart/form-data`
- Body: Audio file (WAV format)

**Response**
```json
{
    "text": "Transcribed text from audio",
    "confidence": 0.95
}
```

### Analyze Image

```http
POST /analyze-image
```

Analyze an image based on a prompt.

**Request**
- Content-Type: `multipart/form-data`
- Body: 
  - `file`: Image file (JPEG/PNG)
  - `prompt`: Analysis prompt
  - `temperature`: (optional) 0.0 to 1.0
  - `max_tokens`: (optional) Maximum tokens
  - `detail_level`: (optional) "low", "medium", "high"

**Response**
```json
{
    "analysis": "Detailed analysis of the image",
    "objects": ["detected", "objects", "in", "image"],
    "confidence": 0.95
}
```

### Get Available Voices

```http
GET /voices
```

Get a list of available voices for text-to-speech.

**Response**
```json
[
    {
        "id": "en_US-hfc_female-medium",
        "name": "English (US) Female",
        "language": "en-US",
        "gender": "female",
        "style": "medium"
    }
]
```

### Get Available Styles

```http
GET /styles
```

Get a list of available styles for image analysis.

**Response**
```json
[
    {
        "id": "natural",
        "name": "Natural",
        "description": "Natural, conversational style"
    }
]
```

## WebSocket API

### Connect

```http
GET /ws/{client_id}
```

Establish a WebSocket connection for real-time communication.

**Parameters**
- `client_id`: Unique identifier for the client

### Message Types

#### Text Message
```json
{
    "type": "text",
    "text": "Hello, how are you?",
    "voice": "en_US-hfc_female-medium",
    "model": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "temperature": 0.7,
    "max_tokens": 1000,
    "animation_style": "natural",
    "motion_scale": 1.0,
    "stillness": 0.5
}
```

#### Audio Message
Binary WebSocket message containing audio data (WAV format).

### Response Format

```json
{
    "type": "response",
    "text": "Response text",
    "audio": "base64_encoded_audio_data",
    "video": "base64_encoded_video_data"
}
```

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
    "error": "Invalid request parameters"
}
```

### 422 Unprocessable Entity
```json
{
    "error": "Invalid input data",
    "details": {
        "field": ["error message"]
    }
}
```

### 500 Internal Server Error
```json
{
    "error": "Internal server error",
    "details": "Error message"
}
```

## Rate Limiting

Currently, there are no rate limits implemented. However, it is recommended to implement rate limiting in production environments.

## Best Practices

1. Use appropriate content types for file uploads
2. Handle WebSocket connection errors and reconnection
3. Implement proper error handling for all API calls
4. Use appropriate timeouts for long-running operations
5. Implement proper logging and monitoring
6. Use appropriate security measures in production

## Examples

### Python Example

```python
import requests
import json

# Chat example
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "text": "Hello, how are you?",
        "voice": "en_US-hfc_female-medium"
    }
)
print(response.json())

# Transcribe example
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f}
    )
print(response.json())
```

### JavaScript Example

```javascript
// WebSocket example
const ws = new WebSocket("ws://localhost:8000/ws/client123");

ws.onopen = () => {
    // Send text message
    ws.send(JSON.stringify({
        type: "text",
        text: "Hello, how are you?",
        voice: "en_US-hfc_female-medium"
    }));
};

ws.onmessage = (event) => {
    const response = JSON.parse(event.data);
    console.log(response);
};

// Send audio data
const audioBlob = new Blob([audioData], { type: "audio/wav" });
ws.send(audioBlob);
``` 