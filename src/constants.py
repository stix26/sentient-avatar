from enum import Enum
from typing import Dict, List

# API Constants
API_V1_STR: str = "/api/v1"
PROJECT_NAME: str = "Sentient Avatar"
VERSION: str = "0.1.0"
DESCRIPTION: str = "A sentient avatar system with emotional intelligence"

# Security Constants
SECRET_KEY: str = "your-secret-key-here"  # Change in production
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
REFRESH_TOKEN_EXPIRE_DAYS: int = 7

# Database Constants
DEFAULT_PAGE_SIZE: int = 20
MAX_PAGE_SIZE: int = 100

# Avatar Constants
class EmotionType(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    ANXIOUS = "anxious"
    CALM = "calm"
    SURPRISED = "surprised"

class CognitiveState(str, Enum):
    FOCUSED = "focused"
    DISTRACTED = "distracted"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    RELAXED = "relaxed"
    STRESSED = "stressed"

class PhysicalState(str, Enum):
    ACTIVE = "active"
    RESTING = "resting"
    TENSE = "tense"
    RELAXED = "relaxed"
    ENERGETIC = "energetic"
    FATIGUED = "fatigued"

# Streaming Constants
STREAM_UPDATE_INTERVAL: float = 0.1  # seconds
MAX_STREAM_DURATION: int = 3600  # 1 hour in seconds

# Cache Constants
CACHE_TTL: int = 300  # 5 minutes
AVATAR_CACHE_PREFIX: str = "avatar:"
USER_CACHE_PREFIX: str = "user:"

# Rate Limiting
RATE_LIMIT_PER_MINUTE: int = 60
RATE_LIMIT_PER_HOUR: int = 1000

# Monitoring
METRICS_NAMESPACE: str = "sentient_avatar"
METRICS_SUBSYSTEM: str = "api"

# Error Messages
ERROR_MESSAGES: Dict[str, str] = {
    "invalid_credentials": "Invalid username or password",
    "user_not_found": "User not found",
    "avatar_not_found": "Avatar not found",
    "permission_denied": "Permission denied",
    "invalid_token": "Invalid or expired token",
    "rate_limit_exceeded": "Rate limit exceeded",
    "validation_error": "Validation error",
    "internal_error": "Internal server error"
}

# Success Messages
SUCCESS_MESSAGES: Dict[str, str] = {
    "user_created": "User created successfully",
    "user_updated": "User updated successfully",
    "user_deleted": "User deleted successfully",
    "avatar_created": "Avatar created successfully",
    "avatar_updated": "Avatar updated successfully",
    "avatar_deleted": "Avatar deleted successfully",
    "login_successful": "Login successful",
    "logout_successful": "Logout successful"
}

# File Upload
ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "gif"]
MAX_FILE_SIZE: int = 5 * 1024 * 1024  # 5MB 