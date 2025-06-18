from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class AvatarBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    personality: Dict[str, Any] = Field(default_factory=dict)
    appearance: Dict[str, Any] = Field(default_factory=dict)
    voice: Dict[str, Any] = Field(default_factory=dict)


class AvatarCreate(AvatarBase):
    pass


class AvatarUpdate(AvatarBase):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    personality: Optional[Dict[str, Any]] = None
    appearance: Optional[Dict[str, Any]] = None
    voice: Optional[Dict[str, Any]] = None


class EmotionUpdate(BaseModel):
    emotion: str = Field(..., min_length=1, max_length=50)
    intensity: float = Field(..., ge=0.0, le=1.0)
    context: Optional[Dict[str, Any]] = None


class CognitiveUpdate(BaseModel):
    operation: str = Field(..., min_length=1, max_length=50)
    input_data: Dict[str, Any] = Field(default_factory=dict)
    parameters: Optional[Dict[str, Any]] = None


class PhysicalUpdate(BaseModel):
    action: str = Field(..., min_length=1, max_length=50)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    duration: Optional[float] = Field(None, ge=0.0)


class AvatarResponse(AvatarBase):
    id: int
    owner_id: int
    created_at: datetime
    updated_at: datetime
    current_emotion: Optional[Dict[str, Any]] = None
    current_cognitive_state: Optional[Dict[str, Any]] = None
    current_physical_state: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True
