from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket
from sqlalchemy.orm import Session
from src.database import get_db
from src.models.user import User
from src.schemas.avatar import (
    AvatarCreate,
    AvatarUpdate,
    AvatarResponse,
    EmotionUpdate,
    CognitiveUpdate,
    PhysicalUpdate
)
from src.security import get_current_active_user
from src.services.avatar import AvatarService
from src.monitoring import (
    AVATAR_CREATIONS,
    AVATAR_UPDATES,
    EMOTION_CHANGES,
    COGNITIVE_PROCESSING_TIME,
    PHYSICAL_ACTION_TIME
)

router = APIRouter(prefix="/avatar", tags=["avatar"])
avatar_service = AvatarService()

@router.post("/", response_model=AvatarResponse)
def create_avatar(
    avatar_in: AvatarCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Create a new avatar.
    """
    AVATAR_CREATIONS.inc()
    return avatar_service.create_avatar(avatar_in, current_user, db)

@router.get("/{avatar_id}", response_model=AvatarResponse)
def get_avatar(
    avatar_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get avatar by ID.
    """
    avatar = avatar_service.get_avatar(avatar_id, current_user, db)
    if not avatar:
        raise HTTPException(
            status_code=404,
            detail="Avatar not found"
        )
    return avatar

@router.put("/{avatar_id}", response_model=AvatarResponse)
def update_avatar(
    avatar_id: int,
    avatar_in: AvatarUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Update avatar.
    """
    AVATAR_UPDATES.inc()
    avatar = avatar_service.update_avatar(avatar_id, avatar_in, current_user, db)
    if not avatar:
        raise HTTPException(
            status_code=404,
            detail="Avatar not found"
        )
    return avatar

@router.post("/{avatar_id}/emotion", response_model=AvatarResponse)
def update_emotion(
    avatar_id: int,
    emotion_in: EmotionUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Update avatar's emotion.
    """
    EMOTION_CHANGES.labels(emotion=emotion_in.emotion).inc()
    avatar = avatar_service.update_emotion(avatar_id, emotion_in, current_user, db)
    if not avatar:
        raise HTTPException(
            status_code=404,
            detail="Avatar not found"
        )
    return avatar

@router.post("/{avatar_id}/cognitive", response_model=AvatarResponse)
def update_cognitive(
    avatar_id: int,
    cognitive_in: CognitiveUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Update avatar's cognitive state.
    """
    with COGNITIVE_PROCESSING_TIME.labels(operation=cognitive_in.operation).time():
        avatar = avatar_service.update_cognitive(avatar_id, cognitive_in, current_user, db)
        if not avatar:
            raise HTTPException(
                status_code=404,
                detail="Avatar not found"
            )
        return avatar

@router.post("/{avatar_id}/physical", response_model=AvatarResponse)
def update_physical(
    avatar_id: int,
    physical_in: PhysicalUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Update avatar's physical state.
    """
    with PHYSICAL_ACTION_TIME.labels(action=physical_in.action).time():
        avatar = avatar_service.update_physical(avatar_id, physical_in, current_user, db)
        if not avatar:
            raise HTTPException(
                status_code=404,
                detail="Avatar not found"
            )
        return avatar

@router.websocket("/{avatar_id}/stream")
async def stream_avatar(
    websocket: WebSocket,
    avatar_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Stream avatar updates in real-time.
    """
    await websocket.accept()
    try:
        async for update in avatar_service.stream_avatar(avatar_id, current_user, db):
            await websocket.send_json(update)
    except Exception as e:
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) 