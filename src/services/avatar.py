from typing import Optional, Dict, Any, AsyncGenerator
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from src.models.avatar import Avatar
from src.models.user import User
from src.schemas.avatar import (
    AvatarCreate,
    AvatarUpdate,
    EmotionUpdate,
    CognitiveUpdate,
    PhysicalUpdate
)
from src.services.emotion import EmotionService
from src.services.cognitive import CognitiveService
from src.services.physical import PhysicalService
from src.services.streaming import StreamingService

class AvatarService:
    def __init__(self):
        self.emotion_service = EmotionService()
        self.cognitive_service = CognitiveService()
        self.physical_service = PhysicalService()
        self.streaming_service = StreamingService()

    def create_avatar(
        self,
        avatar_in: AvatarCreate,
        current_user: User,
        db: Session
    ) -> Avatar:
        """
        Create a new avatar for the current user.
        """
        avatar = Avatar(
            **avatar_in.dict(),
            owner_id=current_user.id
        )
        db.add(avatar)
        db.commit()
        db.refresh(avatar)
        return avatar

    def get_avatar(
        self,
        avatar_id: int,
        current_user: User,
        db: Session
    ) -> Optional[Avatar]:
        """
        Get avatar by ID if it belongs to the current user.
        """
        avatar = db.query(Avatar).filter(
            Avatar.id == avatar_id,
            Avatar.owner_id == current_user.id
        ).first()
        return avatar

    def update_avatar(
        self,
        avatar_id: int,
        avatar_in: AvatarUpdate,
        current_user: User,
        db: Session
    ) -> Optional[Avatar]:
        """
        Update avatar if it belongs to the current user.
        """
        avatar = self.get_avatar(avatar_id, current_user, db)
        if not avatar:
            return None

        update_data = avatar_in.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(avatar, field, value)

        db.commit()
        db.refresh(avatar)
        return avatar

    def update_emotion(
        self,
        avatar_id: int,
        emotion_in: EmotionUpdate,
        current_user: User,
        db: Session
    ) -> Optional[Avatar]:
        """
        Update avatar's emotion state.
        """
        avatar = self.get_avatar(avatar_id, current_user, db)
        if not avatar:
            return None

        emotion_state = self.emotion_service.process_emotion(
            emotion_in.emotion,
            emotion_in.intensity,
            emotion_in.context
        )
        avatar.current_emotion = emotion_state

        db.commit()
        db.refresh(avatar)
        return avatar

    def update_cognitive(
        self,
        avatar_id: int,
        cognitive_in: CognitiveUpdate,
        current_user: User,
        db: Session
    ) -> Optional[Avatar]:
        """
        Update avatar's cognitive state.
        """
        avatar = self.get_avatar(avatar_id, current_user, db)
        if not avatar:
            return None

        cognitive_state = self.cognitive_service.process_cognitive(
            cognitive_in.operation,
            cognitive_in.input_data,
            cognitive_in.parameters
        )
        avatar.current_cognitive_state = cognitive_state

        db.commit()
        db.refresh(avatar)
        return avatar

    def update_physical(
        self,
        avatar_id: int,
        physical_in: PhysicalUpdate,
        current_user: User,
        db: Session
    ) -> Optional[Avatar]:
        """
        Update avatar's physical state.
        """
        avatar = self.get_avatar(avatar_id, current_user, db)
        if not avatar:
            return None

        physical_state = self.physical_service.process_physical(
            physical_in.action,
            physical_in.parameters,
            physical_in.duration
        )
        avatar.current_physical_state = physical_state

        db.commit()
        db.refresh(avatar)
        return avatar

    async def stream_avatar(
        self,
        avatar_id: int,
        current_user: User,
        db: Session
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream avatar updates in real-time.
        """
        avatar = self.get_avatar(avatar_id, current_user, db)
        if not avatar:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Avatar not found"
            )

        async for update in self.streaming_service.stream_avatar_updates(avatar):
            yield update 