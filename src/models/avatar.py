from sqlalchemy import Column, Integer, String, JSON, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from src.database import Base

class Avatar(Base):
    __tablename__ = "avatars"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    personality = Column(JSON, nullable=False, default=dict)
    appearance = Column(JSON, nullable=False, default=dict)
    voice = Column(JSON, nullable=False, default=dict)
    current_emotion = Column(JSON)
    current_cognitive_state = Column(JSON)
    current_physical_state = Column(JSON)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    owner = relationship("User", back_populates="avatars")

    def __repr__(self):
        return f"<Avatar {self.name}>"

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "personality": self.personality,
            "appearance": self.appearance,
            "voice": self.voice,
            "current_emotion": self.current_emotion,
            "current_cognitive_state": self.current_cognitive_state,
            "current_physical_state": self.current_physical_state,
            "owner_id": self.owner_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        } 