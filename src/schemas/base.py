from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        from_attributes=True, json_encoders={datetime: lambda v: v.isoformat()}
    )


class TimestampSchema(BaseSchema):
    created_at: datetime
    updated_at: datetime


class IDSchema(BaseSchema):
    id: int


class BaseResponse(BaseSchema):
    success: bool = True
    message: Optional[str] = None


class ErrorResponse(BaseSchema):
    success: bool = False
    error: str
    details: Optional[dict] = None
