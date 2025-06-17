from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import ValidationError
from sqlalchemy.orm import Session

from src import constants
from src.database import SessionLocal
from src.models.user import User
from src.schemas.user import TokenPayload
from src.security import verify_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{constants.API_V1_STR}/auth/login")

def get_db() -> Generator:
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

async def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    try:
        payload = verify_token(token)
        token_data = TokenPayload(**payload)
    except (JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=constants.ERROR_MESSAGES["invalid_token"],
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = db.query(User).filter(User.id == token_data.sub).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=constants.ERROR_MESSAGES["user_not_found"]
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

async def get_current_active_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=constants.ERROR_MESSAGES["permission_denied"]
        )
    return current_user

def get_pagination_params(
    skip: Optional[int] = 0,
    limit: Optional[int] = constants.DEFAULT_PAGE_SIZE
) -> dict:
    if limit > constants.MAX_PAGE_SIZE:
        limit = constants.MAX_PAGE_SIZE
    return {"skip": skip, "limit": limit} 