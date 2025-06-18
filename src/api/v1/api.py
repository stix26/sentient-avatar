from fastapi import APIRouter
from src.api.v1.endpoints import auth, users, avatar

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(avatar.router, prefix="/avatar", tags=["avatar"])
