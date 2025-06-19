import base64
import uuid
from datetime import datetime, timedelta
from io import BytesIO
from typing import List, Optional

import jwt
import pyotp
import qrcode
import redis
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class User(BaseModel):
    id: str
    email: EmailStr
    username: str
    hashed_password: str
    is_active: bool = True
    is_superuser: bool = False
    roles: List[str] = []
    permissions: List[str] = []
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    last_login: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    user_id: str
    roles: List[str]
    permissions: List[str]
    exp: datetime


class AuthService:
    def __init__(self, redis_url: str, jwt_secret: str, jwt_algorithm: str = "HS256"):
        self.redis = redis.from_url(redis_url)
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        return self.pwd_context.hash(password)

    def create_access_token(
        self, data: dict, expires_delta: Optional[timedelta] = None
    ) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.jwt_secret, algorithm=self.jwt_algorithm)

    def create_refresh_token(self, data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=7)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.jwt_secret, algorithm=self.jwt_algorithm)

    def verify_token(self, token: str) -> TokenData:
        try:
            payload = jwt.decode(
                token, self.jwt_secret, algorithms=[self.jwt_algorithm]
            )
            return TokenData(**payload)
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    async def get_current_user(self, token: str = Depends(oauth2_scheme)) -> User:
        token_data = self.verify_token(token)
        user_data = self.redis.get(f"user:{token_data.user_id}")
        if not user_data:
            raise HTTPException(status_code=401, detail="User not found")
        return User.parse_raw(user_data)

    def check_permission(self, user: User, required_permission: str) -> bool:
        return required_permission in user.permissions

    def check_role(self, user: User, required_role: str) -> bool:
        return required_role in user.roles

    def generate_mfa_secret(self) -> str:
        return pyotp.random_base32()

    def generate_mfa_qr(self, secret: str, email: str) -> str:
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(email, issuer_name="Sentient Avatar")

        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def verify_mfa(self, secret: str, token: str) -> bool:
        totp = pyotp.TOTP(secret)
        return totp.verify(token)

    async def create_user(
        self, email: str, username: str, password: str, roles: List[str] = None
    ) -> User:
        user_id = str(uuid.uuid4())
        hashed_password = self.get_password_hash(password)

        user = User(
            id=user_id,
            email=email,
            username=username,
            hashed_password=hashed_password,
            roles=roles or ["user"],
            permissions=self._get_permissions_for_roles(roles or ["user"]),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        self.redis.set(f"user:{user_id}", user.json())
        return user

    def _get_permissions_for_roles(self, roles: List[str]) -> List[str]:
        role_permissions = {
            "admin": ["*"],
            "moderator": ["read:*", "write:chat", "write:avatar"],
            "user": ["read:chat", "read:avatar", "write:chat"],
        }

        permissions = set()
        for role in roles:
            if role in role_permissions:
                permissions.update(role_permissions[role])
        return list(permissions)

    async def update_user(self, user_id: str, **kwargs) -> User:
        user_data = self.redis.get(f"user:{user_id}")
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")

        user = User.parse_raw(user_data)
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)

        user.updated_at = datetime.utcnow()
        self.redis.set(f"user:{user_id}", user.json())
        return user

    async def delete_user(self, user_id: str) -> None:
        if not self.redis.delete(f"user:{user_id}"):
            raise HTTPException(status_code=404, detail="User not found")

    async def login(
        self, email: str, password: str, mfa_token: Optional[str] = None
    ) -> Token:
        user_data = self.redis.get(f"user:email:{email}")
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        user = User.parse_raw(user_data)
        if not self.verify_password(password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        if user.mfa_enabled:
            if not mfa_token:
                raise HTTPException(status_code=400, detail="MFA token required")
            if not self.verify_mfa(user.mfa_secret, mfa_token):
                raise HTTPException(status_code=401, detail="Invalid MFA token")

        access_token = self.create_access_token(
            {"user_id": user.id, "roles": user.roles, "permissions": user.permissions}
        )

        refresh_token = self.create_refresh_token({"user_id": user.id})

        user.last_login = datetime.utcnow()
        self.redis.set(f"user:{user.id}", user.json())

        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=900,  # 15 minutes
        )

    async def refresh_token(self, refresh_token: str) -> Token:
        try:
            payload = jwt.decode(
                refresh_token, self.jwt_secret, algorithms=[self.jwt_algorithm]
            )
            user_id = payload.get("user_id")

            user_data = self.redis.get(f"user:{user_id}")
            if not user_data:
                raise HTTPException(status_code=401, detail="User not found")

            user = User.parse_raw(user_data)

            access_token = self.create_access_token(
                {
                    "user_id": user.id,
                    "roles": user.roles,
                    "permissions": user.permissions,
                }
            )

            new_refresh_token = self.create_refresh_token({"user_id": user.id})

            return Token(
                access_token=access_token,
                refresh_token=new_refresh_token,
                expires_in=900,
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Refresh token has expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid refresh token")

    def require_permission(self, permission: str):
        async def permission_dependency(user: User = Depends(self.get_current_user)):
            if not self.check_permission(user, permission):
                raise HTTPException(status_code=403, detail="Permission denied")
            return user

        return permission_dependency

    def require_role(self, role: str):
        async def role_dependency(user: User = Depends(self.get_current_user)):
            if not self.check_role(user, role):
                raise HTTPException(status_code=403, detail="Role required")
            return user

        return role_dependency
