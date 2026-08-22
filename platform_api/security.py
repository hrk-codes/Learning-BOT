from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError
from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.orm import Session

from platform_api.config import get_platform_settings
from platform_api.database import get_db
from platform_api.errors import ApiError
from platform_api.models import User


password_hash = PasswordHash.recommended()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")
DUMMY_HASH = password_hash.hash("not-a-real-account-password")


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = db.scalar(select(User).where(User.email == email.strip().lower()))
    if user is None:
        password_hash.verify(password, DUMMY_HASH)
        return None
    if not password_hash.verify(password, user.password_hash):
        return None
    return user if user.is_active else None


def create_access_token(user: User) -> tuple[str, int]:
    settings = get_platform_settings()
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_minutes)
    token = jwt.encode(
        {"sub": user.id, "role": user.role, "exp": expires, "iat": datetime.now(timezone.utc)},
        settings.jwt_secret,
        algorithm="HS256",
    )
    return token, settings.access_token_minutes * 60


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    try:
        payload = jwt.decode(token, get_platform_settings().jwt_secret, algorithms=["HS256"])
        user_id = payload.get("sub")
        if not isinstance(user_id, str):
            raise InvalidTokenError("Missing subject")
    except InvalidTokenError as exc:
        raise ApiError(401, "invalid_token", "Your session is invalid or expired.") from exc
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise ApiError(401, "invalid_token", "Your session is invalid or expired.")
    return user

