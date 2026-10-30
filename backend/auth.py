"""
Authentication and JWT Utilities for EchoSafe.
"""

import hashlib
import hmac
import os
from datetime import datetime, timedelta
from typing import Optional, List
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from backend.database import get_db
from backend.models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

def hash_password(password: str) -> str:
    """Generate salted SHA-256 password hash."""
    salt = os.urandom(16).hex()
    hash_obj = hashlib.sha256((salt + password).encode("utf-8"))
    return f"{salt}${hash_obj.hexdigest()}"

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against stored salted hash."""
    if not hashed_password or "$" not in hashed_password:
        return False
    salt, original_hash = hashed_password.split("$", 1)
    test_hash = hashlib.sha256((salt + plain_password).encode("utf-8")).hexdigest()
    return hmac.compare_digest(original_hash, test_hash)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> dict:
    """Decode and validate signed JWT access token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again."
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials."
        )

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """FastAPI dependency to retrieve currently authenticated user."""
    payload = decode_access_token(token)
    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload.")
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    
    # Check if account is blocked permanently
    if user.is_restricted and user.restriction_type == "block":
        if user.restriction_until and user.restriction_until > datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Account is currently suspended until {user.restriction_until.strftime('%Y-%m-%d %H:%M:%S')}."
            )

    return user

def require_roles(allowed_roles: List[str]):
    """Role-based authorization guard."""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Requires one of roles: {allowed_roles}"
            )
        return current_user
    return role_checker
