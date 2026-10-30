"""
Authentication API Routes for EchoSafe.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.auth import hash_password, verify_password, create_access_token, get_current_user
from backend.schemas import UserRegister, UserLogin, TokenResponse, UserResponse

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

@router.post("/register", response_model=TokenResponse)
def register(payload: UserRegister, db: Session = Depends(get_db)):
    # Check if username or email already exists
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(status_code=400, detail="Username already registered.")
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered.")

    avatar = f"https://api.dicebear.com/7.x/bottts/svg?seed={payload.username}"
    new_user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        role=payload.role or "follower",
        avatar_url=avatar,
        bio=payload.bio or ""
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token({"sub": new_user.id, "role": new_user.role, "username": new_user.username})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": new_user.id,
        "username": new_user.username,
        "role": new_user.role,
        "avatar_url": new_user.avatar_url
    }

@router.post("/login", response_model=TokenResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password."
        )

    token = create_access_token({"sub": user.id, "role": user.role, "username": user.username})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username,
        "role": user.role,
        "avatar_url": user.avatar_url
    }

@router.post("/token", response_model=TokenResponse)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """OAuth2 compatible token login for Swagger UI."""
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password."
        )

    token = create_access_token({"sub": user.id, "role": user.role, "username": user.username})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username,
        "role": user.role,
        "avatar_url": user.avatar_url
    }

@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Return currently authenticated user profile."""
    return current_user

@router.post("/demo-login/{role}", response_model=TokenResponse)
def demo_switch_account(role: str, db: Session = Depends(get_db)):
    """One-click demo account switcher for presentations."""
    role_map = {
        "admin": "admin",
        "moderator": "moderator_sarah",
        "influencer": "creator_0_competitive",
        "creator_gaming": "creator_0_competitive",
        "creator_comedy": "creator_5_regional",
        "positive": "user_positive",
        "critic": "user_critic",
        "abuser": "user_abuser"
    }

    target_username = role_map.get(role.lower(), "user_positive")
    user = db.query(User).filter(User.username == target_username).first()
    if not user:
        # Fallback to first user matching role
        user = db.query(User).filter(User.role == role.lower()).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"No demo user found for role: {role}")

    token = create_access_token({"sub": user.id, "role": user.role, "username": user.username})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username,
        "role": user.role,
        "avatar_url": user.avatar_url
    }
