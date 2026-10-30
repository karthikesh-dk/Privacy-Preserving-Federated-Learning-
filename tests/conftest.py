"""
Pytest configuration and global fixtures.
Resets demo user restriction states before and after every test
to guarantee clean, deterministic test isolation.
"""

import pytest
from backend.database import SessionLocal
from backend.models import User

@pytest.fixture(autouse=True)
def reset_demo_users_state():
    db = SessionLocal()
    try:
        for u in db.query(User).all():
            u.is_restricted = False
            u.restriction_type = None
            u.restriction_until = None
        db.commit()
    finally:
        db.close()

    yield

    db = SessionLocal()
    try:
        for u in db.query(User).all():
            u.is_restricted = False
            u.restriction_type = None
            u.restriction_until = None
        db.commit()
    finally:
        db.close()
