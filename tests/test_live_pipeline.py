"""
Unit & Integration Tests for EchoSafe Live Moderation Pipeline & Auth.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def admin_token(client):
    res = client.post("/api/auth/login", json={"username": "admin", "password": "password123"})
    assert res.status_code == 200
    return res.json()["access_token"]

@pytest.fixture
def mod_token(client):
    res = client.post("/api/auth/login", json={"username": "moderator_sarah", "password": "password123"})
    assert res.status_code == 200
    return res.json()["access_token"]

@pytest.fixture
def user_token(client):
    res = client.post("/api/auth/login", json={"username": "user_positive", "password": "password123"})
    assert res.status_code == 200
    return res.json()["access_token"]

def test_auth_login_and_me(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    res = client.get("/api/auth/me", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["username"] == "admin"
    assert data["role"] == "admin"

def test_demo_role_switching(client):
    for role in ["positive", "critic", "abuser", "moderator", "admin"]:
        res = client.post(f"/api/auth/demo-login/{role}")
        assert res.status_code == 200
        assert "access_token" in res.json()

def test_feed_retrieval(client, user_token):
    headers = {"Authorization": f"Bearer {user_token}"}
    res = client.get("/api/posts?limit=10", headers=headers)
    assert res.status_code == 200
    posts = res.json()
    assert len(posts) > 0
    assert "post_type" in posts[0]

def test_comment_pipeline_positive(client, user_token):
    headers = {"Authorization": f"Bearer {user_token}"}
    res = client.post("/api/comments", json={
        "post_id": "post_0001",
        "text": "Your tutorial video is brilliant, learned so much!"
    }, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["published", "flagged_warning"]
    assert data["verdict"] == "APPROVED"
    assert data["label"] == "positive"

def test_comment_pipeline_criticism_allowed(client):
    res_crit = client.post("/api/auth/demo-login/critic").json()
    headers = {"Authorization": f"Bearer {res_crit['access_token']}"}
    res = client.post("/api/comments", json={
        "post_id": "post_0001",
        "text": "I disagree with your benchmark conclusions, seems biased toward one phone."
    }, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["published", "flagged_warning"]
    assert data["label"] == "negative-but-not-abusive"

def test_comment_pipeline_severe_abuse(client):
    # Log in as abuser account
    res_login = client.post("/api/auth/login", json={"username": "user_abuser", "password": "password123"})
    token = res_login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    res = client.post("/api/comments", json={
        "post_id": "post_0001",
        "text": "This is the worst fucking video, go delete your channel you useless asshole."
    }, headers=headers)
    assert res.status_code in [200, 403]
    if res.status_code == 200:
        data = res.json()
        assert data["status"] in ["held_for_review", "rejected"]
        assert data["label"] == "abusive-vulgar"

def test_moderator_queue_and_action(client, mod_token):
    headers = {"Authorization": f"Bearer {mod_token}"}
    res = client.get("/api/moderation/queue?status_filter=all&limit=5", headers=headers)
    assert res.status_code == 200
    queue = res.json()
    assert len(queue) > 0

    target_comment_id = queue[0]["id"]
    action_res = client.post(f"/api/moderation/comments/{target_comment_id}/action", json={
        "action": "warn",
        "reason": "Test warning issued by moderator."
    }, headers=headers)
    assert action_res.status_code == 200
    assert action_res.json()["success"] is True

def test_federated_learning_status_and_proof(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    res = client.get("/api/federated/status", headers=headers)
    assert res.status_code == 200
    assert res.json()["total_clients"] == 8

    proof_res = client.get("/api/federated/proof", headers=headers)
    assert proof_res.status_code == 200
    assert proof_res.json()["data_locality_guarantee"]["status"] == "VERIFIED_PASS"
