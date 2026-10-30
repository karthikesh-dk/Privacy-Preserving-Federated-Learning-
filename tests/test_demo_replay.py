"""
Automated Test that executes the full 9-step demo scenario against TestClient.
"""

import pytest
from starlette.testclient import TestClient
from backend.main import app
@pytest.fixture
def client():
    return TestClient(app)

def test_full_scenario_replay(client):
    # 1. Positive Praise
    res = client.post("/api/auth/demo-login/positive").json()
    headers_pos = {"Authorization": f"Bearer {res['access_token']}"}
    r = client.post("/api/comments", json={
        "post_id": "post_0001",
        "text": "Your breakdown and detailed explanation is fantastic! Learned a ton."
    }, headers=headers_pos).json()
    assert r["status"] == "published"

    # 2. Constructive Criticism
    res = client.post("/api/auth/demo-login/critic").json()
    headers_crit = {"Authorization": f"Bearer {res['access_token']}"}
    r = client.post("/api/comments", json={
        "post_id": "post_0001",
        "text": "I disagree with your conclusion on the benchmarks. The methodology seems biased."
    }, headers=headers_crit).json()
    assert r["status"] == "published"

    # 3. Warning / Hostile Flow
    r_warn = client.post("/api/comments", json={
        "post_id": "post_0001",
        "text": "This whole design choice is completely brainless and trash."
    }, headers=headers_pos).json()
    assert r_warn["status"] in ["published", "warning", "held_for_review", "rejected"]

    # 4. Severe Hostility
    res_abuser = client.post("/api/auth/demo-login/abuser").json()
    headers_abuser = {"Authorization": f"Bearer {res_abuser['access_token']}"}
    abuser_id = res_abuser["user_id"]
    r_abuse = client.post("/api/comments", json={
        "post_id": "post_0001",
        "text": "You are an incompetent clown, stop making garbage content and leave."
    }, headers=headers_abuser).json()
    assert r_abuse["status"] in ["held_for_review", "rejected"]
    assert r_abuse["abuse_severity"] in ["severe", "medium"]

    # 5. Burst Attack / Restriction Interception
    r_burst = client.post("/api/comments", json={
        "post_id": "post_0001",
        "text": "Garbage trash clown attack 2!"
    }, headers=headers_abuser)
    assert r_burst.status_code in [200, 403]
    if r_burst.status_code == 403:
        assert "restricted" in r_burst.json()["detail"].lower()
    else:
        assert r_burst.json()["status"] in ["held_for_review", "rate_limited", "slow_mode", "rejected"]

    # 6. User Appeal
    r_appeal = client.post(f"/api/comments/{r_abuse['comment_id']}/appeal", json={
        "comment_id": r_abuse["comment_id"],
        "reason": "I was expressing critical frustration, not a targeted attack."
    }, headers=headers_abuser).json()
    assert r_appeal["status"] == "pending"

    # 7. Moderator Console & Profile Inspection
    res_mod = client.post("/api/auth/demo-login/moderator").json()
    headers_mod = {"Authorization": f"Bearer {res_mod['access_token']}"}
    queue = client.get("/api/moderation/queue", headers=headers_mod).json()
    assert len(queue) > 0
    prof = client.get(f"/api/moderation/users/{abuser_id}/profile", headers=headers_mod).json()
    assert prof["wellbeing_signal"] in ["low", "medium", "high"]
    assert "Not a clinical diagnosis" in prof["wellbeing_disclaimer"] or "NOT A CLINICAL DIAGNOSIS" in prof["wellbeing_disclaimer"]

    # 8. Federated Learning Round
    res_admin = client.post("/api/auth/demo-login/admin").json()
    headers_admin = {"Authorization": f"Bearer {res_admin['access_token']}"}
    fl_round = client.post("/api/federated/run-round", json={
        "strategy": "fedavg",
        "enable_secagg": True,
        "enable_dp": False,
        "proximal_mu": 0.01
    }, headers=headers_admin).json()
    assert fl_round["round_number"] >= 1
    assert fl_round["accuracy"] > 0.50
    assert fl_round["comm_overhead_kb"] > 0

    # 9. Cryptographic Proof
    proof = client.get("/api/federated/proof", headers=headers_admin).json()
    assert proof["data_locality_guarantee"]["status"] == "VERIFIED_PASS"
    assert proof["data_locality_guarantee"]["raw_text_leakage_bytes"] == 0
