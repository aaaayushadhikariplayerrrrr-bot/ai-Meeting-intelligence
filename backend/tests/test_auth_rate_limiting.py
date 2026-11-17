"""Comprehensive test suite for Phase 6C.3 Step 2: Authentication Rate Limiting.

Covers:
1. First five login attempts succeed; sixth returns HTTP 429.
2. Sixth login request is rejected before password verification (verify_password).
3. First three registration attempts succeed; fourth returns HTTP 429.
4. Login and registration quotas are strictly independent.
5. Advancing the injected clock beyond 60 seconds restores the quota.
6. Retry-After header exists, is a valid positive integer, and reflects the remaining window.
7. Different client IPs receive independent quotas.
8. Spoofed X-Forwarded-For headers cannot bypass rate limiting for the same socket IP.
9. Concurrent requests cannot exceed the configured limit (thread-safety under concurrency).
10. Memory cleanup and maximum-key capacity safeguard fail-securely without bypass.
11. Test isolation fixture resets limiter state between tests.
"""

import concurrent.futures
from fastapi import Request
from fastapi.testclient import TestClient
import pytest

from app.main import app as fastapi_app
from app.core.rate_limit import (
    InMemoryRateLimiter,
    login_rate_limiter,
    register_rate_limiter,
)
import app.core.security as security


# --- 1. Login Rate Limiting (5 per 60s) ---

def test_login_rate_limit_enforced_at_5_attempts(client):
    """Verify first 5 login attempts are evaluated normally and the 6th is rejected with HTTP 429."""
    for attempt in range(1, 6):
        resp = client.post(
            "/api/v1/auth/login",
            json={"email": "nobody@example.com", "password": "WrongPassword123!"},
        )
        assert resp.status_code == 401, f"Attempt {attempt} should return 401, got {resp.status_code}"

    # 6th attempt must be rejected with 429
    resp_sixth = client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "WrongPassword123!"},
    )
    assert resp_sixth.status_code == 429
    assert resp_sixth.json()["detail"] == "Too many requests. Please try again later."
    assert "Retry-After" in resp_sixth.headers
    assert int(resp_sixth.headers["Retry-After"]) > 0


def test_sixth_login_request_rejected_before_password_verification(client, monkeypatch):
    """Verify the 6th login request is rejected by the limiter BEFORE verify_password executes."""
    verify_call_count = 0
    orig_verify = security.verify_password

    def spy_verify(*args, **kwargs):
        nonlocal verify_call_count
        verify_call_count += 1
        return orig_verify(*args, **kwargs)

    monkeypatch.setattr("app.api.endpoints.auth.verify_password", spy_verify)

    for _ in range(5):
        resp = client.post(
            "/api/v1/auth/login",
            json={"email": "nobody@example.com", "password": "WrongPassword123!"},
        )
        assert resp.status_code == 401

    assert verify_call_count == 5

    # 6th attempt must return 429 without invoking verify_password
    resp_sixth = client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "WrongPassword123!"},
    )
    assert resp_sixth.status_code == 429
    assert verify_call_count == 5, "verify_password was called on a rate-limited request!"


# --- 2. Registration Rate Limiting (3 per 60s) ---

def test_register_rate_limit_enforced_at_3_attempts(client):
    """Verify first 3 registration attempts succeed and the 4th is rejected with HTTP 429."""
    for attempt in range(1, 4):
        resp = client.post(
            "/api/v1/auth/register",
            json={
                "email": f"user{attempt}@example.com",
                "full_name": f"User {attempt}",
                "password": "Password123!",
                "organization_name": f"Org {attempt}",
            },
        )
        assert resp.status_code == 201, f"Registration {attempt} should return 201, got {resp.status_code}"

    # 4th attempt must be rejected with 429
    resp_fourth = client.post(
        "/api/v1/auth/register",
        json={
            "email": "user4@example.com",
            "full_name": "User 4",
            "password": "Password123!",
            "organization_name": "Org 4",
        },
    )
    assert resp_fourth.status_code == 429
    assert resp_fourth.json()["detail"] == "Too many requests. Please try again later."
    assert "Retry-After" in resp_fourth.headers


# --- 3. Independent Limits for Login and Registration ---

def test_login_and_register_counters_are_independent(client):
    """Verify exhausting login rate limit does not block registration and vice versa."""
    # Exhaust login quota (5 requests)
    for _ in range(5):
        resp = client.post(
            "/api/v1/auth/login",
            json={"email": "nobody@example.com", "password": "WrongPassword123!"},
        )
        assert resp.status_code == 401

    # 6th login is blocked
    r_login_blocked = client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "WrongPassword123!"},
    )
    assert r_login_blocked.status_code == 429

    # Registration from the same client IP must succeed normally
    r_reg = client.post(
        "/api/v1/auth/register",
        json={
            "email": "independent@example.com",
            "full_name": "Independent User",
            "password": "Password123!",
            "organization_name": "Independent Org",
        },
    )
    assert r_reg.status_code == 201, f"Registration should succeed despite login 429, got {r_reg.status_code}"


# --- 4. Clock Injection and Quota Restoration ---

def test_advancing_clock_beyond_window_restores_quota(client):
    """Verify advancing the injected monotonic clock beyond 60s resets the quota deterministically."""
    simulated_time = 1000.0
    login_rate_limiter.clock = lambda: simulated_time

    # Send 5 requests at t = 1000.0
    for _ in range(5):
        assert client.post(
            "/api/v1/auth/login",
            json={"email": "nobody@example.com", "password": "WrongPassword123!"},
        ).status_code == 401

    # 6th request at t = 1000.0 is rate-limited
    assert client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "WrongPassword123!"},
    ).status_code == 429

    # Advance clock by 61 seconds (beyond window)
    simulated_time += 61.0

    # Next request should be admitted and return 401, not 429
    resp_after_window = client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "WrongPassword123!"},
    )
    assert resp_after_window.status_code == 401


# --- 5. Retry-After Header Accuracy ---

def test_retry_after_header_validity(client):
    """Verify Retry-After is a valid positive integer reflecting the remaining window time."""
    simulated_time = 1000.0
    login_rate_limiter.clock = lambda: simulated_time

    # Send 5 requests at t = 1000.0
    for _ in range(5):
        client.post(
            "/api/v1/auth/login",
            json={"email": "nobody@example.com", "password": "WrongPassword123!"},
        )

    # Advance clock by 25 seconds -> remaining window is 35 seconds
    simulated_time += 25.0

    resp = client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "WrongPassword123!"},
    )
    assert resp.status_code == 429
    retry_after_str = resp.headers.get("Retry-After")
    assert retry_after_str is not None
    assert retry_after_str.isdigit()

    retry_after_val = int(retry_after_str)
    assert retry_after_val == 35, f"Expected Retry-After to be 35, got {retry_after_val}"


# --- 6. Multi-IP Isolation ---

def test_different_client_ips_receive_independent_quotas():
    """Verify clients connecting from different socket IPs maintain separate quotas."""
    client_ip1 = TestClient(fastapi_app, client=("192.168.10.1", 50000))
    client_ip2 = TestClient(fastapi_app, client=("192.168.10.2", 50000))

    # Exhaust quota for IP1
    for _ in range(5):
        assert client_ip1.post(
            "/api/v1/auth/login",
            json={"email": "nobody@example.com", "password": "WrongPassword123!"},
        ).status_code == 401

    assert client_ip1.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "WrongPassword123!"},
    ).status_code == 429

    # IP2 must still have full quota
    for _ in range(5):
        assert client_ip2.post(
            "/api/v1/auth/login",
            json={"email": "nobody@example.com", "password": "WrongPassword123!"},
        ).status_code == 401

    # IP2 6th request is now rate-limited
    assert client_ip2.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "WrongPassword123!"},
    ).status_code == 429


# --- 7. Header Spoofing Resistance ---

def test_spoofed_forwarded_headers_do_not_bypass_rate_limit():
    """Verify arbitrary X-Forwarded-For or X-Real-IP headers cannot bypass the limiter."""
    test_client = TestClient(fastapi_app, client=("10.0.0.1", 50000))

    # Send 5 requests with spoofed headers pretending to be from different IPs
    for i in range(1, 6):
        headers = {
            "X-Forwarded-For": f"203.0.113.{i}",
            "X-Real-IP": f"203.0.113.{i}",
        }
        resp = test_client.post(
            "/api/v1/auth/login",
            json={"email": "nobody@example.com", "password": "WrongPassword123!"},
            headers=headers,
        )
        assert resp.status_code == 401

    # 6th request with another spoofed header is still blocked
    resp_sixth = test_client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "WrongPassword123!"},
        headers={"X-Forwarded-For": "203.0.113.99"},
    )
    assert resp_sixth.status_code == 429, "Spoofed X-Forwarded-For allowed rate-limit bypass!"


# --- 8. Thread-Safe Concurrency ---

def test_concurrent_requests_cannot_exceed_limit():
    """Verify concurrent requests from the same IP cannot exceed the limit due to race conditions."""
    concurrent_client = TestClient(fastapi_app, client=("172.16.0.1", 50000))

    def make_login_call():
        return concurrent_client.post(
            "/api/v1/auth/login",
            json={"email": "nobody@example.com", "password": "WrongPassword123!"},
        ).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_login_call) for _ in range(10)]
        results = [f.result() for f in futures]

    count_401 = results.count(401)
    count_429 = results.count(429)

    assert count_401 == 5, f"Expected exactly 5 requests admitted (401), got {count_401}"
    assert count_429 == 5, f"Expected exactly 5 requests rejected (429), got {count_429}"
    assert len(results) == 10


# --- 9. Memory Safety & Maximum-Key Safeguard ---

def test_memory_cleanup_and_max_keys_safeguard():
    """Verify expired entries are purged and max_keys safeguard rejects conservatively without bypass."""
    sim_time = 100.0
    limiter = InMemoryRateLimiter(
        max_requests=2,
        window_seconds=60,
        namespace="test",
        max_keys=2,
        clock_fn=lambda: sim_time,
    )

    class MockRequest:
        def __init__(self, host: str):
            self.client = type("Addr", (), {"host": host})()

    req_ip1 = MockRequest("1.1.1.1")
    req_ip2 = MockRequest("2.2.2.2")
    req_ip3 = MockRequest("3.3.3.3")

    # Admit IP1 and IP2 -> limiter reaches max_keys (2)
    limiter(req_ip1)
    limiter(req_ip2)
    assert len(limiter._records) == 2

    # IP3 arrives at t = 110.0 (IP1 and IP2 still active)
    sim_time = 110.0
    with pytest.raises(Exception) as exc_info:
        limiter(req_ip3)
    assert exc_info.value.status_code == 429
    assert exc_info.value.headers["Retry-After"] == "60"

    # Advance time to t = 165.0 so IP1 and IP2 are expired
    sim_time = 165.0

    # IP3 arrives now -> expired entries are swept and IP3 is safely accommodated
    limiter(req_ip3)
    assert "test:3.3.3.3" in limiter._records
    assert len(limiter._records) == 1  # IP1 and IP2 were purged


# --- 10. Test Isolation Fixture Verification ---

def test_existing_auth_test_isolation_fixture(client):
    """Verify limiter reset fixture isolates tests so state does not leak."""
    # Ensure limiter starts with clean records
    assert len(login_rate_limiter._records) == 0
    assert len(register_rate_limiter._records) == 0

    # Perform 2 login requests
    client.post("/api/v1/auth/login", json={"email": "nobody@example.com", "password": "WrongPassword123!"})
    client.post("/api/v1/auth/login", json={"email": "nobody@example.com", "password": "WrongPassword123!"})

    assert len(login_rate_limiter._records) == 1
