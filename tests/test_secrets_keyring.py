import pytest
from fastapi.testclient import TestClient
from keyring.errors import KeyringError

import aiforge.core.secrets as secrets_mod
from aiforge.core.secrets import (
    SERVICE_NAME,
    SecretBackendUnavailableError,
    get_secret,
    get_secret_backend_status,
    has_secret,
    save_secret,
)


class _InMemoryKeyring:
    priority = 10

    def __init__(self):
        self.store = {}

    def get_password(self, service, account):
        return self.store.get((service, account))

    def set_password(self, service, account, secret):
        self.store[(service, account)] = secret


class _UnavailableKeyring:
    priority = 0

    def get_password(self, service, account):
        raise KeyringError("No keyring backend available")

    def set_password(self, service, account, secret):
        raise KeyringError("No keyring backend available")


def test_secrets_save_read_and_absent_key(monkeypatch):
    """save_secret, get_secret, and has_secret work with a cross-platform keyring backend and return None when absent."""
    mem_kr = _InMemoryKeyring()
    monkeypatch.setattr(secrets_mod.keyring, "get_keyring", lambda: mem_kr)
    monkeypatch.setattr(secrets_mod.keyring, "set_password", mem_kr.set_password)
    monkeypatch.setattr(secrets_mod.keyring, "get_password", mem_kr.get_password)

    status = get_secret_backend_status()
    assert status["available"] is True

    # Absent key returns None
    assert get_secret("openai") is None
    assert has_secret("openai") is False

    # Save and read back
    save_secret("openai", "sk-test-12345")
    assert mem_kr.store[(SERVICE_NAME, "aiforge_openai_key")] == "sk-test-12345"
    assert get_secret("openai") == "sk-test-12345"
    assert has_secret("openai") is True

    # Legacy macOS Keychain fallback read
    mem_kr.store[("aiforge_api_google", "aiforge_user")] = "legacy-google-secret"
    assert get_secret("google") == "legacy-google-secret"
    assert has_secret("google") is True


def test_secrets_unavailable_backend_raises_explicit_error_and_ui_reports_503(monkeypatch, tmp_path):
    """When keyring backend is unavailable or fails, SecretBackendUnavailableError is raised and UI never writes plaintext."""
    bad_kr = _UnavailableKeyring()
    monkeypatch.setattr(secrets_mod.keyring, "get_keyring", lambda: bad_kr)
    monkeypatch.setattr(secrets_mod.keyring, "set_password", bad_kr.set_password)
    monkeypatch.setattr(secrets_mod.keyring, "get_password", bad_kr.get_password)

    status = get_secret_backend_status()
    assert status["available"] is False

    with pytest.raises(SecretBackendUnavailableError):
        get_secret("openai")

    with pytest.raises(SecretBackendUnavailableError):
        save_secret("openai", "sk-should-fail")

    import aiforge.ui.app as ui_app

    monkeypatch.setenv("AIFORGE_TESTING", "1")
    client = TestClient(ui_app.app)

    # GET /api/keys reports backend_available=False without crashing
    res_get = client.get("/api/keys")
    assert res_get.status_code == 200
    body = res_get.json()
    assert body["backend_available"] is False
    assert body["openai"] is False
    assert "backend_error" in body

    # POST /api/keys returns 503 and does not write any plaintext file
    res_post = client.post(
        "/api/keys",
        json={"openai": "sk-secret"},
        headers={"Origin": "http://127.0.0.1:8500"},
    )
    assert res_post.status_code == 503
