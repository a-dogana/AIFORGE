import os
import logging
from typing import Optional, Dict, Any

import keyring
from keyring.errors import KeyringError, NoKeyringError

logger = logging.getLogger("aiforge.secrets")

SERVICE_NAME = "aiforge-Secrets"
LEGACY_ACCOUNT_NAME = "aiforge_user"


class SecretBackendUnavailableError(RuntimeError):
    """Raised when the OS credential store backend is unavailable, locked, or fails."""


def _account_for(provider: str) -> str:
    return f"aiforge_{provider}_key"


def _legacy_service_for(provider: str) -> str:
    return f"aiforge_api_{provider}"


def get_secret_backend_status() -> Dict[str, Any]:
    """Return diagnostic status of the active OS credential backend."""
    try:
        kr = keyring.get_keyring()
        backend_cls = f"{type(kr).__module__}.{type(kr).__name__}"
        priority = getattr(kr, "priority", 0)
        is_fail = "fail" in type(kr).__module__.lower() or priority <= 0
        if is_fail:
            return {
                "available": False,
                "backend": backend_cls,
                "error": "No supported OS credential store backend available.",
            }
        return {"available": True, "backend": backend_cls, "error": None}
    except Exception as e:
        return {"available": False, "backend": "unknown", "error": str(e)}


def _ensure_backend_available() -> None:
    status = get_secret_backend_status()
    if not status["available"]:
        raise SecretBackendUnavailableError(status["error"] or "Credential backend unavailable.")


def save_secret(provider: str, secret: str) -> None:
    """
    Saves a secret to the OS credential store via ``keyring``.
    Never falls back to plaintext files.
    """
    if not secret:
        return
    _ensure_backend_available()
    try:
        keyring.set_password(SERVICE_NAME, _account_for(provider), secret)
    except (KeyringError, NoKeyringError, Exception) as e:
        logger.error(f"Failed to save secret for '{provider}' in OS credential store: {e}")
        raise SecretBackendUnavailableError(
            f"Credential store failed for provider '{provider}': {e}"
        ) from e


def get_secret(provider: str) -> Optional[str]:
    """
    Retrieves a secret from the OS credential store via ``keyring``.
    - Returns ``None`` if the credential does not exist.
    - Raises ``SecretBackendUnavailableError`` if the credential backend is unavailable or fails.
    """
    _ensure_backend_available()
    try:
        val = keyring.get_password(SERVICE_NAME, _account_for(provider))
        if val is not None:
            return val.strip() or None
        # Backward-compatible read for pre-2.1.1 macOS Keychain entries
        legacy_val = keyring.get_password(_legacy_service_for(provider), LEGACY_ACCOUNT_NAME)
        if legacy_val is not None:
            return legacy_val.strip() or None
        return None
    except (KeyringError, NoKeyringError, Exception) as e:
        logger.error(f"Failed to read secret for '{provider}' from OS credential store: {e}")
        raise SecretBackendUnavailableError(
            f"Credential store unavailable while reading '{provider}': {e}"
        ) from e


def has_secret(provider: str) -> bool:
    """Returns True if a secret is stored for ``provider``, False if absent."""
    return get_secret(provider) is not None


def check_plaintext_migration() -> bool:
    """
    Checks if legacy plaintext api_keys.json exists and returns a warning flag.
    Does not delete it without consent.
    """
    path = "registry/api_keys.json"
    return os.path.exists(path)
