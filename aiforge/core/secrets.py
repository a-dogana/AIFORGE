import subprocess
import os
import json
import logging

logger = logging.getLogger("aiforge.secrets")

def save_secret(provider: str, secret: str):
    """
    Saves a secret to the macOS Keychain securely.
    """
    if not secret:
        return
    # -U means update if exists
    try:
        subprocess.run(
            ["security", "add-generic-password", "-s", f"aiforge_api_{provider}", "-a", "aiforge_user", "-w", secret, "-U"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to save secret for {provider} in Keychain.")
        raise RuntimeError(f"Keychain storage failed for {provider}")

def get_secret(provider: str) -> str:
    """
    Retrieves a secret from the macOS Keychain. Returns None if not found.
    """
    try:
        res = subprocess.run(
            ["security", "find-generic-password", "-s", f"aiforge_api_{provider}", "-a", "aiforge_user", "-w"],
            capture_output=True, text=True, check=True
        )
        return res.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def has_secret(provider: str) -> bool:
    return get_secret(provider) is not None

def check_plaintext_migration():
    """
    Checks if legacy plaintext api_keys.json exists and returns a warning.
    Does not delete it without consent.
    """
    path = "registry/api_keys.json"
    if os.path.exists(path):
        return True
    return False
