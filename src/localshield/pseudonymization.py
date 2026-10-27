"""Optional encrypted reversible pseudonymization storage.

Uses standard authenticated symmetric encryption (Fernet / AES-HMAC) from the
cryptography library to store and retrieve reversible entity mappings locally.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from cryptography.fernet import Fernet, InvalidToken


class EncryptedMappingStore:
    """Manages secure local storage for reversible entity pseudonymization.

    Keys are kept strictly separate from data. Mappings are encrypted at rest.
    """

    def __init__(
        self,
        storage_dir: str = "./localshield_storage",
        encryption_key: Optional[bytes] = None,
        ttl_seconds: int = 86400,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds

        if encryption_key is None:
            # Check environment
            env_key = os.getenv("LOCALSHIELD_REVERSIBLE_KEY")
            if env_key:
                self.key = env_key.encode("utf-8")
            else:
                # Generate new key for session if not provided
                self.key = Fernet.generate_key()
        else:
            self.key = encryption_key

        try:
            self._cipher = Fernet(self.key)
        except Exception as e:
            raise ValueError(f"Invalid encryption key provided for pseudonymization store: {e}")

    def save_session(self, mapping: Dict[str, str], session_id: Optional[str] = None) -> str:
        """Encrypt and persist session pseudonym-to-original mappings."""
        if not session_id:
            session_id = str(uuid.uuid4())

        payload = {
            "session_id": session_id,
            "created_at": time.time(),
            "expires_at": time.time() + self.ttl_seconds,
            "mapping": mapping,
        }

        plaintext_bytes = json.dumps(payload).encode("utf-8")
        encrypted_data = self._cipher.encrypt(plaintext_bytes)

        session_file = self.storage_dir / f"{session_id}.enc"
        session_file.write_bytes(encrypted_data)

        # Restrict permissions on Windows/POSIX where supported
        try:
            os.chmod(session_file, 0o600)
        except Exception:
            pass

        return session_id

    def load_session(self, session_id: str) -> Dict[str, str]:
        """Load and decrypt session mappings if session is valid and unexpired."""
        session_file = self.storage_dir / f"{session_id}.enc"
        if not session_file.exists():
            raise FileNotFoundError(f"Pseudonymization session {session_id} not found")

        encrypted_data = session_file.read_bytes()
        try:
            decrypted_bytes = self._cipher.decrypt(encrypted_data)
        except InvalidToken:
            raise PermissionError("Decryption failed: invalid encryption key or corrupted data")

        payload = json.loads(decrypted_bytes.decode("utf-8"))
        if time.time() > payload.get("expires_at", 0):
            # Expired session; delete and fail
            self.delete_session(session_id)
            raise ValueError(f"Pseudonymization session {session_id} has expired")

        return payload.get("mapping", {})

    def delete_session(self, session_id: str) -> bool:
        """Securely remove encrypted session mapping file."""
        session_file = self.storage_dir / f"{session_id}.enc"
        if session_file.exists():
            session_file.unlink()
            return True
        return False

    def reidentify(self, redacted_text: str, session_id: str, authorized: bool = False) -> str:
        """Reconstruct original text from placeholders. Requires explicit authorization."""
        if not authorized:
            raise PermissionError(
                "Explicit authorization required to re-identify pseudonymized text."
            )

        mapping = self.load_session(session_id)
        reidentified_text = redacted_text
        # Replace each placeholder with original sensitive value
        for placeholder, original in mapping.items():
            reidentified_text = reidentified_text.replace(placeholder, original)

        return reidentified_text
