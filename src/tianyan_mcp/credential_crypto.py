from __future__ import annotations

import hmac
import os
import re
import secrets
from dataclasses import dataclass, field
from typing import Final

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import SecretStr

_API_KEY_PATTERN: Final = re.compile(r"^tymcp_([0-9a-f]{16})_([A-Za-z0-9_-]{43})$")
_NONCE_BYTES: Final = 12
_KEY_BYTES: Final = 32


@dataclass(frozen=True, slots=True)
class GeneratedApiKey:
    key_id: str
    plaintext: str = field(repr=False)
    verification_tag: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class EncryptedCredential:
    ciphertext: bytes = field(repr=False)
    nonce: bytes = field(repr=False)
    key_version: int


class CredentialDecryptionError(Exception):
    """Raised when a stored credential cannot be safely decrypted."""


class ApiKeyCodec:
    def __init__(self, pepper: bytes) -> None:
        if len(pepper) < _KEY_BYTES:
            raise ValueError("API key pepper must contain at least 32 bytes")
        self._pepper = pepper

    def generate(self) -> GeneratedApiKey:
        key_id = secrets.token_hex(8)
        secret = secrets.token_urlsafe(32)
        plaintext = f"tymcp_{key_id}_{secret}"
        return GeneratedApiKey(
            key_id=key_id,
            plaintext=plaintext,
            verification_tag=self.verification_tag(plaintext),
        )

    @staticmethod
    def parse_key_id(plaintext: str) -> str | None:
        match = _API_KEY_PATTERN.fullmatch(plaintext)
        return match.group(1) if match is not None else None

    def verification_tag(self, plaintext: str) -> bytes:
        return hmac.digest(self._pepper, plaintext.encode("utf-8"), "sha256")

    def verify(self, plaintext: str, expected_tag: bytes) -> bool:
        if self.parse_key_id(plaintext) is None or len(expected_tag) != 32:
            return False
        return hmac.compare_digest(self.verification_tag(plaintext), expected_tag)


class CredentialCipher:
    def __init__(self, *, active_version: int, keys: dict[int, bytes]) -> None:
        if active_version not in keys:
            raise ValueError("active credential key is unavailable")
        if any(len(key) != _KEY_BYTES for key in keys.values()):
            raise ValueError("credential encryption keys must be 32 bytes")
        self._active_version = active_version
        self._keys = dict(keys)

    def encrypt(self, tenant_id: str, value: SecretStr) -> EncryptedCredential:
        nonce = os.urandom(_NONCE_BYTES)
        ciphertext = AESGCM(self._keys[self._active_version]).encrypt(
            nonce,
            value.get_secret_value().encode("utf-8"),
            self._aad(tenant_id),
        )
        return EncryptedCredential(
            ciphertext=ciphertext,
            nonce=nonce,
            key_version=self._active_version,
        )

    def decrypt(self, tenant_id: str, encrypted: EncryptedCredential) -> SecretStr:
        key = self._keys.get(encrypted.key_version)
        if key is None:
            raise CredentialDecryptionError("stored credential cannot be decrypted")
        try:
            plaintext = AESGCM(key).decrypt(
                encrypted.nonce,
                encrypted.ciphertext,
                self._aad(tenant_id),
            )
            return SecretStr(plaintext.decode("utf-8"))
        except (InvalidTag, UnicodeDecodeError, ValueError) as error:
            raise CredentialDecryptionError("stored credential cannot be decrypted") from error

    @staticmethod
    def _aad(tenant_id: str) -> bytes:
        return tenant_id.encode("utf-8")
