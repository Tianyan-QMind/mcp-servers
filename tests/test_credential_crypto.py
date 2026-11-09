from __future__ import annotations

import base64
import re
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import SecretStr

from tianyan_mcp.config import Settings
from tianyan_mcp.credential_crypto import (
    ApiKeyCodec,
    CredentialCipher,
    CredentialDecryptionError,
    EncryptedCredential,
)


def test_generated_api_key_round_trips_without_storing_plaintext() -> None:
    codec = ApiKeyCodec(b"p" * 32)

    generated = codec.generate()

    assert re.fullmatch(r"tymcp_[0-9a-f]{16}_[A-Za-z0-9_-]{43}", generated.plaintext)
    assert generated.plaintext.startswith(f"tymcp_{generated.key_id}_")
    assert codec.parse_key_id(generated.plaintext) == generated.key_id
    assert codec.verify(generated.plaintext, generated.verification_tag)
    assert not codec.verify(generated.plaintext + "x", generated.verification_tag)
    assert generated.plaintext not in repr(generated)


def test_key_parser_rejects_malformed_and_oversized_keys() -> None:
    codec = ApiKeyCodec(b"p" * 32)

    for candidate in ("", "tymcp_short_secret", "tymcp_" + "a" * 16 + "_x" * 1000):
        assert codec.parse_key_id(candidate) is None
        assert not codec.verify(candidate, b"x" * 32)


def test_api_key_codec_rejects_short_pepper() -> None:
    with pytest.raises(ValueError, match="at least 32 bytes"):
        ApiKeyCodec(b"p" * 31)


def test_cipher_binds_ciphertext_to_tenant() -> None:
    cipher = CredentialCipher(active_version=1, keys={1: b"k" * 32})
    secret = SecretStr("fake-open-id")

    encrypted = cipher.encrypt("tenant-a", secret)

    assert cipher.decrypt("tenant-a", encrypted).get_secret_value() == "fake-open-id"
    with pytest.raises(CredentialDecryptionError) as error:
        cipher.decrypt("tenant-b", encrypted)
    assert "fake-open-id" not in str(error.value)
    assert "fake-open-id" not in repr(encrypted)


def test_cipher_uses_distinct_12_byte_nonces() -> None:
    cipher = CredentialCipher(active_version=1, keys={1: b"k" * 32})

    first = cipher.encrypt("tenant-a", SecretStr("fake-open-id"))
    second = cipher.encrypt("tenant-a", SecretStr("fake-open-id"))

    assert len(first.nonce) == 12
    assert len(second.nonce) == 12
    assert first.nonce != second.nonce


def test_cipher_rejects_unknown_key_version_without_secret_details() -> None:
    cipher = CredentialCipher(active_version=1, keys={1: b"k" * 32})
    encrypted = EncryptedCredential(ciphertext=b"not-used", nonce=b"n" * 12, key_version=2)

    with pytest.raises(CredentialDecryptionError) as error:
        cipher.decrypt("tenant-a", encrypted)

    assert "not-used" not in str(error.value)


def test_cipher_converts_non_utf8_plaintext_to_secret_free_error() -> None:
    key = b"k" * 32
    encrypted = EncryptedCredential(
        ciphertext=AESGCM(key).encrypt(b"n" * 12, b"\xff", b"tenant-a"),
        nonce=b"n" * 12,
        key_version=1,
    )

    with pytest.raises(CredentialDecryptionError) as error:
        CredentialCipher(active_version=1, keys={1: key}).decrypt("tenant-a", encrypted)

    assert "ff" not in str(error.value)


def test_settings_rejects_malformed_base64_credential_key(tmp_path: Path) -> None:
    env = {
        "TIANYAN_IDENTITY_MODE": "multi_tenant",
        "TIANYAN_DATABASE_PATH": str(tmp_path / "tenant.db"),
        "TIANYAN_API_KEY_PEPPER": "p" * 32,
        "TIANYAN_CREDENTIAL_ENCRYPTION_KEY_VERSION": "1",
        "TIANYAN_CREDENTIAL_ENCRYPTION_KEY_V1": "not-base64!",
    }

    with pytest.raises(ValueError, match="Base64-encoded 32-byte"):
        Settings.from_env(env)


def test_settings_decodes_exactly_32_base64_credential_key(tmp_path: Path) -> None:
    expected_key = b"k" * 32
    env = {
        "TIANYAN_IDENTITY_MODE": "multi_tenant",
        "TIANYAN_DATABASE_PATH": str(tmp_path / "tenant.db"),
        "TIANYAN_API_KEY_PEPPER": "p" * 32,
        "TIANYAN_CREDENTIAL_ENCRYPTION_KEY_VERSION": "1",
        "TIANYAN_CREDENTIAL_ENCRYPTION_KEY_V1": base64.b64encode(expected_key).decode(),
    }

    settings = Settings.from_env(env)

    assert settings.credential_keys[1].get_secret_value() == expected_key
