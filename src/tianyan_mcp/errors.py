from __future__ import annotations

import json
import re
from typing import cast


class ServiceError(Exception):
    def __init__(self, code: str, message: str, retryable: bool) -> None:
        self.code = code
        self.message = message
        self.retryable = retryable
        super().__init__(f"{code}: {message}")

    def as_json(self) -> str:
        return json.dumps(
            {
                "error": {
                    "code": self.code,
                    "message": self.message,
                    "retryable": self.retryable,
                }
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


_UPSTREAM_MESSAGE_KEYS = ("message", "msg")
_UPSTREAM_NESTED_KEYS = ("CN", "EN", "message", "msg")

# Upstream business envelopes occasionally contain server-side diagnostics
# (paths, URLs, Java stack frames, key material file names). Such fragments
# are replaced before the text reaches MCP callers; ordinary short business
# messages (Chinese or English) contain none of these shapes.
_REDACTION = "[redacted]"
_REDACT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"https?://[^\s，。；;：]+",
        r"(?<![A-Za-z0-9])[A-Za-z]:\\[^\s,，。；;：:）)]+",
        r"~[/\\][^\s,，。；;：:）)]+",
        r"(?<![\w.~-])/(?:[A-Za-z0-9._~-]+/)*[A-Za-z0-9._~-]+",
        r"\bat\s+[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+\s*\([^)]*\)",
        r"(?<![A-Za-z0-9_$])[A-Za-z_$][\w$]*(?:Exception|Throwable)(?![\w$])",
        r"(?<![A-Za-z0-9_$])[A-Za-z_$][\w$]*Error(?![\w$])",
        r"(?<![\w.-])\.env(?:\.[A-Za-z0-9_-]+)?(?![\w.-])",
        r"(?<![\w.-])id_rsa(?:\.pub)?(?![\w.-])",
        r"(?<![\w.~-])[\w.~-]+\.(?:pem|key|p12|pfx|jks|keystore|crt|cer|der)(?![\w])",
    )
)


def upstream_business_message(payload: object, *, limit: int = 300) -> str:
    """Extract bounded human-readable text from a Tianyan business error envelope.

    Only structured ``message``/``msg`` fields of a parsed JSON object are
    inspected; callers must never pass raw HTTP response bodies. Live upstreams
    emit errors either in ``message`` or in ``msg``; ``msg`` is frequently a
    JSON-encoded ``{"CN": ..., "EN": ...}`` string. Control characters are
    removed and the result is truncated to ``limit`` characters so it can be
    appended safely to a fixed MCP-layer message.
    """
    if not isinstance(payload, dict):
        return ""
    envelope = cast(dict[str, object], payload)
    parts: list[str] = []
    for key in _UPSTREAM_MESSAGE_KEYS:
        raw = envelope.get(key)
        if type(raw) is not str or not raw.strip():
            continue
        text = raw.strip()
        try:
            decoded: object = json.loads(text)
        except ValueError:
            parts.append(text)
            continue
        if isinstance(decoded, dict):
            decoded_dict = cast(dict[str, object], decoded)
            for sub_key in _UPSTREAM_NESTED_KEYS:
                value = decoded_dict.get(sub_key)
                if type(value) is str and value.strip():
                    parts.append(value.strip())
        elif type(decoded) is str and decoded.strip():
            parts.append(decoded.strip())
        else:
            parts.append(text)

    cleaned_parts: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = _sanitize_text(_redact_text(part))
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            cleaned_parts.append(cleaned)
    result = " ".join(cleaned_parts)
    if len(result) > limit > 0:
        result = result[: limit - 1].rstrip() + "…"
    return result


def rejection_message(fixed_message: str, payload: object) -> str:
    """Append structured upstream detail to a fixed MCP-layer rejection text."""
    detail = upstream_business_message(payload)
    return f"{fixed_message}：{detail}" if detail else fixed_message


def _redact_text(value: str) -> str:
    redacted = value
    for pattern in _REDACT_PATTERNS:
        redacted = pattern.sub(_REDACTION, redacted)
    return redacted


def _sanitize_text(value: str) -> str:
    sanitized = "".join(char if ord(char) >= 32 else " " for char in value)
    return " ".join(sanitized.split())
