"""Day 27: Security hardening, secret rotation, and threat modeling.

Learning goals
--------------
- Score STRIDE-style threats with explicit mitigations.
- Rotate HMAC secrets with a bounded overlap window.
- Add safe browser headers and redacted audit events.
- Make security decisions deterministic and testable.

Teaching notes
--------------
Threat models turn assets, actors, and failure scenarios into controls and tests.
Likelihood times impact is a review signal, not a probability. Rotation signs
with one current key, temporarily verifies with previous keys, and then retires
them. Never print key material. Security headers are defense in depth; they do
not replace authorization, validation, or output encoding. Audit records should
explain what happened without copying credentials or untrusted payloads.

Run this file with Python 3.10+; it uses only the standard library and no network.

Practice exercises
------------------
1. Implement rank_threats(threats), returning an immutable risk-descending tuple.
2. Implement rotate_and_verify(ring, message, new_id, new_secret), keeping the
   old signature valid while new signatures use the new ID.
3. Implement redact_audit_fields(values) without mutating values.

Expert challenge: build a multi-service rotation controller. Stage and activate
versioned keys, measure old-key use, retire only after consumer acknowledgements,
and test partial rollout, rollback, clock skew, replay, and refresh failure.

Solution guidance
-----------------
Use separate signing, verification, storage, and orchestration interfaces.
Record only key IDs, service IDs, event IDs, and outcome categories. Model
proposed -> staged -> active -> retired as idempotent transitions and keep the
accepted-key window bounded.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
from typing import Literal

ThreatCategory = Literal[
    "spoofing", "tampering", "repudiation", "information_disclosure",
    "denial_of_service", "elevation_of_privilege",
]
_CATEGORIES = frozenset({
    "spoofing", "tampering", "repudiation", "information_disclosure",
    "denial_of_service", "elevation_of_privilege",
})
_SENSITIVE = ("token", "secret", "password", "cookie", "authorization", "key")


def _text(value: str, field: str, limit: int = 200) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    value = value.strip()
    if not value or len(value) > limit:
        raise ValueError(f"{field} must be non-empty and at most {limit} characters")
    return value


def _rating(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise ValueError(f"{field} must be an integer from 1 through 5")
    return value


@dataclass(frozen=True, slots=True)
class Threat:
    """A small, reviewable threat-register entry."""
    threat_id: str
    category: ThreatCategory
    asset: str
    scenario: str
    likelihood: int
    impact: int
    mitigation: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "threat_id", _text(self.threat_id, "threat_id", 64))
        category = _text(self.category, "category")
        if category not in _CATEGORIES:
            raise ValueError("category must be a STRIDE category")
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "asset", _text(self.asset, "asset"))
        object.__setattr__(self, "scenario", _text(self.scenario, "scenario", 500))
        object.__setattr__(self, "mitigation", _text(self.mitigation, "mitigation", 500))
        _rating(self.likelihood, "likelihood")
        _rating(self.impact, "impact")

    @property
    def risk_score(self) -> int:
        return self.likelihood * self.impact


def rank_threats(threats: Iterable[Threat]) -> tuple[Threat, ...]:
    """Practice solution: highest risk first, then deterministic ID."""
    values = tuple(threats)
    if any(not isinstance(item, Threat) for item in values):
        raise TypeError("threats must contain Threat objects")
    return tuple(sorted(values, key=lambda item: (-item.risk_score, item.threat_id)))


@dataclass(frozen=True, slots=True, repr=False)
class SecretVersion:
    """Key material is deliberately hidden from repr output."""
    key_id: str
    material: bytes

    def __post_init__(self) -> None:
        key_id = _text(self.key_id, "key_id", 64)
        if any(char.isspace() or char == ":" for char in key_id):
            raise ValueError("key_id cannot contain whitespace or ':'")
        if not isinstance(self.material, bytes) or len(self.material) < 16:
            raise ValueError("material must contain at least 16 bytes")
        object.__setattr__(self, "key_id", key_id)

    def __repr__(self) -> str:
        return f"SecretVersion(key_id={self.key_id!r}, material='[REDACTED]')"


class SecretRing:
    """Sign with current and verify against a bounded previous-key window."""

    def __init__(
        self, current: SecretVersion, *,
        previous: Sequence[SecretVersion] = (), max_previous: int = 2,
    ) -> None:
        if not isinstance(current, SecretVersion):
            raise TypeError("current must be a SecretVersion")
        if isinstance(previous, (str, bytes)):
            raise TypeError("previous must be a sequence")
        if isinstance(max_previous, bool) or not isinstance(max_previous, int) or max_previous < 0:
            raise ValueError("max_previous must be a non-negative integer")
        versions = (current, *tuple(previous))
        if any(not isinstance(item, SecretVersion) for item in versions):
            raise TypeError("all versions must be SecretVersion objects")
        if len({item.key_id for item in versions}) != len(versions):
            raise ValueError("key IDs must be unique")
        self._current = current
        self._previous = tuple(versions[1:])[:max_previous]
        self._max_previous = max_previous

    @property
    def accepted_key_ids(self) -> tuple[str, ...]:
        return (self._current.key_id, *(item.key_id for item in self._previous))

    def rotate(self, new_version: SecretVersion) -> None:
        if not isinstance(new_version, SecretVersion):
            raise TypeError("new_version must be a SecretVersion")
        if new_version.key_id in self.accepted_key_ids:
            raise ValueError("key ID is already accepted")
        self._previous = (self._current, *self._previous)[:self._max_previous]
        self._current = new_version

    def sign(self, message: bytes) -> str:
        if not isinstance(message, bytes):
            raise TypeError("message must be bytes")
        digest = hmac.new(self._current.material, message, hashlib.sha256).hexdigest()
        return f"{self._current.key_id}:{digest}"

    def verify(self, message: bytes, signature: str) -> str | None:
        if not isinstance(message, bytes) or not isinstance(signature, str):
            raise TypeError("message must be bytes and signature must be text")
        try:
            key_id, supplied = signature.split(":", 1)
        except ValueError:
            return None
        for version in (self._current, *self._previous):
            if hmac.compare_digest(version.key_id, key_id):
                expected = hmac.new(version.material, message, hashlib.sha256).hexdigest()
                return version.key_id if hmac.compare_digest(expected, supplied) else None
        return None


def rotate_and_verify(
    ring: SecretRing, message: bytes, new_id: str, new_secret: bytes
) -> tuple[str, str]:
    """Practice solution: overlap old verification and new signing."""
    if not isinstance(ring, SecretRing):
        raise TypeError("ring must be a SecretRing")
    old_id = ring.verify(message, ring.sign(message))
    ring.rotate(SecretVersion(new_id, new_secret))
    new_id = ring.verify(message, ring.sign(message))
    if old_id is None or new_id is None:
        raise AssertionError("rotation did not preserve verification")
    return old_id, new_id


def build_security_headers(*, nonce: str | None = None) -> dict[str, str]:
    """Build conservative headers for an HTML response."""
    source = "'none'"
    if nonce is not None:
        nonce = _text(nonce, "nonce", 128)
        if any(char.isspace() for char in nonce):
            raise ValueError("nonce cannot contain whitespace")
        source = f"'nonce-{nonce}'"
    return {
        "Content-Security-Policy": (
            "default-src 'self'; object-src 'none'; base-uri 'none'; "
            f"script-src 'self' {source}; frame-ancestors 'none'"
        ),
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "Cache-Control": "no-store",
    }


def redact_audit_fields(values: Mapping[str, object]) -> dict[str, str]:
    """Practice solution: copy values and redact sensitive field names."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Mapping):
        raise TypeError("values must be a mapping")
    result: dict[str, str] = {}
    for raw_name, raw_value in values.items():
        name = _text(raw_name, "field name", 100)
        rendered = (
            type(raw_value).__name__
            if not isinstance(raw_value, (str, int, float, bool)) and raw_value is not None
            else ("None" if raw_value is None else str(raw_value))
        )
        rendered = rendered if len(rendered) <= 120 else rendered[:117] + "..."
        result[name] = (
            "[REDACTED]"
            if any(word in name.casefold() for word in _SENSITIVE)
            else rendered
        )
    return result


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """Bounded, safe-to-log event envelope."""
    event_id: str
    action: str
    actor: str
    fields: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_mapping(
        cls, event_id: str, action: str, actor: str, values: Mapping[str, object]
    ) -> "AuditEvent":
        return cls(
            _text(event_id, "event_id", 80),
            _text(action, "action", 120),
            _text(actor, "actor", 120),
            tuple(sorted(redact_audit_fields(values).items())),
        )


def run_self_checks() -> None:
    threats = (
        Threat("T-2", "tampering", "invoice", "A caller changes a total.", 4, 5, "Authenticate and validate."),
        Threat("T-1", "information_disclosure", "token", "A debug log copies a credential.", 3, 5, "Redact and restrict logs."),
    )
    assert [item.threat_id for item in rank_threats(threats)] == ["T-2", "T-1"]

    first = SecretVersion("k-old", b"first-secret-material")
    ring = SecretRing(first, max_previous=1)
    old_signature = ring.sign(b"rotate")
    assert ring.verify(b"rotate", old_signature) == "k-old"
    assert rotate_and_verify(ring, b"rotate", "k-new", b"second-secret-material") == ("k-old", "k-new")
    assert ring.accepted_key_ids == ("k-new", "k-old")
    assert ring.verify(b"rotate", "k-new:tampered") is None
    assert "first-secret-material" not in repr(first)

    headers = build_security_headers(nonce="abc123")
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert "'nonce-abc123'" in headers["Content-Security-Policy"]
    assert build_security_headers()["Cache-Control"] == "no-store"

    original = {"Actor": "service-a", "Authorization": "Bearer secret", "count": 4}
    assert redact_audit_fields(original) == {
        "Actor": "service-a", "Authorization": "[REDACTED]", "count": "4"
    }
    assert original["Authorization"] == "Bearer secret"
    assert dict(AuditEvent.from_mapping("evt-27", "key.rotated", "controller", original).fields)["Authorization"] == "[REDACTED]"

    for invalid in (
        lambda: Threat("bad", "unknown", "asset", "scenario", 1, 1, "fix"),
        lambda: SecretVersion("bad", b"short"),
        lambda: ring.rotate(first),
        lambda: build_security_headers(nonce="bad nonce"),
    ):
        try:
            invalid()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid input was accepted")


def main() -> None:
    run_self_checks()
    threat = Threat(
        "T-27", "denial_of_service", "webhook",
        "An unbounded body consumes memory.", 3, 4,
        "Bound body size before parsing.",
    )
    print("Highest risk:", rank_threats((threat,))[0].risk_score)
    ring = SecretRing(SecretVersion("k-current", b"demo-secret-material"))
    print("Rotation overlap:", rotate_and_verify(
        ring, b"event-27", "k-next", b"next-secret-material"
    ))
    print("Audit sample:", AuditEvent.from_mapping(
        "evt-27", "key.rotated", "controller",
        {"key_id": "k-next", "Authorization": "Bearer hidden"},
    ))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
