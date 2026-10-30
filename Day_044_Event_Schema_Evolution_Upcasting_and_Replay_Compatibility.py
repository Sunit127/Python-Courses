"""Day 44: Event schema evolution, upcasting, and replay compatibility.

Learning goals
--------------
1. Explain why an append-only event log must preserve old payloads.
2. Separate a stored event envelope from the canonical shape used by code.
3. Build deterministic, non-mutating upcasters for multiple schema versions.
4. Reject future or malformed versions without silently corrupting history.
5. Replay mixed-version events through one compatibility boundary.

Teaching notes
--------------
- Event sourcing makes old data part of the public contract. A class or JSON
  shape that is convenient today may still need to be understood years later.
- Keep the stored envelope stable: event id, kind, schema version, and payload.
  Upcast only at the read boundary; never rewrite the historical log in place.
- An upcaster is a pure function from one version to the next. It should be
  deterministic, side-effect free, and safe to run again during a retry.
- Additive changes are usually easier to roll out than destructive renames.
  During a rename, accept the old field, produce the new canonical field, and
  decide explicitly how missing values are represented.
- Reject a future schema version. Guessing how an unknown payload should work
  can create a valid-looking but incorrect balance, permission, or audit record.
- Verify signatures or hash chains over the raw stored envelope before
  upcasting. The compatibility layer must not weaken tamper detection.
- A replay test should include the oldest supported version, the newest
  version, repeated upcasting, unknown fields, and malformed input.
- In production, record schema ownership, retention policy, deprecation dates,
  and a migration runbook. If an old version is no longer supportable, build a
  separately verified migration rather than deleting history.

Run this file with Python 3.10+.
It uses only the standard library, opens no network connections, and keeps all
events in memory.

Practice exercises
------------------
1. Implement normalize_event_batch(events): preserve first-seen event order,
   ignore an exact duplicate event id, and reject conflicting duplicates.
2. Implement compatibility_report(events): count events by stored schema
   version and list kinds that still need an upcaster.
3. Extend ProfileProjector with a deactivation event and keep replay pure.

Solutions appear in the executable code below.

Expert challenge: long-lived event contract
--------------------------------------------
Design a versioned event registry for a multi-team platform. Store raw JSON
envelopes with a hash chain, signatures, producer metadata, and retention
classification. Add property-based tests for upcaster idempotence and
round-trip invariants, a dead-letter queue for poison events, replay
checkpoints, metrics for per-version volume and upcast latency, and a canary
replay against the previous release. Document when a breaking change requires
a new event kind, how GDPR deletion is represented without rewriting audit
history, and how operators roll back a bad upcaster safely.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json


class SchemaError(ValueError):
    """Base class for malformed or unsupported event schemas."""


class FutureSchemaError(SchemaError):
    """Raised when a producer is newer than this consumer."""


class ConflictingDuplicateError(SchemaError):
    """Raised when one event id is used for two different payloads."""


CURRENT_SCHEMA_VERSION = 3
SUPPORTED_KINDS = {"ProfileRegistered", "ProfileUpdated"}


def _text(value: object, *, field: str, limit: int = 120) -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{field} must be text")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise SchemaError(f"{field} must not be empty")
    if len(cleaned) > limit:
        raise SchemaError(f"{field} must be at most {limit} characters")
    return cleaned


def _version(value: object, *, field: str = "schema_version") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SchemaError(f"{field} must be a positive integer")
    return value


def _json_object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{field} must be an object")
    copied = dict(value)
    try:
        json.dumps(copied, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise SchemaError(f"{field} must be JSON-compatible") from error
    return copied


@dataclass(frozen=True, slots=True)
class StoredEvent:
    """The immutable envelope as it appears in the event log."""

    event_id: str
    kind: str
    schema_version: int
    data: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _text(self.event_id, field="event_id"))
        object.__setattr__(self, "kind", _text(self.kind, field="kind"))
        object.__setattr__(
            self, "schema_version", _version(self.schema_version)
        )
        object.__setattr__(self, "data", _json_object(self.data, field="data"))


@dataclass(frozen=True, slots=True)
class CanonicalEvent:
    """The shape consumed by current application code."""

    event_id: str
    kind: str
    schema_version: int
    data: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise SchemaError("canonical events must use the current version")
        object.__setattr__(self, "event_id", _text(self.event_id, field="event_id"))
        object.__setattr__(self, "kind", _text(self.kind, field="kind"))
        object.__setattr__(self, "data", _json_object(self.data, field="data"))


def _stored_from_mapping(raw: Mapping[str, object]) -> StoredEvent:
    """Parse an external envelope without changing the caller's mapping."""
    if not isinstance(raw, Mapping):
        raise SchemaError("event must be an object")
    return StoredEvent(
        event_id=raw.get("event_id"),
        kind=raw.get("kind"),
        schema_version=raw.get("schema_version"),
        data=raw.get("data"),
    )


def _v1_to_v2(data: Mapping[str, object]) -> dict[str, object]:
    """Rename name -> display_name and introduce an explicit email field."""
    result = dict(data)
    if "display_name" not in result:
        result["display_name"] = result.pop("name", None)
    if result["display_name"] is None:
        raise SchemaError("v1 event is missing name")
    result["display_name"] = _text(result["display_name"], field="display_name")
    result.setdefault("email", None)
    return result


def _v2_to_v3(data: Mapping[str, object]) -> dict[str, object]:
    """Normalize display names and add an explicit consent default."""
    result = dict(data)
    result["display_name"] = _text(result.get("display_name"), field="display_name")
    email = result.get("email")
    if email is not None:
        result["email"] = _text(email, field="email", limit=240)
    result.setdefault("marketing_opt_in", False)
    if not isinstance(result["marketing_opt_in"], bool):
        raise SchemaError("marketing_opt_in must be boolean")
    return result


def upcast_event(
    event: StoredEvent | CanonicalEvent | Mapping[str, object],
) -> CanonicalEvent:
    """Convert any supported historical version into the current contract."""
    if isinstance(event, CanonicalEvent):
        return event
    stored = event if isinstance(event, StoredEvent) else _stored_from_mapping(event)
    if stored.kind not in SUPPORTED_KINDS:
        raise SchemaError(f"unsupported event kind: {stored.kind}")
    if stored.schema_version > CURRENT_SCHEMA_VERSION:
        raise FutureSchemaError(
            f"{stored.kind} uses schema {stored.schema_version}; "
            f"consumer supports {CURRENT_SCHEMA_VERSION}"
        )

    data = dict(stored.data)
    version = stored.schema_version
    while version < CURRENT_SCHEMA_VERSION:
        if version == 1:
            data = _v1_to_v2(data)
        elif version == 2:
            data = _v2_to_v3(data)
        else:
            raise SchemaError(f"no upcaster registered for version {version}")
        version += 1

    return CanonicalEvent(stored.event_id, stored.kind, version, data)


def normalize_event_batch(
    events: Iterable[StoredEvent],
) -> tuple[StoredEvent, ...]:
    """Solution 1: keep first occurrences and reject conflicting duplicates."""
    try:
        items = tuple(events)
    except TypeError as error:
        raise SchemaError("events must be iterable") from error
    result: list[StoredEvent] = []
    seen: dict[str, StoredEvent] = {}
    for event in items:
        if not isinstance(event, StoredEvent):
            raise SchemaError("events must contain StoredEvent values")
        previous = seen.get(event.event_id)
        if previous is None:
            seen[event.event_id] = event
            result.append(event)
        elif previous != event:
            raise ConflictingDuplicateError(
                f"event id {event.event_id!r} has conflicting payloads"
            )
    return tuple(result)


def compatibility_report(
    events: Iterable[StoredEvent],
) -> dict[str, object]:
    """Solution 2: report stored versions and events needing migration."""
    normalized = normalize_event_batch(events)
    versions = Counter(event.schema_version for event in normalized)
    needs_upcast = sorted(
        event.kind for event in normalized
        if event.schema_version < CURRENT_SCHEMA_VERSION
    )
    return {
        "count": len(normalized),
        "versions": dict(sorted(versions.items())),
        "needs_upcast": tuple(needs_upcast),
    }


@dataclass(frozen=True, slots=True)
class Profile:
    profile_id: str
    display_name: str
    email: str | None
    marketing_opt_in: bool
    active: bool


class ProfileProjector:
    """Pure fold over canonical events; no event or state is mutated."""

    def apply(self, profile: Profile | None, event: CanonicalEvent) -> Profile:
        if not isinstance(event, CanonicalEvent):
            raise SchemaError("projector requires canonical events")
        profile_id = event.data.get("profile_id")
        if profile_id is None:
            raise SchemaError("event data requires profile_id")
        clean_id = _text(profile_id, field="profile_id")
        if event.kind == "ProfileRegistered":
            if profile is not None:
                raise SchemaError("profile is already registered")
            return Profile(
                clean_id,
                _text(event.data.get("display_name"), field="display_name"),
                event.data.get("email"),
                event.data["marketing_opt_in"],
                True,
            )
        if profile is None:
            raise SchemaError("profile must exist before updates")
        if event.kind == "ProfileUpdated":
            name = event.data.get("display_name", profile.display_name)
            email = event.data.get("email", profile.email)
            opt_in = event.data.get("marketing_opt_in", profile.marketing_opt_in)
            return Profile(
                clean_id,
                _text(name, field="display_name"),
                None if email is None else _text(email, field="email", limit=240),
                opt_in,
                profile.active,
            )
        raise SchemaError(f"unsupported projection kind: {event.kind}")

    def replay(self, events: Iterable[StoredEvent]) -> Profile:
        """Upcast and apply a stream in order."""
        profile: Profile | None = None
        for stored in normalize_event_batch(events):
            canonical = upcast_event(stored)
            profile = self.apply(profile, canonical)
        if profile is None:
            raise SchemaError("stream did not register a profile")
        return profile


def _expect_error(callback: object, expected: type[BaseException]) -> None:
    if not callable(callback):
        raise TypeError("callback must be callable")
    try:
        callback()
    except expected:
        return
    raise AssertionError(f"expected {expected.__name__}")


def run_self_checks() -> None:
    """Exercise old, current, malformed, and future schema paths."""
    old = StoredEvent(
        "evt-1", "ProfileRegistered", 1,
        {"profile_id": "p-1", "name": " Ada  Lovelace "},
    )
    current = StoredEvent(
        "evt-2", "ProfileUpdated", 3,
        {"profile_id": "p-1", "display_name": "Ada Lovelace",
         "marketing_opt_in": True},
    )
    canonical = upcast_event(old)
    assert canonical.schema_version == CURRENT_SCHEMA_VERSION
    assert canonical.data["display_name"] == "Ada Lovelace"
    assert canonical.data["marketing_opt_in"] is False
    assert upcast_event(canonical).data == canonical.data

    projector = ProfileProjector()
    profile = projector.replay((old, current, current))
    assert profile == Profile("p-1", "Ada Lovelace", None, True, True)
    assert compatibility_report((old, current, current)) == {
        "count": 2, "versions": {1: 1, 3: 1},
        "needs_upcast": ("ProfileRegistered",),
    }

    _expect_error(
        lambda: upcast_event(
            StoredEvent("future", "ProfileRegistered", 4, {})
        ),
        FutureSchemaError,
    )
    _expect_error(
        lambda: upcast_event(
            StoredEvent("bad", "ProfileRegistered", 1, {"profile_id": "p-2"})
        ),
        SchemaError,
    )
    duplicate = StoredEvent("evt-1", "ProfileRegistered", 1,
                            {"profile_id": "p-1", "name": "Ada Lovelace"})
    _expect_error(
        lambda: normalize_event_batch((old, duplicate)),
        ConflictingDuplicateError,
    )
    assert normalize_event_batch((old, old, current)) == (old, current)


def main() -> None:
    run_self_checks()
    stream = (
        StoredEvent("demo-1", "ProfileRegistered", 1,
                    {"profile_id": "demo", "name": "Grace Hopper"}),
        StoredEvent("demo-2", "ProfileUpdated", 2,
                    {"profile_id": "demo", "display_name": "Grace Hopper",
                     "email": "grace@example.test"}),
    )
    print("Stored versions:", compatibility_report(stream)["versions"])
    print("Replayed profile:", ProfileProjector().replay(stream))
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
