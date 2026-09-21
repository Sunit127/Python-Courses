"""Day 17: Serialization, configuration, and schema-aware data boundaries.

Learning goals
--------------
1. Distinguish serialization from validation and domain modeling.
2. Encode and decode JSON without losing control of accepted data.
3. Validate unknown input before constructing trusted domain objects.
4. Layer defaults, file settings, and environment overrides predictably.
5. evolve stored data with explicit schema versions and migrations.

Teaching notes
--------------
- Serialization converts in-memory values to a transport or storage format.
  Deserialization reverses that step, but decoded data is still untrusted.
- JSON supports objects, arrays, strings, numbers, Booleans, and null. It does
  not directly preserve tuples, sets, Decimal values, dates, or custom classes.
  Convert those values explicitly and document the representation.
- json.loads checks JSON syntax, not your application's schema. Validate keys,
  types, ranges, and cross-field rules before constructing a domain object.
- Reject unknown keys at strict boundaries. This catches misspellings and makes
  version changes deliberate instead of silently ignoring data.
- A schema version belongs in durable or exchanged records. Migrations should
  be small, deterministic functions that transform one known version into the
  next and never mutate the caller's dictionary.
- Configuration commonly follows a precedence rule such as defaults, then a
  file, then environment overrides. Merge sources first, validate once, and
  make the winning source easy to explain.
- Treat environment values as strings. Parse Booleans, integers, and numbers
  explicitly; bool("false") is True and is almost never the desired result.
- Never print or serialize secrets accidentally. Keep secret retrieval at the
  application boundary and redact sensitive fields in diagnostics.
- TOML is available as tomllib in Python 3.11+. This lesson uses JSON so it
  remains compatible with Python 3.10 and requires no third-party package.

Run this file with Python 3.10+ to execute deterministic examples and checks.
No network connection, environment mutation, or persistent file is required.

Practice exercises
------------------
1. Implement parse_bool(value). Accept common true/false strings
   case-insensitively and reject every other value.
2. Implement apply_environment_overrides(settings, environment). Recognize
   COURSE_TIMEOUT_SECONDS, COURSE_RETRY_LIMIT, and COURSE_DEBUG; ignore
   unrelated variables and do not mutate settings.
3. Implement load_json_lines(text). Parse one JSON object per non-empty line,
   report the failing line number, and return an immutable tuple.

Solutions appear below the main example.

Expert challenge: versioned job manifest
----------------------------------------
Build a loader for a job manifest containing an ID, task name, arguments,
creation timestamp, retry policy, and schema version. Migrate old versions,
validate every field, reject unknown keys, and create an immutable domain
object. Add a safe diagnostic representation that redacts arguments whose keys
contain token, secret, or password.

Solution guidance
-----------------
1. Decode JSON into object, then validate its shape before domain construction.
2. Keep one pure migration per version transition and test each independently.
3. Separate public metadata from sensitive arguments in logs and exceptions.
4. Round-trip supported records and test malformed JSON, unknown fields,
   unsupported versions, invalid timestamps, and non-finite numbers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from math import isfinite
from typing import Any, Literal, TypeAlias


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
EnvironmentName: TypeAlias = Literal["development", "test", "production"]

CURRENT_SCHEMA_VERSION = 2


class ConfigurationError(ValueError):
    """Raised when serialized or configured data violates the contract."""


def _expect_object(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{field_name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ConfigurationError(f"{field_name} keys must be strings")
    return dict(value)


def _reject_unknown_keys(
    value: Mapping[str, object],
    allowed: set[str],
    field_name: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ConfigurationError(f"{field_name} has unknown keys: {names}")


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ConfigurationError(f"{field_name} cannot be empty")
    return cleaned


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{field_name} must be an integer")
    if value < 0:
        raise ConfigurationError(f"{field_name} cannot be negative")
    return value


def _positive_finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{field_name} must be a number")
    converted = float(value)
    if not isfinite(converted) or converted <= 0:
        raise ConfigurationError(f"{field_name} must be positive and finite")
    return converted


@dataclass(frozen=True, slots=True)
class CourseSettings:
    """Trusted, validated application settings."""

    environment: EnvironmentName
    timeout_seconds: float
    retry_limit: int
    debug: bool
    enabled_topics: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.environment not in {"development", "test", "production"}:
            raise ConfigurationError("environment is not supported")
        _positive_finite_number(self.timeout_seconds, "timeout_seconds")
        _non_negative_int(self.retry_limit, "retry_limit")
        if not isinstance(self.debug, bool):
            raise ConfigurationError("debug must be a Boolean")
        if not isinstance(self.enabled_topics, tuple):
            raise ConfigurationError("enabled_topics must be a tuple")

        normalized: list[str] = []
        seen: set[str] = set()
        for topic in self.enabled_topics:
            clean_topic = _required_text(topic, "enabled_topics item")
            key = clean_topic.casefold()
            if key in seen:
                raise ConfigurationError("enabled_topics cannot contain duplicates")
            seen.add(key)
            normalized.append(clean_topic)

        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        object.__setattr__(self, "enabled_topics", tuple(normalized))


DEFAULT_SETTINGS: dict[str, object] = {
    "environment": "development",
    "timeout_seconds": 5.0,
    "retry_limit": 2,
    "debug": False,
    "enabled_topics": ["serialization"],
}


def migrate_settings(document: Mapping[str, object]) -> dict[str, object]:
    """Return a current-version copy of a supported settings document."""
    migrated = dict(document)
    version = migrated.get("schema_version", 1)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ConfigurationError("schema_version must be an integer")

    if version == 1:
        if "timeout" in migrated and "timeout_seconds" in migrated:
            raise ConfigurationError("version 1 cannot define both timeout fields")
        if "timeout" in migrated:
            migrated["timeout_seconds"] = migrated.pop("timeout")
        migrated.setdefault("enabled_topics", ["serialization"])
        migrated["schema_version"] = 2
        version = 2

    if version != CURRENT_SCHEMA_VERSION:
        raise ConfigurationError(f"unsupported schema_version: {version}")

    return migrated


def settings_from_mapping(value: Mapping[str, object]) -> CourseSettings:
    """Validate a merged, current-version mapping."""
    allowed = {
        "environment",
        "timeout_seconds",
        "retry_limit",
        "debug",
        "enabled_topics",
    }
    _reject_unknown_keys(value, allowed, "settings")

    environment = value.get("environment")
    if environment not in {"development", "test", "production"}:
        raise ConfigurationError("environment is not supported")

    debug = value.get("debug")
    if not isinstance(debug, bool):
        raise ConfigurationError("debug must be a Boolean")

    topics_value = value.get("enabled_topics")
    if not isinstance(topics_value, list):
        raise ConfigurationError("enabled_topics must be a JSON array")
    if any(not isinstance(item, str) for item in topics_value):
        raise ConfigurationError("enabled_topics items must be strings")

    return CourseSettings(
        environment=environment,
        timeout_seconds=_positive_finite_number(
            value.get("timeout_seconds"),
            "timeout_seconds",
        ),
        retry_limit=_non_negative_int(value.get("retry_limit"), "retry_limit"),
        debug=debug,
        enabled_topics=tuple(topics_value),
    )


def decode_settings(text: str) -> dict[str, object]:
    """Decode and migrate a JSON settings document."""
    try:
        decoded: object = json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigurationError(
            f"invalid JSON at line {error.lineno}, column {error.colno}"
        ) from error

    document = _expect_object(decoded, "settings document")
    migrated = migrate_settings(document)
    _reject_unknown_keys(
        migrated,
        {
            "schema_version",
            "environment",
            "timeout_seconds",
            "retry_limit",
            "debug",
            "enabled_topics",
        },
        "settings document",
    )
    migrated.pop("schema_version")
    return migrated


def merge_settings(
    defaults: Mapping[str, object],
    file_settings: Mapping[str, object],
    environment_settings: Mapping[str, object],
) -> dict[str, object]:
    """Merge configuration in increasing precedence without mutating inputs."""
    return {**defaults, **file_settings, **environment_settings}


def encode_settings(settings: CourseSettings) -> str:
    """Serialize settings to stable, readable, versioned JSON."""
    payload: dict[str, object] = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        **asdict(settings),
    }
    payload["enabled_topics"] = list(settings.enabled_topics)
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


# Practice exercise solutions


def parse_bool(value: str) -> bool:
    """Parse a human configuration Boolean explicitly."""
    if not isinstance(value, str):
        raise ConfigurationError("Boolean value must be a string")
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"invalid Boolean value: {value!r}")


def apply_environment_overrides(
    settings: Mapping[str, object],
    environment: Mapping[str, str],
) -> dict[str, object]:
    """Return recognized, parsed overrides layered on a settings copy."""
    result = dict(settings)

    if "COURSE_TIMEOUT_SECONDS" in environment:
        raw_timeout = environment["COURSE_TIMEOUT_SECONDS"]
        try:
            timeout = float(raw_timeout)
        except ValueError as error:
            raise ConfigurationError(
                "COURSE_TIMEOUT_SECONDS must be a number"
            ) from error
        result["timeout_seconds"] = _positive_finite_number(
            timeout,
            "COURSE_TIMEOUT_SECONDS",
        )

    if "COURSE_RETRY_LIMIT" in environment:
        raw_retries = environment["COURSE_RETRY_LIMIT"].strip()
        try:
            retries = int(raw_retries)
        except ValueError as error:
            raise ConfigurationError(
                "COURSE_RETRY_LIMIT must be an integer"
            ) from error
        result["retry_limit"] = _non_negative_int(
            retries,
            "COURSE_RETRY_LIMIT",
        )

    if "COURSE_DEBUG" in environment:
        result["debug"] = parse_bool(environment["COURSE_DEBUG"])

    return result


def load_json_lines(text: str) -> tuple[dict[str, object], ...]:
    """Parse non-empty JSON object lines with useful line-number errors."""
    records: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            decoded: object = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ConfigurationError(
                f"line {line_number}: invalid JSON at column {error.colno}"
            ) from error
        try:
            record = _expect_object(decoded, "record")
        except ConfigurationError as error:
            raise ConfigurationError(f"line {line_number}: {error}") from error
        records.append(record)
    return tuple(records)


def redact_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Create a shallow diagnostic copy with common secret fields hidden."""
    sensitive_fragments = ("password", "secret", "token")
    return {
        key: (
            "<redacted>"
            if any(fragment in key.casefold() for fragment in sensitive_fragments)
            else item
        )
        for key, item in value.items()
    }


def run_self_checks() -> None:
    """Exercise normal, boundary, migration, and failure behavior."""
    version_one = {
        "schema_version": 1,
        "environment": "test",
        "timeout": 3,
        "retry_limit": 1,
        "debug": False,
    }
    original = dict(version_one)
    decoded = decode_settings(json.dumps(version_one))
    assert version_one == original
    assert decoded["timeout_seconds"] == 3
    assert decoded["enabled_topics"] == ["serialization"]

    overrides = apply_environment_overrides(
        decoded,
        {
            "COURSE_TIMEOUT_SECONDS": "2.5",
            "COURSE_RETRY_LIMIT": "4",
            "COURSE_DEBUG": "yes",
            "UNRELATED": "ignored",
        },
    )
    settings = settings_from_mapping(
        merge_settings(DEFAULT_SETTINGS, decoded, overrides)
    )
    assert settings == CourseSettings(
        environment="test",
        timeout_seconds=2.5,
        retry_limit=4,
        debug=True,
        enabled_topics=("serialization",),
    )

    encoded = encode_settings(settings)
    round_tripped = settings_from_mapping(decode_settings(encoded))
    assert round_tripped == settings
    assert '"schema_version": 2' in encoded

    assert parse_bool(" OFF ") is False
    assert parse_bool("1") is True
    assert load_json_lines('{"id": 1}\n\n{"id": 2}') == (
        {"id": 1},
        {"id": 2},
    )
    assert redact_mapping(
        {"user": "learner", "api_token": "abc", "PasswordHint": "private"}
    ) == {
        "user": "learner",
        "api_token": "<redacted>",
        "PasswordHint": "<redacted>",
    }

    invalid_calls = (
        lambda: decode_settings("[1, 2]"),
        lambda: decode_settings('{"schema_version": 99}'),
        lambda: decode_settings(
            '{"schema_version": 2, "environment": "test", "typo": true}'
        ),
        lambda: settings_from_mapping(
            {**DEFAULT_SETTINGS, "timeout_seconds": float("inf")}
        ),
        lambda: settings_from_mapping(
            {**DEFAULT_SETTINGS, "enabled_topics": ["JSON", "json"]}
        ),
        lambda: parse_bool("sometimes"),
        lambda: apply_environment_overrides(
            DEFAULT_SETTINGS,
            {"COURSE_RETRY_LIMIT": "-1"},
        ),
        lambda: apply_environment_overrides(
            DEFAULT_SETTINGS,
            {"COURSE_TIMEOUT_SECONDS": "nan"},
        ),
        lambda: load_json_lines('{"ok": true}\nnot-json'),
        lambda: load_json_lines("[1, 2, 3]"),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except ConfigurationError:
            pass
        else:
            raise AssertionError("invalid input must raise ConfigurationError")


def main() -> None:
    """Run a safe demonstration using in-memory configuration sources."""
    run_self_checks()

    file_text = """
    {
      "schema_version": 2,
      "environment": "production",
      "timeout_seconds": 8,
      "retry_limit": 3,
      "debug": false,
      "enabled_topics": ["serialization", "configuration"]
    }
    """
    file_settings = decode_settings(file_text)
    environment_settings = apply_environment_overrides(
        {},
        {
            "COURSE_TIMEOUT_SECONDS": "4.5",
            "COURSE_DEBUG": "false",
        },
    )
    settings = settings_from_mapping(
        merge_settings(DEFAULT_SETTINGS, file_settings, environment_settings)
    )

    print("Validated settings:", settings)
    print("Versioned JSON:")
    print(encode_settings(settings))
    print(
        "JSON Lines:",
        load_json_lines('{"event": "started"}\n{"event": "completed"}'),
    )
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
