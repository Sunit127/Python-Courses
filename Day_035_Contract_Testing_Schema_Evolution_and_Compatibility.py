"""Day 35: Contract testing, schema evolution, and compatibility.

Learning goals
--------------
1. Treat an API schema as an executable agreement between producers and consumers.
2. Distinguish backward-compatible evolution from changes that require coordination.
3. Build tolerant readers while keeping required fields and value types explicit.
4. Verify consumer-driven contracts against a provider without network dependencies.
5. Upcast versioned events so domain code handles one current representation.

Teaching notes
--------------
- A contract describes observable behavior: accepted inputs, response fields, types,
  error semantics, and invariants. An implementation detail is not a contract.
- Compatibility is directional. This lesson asks whether payloads valid under an
  old schema remain valid under a new schema. Adding an optional field is usually
  safe; removing a field, narrowing its type, or adding a required field is not.
- Consumers should validate what they depend on and ignore unrelated fields.
  Rejecting every unknown response field makes harmless provider additions break.
- Producers still need strict validation at write boundaries. Tolerant reading is
  not permission to accept malformed required data or non-JSON values.
- Consumer-driven contracts are focused examples owned by consumers and replayed
  in provider CI. They complement unit, integration, and end-to-end tests.
- Schema version numbers communicate a transition; they do not make it safe.
  Document compatibility rules and provide deterministic migration or upcasting.
- Never silently reinterpret ambiguous data. Reject unknown versions and preserve
  enough context to diagnose which contract failed without logging secrets.
- A production schema format may be JSON Schema, OpenAPI, Protocol Buffers, or
  Avro. The small standard-library model here exposes the underlying reasoning.

Run this file with Python 3.10+.
It uses only the standard library, opens no network connections, and writes no files.

Practice exercises
------------------
1. Implement compare_schemas(old, new). Report removed fields, newly required
   fields, requiredness changes, type widening or narrowing, and stricter handling
   of unknown fields.
2. Implement failed_contracts(results). Return failed consumer/case identifiers in
   execution order as an immutable tuple.
3. Implement upcast_profile_event(payload). Convert version 1 profile events to
   version 2, accept already-current events, and reject unknown versions.

Solutions appear in the executable code below.

Expert challenge: a compatibility gate
---------------------------------------
Build a CI gate for an HTTP or event-driven service. Store versioned schemas and
consumer contract cases in the repository. On every change, compare the proposed
schema with the released schema, replay all consumer contracts, test historical
payload fixtures, and require an explicit migration plan for breaking changes.
Generate a machine-readable report that identifies the affected consumers.

Solution guidance
-----------------
1. Define compatibility separately for requests, responses, and stored events.
   A rule that is safe for a response producer may be unsafe for a request reader.
2. Parse schemas into a canonical model before comparing them. Reject duplicate
   fields, unknown type names, invalid versions, and nondeterministic ordering.
3. Classify every change with a stable code. Fail CI on breaking codes instead of
   searching human-readable messages.
4. Run contracts in isolation with deterministic providers or test doubles. Bound
   failure details and keep credentials and personal data out of reports.
5. Keep one upcaster per adjacent version when migrations become complex. Test
   direct upgrades, chained upgrades, repeated reads, and unsupported versions.
6. Deploy additive changes before code that relies on them. Remove deprecated
   fields only after telemetry and contract results show that consumers moved.
7. Test empty contracts, provider exceptions, Boolean-versus-integer edge cases,
   non-finite numbers, nested non-JSON values, and duplicate case identifiers.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
import re


class ContractError(Exception):
    """Base class for expected contract failures."""


class ValidationError(ContractError, ValueError):
    """A schema, payload, or contract definition is invalid."""


class CompatibilityError(ContractError):
    """A payload version cannot be upgraded safely."""


_JSON_TYPES = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9._-]*$")


def _clean_name(value: str, *, field: str) -> str:
    """Return a bounded machine-readable name."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 80 or not _NAME_PATTERN.fullmatch(cleaned):
        raise ValidationError(f"{field} must be a lowercase machine-readable name")
    return cleaned


def _json_type(value: object) -> str:
    """Return the precise JSON type name for a Python value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError("JSON numbers must be finite")
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    raise ValidationError(f"{type(value).__name__} is not a JSON value")


def _validate_json(value: object, *, path: str = "$") -> None:
    """Recursively reject values that cannot cross a JSON boundary."""
    kind = _json_type(value)
    if kind == "array":
        for index, item in enumerate(value):
            _validate_json(item, path=f"{path}[{index}]")
    elif kind == "object":
        assert isinstance(value, Mapping)
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"{path} contains a non-string object key")
            _validate_json(item, path=f"{path}.{key}")


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One named field and the JSON types a consumer accepts."""

    name: str
    types: frozenset[str]
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_name(self.name, field="field name"))
        try:
            normalized_types = frozenset(self.types)
        except TypeError as error:
            raise TypeError("field types must be an iterable of strings") from error
        if not normalized_types or not normalized_types <= _JSON_TYPES:
            raise ValidationError("field types must use supported JSON type names")
        if any(not isinstance(kind, str) for kind in normalized_types):
            raise TypeError("field types must contain strings")
        object.__setattr__(self, "types", normalized_types)
        if not isinstance(self.required, bool):
            raise TypeError("required must be a Boolean")


@dataclass(frozen=True, slots=True)
class ObjectSchema:
    """A small executable object schema for compatibility exercises."""

    name: str
    version: int
    fields: tuple[FieldSpec, ...]
    allow_unknown: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_name(self.name, field="schema name"))
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version < 1
        ):
            raise ValidationError("schema version must be a positive integer")
        normalized_fields = tuple(self.fields)
        if any(not isinstance(field, FieldSpec) for field in normalized_fields):
            raise TypeError("schema fields must contain FieldSpec values")
        names = tuple(field.name for field in normalized_fields)
        if len(set(names)) != len(names):
            raise ValidationError("schema field names must be unique")
        object.__setattr__(self, "fields", normalized_fields)
        if not isinstance(self.allow_unknown, bool):
            raise TypeError("allow_unknown must be a Boolean")

    @property
    def by_name(self) -> dict[str, FieldSpec]:
        """Return a fresh lookup so callers cannot mutate schema state."""
        return {field.name: field for field in self.fields}


def validate_payload(
    schema: ObjectSchema,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate a JSON object and return a shallow defensive copy."""
    if not isinstance(schema, ObjectSchema):
        raise TypeError("schema must be an ObjectSchema")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    _validate_json(payload)
    fields = schema.by_name
    missing = tuple(
        field.name
        for field in schema.fields
        if field.required and field.name not in payload
    )
    if missing:
        raise ValidationError("missing required fields: " + ", ".join(missing))
    unknown = tuple(sorted(set(payload) - set(fields)))
    if unknown and not schema.allow_unknown:
        raise ValidationError("unknown fields: " + ", ".join(unknown))
    for name, field in fields.items():
        if name in payload:
            actual = _json_type(payload[name])
            if actual not in field.types:
                expected = ", ".join(sorted(field.types))
                raise ValidationError(
                    f"field {name!r} has type {actual}; expected {expected}"
                )
    return dict(payload)


@dataclass(frozen=True, slots=True)
class SchemaChange:
    """One stable compatibility finding."""

    code: str
    field: str
    breaking: bool
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _clean_name(self.code, field="change code"))
        if self.field != "*" and not _NAME_PATTERN.fullmatch(self.field):
            raise ValidationError("change field must be '*' or a field name")
        if not isinstance(self.breaking, bool):
            raise TypeError("breaking must be a Boolean")
        if (
            not isinstance(self.detail, str)
            or not self.detail
            or len(self.detail) > 240
        ):
            raise ValidationError("change detail must be bounded non-empty text")


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Directional changes from a released schema to a proposed schema."""

    old_version: int
    new_version: int
    changes: tuple[SchemaChange, ...]

    @property
    def backward_compatible(self) -> bool:
        """Whether old valid payloads remain acceptable under the new schema."""
        return not any(change.breaking for change in self.changes)

    @property
    def breaking_codes(self) -> tuple[str, ...]:
        """Return breaking change codes in deterministic order."""
        return tuple(change.code for change in self.changes if change.breaking)


def compare_schemas(
    old: ObjectSchema,
    new: ObjectSchema,
) -> CompatibilityReport:
    """Solution 1: compare schemas using explicit directional rules."""
    if not isinstance(old, ObjectSchema) or not isinstance(new, ObjectSchema):
        raise TypeError("old and new must be ObjectSchema values")
    if old.name != new.name:
        raise ValidationError("schemas must have the same name")
    if new.version <= old.version:
        raise ValidationError("the proposed schema version must increase")

    old_fields = old.by_name
    new_fields = new.by_name
    changes: list[SchemaChange] = []

    for name in sorted(old_fields.keys() - new_fields.keys()):
        changes.append(
            SchemaChange(
                "field.removed",
                name,
                True,
                f"field {name!r} no longer exists",
            )
        )

    for name in sorted(new_fields.keys() - old_fields.keys()):
        field = new_fields[name]
        changes.append(
            SchemaChange(
                "field.added-required" if field.required else "field.added-optional",
                name,
                field.required,
                f"field {name!r} was added as "
                f"{'required' if field.required else 'optional'}",
            )
        )

    for name in sorted(old_fields.keys() & new_fields.keys()):
        before = old_fields[name]
        after = new_fields[name]
        if before.required != after.required:
            became_required = not before.required and after.required
            changes.append(
                SchemaChange(
                    "field.became-required"
                    if became_required
                    else "field.became-optional",
                    name,
                    became_required,
                    f"field {name!r} changed requiredness",
                )
            )
        if before.types != after.types:
            widened = before.types < after.types
            changes.append(
                SchemaChange(
                    "field.type-widened" if widened else "field.type-changed",
                    name,
                    not widened,
                    f"field {name!r} types changed from "
                    f"{sorted(before.types)} to {sorted(after.types)}",
                )
            )

    if old.allow_unknown and not new.allow_unknown:
        changes.append(
            SchemaChange(
                "object.unknown-fields-rejected",
                "*",
                True,
                "the new reader rejects fields accepted by the old reader",
            )
        )
    elif not old.allow_unknown and new.allow_unknown:
        changes.append(
            SchemaChange(
                "object.unknown-fields-allowed",
                "*",
                False,
                "the new reader accepts additional object fields",
            )
        )

    return CompatibilityReport(old.version, new.version, tuple(changes))


ContractAssertion = Callable[[Mapping[str, object]], None]
Provider = Callable[[Mapping[str, object]], Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class ContractCase:
    """One consumer-owned request, response schema, and behavioral assertion."""

    consumer: str
    case: str
    request: Mapping[str, object]
    response_schema: ObjectSchema
    assertion: ContractAssertion

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "consumer", _clean_name(self.consumer, field="consumer")
        )
        object.__setattr__(self, "case", _clean_name(self.case, field="case"))
        if not isinstance(self.request, Mapping):
            raise TypeError("contract request must be a mapping")
        _validate_json(self.request)
        object.__setattr__(self, "request", dict(self.request))
        if not isinstance(self.response_schema, ObjectSchema):
            raise TypeError("response_schema must be an ObjectSchema")
        if not callable(self.assertion):
            raise TypeError("contract assertion must be callable")

    @property
    def identifier(self) -> str:
        """Return a stable consumer/case identifier."""
        return f"{self.consumer}/{self.case}"


@dataclass(frozen=True, slots=True)
class ContractResult:
    """The bounded outcome of one provider verification."""

    identifier: str
    passed: bool
    detail: str

    def __post_init__(self) -> None:
        parts = self.identifier.split("/")
        if len(parts) != 2:
            raise ValidationError("result identifier must be consumer/case")
        _clean_name(parts[0], field="result consumer")
        _clean_name(parts[1], field="result case")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a Boolean")
        if (
            not isinstance(self.detail, str)
            or not self.detail
            or len(self.detail) > 300
        ):
            raise ValidationError("result detail must be bounded non-empty text")


def _bounded_error(error: Exception) -> str:
    """Render an exception without multiline or unbounded diagnostic output."""
    message = str(error).replace("\n", " ").strip() or "no detail"
    return f"{type(error).__name__}: {message}"[:300]


def verify_provider(
    provider: Provider,
    contracts: Iterable[ContractCase],
) -> tuple[ContractResult, ...]:
    """Replay consumer contracts against a provider in deterministic order."""
    if not callable(provider):
        raise TypeError("provider must be callable")
    materialized = tuple(contracts)
    if not materialized:
        raise ValidationError("at least one contract case is required")
    identifiers = tuple(contract.identifier for contract in materialized)
    if len(set(identifiers)) != len(identifiers):
        raise ValidationError("contract identifiers must be unique")

    results: list[ContractResult] = []
    for contract in materialized:
        try:
            response = provider(dict(contract.request))
            validated = validate_payload(contract.response_schema, response)
            contract.assertion(validated)
        except Exception as error:
            results.append(
                ContractResult(contract.identifier, False, _bounded_error(error))
            )
        else:
            results.append(ContractResult(contract.identifier, True, "passed"))
    return tuple(results)


def failed_contracts(results: Iterable[ContractResult]) -> tuple[str, ...]:
    """Solution 2: return failed identifiers in execution order."""
    materialized = tuple(results)
    if any(not isinstance(result, ContractResult) for result in materialized):
        raise TypeError("results must contain ContractResult values")
    return tuple(result.identifier for result in materialized if not result.passed)


PROFILE_EVENT_V2 = ObjectSchema(
    "profile-event",
    2,
    (
        FieldSpec("schema_version", frozenset({"integer"})),
        FieldSpec("user_id", frozenset({"string"})),
        FieldSpec("display_name", frozenset({"string"})),
        FieldSpec("locale", frozenset({"string"}), required=False),
    ),
    allow_unknown=False,
)


def upcast_profile_event(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Solution 3: upgrade a profile event to the current representation."""
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    _validate_json(payload)
    version = payload.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise CompatibilityError("schema_version must be an integer")

    upgraded = dict(payload)
    if version == 1:
        if "display_name" in upgraded:
            raise CompatibilityError("version 1 must not contain display_name")
        full_name = upgraded.pop("full_name", None)
        if not isinstance(full_name, str) or not full_name.strip():
            raise CompatibilityError("version 1 requires a non-empty full_name")
        upgraded["display_name"] = full_name.strip()
        upgraded["schema_version"] = 2
    elif version != 2:
        raise CompatibilityError(f"unsupported profile event version: {version}")

    try:
        return validate_payload(PROFILE_EVENT_V2, upgraded)
    except ValidationError as error:
        raise CompatibilityError("profile event is invalid after upcasting") from error


def _require_active(response: Mapping[str, object]) -> None:
    """Example consumer invariant beyond structural schema validation."""
    if response["status"] != "active":
        raise AssertionError("consumer requires an active user")


def run_self_checks() -> None:
    """Exercise compatible, breaking, contract, and versioning paths."""
    released = ObjectSchema(
        "user-response",
        1,
        (
            FieldSpec("id", frozenset({"string"})),
            FieldSpec("status", frozenset({"string"})),
            FieldSpec("nickname", frozenset({"string"}), required=False),
        ),
    )
    additive = ObjectSchema(
        "user-response",
        2,
        (
            FieldSpec("id", frozenset({"string"})),
            FieldSpec("status", frozenset({"string"})),
            FieldSpec(
                "nickname", frozenset({"string", "null"}), required=False
            ),
            FieldSpec("avatar_url", frozenset({"string"}), required=False),
        ),
    )
    additive_report = compare_schemas(released, additive)
    assert additive_report.backward_compatible
    assert [change.code for change in additive_report.changes] == [
        "field.added-optional",
        "field.type-widened",
    ]

    breaking = ObjectSchema(
        "user-response",
        3,
        (
            FieldSpec("id", frozenset({"integer"})),
            FieldSpec("email", frozenset({"string"})),
        ),
        allow_unknown=False,
    )
    breaking_report = compare_schemas(additive, breaking)
    assert not breaking_report.backward_compatible
    assert "field.removed" in breaking_report.breaking_codes
    assert "field.added-required" in breaking_report.breaking_codes
    assert "field.type-changed" in breaking_report.breaking_codes
    assert "object.unknown-fields-rejected" in breaking_report.breaking_codes

    consumer_schema = ObjectSchema(
        "mobile-user",
        1,
        (
            FieldSpec("id", frozenset({"string"})),
            FieldSpec("status", frozenset({"string"})),
        ),
        allow_unknown=True,
    )
    contract = ContractCase(
        "mobile-app",
        "active-user",
        {"user_id": "u-123"},
        consumer_schema,
        _require_active,
    )

    def good_provider(request: Mapping[str, object]) -> Mapping[str, object]:
        assert request["user_id"] == "u-123"
        return {
            "id": "u-123",
            "status": "active",
            "new_provider_field": "ignored by tolerant consumer",
        }

    passed = verify_provider(good_provider, (contract,))
    assert failed_contracts(passed) == ()

    def bad_provider(_: Mapping[str, object]) -> Mapping[str, object]:
        return {"id": 123, "status": "disabled"}

    failed = verify_provider(bad_provider, (contract,))
    assert failed_contracts(failed) == ("mobile-app/active-user",)
    assert failed[0].detail.startswith("ValidationError:")

    upgraded = upcast_profile_event(
        {
            "schema_version": 1,
            "user_id": "u-7",
            "full_name": "  Ada Lovelace  ",
        }
    )
    assert upgraded == {
        "schema_version": 2,
        "user_id": "u-7",
        "display_name": "Ada Lovelace",
    }
    current = {
        "schema_version": 2,
        "user_id": "u-8",
        "display_name": "Grace Hopper",
        "locale": "en",
    }
    assert upcast_profile_event(current) == current

    invalid_calls: tuple[Callable[[], object], ...] = (
        lambda: FieldSpec("id", frozenset()),
        lambda: FieldSpec("id", frozenset({"integer"}), required="yes"),
        lambda: ObjectSchema(
            "duplicate",
            1,
            (
                FieldSpec("id", frozenset({"string"})),
                FieldSpec("id", frozenset({"string"})),
            ),
        ),
        lambda: validate_payload(
            consumer_schema,
            {"id": "u-1", "status": float("nan")},
        ),
        lambda: compare_schemas(additive, released),
        lambda: verify_provider(good_provider, ()),
        lambda: verify_provider(good_provider, (contract, contract)),
        lambda: upcast_profile_event(
            {"schema_version": True, "user_id": "u-1", "display_name": "Ada"}
        ),
        lambda: upcast_profile_event(
            {"schema_version": 99, "user_id": "u-1", "display_name": "Ada"}
        ),
        lambda: upcast_profile_event(
            {"schema_version": 1, "user_id": "u-1", "full_name": " "}
        ),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, ValueError, ContractError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run a compatibility comparison and consumer contract demo."""
    run_self_checks()

    old = ObjectSchema(
        "task-response",
        1,
        (
            FieldSpec("id", frozenset({"string"})),
            FieldSpec("title", frozenset({"string"})),
        ),
    )
    proposed = ObjectSchema(
        "task-response",
        2,
        (
            FieldSpec("id", frozenset({"string"})),
            FieldSpec("title", frozenset({"string"})),
            FieldSpec("completed", frozenset({"boolean"}), required=False),
        ),
    )
    report = compare_schemas(old, proposed)
    print("Backward compatible:", report.backward_compatible)
    for change in report.changes:
        print(f"- {change.code}: {change.detail}")

    print(
        "Upcast event:",
        upcast_profile_event(
            {
                "schema_version": 1,
                "user_id": "u-42",
                "full_name": "Margaret Hamilton",
            }
        ),
    )
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
