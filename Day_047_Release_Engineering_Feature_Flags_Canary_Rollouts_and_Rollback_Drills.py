"""Day 47: Release engineering with feature flags, canaries, and rollback drills.

This lesson turns a safe deployment from a hopeful button click into an observable,
reversible process. The example is deliberately in-memory and standard-library
only, so it can run on any modern Python 3 interpreter.

Learning goals
--------------
1. Separate deploying code from exposing behavior with an explicit feature flag.
2. Assign a stable canary cohort without relying on random per-request choices.
3. Promote or roll back from measured error-rate and latency evidence.
4. Make a rollback drill auditable, deterministic, and safe to repeat.

The patterns here are teaching examples, not a replacement for a production
deployment platform. Real systems should persist decisions and protect the
control plane with authentication, authorization, and change review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Iterable, Sequence


def utc_now() -> datetime:
    """Return an aware UTC timestamp so comparisons never mix naive and aware times."""
    return datetime.now(timezone.utc)


def stable_bucket(identity: str, salt: str = "python-course-day-47") -> int:
    """Map an identity to a repeatable bucket in the inclusive range 0..9_999.

    A cryptographic hash avoids Python's process-randomized hash() and keeps a
    user in the same cohort across requests and process restarts.
    """
    if not identity:
        raise ValueError("identity must not be empty")
    digest = sha256(f"{salt}:{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10_000


@dataclass(frozen=True)
class User:
    user_id: str
    environment: str = "production"
    plan: str = "free"

    def __post_init__(self) -> None:
        if not self.user_id.strip():
            raise ValueError("user_id must not be blank")
        if not self.environment.strip():
            raise ValueError("environment must not be blank")


@dataclass(frozen=True)
class FeatureFlag:
    """A small, auditable release toggle.

    The evaluation order is intentional: an explicit kill switch wins, then an
    allow-list, then environment/plan targeting, and finally the percentage
    cohort. A percentage is a rollout decision, not a security boundary.
    """

    name: str
    enabled: bool = False
    environments: frozenset[str] = frozenset()
    plans: frozenset[str] = frozenset()
    allow_users: frozenset[str] = frozenset()
    percentage: int = 0
    salt: str = "python-course-day-47"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("flag name must not be blank")
        if not 0 <= self.percentage <= 10_000:
            raise ValueError("percentage must be between 0 and 10_000")

    def enabled_for(self, user: User) -> bool:
        if not self.enabled:
            return False
        if user.user_id in self.allow_users:
            return True
        if self.environments and user.environment not in self.environments:
            return False
        if self.plans and user.plan not in self.plans:
            return False
        return stable_bucket(user.user_id, self.salt) < self.percentage


class ReleaseState(str, Enum):
    CANDIDATE = "candidate"
    CANARY = "canary"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class Release:
    version: str
    flag: FeatureFlag
    state: ReleaseState = ReleaseState.CANDIDATE
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("version must not be blank")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")


@dataclass(frozen=True)
class Observation:
    """One request outcome from the candidate release."""

    latency_ms: float
    succeeded: bool
    cohort: str = "canary"

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if not self.cohort.strip():
            raise ValueError("cohort must not be blank")


@dataclass(frozen=True)
class HealthReport:
    samples: int
    error_rate: float
    p95_latency_ms: float

    @property
    def healthy(self) -> bool:
        return self.error_rate <= 0.02 and self.p95_latency_ms <= 500.0


def percentile95(values: Sequence[float]) -> float:
    """Return an interpolated p95 without requiring an external dependency."""
    if not values:
        raise ValueError("at least one value is required")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = 0.95 * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_health(observations: Iterable[Observation]) -> HealthReport:
    records = list(observations)
    if not records:
        raise ValueError("cannot assess an empty canary")
    failures = sum(not item.succeeded for item in records)
    return HealthReport(
        samples=len(records),
        error_rate=failures / len(records),
        p95_latency_ms=percentile95([item.latency_ms for item in records]),
    )


@dataclass(frozen=True)
class CanaryStage:
    percentage: int
    minimum_samples: int = 20

    def __post_init__(self) -> None:
        if not 0 < self.percentage <= 10_000:
            raise ValueError("stage percentage must be between 1 and 10_000")
        if self.minimum_samples < 1:
            raise ValueError("minimum_samples must be positive")


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    report: HealthReport | None = None


class RolloutController:
    """A deterministic state machine for promotion and rollback decisions."""

    def __init__(self, release: Release, stages: Sequence[CanaryStage]) -> None:
        if not stages:
            raise ValueError("at least one canary stage is required")
        percentages = [stage.percentage for stage in stages]
        if percentages != sorted(percentages):
            raise ValueError("stages must be in ascending percentage order")
        if percentages[-1] != 10_000:
            raise ValueError("the final stage must represent 100 percent")
        self.release = release
        self.stages = tuple(stages)
        self.stage_index = 0
        self.history: list[Decision] = []

    @property
    def current_stage(self) -> CanaryStage:
        return self.stages[self.stage_index]

    def collect_decision(self, observations: Iterable[Observation]) -> Decision:
        report = summarize_health(observations)
        stage = self.current_stage
        if report.samples < stage.minimum_samples:
            decision = Decision(
                "hold",
                f"need {stage.minimum_samples - report.samples} more samples",
                report,
            )
        elif not report.healthy:
            decision = Decision(
                "rollback",
                "error rate or p95 latency crossed the release guardrail",
                report,
            )
        elif stage.percentage == 10_000:
            decision = Decision("promote", "all stages are healthy", report)
        else:
            decision = Decision(
                "promote",
                "health is within guardrails; advance to the next stage",
                report,
            )
        self.history.append(decision)
        return decision

    def apply(self, decision: Decision) -> Release:
        if decision not in self.history:
            raise ValueError("decision must come from collect_decision")
        if decision.action == "rollback":
            self.release = Release(
                self.release.version,
                FeatureFlag(
                    name=self.release.flag.name,
                    enabled=False,
                    salt=self.release.flag.salt,
                ),
                ReleaseState.ROLLED_BACK,
                self.release.created_at,
            )
        elif decision.action == "promote":
            if self.current_stage.percentage == 10_000:
                self.release = Release(
                    self.release.version,
                    self.release.flag,
                    ReleaseState.PROMOTED,
                    self.release.created_at,
                )
            else:
                self.stage_index += 1
                next_stage = self.current_stage
                self.release = Release(
                    self.release.version,
                    FeatureFlag(
                        name=self.release.flag.name,
                        enabled=True,
                        environments=self.release.flag.environments,
                        plans=self.release.flag.plans,
                        allow_users=self.release.flag.allow_users,
                        percentage=next_stage.percentage,
                        salt=self.release.flag.salt,
                    ),
                    ReleaseState.CANARY,
                    self.release.created_at,
                )
        return self.release


@dataclass(frozen=True)
class AuditEvent:
    at: datetime
    action: str
    version: str
    reason: str


class ReleaseLedger:
    """Append-only audit log for release actions.

    Production ledgers should be durable and tamper-evident. This in-memory
    version makes the ordering and immutability contract easy to inspect.
    """

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def record(self, action: str, version: str, reason: str) -> None:
        if not action.strip() or not version.strip() or not reason.strip():
            raise ValueError("audit fields must not be blank")
        self._events.append(AuditEvent(utc_now(), action, version, reason))

    def events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._events)


def healthy_samples(count: int, latency_ms: float = 120.0) -> list[Observation]:
    if count < 1:
        raise ValueError("count must be positive")
    return [Observation(latency_ms, True) for _ in range(count)]


def failed_samples(count: int, latency_ms: float = 850.0) -> list[Observation]:
    if count < 1:
        raise ValueError("count must be positive")
    return [Observation(latency_ms, False) for _ in range(count)]


def demo_successful_rollout() -> tuple[Release, tuple[AuditEvent, ...]]:
    """Run a 1% -> 10% -> 100% rollout with deterministic evidence."""
    flag = FeatureFlag(
        name="checkout_v2",
        enabled=True,
        environments=frozenset({"production"}),
        percentage=100,
    )
    release = Release("2026.11.13.1", flag)
    controller = RolloutController(
        release,
        (
            CanaryStage(100, minimum_samples=5),
            CanaryStage(1_000, minimum_samples=5),
            CanaryStage(10_000, minimum_samples=5),
        ),
    )
    ledger = ReleaseLedger()
    for samples in (healthy_samples(5), healthy_samples(5), healthy_samples(5)):
        decision = controller.collect_decision(samples)
        assert decision.action == "promote"
        prior_stage = controller.current_stage.percentage
        release = controller.apply(decision)
        ledger.record("promote", release.version, f"stage {prior_stage} passed")
    assert release.state is ReleaseState.PROMOTED
    return release, ledger.events()


def demo_rollback() -> Release:
    """Show that a slow/erroring canary is disabled immediately."""
    flag = FeatureFlag(
        name="search_v3",
        enabled=True,
        environments=frozenset({"production"}),
        percentage=100,
    )
    release = Release("2026.11.13.2", flag)
    controller = RolloutController(
        release,
        (CanaryStage(100, minimum_samples=5), CanaryStage(10_000, minimum_samples=5)),
    )
    decision = controller.collect_decision(healthy_samples(4) + failed_samples(2))
    assert decision.action == "rollback"
    rolled_back = controller.apply(decision)
    assert rolled_back.state is ReleaseState.ROLLED_BACK
    assert not rolled_back.flag.enabled
    return rolled_back


def run_examples() -> None:
    # Stable assignment: the same user stays in the same cohort.
    alice = User("alice", plan="pro")
    assert stable_bucket(alice.user_id) == stable_bucket(alice.user_id)
    flag = FeatureFlag(
        "reports_v2",
        enabled=True,
        environments=frozenset({"production"}),
        plans=frozenset({"pro"}),
        allow_users=frozenset({"vip"}),
        percentage=0,
    )
    assert flag.enabled_for(alice) is False
    assert flag.enabled_for(User("vip")) is True
    assert flag.enabled_for(User("other", environment="staging", plan="pro")) is False
    promoted, events = demo_successful_rollout()
    assert promoted.flag.percentage == 10_000
    assert len(events) == 3
    rolled_back = demo_rollback()
    assert rolled_back.state is ReleaseState.ROLLED_BACK

    # Edge cases that should fail loudly and locally.
    for bad_percentage in (-1, 10_001):
        try:
            FeatureFlag("bad", percentage=bad_percentage)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid percentage was accepted")
    try:
        summarize_health([])
    except ValueError:
        pass
    else:
        raise AssertionError("empty health report was accepted")

    print("Day 47 examples passed: flags, canary promotion, and rollback drills")


# Practice exercises
# ------------------
# 1. Add a blocked_users: frozenset[str] field. Make it override every other
#    targeting rule, including allow_users, and add tests for both.
# 2. Replace the fixed 2%/500 ms thresholds with a Guardrails dataclass that
#    accepts an error-rate and latency limit. Inject it into RolloutController
#    and keep the existing defaults.
# 3. Add a hold path to apply that leaves the release and stage unchanged.
#    Record a ledger event for hold decisions without advancing the rollout.
# 4. Make ReleaseLedger serialize its events to JSON Lines using only the
#    standard library. Include an ISO-8601 timestamp and write atomically
#    through a temporary file plus Path.replace.

# Solutions / clear guidance
# --------------------------
# 1. Check blocked_users first in enabled_for, before the kill switch's normal
#    targeting branches; a blocked user should return False immediately.
# 2. Define Guardrails(max_error_rate: float = 0.02,
#    max_p95_latency_ms: float = 500.0). Validate both are non-negative, then
#    compare the HealthReport against those values. Pass one guardrail object to
#    the controller rather than reading global constants.
# 3. Return self.release for hold; do not increment stage_index. The important
#    invariant is that applying the same hold decision is idempotent.
# 4. Use json.dumps(asdict(event), default=str) or an explicit encoder,
#    newline-delimited records, and Path(temp_name).replace(destination).
#    Restrict the destination to a caller-provided safe directory.

# Expert challenge / mini-project
# --------------------------------
# Build a local "release commander" around these primitives:
#
# * Read a version, flag name, stage percentages, and guardrails from a JSON
#   configuration file.
# * Replay a JSON Lines observation file and calculate one HealthReport per
#   stage. Do not mix observations from different versions.
# * Require an explicit operator approval token before promotion; never accept
#   the token from a command-line argument that may leak into shell history.
# * Write an append-only audit log and a rollback marker. Make rerunning the
#   command after a successful rollback a no-op.
# * Add unit tests for stable cohorts, malformed records, insufficient samples,
#   rollback idempotency, and a canary that is healthy at 1% but unhealthy at 10%.
#
# A strong solution keeps policy (pure calculations), orchestration (state
# transitions), and adapters (JSONL/files/CLI) in separate modules. Document
# how an operator would disable the flag, verify the previous version, and
# re-enable traffic after the incident review.


if __name__ == "__main__":
    run_examples()
