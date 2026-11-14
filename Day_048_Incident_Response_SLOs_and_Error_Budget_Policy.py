"""Day 48: Incident response, SLOs, and error-budget policy.

This lesson connects operational signals to an explicit decision process. The
examples are standard-library only and run entirely in memory, so they are safe
to study and execute on a local Python 3.10+ interpreter.

Learning goals
--------------
1. Define an availability SLO and calculate its error budget from observations.
2. Measure a rolling window without mixing naive and timezone-aware timestamps.
3. Model incident severity and safe state transitions with a small state machine.
4. Turn remaining error budget into a deployment policy with explainable reasons.
5. Record a timeline that helps responders communicate, learn, and audit changes.

Teaching notes
--------------
- An SLI is the measured signal (for example, successful requests divided by
  valid requests). An SLO is the target for that signal over a stated window.
- An error budget is the tolerated unreliability: 1 - SLO target. A 99.9%
  availability target permits 0.1% unsuccessful requests in the window.
- A budget is a decision tool, not permission to spend failures carelessly.
  Protect users first, then use healthy budget for measured delivery.
- Burn rate compares the observed error rate with the allowed error rate. A
  burn rate of 2 means the service is consuming budget twice as fast as the
  planned rate.
- Incident response needs a shared vocabulary. Detect, acknowledge, mitigate,
  resolve, and review are distinct transitions; timestamps and owners make
  handoffs reliable.
- Keep policy calculations pure and inject the clock. Deterministic code is
  easier to test during a stressful incident and easier to replay afterward.
- A real service would persist events, page an on-call engineer, integrate with
  telemetry, and enforce authorization. This lesson intentionally demonstrates
  the domain logic without contacting any external system.

Run this file directly to see a healthy window, a budget breach, and a small
incident timeline. The self-checks exercise normal paths and important edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Iterable


def utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    """Build an explicit UTC timestamp for deterministic examples and tests."""
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def require_aware(value: datetime, field_name: str) -> datetime:
    """Reject naive datetimes before they can corrupt a rolling-window query."""
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


@dataclass(frozen=True, slots=True)
class SLO:
    """An availability target measured across a finite rolling window."""

    name: str
    target: float
    window: timedelta

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("SLO name must not be blank")
        if not 0.0 < self.target <= 1.0:
            raise ValueError("target must be greater than 0 and at most 1")
        if self.window <= timedelta(0):
            raise ValueError("window must be positive")

    @property
    def allowed_error_rate(self) -> float:
        """Return the fraction of requests the budget can tolerate."""
        return 1.0 - self.target


@dataclass(frozen=True, slots=True)
class RequestObservation:
    """One request outcome used to calculate the availability SLI."""

    at: datetime
    succeeded: bool
    request_id: str = ""

    def __post_init__(self) -> None:
        require_aware(self.at, "observation timestamp")
        if self.request_id and not self.request_id.strip():
            raise ValueError("request_id must be blank or non-whitespace")


@dataclass(frozen=True, slots=True)
class WindowReport:
    """A measured window and the budget state derived from it."""

    start: datetime
    end: datetime
    samples: int
    failures: int
    availability: float
    error_rate: float
    budget_remaining: float

    @property
    def exhausted(self) -> bool:
        return self.budget_remaining <= 0.0


def measure_window(
    observations: Iterable[RequestObservation],
    slo: SLO,
    *,
    end: datetime,
) -> WindowReport:
    """Measure observations in [end - window, end], inclusive at both edges."""
    checked_end = require_aware(end, "window end")
    start = checked_end - slo.window
    records = [
        item
        for item in observations
        if start <= require_aware(item.at, "observation timestamp") <= checked_end
    ]
    if not records:
        raise ValueError("window contains no observations")
    failures = sum(not item.succeeded for item in records)
    error_rate = failures / len(records)
    availability = 1.0 - error_rate
    # A negative value is useful: it quantifies how far beyond the budget we are.
    budget_remaining = slo.allowed_error_rate - error_rate
    return WindowReport(
        start=start,
        end=checked_end,
        samples=len(records),
        failures=failures,
        availability=availability,
        error_rate=error_rate,
        budget_remaining=budget_remaining,
    )


def burn_rate(report: WindowReport, slo: SLO) -> float:
    """Compare current error consumption with the allowed error rate."""
    if slo.allowed_error_rate == 0.0:
        return 0.0 if report.error_rate == 0.0 else float("inf")
    return report.error_rate / slo.allowed_error_rate


class IncidentSeverity(str, Enum):
    SEV3 = "sev3"  # limited impact, normal business-hours response
    SEV2 = "sev2"  # material customer or business impact
    SEV1 = "sev1"  # widespread or safety-critical impact


class IncidentStatus(str, Enum):
    DETECTED = "detected"
    ACKNOWLEDGED = "acknowledged"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"


_ALLOWED_TRANSITIONS: dict[IncidentStatus, frozenset[IncidentStatus]] = {
    IncidentStatus.DETECTED: frozenset({IncidentStatus.ACKNOWLEDGED}),
    IncidentStatus.ACKNOWLEDGED: frozenset(
        {IncidentStatus.MITIGATING, IncidentStatus.RESOLVED}
    ),
    IncidentStatus.MITIGATING: frozenset({IncidentStatus.RESOLVED}),
    IncidentStatus.RESOLVED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    at: datetime
    message: str
    actor: str

    def __post_init__(self) -> None:
        require_aware(self.at, "timeline timestamp")
        if not self.message.strip() or not self.actor.strip():
            raise ValueError("timeline message and actor must not be blank")


@dataclass
class Incident:
    """A deliberately small, auditable incident state machine."""

    incident_id: str
    service: str
    severity: IncidentSeverity
    started_at: datetime
    status: IncidentStatus = IncidentStatus.DETECTED
    timeline: list[TimelineEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.incident_id.strip() or not self.service.strip():
            raise ValueError("incident_id and service must not be blank")
        require_aware(self.started_at, "incident start")
        self.timeline.append(
            TimelineEvent(self.started_at, "incident detected", "monitor")
        )

    def transition(
        self,
        new_status: IncidentStatus,
        *,
        at: datetime,
        actor: str,
        message: str,
    ) -> None:
        """Apply one legal transition and append its audit event."""
        require_aware(at, "transition timestamp")
        if new_status not in _ALLOWED_TRANSITIONS[self.status]:
            raise ValueError(
                f"cannot move from {self.status.value} to {new_status.value}"
            )
        self.status = new_status
        self.timeline.append(TimelineEvent(at, message, actor))


@dataclass(frozen=True, slots=True)
class DeploymentDecision:
    allowed: bool
    reason: str
    burn_rate: float
    budget_remaining: float


def deployment_policy(
    report: WindowReport,
    slo: SLO,
    *,
    active_incident: bool,
    max_burn_rate: float = 1.0,
) -> DeploymentDecision:
    """Decide whether a risky deployment may proceed.

    The policy freezes risky changes during an active incident or when the
    measured burn rate exceeds the configured limit. A healthy budget does not
    bypass an incident commander's explicit freeze.
    """
    if max_burn_rate <= 0.0:
        raise ValueError("max_burn_rate must be positive")
    rate = burn_rate(report, slo)
    if active_incident:
        return DeploymentDecision(
            False,
            "deployment frozen while an incident is active",
            rate,
            report.budget_remaining,
        )
    if rate > max_burn_rate:
        return DeploymentDecision(
            False,
            "deployment paused because error budget is burning too quickly",
            rate,
            report.budget_remaining,
        )
    return DeploymentDecision(
        True,
        "deployment allowed: budget burn is within policy",
        rate,
        report.budget_remaining,
    )


def demo() -> None:
    """Run a healthy report, a policy decision, and an incident timeline."""
    slo = SLO("checkout availability", target=0.99, window=timedelta(hours=1))
    now = utc(2026, 11, 14, 12)
    healthy = [
        RequestObservation(now - timedelta(minutes=5 * i), True)
        for i in range(99)
    ]
    healthy.append(RequestObservation(now - timedelta(minutes=2), False))
    report = measure_window(healthy, slo, end=now)
    decision = deployment_policy(report, slo, active_incident=False)

    incident = Incident("inc-048", "checkout", IncidentSeverity.SEV2, now)
    incident.transition(
        IncidentStatus.ACKNOWLEDGED,
        at=now + timedelta(minutes=2),
        actor="on-call",
        message="acknowledged alert and opened bridge",
    )
    incident.transition(
        IncidentStatus.MITIGATING,
        at=now + timedelta(minutes=8),
        actor="on-call",
        message="disabled the newest checkout flag",
    )
    incident.transition(
        IncidentStatus.RESOLVED,
        at=now + timedelta(minutes=20),
        actor="on-call",
        message="error rate returned below the SLO threshold",
    )

    print(
        f"Availability: {report.availability:.2%}; "
        f"budget remaining: {report.budget_remaining:.2%}"
    )
    print(f"Deployment: {'allowed' if decision.allowed else 'paused'}")
    print(f"Incident {incident.incident_id}: {incident.status.value}")
    print(f"Timeline events: {len(incident.timeline)}")


# Practice exercises
# ------------------
# 1. Add a latency observation and create a latency SLO whose SLI is the
#    percentage of requests below a configured threshold.
# 2. Add a notes field to DeploymentDecision and explain which metric caused
#    a freeze when both an incident and a high burn rate are present.
# 3. Write incident_duration that returns a timedelta for a resolved incident
#    and raises for an incident that is still open.
# 4. Make measure_window accept a clock callable instead of an end argument,
#    then test it with a fixed clock and observations exactly on both boundaries.

# Solutions / clear guidance
# --------------------------
# 1. Keep the SLI calculation separate from the policy. Count observations
#    where latency_ms <= threshold_ms, validate non-negative values, and reuse
#    the same aware timestamp filtering.
# 2. Keep the incident check first because a human-led freeze is stronger than
#    an automated threshold. Set notes to the two independent reasons rather
#    than hiding the burn-rate evidence.
# 3. Store a resolved_at timestamp when the state reaches RESOLVED. Return
#    resolved_at - started_at and reject a naive timestamp or an unresolved
#    incident; this makes the contract explicit.
# 4. Inject clock: Callable[[], datetime], call it once per measurement, and
#    use inclusive comparisons so boundary behavior is deterministic.

# Expert challenge / mini-project
# --------------------------------
# Build a local incident commander CLI around these primitives:
#
# * Read SLOs and request observations from JSON Lines. Reject malformed records
#   with a line number; never silently treat missing fields as successes.
# * Maintain an append-only incident journal with detect, acknowledge, mitigate,
#   resolve, and post-incident-review events. Use an atomic temporary-file
#   replacement for writes.
# * Calculate fast-burn and slow-burn alerts over two windows, for example
#   five minutes and one hour. Deduplicate alerts by incident ID.
# * Require a change ticket and an explicit approval before unfreezing deploys.
# * Add unit tests for empty windows, exact boundaries, a zero-error budget,
#   illegal transitions, replaying the same journal, and malformed JSON.
#
# A strong solution keeps pure SLI/policy functions independent from JSONL and
# terminal adapters. Document the SLO target, paging thresholds, escalation
# owner, and the review questions: what happened, why detection worked or did
# not work, and which prevention work has an owner and due date.


if __name__ == "__main__":
    demo()
