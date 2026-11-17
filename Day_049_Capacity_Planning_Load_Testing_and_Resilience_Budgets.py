"""Day 49: Capacity planning, load testing, and resilience budgets.

This lesson turns performance data into an explicit capacity decision. The
examples use deterministic, in-memory samples and only the Python standard
library, so they are safe to run on a modern Python 3.10+ interpreter.

Learning goals
--------------
1. Distinguish offered load, throughput, latency, errors, and dropped work.
2. Summarize a staged load test without hiding the first saturation point.
3. Estimate instance counts from projected demand, target utilization, and reserves.
4. Define a resilience budget for latency, failures, headroom, and recovery time.
5. Build a release gate whose decisions are measurable and explainable.

Teaching notes
--------------
- Capacity is not the largest number seen once. Sustainable capacity is the load
  a service can handle while meeting its reliability and latency objectives.
- Offered requests include successful, failed, and dropped work. Counting only
  successes makes an overloaded system look healthier than it is.
- Percentiles describe a distribution better than an average. This lesson uses
  the nearest-rank p95: at least 95% of observed latencies are at or below it.
- Run a load test as a ramp. Stop claiming additional sustainable capacity after
  the first failed stage, even if a later noisy sample happens to pass.
- Target utilization leaves room for bursts, garbage collection, uneven traffic,
  and measurement error. Reserved instances protect capacity during a failure.
- A resilience budget is a design constraint, not a promise that failures are
  acceptable. It states the worst conditions the release process will permit.
- Load-test only systems you own or are explicitly authorized to test. Use
  bounded concurrency, timeouts, rate limits, and an emergency stop.

Run this file directly to analyze a synthetic ramp, plan failover capacity, and
evaluate a resilience gate. The self-checks cover boundaries and invalid input.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


def _finite_number(value: float, field_name: str, *, positive: bool = False) -> float:
    """Return a checked finite number and reject booleans masquerading as integers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    checked = float(value)
    if not math.isfinite(checked):
        raise ValueError(f"{field_name} must be finite")
    if positive and checked <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return checked


def percentile(values: Iterable[float], percent: float) -> float:
    """Calculate a nearest-rank percentile for non-negative measurements."""
    checked_percent = _finite_number(percent, "percent", positive=True)
    if checked_percent > 100.0:
        raise ValueError("percent must be at most 100")
    checked_values: list[float] = []
    for value in values:
        checked = _finite_number(value, "measurement")
        if checked < 0.0:
            raise ValueError("measurements must not be negative")
        checked_values.append(checked)
    if not checked_values:
        raise ValueError("at least one measurement is required")
    checked_values.sort()
    rank = math.ceil((checked_percent / 100.0) * len(checked_values))
    return checked_values[rank - 1]


@dataclass(frozen=True, slots=True)
class LoadSample:
    """Measurements from one steady stage of a load-test ramp."""

    concurrency: int
    duration_seconds: float
    succeeded: int
    failed: int
    dropped: int
    latencies_ms: tuple[float, ...]

    def __post_init__(self) -> None:
        if isinstance(self.concurrency, bool) or not isinstance(self.concurrency, int):
            raise TypeError("concurrency must be an integer")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be positive")
        duration = _finite_number(
            self.duration_seconds, "duration_seconds", positive=True
        )
        object.__setattr__(self, "duration_seconds", duration)

        for field_name in ("succeeded", "failed", "dropped"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must not be negative")

        if self.offered_requests == 0:
            raise ValueError("a load sample must contain offered requests")
        if len(self.latencies_ms) != self.handled_requests:
            raise ValueError(
                "latencies_ms must contain one value per succeeded or failed request"
            )

        checked_latencies: list[float] = []
        for latency in self.latencies_ms:
            checked = _finite_number(latency, "latency")
            if checked < 0.0:
                raise ValueError("latency must not be negative")
            checked_latencies.append(checked)
        object.__setattr__(self, "latencies_ms", tuple(checked_latencies))

    @property
    def handled_requests(self) -> int:
        return self.succeeded + self.failed

    @property
    def offered_requests(self) -> int:
        return self.handled_requests + self.dropped

    @property
    def offered_rps(self) -> float:
        return self.offered_requests / self.duration_seconds

    @property
    def throughput_rps(self) -> float:
        return self.succeeded / self.duration_seconds

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.offered_requests

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return float("inf")
        return percentile(self.latencies_ms, 95.0)


@dataclass(frozen=True, slots=True)
class LoadTarget:
    """The service objectives every sustainable load stage must satisfy."""

    min_success_rate: float
    max_p95_latency_ms: float

    def __post_init__(self) -> None:
        success_rate = _finite_number(self.min_success_rate, "min_success_rate")
        if not 0.0 < success_rate <= 1.0:
            raise ValueError("min_success_rate must be greater than 0 and at most 1")
        max_latency = _finite_number(
            self.max_p95_latency_ms, "max_p95_latency_ms", positive=True
        )
        object.__setattr__(self, "min_success_rate", success_rate)
        object.__setattr__(self, "max_p95_latency_ms", max_latency)

    def accepts(self, sample: LoadSample) -> bool:
        return (
            sample.success_rate >= self.min_success_rate
            and sample.p95_latency_ms <= self.max_p95_latency_ms
        )


@dataclass(frozen=True, slots=True)
class LoadTestReport:
    """A ramp summary that preserves the first failed stage."""

    target: LoadTarget
    samples: tuple[LoadSample, ...]
    sustainable_rps: float
    peak_offered_rps: float
    first_failing_concurrency: int | None
    worst_success_rate: float
    worst_p95_latency_ms: float


def analyze_load_test(
    samples: Iterable[LoadSample], target: LoadTarget
) -> LoadTestReport:
    """Analyze stages in concurrency order and stop capacity credit at saturation."""
    ordered = tuple(sorted(samples, key=lambda item: item.concurrency))
    if not ordered:
        raise ValueError("at least one load sample is required")
    if len({sample.concurrency for sample in ordered}) != len(ordered):
        raise ValueError("each load-test stage must use a unique concurrency")

    sustainable_rps = 0.0
    first_failure: int | None = None
    for sample in ordered:
        if first_failure is None and target.accepts(sample):
            sustainable_rps = max(sustainable_rps, sample.offered_rps)
        elif first_failure is None:
            first_failure = sample.concurrency

    return LoadTestReport(
        target=target,
        samples=ordered,
        sustainable_rps=sustainable_rps,
        peak_offered_rps=max(sample.offered_rps for sample in ordered),
        first_failing_concurrency=first_failure,
        worst_success_rate=min(sample.success_rate for sample in ordered),
        worst_p95_latency_ms=max(sample.p95_latency_ms for sample in ordered),
    )


@dataclass(frozen=True, slots=True)
class InstanceProfile:
    """Measured capacity for one instance and its safe utilization ceiling."""

    max_rps: float
    target_utilization: float

    def __post_init__(self) -> None:
        max_rps = _finite_number(self.max_rps, "max_rps", positive=True)
        utilization = _finite_number(self.target_utilization, "target_utilization")
        if not 0.0 < utilization <= 1.0:
            raise ValueError("target_utilization must be greater than 0 and at most 1")
        object.__setattr__(self, "max_rps", max_rps)
        object.__setattr__(self, "target_utilization", utilization)

    @property
    def safe_rps(self) -> float:
        return self.max_rps * self.target_utilization


@dataclass(frozen=True, slots=True)
class CapacityPolicy:
    """Capacity kept aside for bursts and a stated number of instance failures."""

    reserve_fraction: float
    tolerated_instance_failures: int

    def __post_init__(self) -> None:
        reserve = _finite_number(self.reserve_fraction, "reserve_fraction")
        if not 0.0 <= reserve < 1.0:
            raise ValueError("reserve_fraction must be at least 0 and below 1")
        if (
            isinstance(self.tolerated_instance_failures, bool)
            or not isinstance(self.tolerated_instance_failures, int)
        ):
            raise TypeError("tolerated_instance_failures must be an integer")
        if self.tolerated_instance_failures < 0:
            raise ValueError("tolerated_instance_failures must not be negative")
        object.__setattr__(self, "reserve_fraction", reserve)


@dataclass(frozen=True, slots=True)
class CapacityPlan:
    projected_peak_rps: float
    healthy_instances: int
    reserve_instances: int
    total_instances: int
    surviving_safe_rps: float
    headroom_fraction: float


def plan_capacity(
    projected_peak_rps: float,
    profile: InstanceProfile,
    policy: CapacityPolicy,
) -> CapacityPlan:
    """Plan enough instances for demand, reserve policy, and failover."""
    peak = _finite_number(
        projected_peak_rps, "projected_peak_rps", positive=True
    )
    healthy_instances = math.ceil(peak / profile.safe_rps)
    if policy.reserve_fraction == 0.0:
        fractional_reserve = 0
    else:
        fractional_reserve = math.ceil(
            healthy_instances
            * policy.reserve_fraction
            / (1.0 - policy.reserve_fraction)
        )
    reserve_instances = max(
        fractional_reserve, policy.tolerated_instance_failures
    )
    total_instances = healthy_instances + reserve_instances
    surviving_instances = total_instances - policy.tolerated_instance_failures
    surviving_safe_rps = surviving_instances * profile.safe_rps
    headroom_fraction = (surviving_safe_rps - peak) / peak

    return CapacityPlan(
        projected_peak_rps=peak,
        healthy_instances=healthy_instances,
        reserve_instances=reserve_instances,
        total_instances=total_instances,
        surviving_safe_rps=surviving_safe_rps,
        headroom_fraction=headroom_fraction,
    )


@dataclass(frozen=True, slots=True)
class ResilienceBudget:
    """Release limits for load quality, failover headroom, and recovery."""

    max_error_rate: float
    max_p95_latency_ms: float
    min_failover_headroom_fraction: float
    max_recovery_seconds: float

    def __post_init__(self) -> None:
        error_rate = _finite_number(self.max_error_rate, "max_error_rate")
        if not 0.0 <= error_rate < 1.0:
            raise ValueError("max_error_rate must be at least 0 and below 1")
        latency = _finite_number(
            self.max_p95_latency_ms, "max_p95_latency_ms", positive=True
        )
        headroom = _finite_number(
            self.min_failover_headroom_fraction,
            "min_failover_headroom_fraction",
        )
        if headroom < 0.0:
            raise ValueError("min_failover_headroom_fraction must not be negative")
        recovery = _finite_number(
            self.max_recovery_seconds, "max_recovery_seconds", positive=True
        )
        object.__setattr__(self, "max_error_rate", error_rate)
        object.__setattr__(self, "max_p95_latency_ms", latency)
        object.__setattr__(self, "min_failover_headroom_fraction", headroom)
        object.__setattr__(self, "max_recovery_seconds", recovery)

    @property
    def load_target(self) -> LoadTarget:
        return LoadTarget(
            min_success_rate=1.0 - self.max_error_rate,
            max_p95_latency_ms=self.max_p95_latency_ms,
        )


@dataclass(frozen=True, slots=True)
class ResilienceDecision:
    approved: bool
    reasons: tuple[str, ...]


def evaluate_resilience(
    report: LoadTestReport,
    plan: CapacityPlan,
    budget: ResilienceBudget,
    *,
    recovery_seconds: float,
) -> ResilienceDecision:
    """Evaluate independent release constraints and report every failure."""
    recovery = _finite_number(recovery_seconds, "recovery_seconds")
    if recovery < 0.0:
        raise ValueError("recovery_seconds must not be negative")
    if report.target != budget.load_target:
        raise ValueError("load report target does not match the resilience budget")

    reasons: list[str] = []
    if report.sustainable_rps < plan.projected_peak_rps:
        reasons.append(
            "load test did not sustain the projected peak within latency/error limits"
        )
    if plan.headroom_fraction < budget.min_failover_headroom_fraction:
        reasons.append("failover capacity does not meet the headroom budget")
    if recovery > budget.max_recovery_seconds:
        reasons.append("measured recovery time exceeds the resilience budget")

    if reasons:
        return ResilienceDecision(False, tuple(reasons))
    return ResilienceDecision(
        True,
        ("load, failover headroom, and recovery all satisfy the budget",),
    )


def synthetic_sample(
    concurrency: int,
    duration_seconds: float,
    *,
    succeeded: int,
    failed: int,
    dropped: int,
    latency_ms: float,
) -> LoadSample:
    """Build a deterministic sample without sending traffic to any service."""
    handled = succeeded + failed
    return LoadSample(
        concurrency=concurrency,
        duration_seconds=duration_seconds,
        succeeded=succeeded,
        failed=failed,
        dropped=dropped,
        latencies_ms=(latency_ms,) * handled,
    )


def run_self_checks() -> None:
    """Exercise normal behavior and important edge cases."""
    assert percentile([10, 20, 30, 40], 50) == 20
    assert percentile([10, 20, 30, 40], 95) == 40

    target = LoadTarget(min_success_rate=0.95, max_p95_latency_ms=150)
    passing = synthetic_sample(
        10,
        10,
        succeeded=95,
        failed=5,
        dropped=0,
        latency_ms=100,
    )
    failing = synthetic_sample(
        20,
        10,
        succeeded=90,
        failed=5,
        dropped=5,
        latency_ms=100,
    )
    report = analyze_load_test((failing, passing), target)
    assert report.sustainable_rps == 10.0
    assert report.first_failing_concurrency == 20

    profile = InstanceProfile(max_rps=100, target_utilization=0.8)
    plan = plan_capacity(
        150,
        profile,
        CapacityPolicy(reserve_fraction=0.25, tolerated_instance_failures=1),
    )
    assert plan.healthy_instances == 2
    assert plan.reserve_instances == 1
    assert plan.total_instances == 3
    assert plan.surviving_safe_rps == 160.0

    invalid_calls = (
        lambda: percentile([], 95),
        lambda: percentile([1], 0),
        lambda: synthetic_sample(
            1, 1, succeeded=0, failed=0, dropped=0, latency_ms=1
        ),
        lambda: analyze_load_test((passing, passing), target),
        lambda: InstanceProfile(max_rps=float("inf"), target_utilization=0.8),
        lambda: CapacityPolicy(
            reserve_fraction=1.0, tolerated_instance_failures=1
        ),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def demo() -> None:
    """Analyze a ramp and make a deterministic release decision."""
    budget = ResilienceBudget(
        max_error_rate=0.04,
        max_p95_latency_ms=250,
        min_failover_headroom_fraction=0.25,
        max_recovery_seconds=120,
    )
    samples: Sequence[LoadSample] = (
        synthetic_sample(
            20,
            10,
            succeeded=990,
            failed=5,
            dropped=5,
            latency_ms=90,
        ),
        synthetic_sample(
            40,
            10,
            succeeded=1930,
            failed=20,
            dropped=50,
            latency_ms=140,
        ),
        synthetic_sample(
            80,
            10,
            succeeded=2500,
            failed=200,
            dropped=300,
            latency_ms=800,
        ),
    )
    report = analyze_load_test(samples, budget.load_target)
    plan = plan_capacity(
        180,
        InstanceProfile(max_rps=100, target_utilization=0.8),
        CapacityPolicy(reserve_fraction=0.25, tolerated_instance_failures=1),
    )
    decision = evaluate_resilience(
        report, plan, budget, recovery_seconds=80
    )

    print(f"Sustainable load: {report.sustainable_rps:.0f} requests/second")
    print(f"First failing concurrency: {report.first_failing_concurrency}")
    print(
        f"Capacity plan: {plan.total_instances} instances; "
        f"{plan.headroom_fraction:.1%} failover headroom"
    )
    print(f"Release approved: {decision.approved}")
    print(f"Reason: {decision.reasons[0]}")


# Practice exercises
# ------------------
# 1. Add p50 and p99 latency values to LoadTestReport. Keep percentile
#    calculation independent from reporting so it remains easy to test.
# 2. Write generate_ramp(start, stop, step) that returns concurrency stages and
#    rejects a non-positive step or a range that cannot advance.
# 3. Add an hourly_instance_cost to InstanceProfile and calculate the monthly
#    cost of a capacity plan without weakening its failure guarantees.
# 4. Model a recovery drill as timestamped states and calculate time to detect,
#    time to mitigate, and time to recover.

# Solutions / clear guidance
# --------------------------
# 1. Reuse percentile for each sample, then report the worst value across the
#    ramp or retain per-stage values. Document which interpretation you choose.
# 2. Validate all three integers first. range(start, stop + 1, step) is useful,
#    but decide explicitly whether stop must be included when it is not aligned.
# 3. Validate a finite, non-negative cost. Multiply total_instances by the
#    documented billing hours; show capacity and cost together so price cannot
#    silently replace the resilience requirement.
# 4. Use timezone-aware datetimes and legal state transitions. Durations are
#    differences between adjacent milestones; reject missing or reversed events.

# Expert challenge / mini-project
# --------------------------------
# Build a local capacity-planning CLI around these primitives:
#
# * Read authorized load-test observations from JSON Lines and reject malformed
#   records with line numbers. Keep raw observations immutable.
# * Compare a baseline and candidate release at identical load stages. Flag
#   regressions in throughput, error rate, and p95/p99 latency.
# * Forecast ordinary and seasonal peak demand, then produce normal and
#   single-instance-failure capacity plans with cost estimates.
# * Run a simulated dependency outage and record detection, mitigation, and
#   recovery times against the resilience budget.
# * Emit a machine-readable decision plus a human explanation. Add tests for
#   empty samples, all-dropped traffic, percentile boundaries, saturation,
#   recovery exactly on the limit, and a concurrent update to the input file.
#
# Never aim this tool at a third-party or production service without explicit
# authorization. A strong solution defaults to offline replay, caps concurrency
# and request rate, uses short timeouts, redacts secrets, and provides an
# immediate stop control. Keep measurement, policy, persistence, and CLI
# adapters in separate modules.


if __name__ == "__main__":
    run_self_checks()
    demo()
