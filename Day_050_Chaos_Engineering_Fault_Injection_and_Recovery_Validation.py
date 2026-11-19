"""Day 50: Chaos engineering, fault injection, and recovery validation.

Chaos engineering is controlled experimentation, not random breakage. Start with a
measurable steady-state hypothesis, constrain the blast radius, inject one known
fault, stop when an abort condition is crossed, and prove that the system recovers.

Learning goals
--------------
1. Express steady state with observable error-rate and latency thresholds.
2. Design bounded fault experiments with environment and blast-radius guards.
3. Separate fault injection from measurement and experiment orchestration.
4. Abort safely and validate recovery with consecutive healthy observations.
5. Produce an auditable report that distinguishes completion, abort, and failure
   to recover.

All examples are deterministic, in-memory simulations. They do not modify the
operating system, contact a network, or disrupt a real service. Only run real
fault experiments on systems you own or are explicitly authorized to test. Use
a staging environment first, obtain approval, announce the experiment, monitor
it continuously, and keep a tested emergency stop available.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Callable, Iterable, Sequence


Number = int | float


def finite_number(
    value: Number,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return a finite float after rejecting booleans and invalid bounds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    checked = float(value)
    if not math.isfinite(checked):
        raise ValueError(f"{field_name} must be finite")
    if minimum is not None and checked < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    if maximum is not None and checked > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}")
    return checked


def positive_int(value: int, field_name: str) -> int:
    """Validate an integer count without accepting bool as an integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def percentile(values: Iterable[Number], percent: Number) -> float:
    """Return a nearest-rank percentile for non-negative measurements."""
    checked_percent = finite_number(
        percent, "percent", minimum=0.000_001, maximum=100.0
    )
    checked_values = [
        finite_number(value, "measurement", minimum=0.0) for value in values
    ]
    if not checked_values:
        raise ValueError("at least one measurement is required")
    checked_values.sort()
    rank = math.ceil(checked_percent / 100.0 * len(checked_values))
    return checked_values[rank - 1]


class Phase(str, Enum):
    BASELINE = "baseline"
    INJECTION = "injection"
    RECOVERY = "recovery"


class FaultKind(str, Enum):
    LATENCY = "latency"
    DEPENDENCY_ERROR = "dependency_error"


class Outcome(str, Enum):
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED_TO_RECOVER = "failed_to_recover"


@dataclass(frozen=True, slots=True)
class Observation:
    """One fixed-duration service measurement window."""

    phase: Phase
    window: int
    requests: int
    successes: int
    latencies_ms: tuple[float, ...]

    def __post_init__(self) -> None:
        positive_int(self.window, "window")
        positive_int(self.requests, "requests")
        if isinstance(self.successes, bool) or not isinstance(self.successes, int):
            raise TypeError("successes must be an integer")
        if not 0 <= self.successes <= self.requests:
            raise ValueError("successes must be between 0 and requests")
        if len(self.latencies_ms) != self.requests:
            raise ValueError("latencies_ms must contain one value per request")
        checked = tuple(
            finite_number(value, "latency", minimum=0.0)
            for value in self.latencies_ms
        )
        object.__setattr__(self, "latencies_ms", checked)

    @property
    def failures(self) -> int:
        return self.requests - self.successes

    @property
    def error_rate(self) -> float:
        return self.failures / self.requests

    @property
    def p95_latency_ms(self) -> float:
        return percentile(self.latencies_ms, 95.0)


@dataclass(frozen=True, slots=True)
class HealthThreshold:
    """A measurable steady-state or emergency-abort boundary."""

    max_error_rate: float
    max_p95_latency_ms: float
    min_requests: int = 1

    def __post_init__(self) -> None:
        error_rate = finite_number(
            self.max_error_rate,
            "max_error_rate",
            minimum=0.0,
            maximum=1.0,
        )
        latency = finite_number(
            self.max_p95_latency_ms,
            "max_p95_latency_ms",
            minimum=0.000_001,
        )
        positive_int(self.min_requests, "min_requests")
        object.__setattr__(self, "max_error_rate", error_rate)
        object.__setattr__(self, "max_p95_latency_ms", latency)

    def accepts(self, observation: Observation) -> bool:
        return (
            observation.requests >= self.min_requests
            and observation.error_rate <= self.max_error_rate
            and observation.p95_latency_ms <= self.max_p95_latency_ms
        )

    def violations(self, observation: Observation) -> tuple[str, ...]:
        reasons: list[str] = []
        if observation.requests < self.min_requests:
            reasons.append("request volume is below the measurement minimum")
        if observation.error_rate > self.max_error_rate:
            reasons.append("error rate crossed the allowed threshold")
        if observation.p95_latency_ms > self.max_p95_latency_ms:
            reasons.append("p95 latency crossed the allowed threshold")
        return tuple(reasons)


@dataclass(frozen=True, slots=True)
class FaultSpec:
    """One bounded fault applied to a known set of simulated targets."""

    kind: FaultKind
    intensity: float
    duration_windows: int
    target_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        intensity = finite_number(
            self.intensity, "intensity", minimum=0.000_001, maximum=1.0
        )
        positive_int(self.duration_windows, "duration_windows")
        if not self.target_ids:
            raise ValueError("at least one target is required")
        normalized = tuple(target.strip() for target in self.target_ids)
        if any(not target for target in normalized):
            raise ValueError("target IDs must not be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("target IDs must be unique")
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "target_ids", normalized)


@dataclass(frozen=True, slots=True)
class SafetyPolicy:
    """Organization-level limits that an experiment cannot override."""

    allowed_environments: frozenset[str]
    max_blast_radius_fraction: float
    max_fault_windows: int

    def __post_init__(self) -> None:
        environments = frozenset(
            environment.strip() for environment in self.allowed_environments
        )
        if not environments or any(not item for item in environments):
            raise ValueError("allowed_environments must contain non-blank names")
        radius = finite_number(
            self.max_blast_radius_fraction,
            "max_blast_radius_fraction",
            minimum=0.000_001,
            maximum=1.0,
        )
        positive_int(self.max_fault_windows, "max_fault_windows")
        object.__setattr__(self, "allowed_environments", environments)
        object.__setattr__(self, "max_blast_radius_fraction", radius)


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    """Reviewed inputs for a hypothesis-driven experiment."""

    experiment_id: str
    environment: str
    hypothesis: str
    population_size: int
    sample_interval_seconds: float
    baseline_windows: int
    recovery_confirmation_windows: int
    max_recovery_windows: int
    steady_state: HealthThreshold
    abort_threshold: HealthThreshold
    fault: FaultSpec

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be blank")
        if not self.environment.strip():
            raise ValueError("environment must not be blank")
        if not self.hypothesis.strip():
            raise ValueError("hypothesis must not be blank")
        positive_int(self.population_size, "population_size")
        interval = finite_number(
            self.sample_interval_seconds,
            "sample_interval_seconds",
            minimum=0.000_001,
        )
        positive_int(self.baseline_windows, "baseline_windows")
        positive_int(
            self.recovery_confirmation_windows,
            "recovery_confirmation_windows",
        )
        positive_int(self.max_recovery_windows, "max_recovery_windows")
        if self.recovery_confirmation_windows > self.max_recovery_windows:
            raise ValueError(
                "recovery_confirmation_windows cannot exceed max_recovery_windows"
            )
        if len(self.fault.target_ids) > self.population_size:
            raise ValueError("fault targets cannot exceed the population")
        if self.abort_threshold.max_error_rate < self.steady_state.max_error_rate:
            raise ValueError("abort error threshold cannot be stricter than steady state")
        if (
            self.abort_threshold.max_p95_latency_ms
            < self.steady_state.max_p95_latency_ms
        ):
            raise ValueError(
                "abort latency threshold cannot be stricter than steady state"
            )
        object.__setattr__(self, "sample_interval_seconds", interval)

    @property
    def blast_radius_fraction(self) -> float:
        return len(self.fault.target_ids) / self.population_size


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    experiment_id: str
    outcome: Outcome
    hypothesis_supported: bool
    observations: tuple[Observation, ...]
    abort_reasons: tuple[str, ...]
    recovery_seconds: float | None

    @property
    def injected_windows(self) -> int:
        return sum(
            observation.phase is Phase.INJECTION
            for observation in self.observations
        )


class SafetyViolation(ValueError):
    """The experiment violates an organization-level safety boundary."""


class PreconditionFailed(RuntimeError):
    """The service was not healthy enough to begin an experiment."""


Observer = Callable[[Phase, int, ExperimentPlan], Observation]


class ExperimentRunner:
    """Run baseline, injection, and recovery phases through an observer adapter."""

    def __init__(self, safety_policy: SafetyPolicy) -> None:
        self.safety_policy = safety_policy

    def validate_safety(self, plan: ExperimentPlan) -> None:
        if plan.environment not in self.safety_policy.allowed_environments:
            raise SafetyViolation("environment is not approved for fault injection")
        if plan.blast_radius_fraction > (
            self.safety_policy.max_blast_radius_fraction
        ):
            raise SafetyViolation("fault blast radius exceeds the safety policy")
        if plan.fault.duration_windows > self.safety_policy.max_fault_windows:
            raise SafetyViolation("fault duration exceeds the safety policy")

    @staticmethod
    def _observe(
        observer: Observer,
        phase: Phase,
        window: int,
        plan: ExperimentPlan,
    ) -> Observation:
        observation = observer(phase, window, plan)
        if observation.phase is not phase or observation.window != window:
            raise ValueError("observer returned mismatched phase or window metadata")
        return observation

    def run(self, plan: ExperimentPlan, observer: Observer) -> ExperimentReport:
        """Execute a safe experiment and always enter recovery observation."""
        self.validate_safety(plan)
        observations: list[Observation] = []

        for window in range(1, plan.baseline_windows + 1):
            observation = self._observe(
                observer, Phase.BASELINE, window, plan
            )
            observations.append(observation)
            violations = plan.steady_state.violations(observation)
            if violations:
                joined = "; ".join(violations)
                raise PreconditionFailed(f"baseline is not healthy: {joined}")

        abort_reasons: tuple[str, ...] = ()
        for window in range(1, plan.fault.duration_windows + 1):
            observation = self._observe(
                observer, Phase.INJECTION, window, plan
            )
            observations.append(observation)
            abort_reasons = plan.abort_threshold.violations(observation)
            if abort_reasons:
                break

        consecutive_healthy = 0
        recovery_seconds: float | None = None
        for window in range(1, plan.max_recovery_windows + 1):
            observation = self._observe(
                observer, Phase.RECOVERY, window, plan
            )
            observations.append(observation)
            if plan.steady_state.accepts(observation):
                consecutive_healthy += 1
            else:
                consecutive_healthy = 0
            if consecutive_healthy >= plan.recovery_confirmation_windows:
                recovery_seconds = (
                    window * plan.sample_interval_seconds
                )
                break

        if recovery_seconds is None:
            outcome = Outcome.FAILED_TO_RECOVER
        elif abort_reasons:
            outcome = Outcome.ABORTED
        else:
            outcome = Outcome.COMPLETED

        return ExperimentReport(
            experiment_id=plan.experiment_id,
            outcome=outcome,
            hypothesis_supported=outcome is Outcome.COMPLETED,
            observations=tuple(observations),
            abort_reasons=abort_reasons,
            recovery_seconds=recovery_seconds,
        )


@dataclass(frozen=True, slots=True)
class CheckoutSimulator:
    """A deterministic service model used only for safe local teaching."""

    requests_per_window: int = 100
    base_latency_ms: float = 80.0
    injected_latency_ms: float = 320.0
    unhealthy_recovery_windows: int = 0

    def __post_init__(self) -> None:
        positive_int(self.requests_per_window, "requests_per_window")
        base = finite_number(
            self.base_latency_ms, "base_latency_ms", minimum=0.0
        )
        injected = finite_number(
            self.injected_latency_ms,
            "injected_latency_ms",
            minimum=0.0,
        )
        if (
            isinstance(self.unhealthy_recovery_windows, bool)
            or not isinstance(self.unhealthy_recovery_windows, int)
        ):
            raise TypeError("unhealthy_recovery_windows must be an integer")
        if self.unhealthy_recovery_windows < 0:
            raise ValueError("unhealthy_recovery_windows must not be negative")
        object.__setattr__(self, "base_latency_ms", base)
        object.__setattr__(self, "injected_latency_ms", injected)

    def observe(
        self,
        phase: Phase,
        window: int,
        plan: ExperimentPlan,
    ) -> Observation:
        affected = math.ceil(
            self.requests_per_window * plan.blast_radius_fraction
        )
        successes = self.requests_per_window
        latencies = [self.base_latency_ms] * self.requests_per_window

        if phase is Phase.INJECTION:
            if plan.fault.kind is FaultKind.LATENCY:
                extra = self.injected_latency_ms * plan.fault.intensity
                for index in range(affected):
                    latencies[-(index + 1)] += extra
            else:
                failures = math.ceil(affected * plan.fault.intensity)
                successes -= failures
                for index in range(failures):
                    latencies[-(index + 1)] += self.injected_latency_ms

        if (
            phase is Phase.RECOVERY
            and window <= self.unhealthy_recovery_windows
        ):
            successes -= affected
            for index in range(affected):
                latencies[-(index + 1)] += self.injected_latency_ms

        return Observation(
            phase=phase,
            window=window,
            requests=self.requests_per_window,
            successes=successes,
            latencies_ms=tuple(latencies),
        )


def build_example_plan(
    *,
    kind: FaultKind = FaultKind.LATENCY,
    intensity: float = 0.75,
) -> ExperimentPlan:
    """Create a reviewed staging plan with a ten-percent blast radius."""
    return ExperimentPlan(
        experiment_id="checkout-dependency-2026-11-19",
        environment="staging",
        hypothesis=(
            "checkout stays below abort limits and returns to steady state "
            "within three observation windows"
        ),
        population_size=50,
        sample_interval_seconds=10.0,
        baseline_windows=2,
        recovery_confirmation_windows=2,
        max_recovery_windows=5,
        steady_state=HealthThreshold(
            max_error_rate=0.02,
            max_p95_latency_ms=150.0,
            min_requests=50,
        ),
        abort_threshold=HealthThreshold(
            max_error_rate=0.08,
            max_p95_latency_ms=500.0,
            min_requests=50,
        ),
        fault=FaultSpec(
            kind=kind,
            intensity=intensity,
            duration_windows=3,
            target_ids=("worker-01", "worker-02", "worker-03", "worker-04", "worker-05"),
        ),
    )


def run_self_checks() -> None:
    """Exercise normal, abort, recovery, and invalid-input paths."""
    assert percentile([10, 20, 30, 40], 50) == 20
    assert percentile([10, 20, 30, 40], 95) == 40

    safety = SafetyPolicy(
        allowed_environments=frozenset({"test", "staging"}),
        max_blast_radius_fraction=0.10,
        max_fault_windows=3,
    )
    runner = ExperimentRunner(safety)

    normal = runner.run(build_example_plan(), CheckoutSimulator().observe)
    assert normal.outcome is Outcome.COMPLETED
    assert normal.hypothesis_supported is True
    assert normal.injected_windows == 3
    assert normal.recovery_seconds == 20.0
    assert len(normal.observations) == 7

    aborted = runner.run(
        build_example_plan(
            kind=FaultKind.DEPENDENCY_ERROR,
            intensity=1.0,
        ),
        CheckoutSimulator().observe,
    )
    assert aborted.outcome is Outcome.ABORTED
    assert aborted.injected_windows == 1
    assert aborted.abort_reasons == (
        "error rate crossed the allowed threshold",
    )
    assert aborted.recovery_seconds == 20.0

    failed_recovery = runner.run(
        build_example_plan(),
        CheckoutSimulator(unhealthy_recovery_windows=5).observe,
    )
    assert failed_recovery.outcome is Outcome.FAILED_TO_RECOVER
    assert failed_recovery.recovery_seconds is None

    unhealthy_baseline = CheckoutSimulator()

    def bad_baseline(
        phase: Phase,
        window: int,
        plan: ExperimentPlan,
    ) -> Observation:
        observation = unhealthy_baseline.observe(phase, window, plan)
        if phase is Phase.BASELINE:
            return Observation(
                phase=phase,
                window=window,
                requests=100,
                successes=90,
                latencies_ms=(80.0,) * 100,
            )
        return observation

    try:
        runner.run(build_example_plan(), bad_baseline)
    except PreconditionFailed:
        pass
    else:
        raise AssertionError("an unhealthy baseline must prevent injection")

    production_plan = replace(build_example_plan(), environment="production")
    try:
        runner.run(production_plan, CheckoutSimulator().observe)
    except SafetyViolation:
        pass
    else:
        raise AssertionError("an unapproved environment must be rejected")

    invalid_calls = (
        lambda: percentile([], 95),
        lambda: HealthThreshold(
            max_error_rate=float("nan"),
            max_p95_latency_ms=100,
        ),
        lambda: FaultSpec(
            FaultKind.LATENCY,
            0.5,
            1,
            ("worker-01", "worker-01"),
        ),
        lambda: Observation(
            Phase.BASELINE,
            1,
            2,
            2,
            (10.0,),
        ),
        lambda: SafetyPolicy(
            frozenset({"staging"}),
            0.0,
            1,
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
    """Run a safe synthetic latency experiment and print its evidence."""
    safety = SafetyPolicy(
        allowed_environments=frozenset({"test", "staging"}),
        max_blast_radius_fraction=0.10,
        max_fault_windows=3,
    )
    plan = build_example_plan()
    report = ExperimentRunner(safety).run(
        plan,
        CheckoutSimulator().observe,
    )

    injection = [
        item
        for item in report.observations
        if item.phase is Phase.INJECTION
    ]
    worst_p95 = max(item.p95_latency_ms for item in injection)

    print(f"Experiment: {report.experiment_id}")
    print(f"Blast radius: {plan.blast_radius_fraction:.0%}")
    print(f"Outcome: {report.outcome.value}")
    print(f"Worst injected p95: {worst_p95:.0f} ms")
    print(f"Recovery validated after: {report.recovery_seconds:.0f} seconds")
    print(f"Hypothesis supported: {report.hypothesis_supported}")


# Practice exercises
# ------------------
# 1. Add a throughput field to Observation and include a minimum throughput in
#    HealthThreshold. Write boundary tests for a value exactly on the limit.
# 2. Add a JITTER fault whose extra latency is deterministic for a supplied seed.
#    Keep the observer repeatable so a failed test can be reproduced.
# 3. Add an approval record containing reviewer, ticket, and expiration time.
#    Reject expired or missing approval before the baseline phase begins.
# 4. Export ExperimentReport to JSON without serializing Enum objects or NaN.
#    Include the plan hash so operators can identify exactly what was reviewed.
# 5. Compare a control cohort with the injected cohort and flag a regression only
#    when the difference exceeds a documented tolerance.

# Solutions / clear guidance
# --------------------------
# 1. Store the window duration, calculate requests / duration, and extend both
#    accepts() and violations(). Equality should pass if the rule says "minimum."
# 2. Use random.Random(seed), never the module-level generator. Precompute the
#    sequence or retain the seed in the report so the experiment is replayable.
# 3. Model approval as an immutable dataclass with timezone-aware datetimes.
#    Validate it in validate_safety(), before calling the observer even once.
# 4. Build a plain dictionary explicitly, use enum.value, reject non-finite
#    numbers, and call json.dumps(..., allow_nan=False, sort_keys=True).
# 5. Measure both cohorts in the same windows. Compare error rate, p95 latency,
#    and throughput separately; do not hide an important regression in one score.

# Expert challenge / mini-project
# --------------------------------
# Build a local "chaos experiment controller" around these primitives:
#
# * Load a reviewed YAML or JSON plan, validate its schema, approval, environment,
#   target allow-list, maximum duration, and blast radius before any action.
# * Implement adapters for a fake service only. One adapter injects latency,
#   errors, and process restarts; another collects timestamped observations.
# * Require a healthy baseline, stream live abort checks, make emergency stop
#   idempotent, and enter recovery monitoring even after an exception.
# * Persist an append-only audit trail containing plan hash, operator, phase
#   transitions, observations, abort reason, cleanup result, and final outcome.
# * Add unit tests, state-machine tests, and a crash-recovery test that resumes
#   cleanup after the controller restarts.
# * Generate a Markdown report comparing the hypothesis with evidence and listing
#   follow-up work with owners and due dates.
#
# A strong solution uses ports and adapters: pure policy code must not know how a
# fault is injected. Default to a simulator, require an explicit opt-in for any
# external adapter, prevent production targets, cap time and concurrency, redact
# secrets, and test the emergency stop before expanding the blast radius.


if __name__ == "__main__":
    run_self_checks()
    demo()
