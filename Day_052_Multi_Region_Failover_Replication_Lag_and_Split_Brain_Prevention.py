"""Day 52: Multi-region failover, replication lag, and split-brain prevention.

High availability is not "run the same service twice." During a partition, two
regions can each believe the other has failed. Safe failover requires fresh
evidence, bounded replication lag, a quorum decision, and fencing that makes an
old primary unable to keep writing.

This lesson builds an in-memory failover control plane with Python's standard
library. It measures log-position lag, records one vote per witness and epoch,
evaluates promotion safety, and rejects writes carrying stale fencing tokens.
No network, cloud account, database, or production service is contacted.

Learning goals
--------------
1. Distinguish replication lag, recovery point risk, and service health.
2. Require fresh observations and quorum evidence before promotion.
3. Explain how quorum intersection limits conflicting promotions.
4. Fence a former primary with monotonically increasing epochs.
5. Treat failover and failback as reviewed, observable state transitions.

Teaching notes
--------------
- Multi-region replication copies state; it does not decide which copy may
  accept writes. A control plane must make that authority explicit.
- Replication lag is measured against a durable source position, not wall-clock
  guesses. A candidate at LSN 995 when the durable source is 1000 may lose five
  records if promoted.
- Health and freshness are separate. A green observation from five minutes ago
  is unsafe when the decision budget is thirty seconds.
- A majority quorum is floor(N / 2) + 1. Any two majorities overlap, so a
  witness that grants at most one vote per epoch cannot support two different
  candidates in the same epoch.
- Quorum alone is not enough. Membership must be controlled, witnesses must use
  durable vote records, and the data plane must enforce the chosen epoch.
- A fencing token is a monotonically increasing generation or epoch attached to
  every mutating request. Storage rejects tokens older than its installed
  authority, so a paused or partitioned former primary cannot resume writes.
- "Primary unreachable" is not equivalent to "primary dead." Before promotion,
  isolate or fence the former primary whenever possible and refuse promotion if
  fresh evidence still shows it writable.
- Synchronous replication can reduce data loss but increases write latency and
  can reduce availability. The policy must make this trade-off explicit.
- Failback is another migration, not an automatic reversal. Reconcile data,
  catch up the former region, verify invariants, drain traffic, and issue a new
  epoch through the same safety path.
- Production systems also need authenticated control-plane calls, durable
  consensus, controlled membership changes, audit logs, rehearsed runbooks,
  and independent failure domains. This model teaches invariants; it is not a
  distributed consensus implementation.

Run this file with Python 3.10+. The self-checks cover exact lag boundaries,
stale and future observations, insufficient quorum, conflicting votes, an
unfenced primary, unsafe writable standbys, and stale-token rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Iterable, Mapping


class FailoverError(Exception):
    """Base class for expected failover failures."""


class ValidationError(FailoverError):
    """Input is malformed or internally inconsistent."""


class UnsafeFailover(FailoverError):
    """A promotion does not satisfy the safety policy."""


class StaleFenceToken(FailoverError):
    """A write used authority older than the installed control state."""


def clean_name(value: str, field_name: str) -> str:
    """Return a bounded non-blank identifier."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    result = value.strip()
    if not result:
        raise ValidationError(f"{field_name} must not be blank")
    if len(result) > 100:
        raise ValidationError(f"{field_name} is too long")
    return result


def non_negative_int(value: int, field_name: str) -> int:
    """Reject bool and return a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValidationError(f"{field_name} must not be negative")
    return value


def positive_int(value: int, field_name: str) -> int:
    """Return a strictly positive integer."""
    result = non_negative_int(value, field_name)
    if result == 0:
        raise ValidationError(f"{field_name} must be positive")
    return result


def aware(value: datetime, field_name: str) -> datetime:
    """Reject non-datetime and timezone-naive timestamps."""
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError(f"{field_name} must be timezone-aware")
    return value


@dataclass(frozen=True, slots=True)
class FailoverPolicy:
    """Safety limits applied before a standby can become primary."""

    max_replication_lag_records: int
    max_observation_age: timedelta

    def __post_init__(self) -> None:
        non_negative_int(
            self.max_replication_lag_records,
            "max_replication_lag_records",
        )
        if not isinstance(self.max_observation_age, timedelta):
            raise TypeError("max_observation_age must be a timedelta")
        if self.max_observation_age <= timedelta(0):
            raise ValidationError("max_observation_age must be positive")


@dataclass(frozen=True, slots=True)
class RegionObservation:
    """One control-plane view of a region at a log position."""

    region: str
    applied_lsn: int
    healthy: bool
    writable: bool
    observed_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "region", clean_name(self.region, "region"))
        non_negative_int(self.applied_lsn, "applied_lsn")
        if not isinstance(self.healthy, bool):
            raise TypeError("healthy must be a bool")
        if not isinstance(self.writable, bool):
            raise TypeError("writable must be a bool")
        object.__setattr__(
            self,
            "observed_at",
            aware(self.observed_at, "observed_at"),
        )


@dataclass(frozen=True, slots=True)
class FencingToken:
    """Write authority for exactly one region and one epoch."""

    epoch: int
    region: str

    def __post_init__(self) -> None:
        positive_int(self.epoch, "epoch")
        object.__setattr__(self, "region", clean_name(self.region, "region"))


@dataclass(frozen=True, slots=True)
class FailoverAssessment:
    """Explainable evidence for or against one promotion."""

    allowed: bool
    candidate: str
    next_epoch: int
    replication_lag_records: int | None
    votes: int
    quorum: int
    reasons: tuple[str, ...]


class WitnessRegistry:
    """Durably model one candidate vote per witness and epoch.

    Real witnesses would persist votes independently and communicate over
    authenticated channels. This in-memory registry makes the safety invariant
    visible and deterministic for the lesson.
    """

    def __init__(self, members: Iterable[str]) -> None:
        normalized = frozenset(clean_name(member, "witness") for member in members)
        if not normalized:
            raise ValidationError("at least one witness is required")
        self._members = normalized
        self._votes: dict[int, dict[str, str]] = {}

    @property
    def members(self) -> frozenset[str]:
        return self._members

    @property
    def quorum_size(self) -> int:
        return len(self._members) // 2 + 1

    def grant(self, epoch: int, candidate: str, witness: str) -> bool:
        """Grant idempotently, but never switch a witness in the same epoch."""
        epoch = positive_int(epoch, "epoch")
        candidate = clean_name(candidate, "candidate")
        witness = clean_name(witness, "witness")
        if witness not in self._members:
            raise ValidationError(f"unknown witness: {witness}")

        epoch_votes = self._votes.setdefault(epoch, {})
        previous = epoch_votes.get(witness)
        if previous is None:
            epoch_votes[witness] = candidate
            return True
        return previous == candidate

    def votes_for(self, epoch: int, candidate: str) -> frozenset[str]:
        """Return witnesses that voted for candidate in epoch."""
        epoch = positive_int(epoch, "epoch")
        candidate = clean_name(candidate, "candidate")
        return frozenset(
            witness
            for witness, voted_candidate in self._votes.get(epoch, {}).items()
            if voted_candidate == candidate
        )


class FailoverController:
    """Evaluate evidence and issue monotonically increasing write authority."""

    def __init__(
        self,
        current_primary: str,
        current_epoch: int,
        policy: FailoverPolicy,
        witnesses: WitnessRegistry,
    ) -> None:
        self._current_primary = clean_name(current_primary, "current_primary")
        self._current_epoch = positive_int(current_epoch, "current_epoch")
        if not isinstance(policy, FailoverPolicy):
            raise TypeError("policy must be a FailoverPolicy")
        if not isinstance(witnesses, WitnessRegistry):
            raise TypeError("witnesses must be a WitnessRegistry")
        self._policy = policy
        self._witnesses = witnesses

    @property
    def current_primary(self) -> str:
        return self._current_primary

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    def current_token(self) -> FencingToken:
        return FencingToken(self._current_epoch, self._current_primary)

    def assess(
        self,
        candidate: str,
        durable_source_lsn: int,
        observations: Iterable[RegionObservation],
        now: datetime,
    ) -> FailoverAssessment:
        """Return a decision without changing primary authority."""
        candidate = clean_name(candidate, "candidate")
        durable_source_lsn = non_negative_int(
            durable_source_lsn,
            "durable_source_lsn",
        )
        now = aware(now, "now")

        by_region: dict[str, RegionObservation] = {}
        for observation in observations:
            if not isinstance(observation, RegionObservation):
                raise TypeError("observations must contain RegionObservation values")
            if observation.region in by_region:
                raise ValidationError(
                    f"duplicate observation for region: {observation.region}"
                )
            by_region[observation.region] = observation

        reasons: list[str] = []
        next_epoch = self._current_epoch + 1
        votes = len(self._witnesses.votes_for(next_epoch, candidate))
        quorum = self._witnesses.quorum_size

        if candidate == self._current_primary:
            reasons.append("candidate is already the recorded primary")

        primary = by_region.get(self._current_primary)
        if primary is None:
            reasons.append("current primary has no isolation observation")
        else:
            primary_age = now - primary.observed_at
            if primary_age < timedelta(0):
                reasons.append("current primary observation is from the future")
            elif primary_age > self._policy.max_observation_age:
                reasons.append("current primary isolation observation is stale")
            elif primary.healthy or primary.writable:
                reasons.append("current primary is not proven isolated and read-only")

        standby = by_region.get(candidate)
        lag: int | None = None
        if standby is None:
            reasons.append("candidate has no observation")
        else:
            candidate_age = now - standby.observed_at
            if candidate_age < timedelta(0):
                reasons.append("candidate observation is from the future")
            elif candidate_age > self._policy.max_observation_age:
                reasons.append("candidate observation is stale")
            if not standby.healthy:
                reasons.append("candidate is not healthy")
            if standby.writable:
                reasons.append("candidate is already writable before promotion")
            if standby.applied_lsn > durable_source_lsn:
                reasons.append("candidate LSN is ahead of the durable source position")
            else:
                lag = durable_source_lsn - standby.applied_lsn
                if lag > self._policy.max_replication_lag_records:
                    reasons.append("candidate replication lag exceeds policy")

        for region, observation in by_region.items():
            if (
                region not in {self._current_primary, candidate}
                and observation.writable
            ):
                reasons.append(f"another standby is writable: {region}")

        if votes < quorum:
            reasons.append("candidate does not have a witness quorum")

        return FailoverAssessment(
            allowed=not reasons,
            candidate=candidate,
            next_epoch=next_epoch,
            replication_lag_records=lag,
            votes=votes,
            quorum=quorum,
            reasons=tuple(reasons) if reasons else ("all failover gates passed",),
        )

    def promote(
        self,
        candidate: str,
        durable_source_lsn: int,
        observations: Iterable[RegionObservation],
        now: datetime,
    ) -> FencingToken:
        """Promote only after re-evaluating all evidence."""
        assessment = self.assess(
            candidate,
            durable_source_lsn,
            observations,
            now,
        )
        if not assessment.allowed:
            raise UnsafeFailover("; ".join(assessment.reasons))
        self._current_epoch = assessment.next_epoch
        self._current_primary = assessment.candidate
        return FencingToken(self._current_epoch, self._current_primary)


class FencedLedger:
    """Small data plane that rejects obsolete or wrong-region authority."""

    def __init__(self) -> None:
        self._authority: FencingToken | None = None
        self._entries: list[str] = []

    @property
    def entries(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def install_authority(self, token: FencingToken) -> None:
        """Install only monotonically newer control-plane state."""
        if not isinstance(token, FencingToken):
            raise TypeError("token must be a FencingToken")
        if self._authority is not None:
            if token.epoch < self._authority.epoch:
                raise StaleFenceToken("cannot install an older epoch")
            if (
                token.epoch == self._authority.epoch
                and token.region != self._authority.region
            ):
                raise StaleFenceToken(
                    "one epoch cannot authorize two different regions"
                )
        self._authority = token

    def append(self, token: FencingToken, value: str) -> None:
        """Accept a write only from the installed region and exact epoch."""
        if not isinstance(token, FencingToken):
            raise TypeError("token must be a FencingToken")
        value = clean_name(value, "value")
        if self._authority is None:
            raise StaleFenceToken("no write authority is installed")
        if token != self._authority:
            raise StaleFenceToken(
                "write token does not match the installed region and epoch"
            )
        self._entries.append(value)


def utc(hour: int, minute: int = 0, second: int = 0) -> datetime:
    """Build deterministic timestamps used by examples and checks."""
    return datetime(
        2026,
        11,
        21,
        hour,
        minute,
        second,
        tzinfo=timezone.utc,
    )


def safe_observations(
    now: datetime,
    candidate_lsn: int = 995,
) -> tuple[RegionObservation, ...]:
    """Build evidence showing an isolated primary and healthy standby."""
    return (
        RegionObservation(
            "us-east",
            applied_lsn=1000,
            healthy=False,
            writable=False,
            observed_at=now - timedelta(seconds=5),
        ),
        RegionObservation(
            "eu-west",
            applied_lsn=candidate_lsn,
            healthy=True,
            writable=False,
            observed_at=now - timedelta(seconds=3),
        ),
        RegionObservation(
            "ap-south",
            applied_lsn=990,
            healthy=True,
            writable=False,
            observed_at=now - timedelta(seconds=4),
        ),
    )


def build_example() -> tuple[
    FailoverController,
    WitnessRegistry,
    FencedLedger,
    datetime,
]:
    """Create a five-witness control plane and its currently fenced ledger."""
    now = utc(12)
    witnesses = WitnessRegistry({"w1", "w2", "w3", "w4", "w5"})
    policy = FailoverPolicy(
        max_replication_lag_records=10,
        max_observation_age=timedelta(seconds=30),
    )
    controller = FailoverController(
        current_primary="us-east",
        current_epoch=7,
        policy=policy,
        witnesses=witnesses,
    )
    ledger = FencedLedger()
    ledger.install_authority(controller.current_token())
    return controller, witnesses, ledger, now


def run_self_checks() -> None:
    """Exercise safety gates, quorum intersection, and data-plane fencing."""
    controller, witnesses, ledger, now = build_example()
    old_primary_token = controller.current_token()

    for witness in ("w1", "w2", "w3"):
        assert witnesses.grant(8, "eu-west", witness)
    assert witnesses.grant(8, "eu-west", "w1")  # idempotent retry
    assert not witnesses.grant(8, "ap-south", "w3")  # cannot switch vote

    # Two candidates cannot both obtain a majority of the same five witnesses.
    assert witnesses.grant(8, "ap-south", "w4")
    assert witnesses.grant(8, "ap-south", "w5")
    assert len(witnesses.votes_for(8, "eu-west")) == 3
    assert len(witnesses.votes_for(8, "ap-south")) == 2

    exact_boundary = controller.assess(
        "eu-west",
        durable_source_lsn=1000,
        observations=safe_observations(now, candidate_lsn=990),
        now=now,
    )
    assert exact_boundary.allowed
    assert exact_boundary.replication_lag_records == 10
    assert exact_boundary.votes == exact_boundary.quorum == 3

    new_token = controller.promote(
        "eu-west",
        durable_source_lsn=1000,
        observations=safe_observations(now),
        now=now,
    )
    assert new_token == FencingToken(8, "eu-west")
    ledger.install_authority(new_token)
    ledger.append(new_token, "order-1001")
    assert ledger.entries == ("order-1001",)

    try:
        ledger.append(old_primary_token, "stale-primary-write")
    except StaleFenceToken:
        pass
    else:
        raise AssertionError("the former primary token must be fenced")

    # Each scenario gets independent epoch-7 state and quorum votes.
    def assessed(
        observations: Iterable[RegionObservation],
        *,
        votes: tuple[str, ...] = ("w1", "w2", "w3"),
        source_lsn: int = 1000,
    ) -> FailoverAssessment:
        local_witnesses = WitnessRegistry({"w1", "w2", "w3", "w4", "w5"})
        for voter in votes:
            local_witnesses.grant(8, "eu-west", voter)
        local = FailoverController(
            "us-east",
            7,
            FailoverPolicy(10, timedelta(seconds=30)),
            local_witnesses,
        )
        return local.assess("eu-west", source_lsn, observations, now)

    insufficient_votes = assessed(
        safe_observations(now),
        votes=("w1", "w2"),
    )
    assert not insufficient_votes.allowed
    assert "candidate does not have a witness quorum" in insufficient_votes.reasons

    stale = list(safe_observations(now))
    stale[1] = RegionObservation(
        "eu-west",
        995,
        True,
        False,
        now - timedelta(seconds=31),
    )
    stale_result = assessed(stale)
    assert not stale_result.allowed
    assert "candidate observation is stale" in stale_result.reasons

    future = list(safe_observations(now))
    future[1] = RegionObservation(
        "eu-west",
        995,
        True,
        False,
        now + timedelta(seconds=1),
    )
    assert "candidate observation is from the future" in assessed(future).reasons

    lagged = assessed(safe_observations(now, candidate_lsn=989))
    assert not lagged.allowed
    assert "candidate replication lag exceeds policy" in lagged.reasons

    primary_still_writable = list(safe_observations(now))
    primary_still_writable[0] = RegionObservation(
        "us-east",
        1000,
        True,
        True,
        now - timedelta(seconds=2),
    )
    result = assessed(primary_still_writable)
    assert not result.allowed
    assert "current primary is not proven isolated and read-only" in result.reasons

    second_writer = list(safe_observations(now))
    second_writer[2] = RegionObservation(
        "ap-south",
        990,
        True,
        True,
        now - timedelta(seconds=2),
    )
    result = assessed(second_writer)
    assert not result.allowed
    assert "another standby is writable: ap-south" in result.reasons

    candidate_writer = list(safe_observations(now))
    candidate_writer[1] = RegionObservation(
        "eu-west",
        995,
        True,
        True,
        now - timedelta(seconds=2),
    )
    assert "candidate is already writable before promotion" in assessed(
        candidate_writer
    ).reasons

    ahead = assessed(safe_observations(now, candidate_lsn=1001))
    assert "candidate LSN is ahead of the durable source position" in ahead.reasons

    invalid_calls = (
        lambda: FailoverPolicy(True, timedelta(seconds=5)),
        lambda: FailoverPolicy(1, timedelta(0)),
        lambda: RegionObservation("x", -1, True, False, now),
        lambda: RegionObservation("x", 1, 1, False, now),
        lambda: FencingToken(0, "x"),
        lambda: WitnessRegistry(set()),
        lambda: WitnessRegistry({"w1"}).grant(2, "x", "unknown"),
        lambda: assessed((*safe_observations(now), safe_observations(now)[1])),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, FailoverError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def demo() -> None:
    """Run a safe promotion and show why the former primary cannot write."""
    controller, witnesses, ledger, now = build_example()
    old_token = controller.current_token()
    for witness in ("w1", "w2", "w3"):
        witnesses.grant(8, "eu-west", witness)

    assessment = controller.assess(
        candidate="eu-west",
        durable_source_lsn=1000,
        observations=safe_observations(now),
        now=now,
    )
    print(f"Promotion allowed: {assessment.allowed}")
    print(
        "Replication lag: "
        f"{assessment.replication_lag_records} durable records"
    )
    print(f"Witness votes: {assessment.votes}/{assessment.quorum}")
    print(f"Decision: {assessment.reasons[0]}")

    new_token = controller.promote(
        "eu-west",
        1000,
        safe_observations(now),
        now,
    )
    ledger.install_authority(new_token)
    ledger.append(new_token, "accepted-after-failover")
    print(f"New authority: {new_token.region} at epoch {new_token.epoch}")

    try:
        ledger.append(old_token, "must-not-commit")
    except StaleFenceToken as error:
        print(f"Former primary rejected: {error}")


# Practice exercises
# ------------------
# 1. Add byte lag and estimated time lag to RegionObservation. Require every
#    configured recovery-point limit to pass without treating time lag as exact.
# 2. Add a planned switchover mode. It must drain writes, record a final durable
#    source LSN, wait for zero lag, and still issue a new fencing epoch.
# 3. Add witness observation timestamps. Reject a quorum containing stale votes
#    and explain why retrying a vote in the same epoch must stay idempotent.
# 4. Model a safe failback from eu-west to us-east. Require reconciliation,
#    catch-up, invariant checks, traffic drain, and a fresh majority vote.
# 5. Produce an audit record with old and new primary, epoch, lag, voters,
#    observations, policy version, operator, decision, and reasons.

# Solutions / clear guidance
# --------------------------
# 1. Validate finite non-negative measurements. Keep record, byte, and time
#    limits separate because throughput changes. A candidate must pass all
#    enabled limits; never average a failed limit into a passing score.
# 2. Add an explicit mode enum. A switchover assessment should require the old
#    primary to be read-only, the candidate at the final LSN, and fresh evidence
#    from both. Promotion still increments the epoch before writes resume.
# 3. Store a grant timestamp beside each durable vote and compare it with the
#    decision time. An idempotent retry returns the original grant; it must not
#    refresh old evidence or change the chosen candidate.
# 4. Treat failback as a new promotion. Do not decrement epochs or reuse the old
#    us-east token. Compare divergent records, choose an authoritative history,
#    repair safely, and test reads before restoring write traffic.
# 5. Use immutable dataclasses and canonical serialization. Keep secrets and
#    customer data out of the record. Hash the policy and evidence, then append
#    the result to tamper-evident storage with retention and access controls.

# Expert challenge / mini-project
# --------------------------------
# Build a local multi-region failover simulator around these primitives:
#
# * Simulate three data regions, five witnesses, asynchronous replication,
#   latency, partitions, process pauses, lost acknowledgements, and recovery.
# * Maintain a durable write-ahead log with monotonic LSNs. Every replica must
#   apply records in order and expose verified lag plus its last observation.
# * Implement planned switchover, emergency failover, and reviewed failback.
#   Require bounded lag, fresh health evidence, a majority, and old-primary
#   isolation. Every promotion must create a higher fencing epoch.
# * Route all mutations through a fenced storage adapter. Inject a delayed write
#   from the former primary after promotion and prove it cannot commit.
# * Test quorum intersection by attempting competing promotions in one epoch.
#   Test stale observations, membership changes, lag exactly on each boundary,
#   witness restart, partial replication, and candidate recovery.
# * Emit an append-only timeline containing evidence, decisions, rejected
#   actions, data-loss estimate, recovery duration, and manual approvals.
# * Add unit tests, state-machine or property tests for invariants, type checks,
#   documentation, and a deterministic chaos test suitable for CI.
#
# A strong solution separates policy, consensus, replication, routing, and data
# storage behind narrow interfaces. State the invariants explicitly: at most one
# write authority per epoch, epochs only increase, a witness never changes its
# vote within an epoch, and storage never accepts stale authority. The simulator
# must default to local synthetic data and must not control real infrastructure.


if __name__ == "__main__":
    run_self_checks()
    demo()
