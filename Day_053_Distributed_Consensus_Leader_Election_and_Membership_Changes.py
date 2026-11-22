"""Day 53: Distributed consensus, leader election, and membership changes.

Consensus is the discipline of making one replicated decision despite delayed,
duplicated, or missing messages. This lesson builds a deliberately small,
in-memory Raft-inspired model with terms, one-vote-per-term elections, log
freshness checks, majority commits, and joint-consensus membership changes.

The model is educational: it does not open sockets, persist data, elect a real
service leader, or claim to implement production Raft. Every state transition
is explicit and deterministic so the invariants can be tested locally.

Learning goals
--------------
1. Explain why terms, quorum intersection, and log freshness protect safety.
2. Run a deterministic leader election and reject stale candidates.
3. Replicate a command and commit it only after a membership quorum acks it.
4. Change membership through old/new joint quorum before finalizing it.
5. Distinguish safety (never two committed values at one index) from liveness
   (eventually making progress when a quorum and a healthy leader exist).

Teaching notes
--------------
- A term is a monotonically increasing logical epoch. A message from an older
  term cannot change state. A node seeing a newer term steps down and forgets
  its previous vote.
- Each node grants at most one vote in a term. A candidate votes for itself,
  then asks peers. Majority quorums intersect, so two candidates cannot both
  collect a majority of the same configuration.
- A candidate's log must be at least as up to date as the voter's log. Compare
  the last entry term first, then the last index. This prevents a lagging
  candidate from becoming leader and overwriting committed history.
- A leader may replicate an entry to many followers, but the entry is committed
  only when acknowledgements satisfy the current quorum. A timeout or a
  follower crash can delay progress without making an uncommitted value true.
- Membership changes are not a set replacement. During a joint phase, a
  decision needs a majority of both the old and new configurations. Only after
  that entry commits can the cluster finalize the new configuration.
- Removing a node that is still the only bridge between quorums can strand the
  cluster. Production systems also need configuration entries, snapshots,
  pre-vote or lease policies, durable storage, message authentication, and
  careful recovery of a node that was offline.
- Safety and liveness are different properties. A conservative timeout can
  preserve safety while harming availability; randomized election timeouts
  reduce repeated ties but do not replace quorum rules.
- This file uses full-log copying when a follower is behind to keep the example
  short. Real implementations stream AppendEntries, detect conflicts with
  previous-log terms, compact committed prefixes into snapshots, and retry
  idempotently.

Run this file with Python 3.10+. The self-checks cover stale terms, one vote per
term, quorum loss, committed writes, stale log rejection, joint membership
quorums, and a blocked configuration finalization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class ConsensusError(Exception):
    """Base class for expected consensus failures."""


class ValidationError(ConsensusError):
    """An input violates a model invariant."""


class NotLeader(ConsensusError):
    """A command requires the current leader."""


class CommitRejected(ConsensusError):
    """A proposal could not reach the required quorum."""


class ElectionRejected(ConsensusError):
    """An election cannot be started by this node."""


def clean_identifier(value: str, field_name: str) -> str:
    """Return a bounded, non-blank identifier."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    result = value.strip()
    if not result:
        raise ValidationError(f"{field_name} must not be blank")
    if len(result) > 64:
        raise ValidationError(f"{field_name} is too long")
    return result


def positive_int(value: int, field_name: str) -> int:
    """Require a strictly positive integer without accepting bool."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValidationError(f"{field_name} must be positive")
    return value


def non_negative_int(value: int, field_name: str) -> int:
    """Require a non-negative integer without accepting bool."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValidationError(f"{field_name} must not be negative")
    return value


def normalized_members(values: Iterable[str], field_name: str = "member") -> frozenset[str]:
    """Normalize a non-empty collection of unique node identifiers."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name}s must be an iterable of strings")
    result = frozenset(clean_identifier(value, field_name) for value in values)
    if not result:
        raise ValidationError("a membership must contain at least one node")
    return result


class Role(str, Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass(frozen=True, slots=True)
class LogEntry:
    """One replicated command at a logical log index."""

    term: int
    index: int
    command: str

    def __post_init__(self) -> None:
        positive_int(self.term, "term")
        positive_int(self.index, "index")
        object.__setattr__(self, "command", clean_identifier(self.command, "command"))


@dataclass
class Node:
    """A single in-memory consensus participant."""

    node_id: str
    current_term: int = 0
    role: Role = Role.FOLLOWER
    voted_for: str | None = None
    log: list[LogEntry] = field(default_factory=list)
    alive: bool = True
    commit_index: int = 0

    def __post_init__(self) -> None:
        self.node_id = clean_identifier(self.node_id, "node_id")
        non_negative_int(self.current_term, "current_term")
        if not isinstance(self.role, Role):
            raise TypeError("role must be a Role")
        if self.voted_for is not None:
            self.voted_for = clean_identifier(self.voted_for, "voted_for")
        if not isinstance(self.alive, bool):
            raise TypeError("alive must be a bool")
        non_negative_int(self.commit_index, "commit_index")
        previous_index = 0
        for entry in self.log:
            if not isinstance(entry, LogEntry):
                raise TypeError("log must contain LogEntry values")
            if entry.index != previous_index + 1:
                raise ValidationError("log indexes must be contiguous from one")
            previous_index = entry.index
        if self.commit_index > len(self.log):
            raise ValidationError("commit_index cannot exceed the local log length")

    @property
    def last_log_index(self) -> int:
        """Return the last local log index, or zero for an empty log."""
        return self.log[-1].index if self.log else 0

    @property
    def last_log_term(self) -> int:
        """Return the last local log term, or zero for an empty log."""
        return self.log[-1].term if self.log else 0

    def log_is_at_least_as_new_as(self, index: int, term: int) -> bool:
        """Compare logs using term first and index second."""
        non_negative_int(index, "index")
        non_negative_int(term, "term")
        return (term, index) >= (self.last_log_term, self.last_log_index)

    def handle_vote_request(self, request: "VoteRequest") -> "VoteResponse":
        """Apply the one-vote-per-term and log-freshness rules."""
        if not isinstance(request, VoteRequest):
            raise TypeError("request must be a VoteRequest")
        if not self.alive:
            return VoteResponse(self.current_term, False, "node is offline")
        if request.term < self.current_term:
            return VoteResponse(self.current_term, False, "request has an older term")
        if request.term > self.current_term:
            self.current_term = request.term
            self.role = Role.FOLLOWER
            self.voted_for = None

        if self.voted_for not in (None, request.candidate_id):
            return VoteResponse(self.current_term, False, "already voted this term")
        if not self.log_is_at_least_as_new_as(
            request.last_log_index,
            request.last_log_term,
        ):
            return VoteResponse(self.current_term, False, "candidate log is stale")

        self.voted_for = request.candidate_id
        return VoteResponse(self.current_term, True, "vote granted")


@dataclass(frozen=True, slots=True)
class VoteRequest:
    """The evidence a candidate sends to a voter."""

    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int

    def __post_init__(self) -> None:
        positive_int(self.term, "term")
        object.__setattr__(
            self,
            "candidate_id",
            clean_identifier(self.candidate_id, "candidate_id"),
        )
        non_negative_int(self.last_log_index, "last_log_index")
        non_negative_int(self.last_log_term, "last_log_term")


@dataclass(frozen=True, slots=True)
class VoteResponse:
    """A deterministic response to one vote request."""

    term: int
    vote_granted: bool
    reason: str

    def __post_init__(self) -> None:
        non_negative_int(self.term, "term")
        if not isinstance(self.vote_granted, bool):
            raise TypeError("vote_granted must be a bool")
        object.__setattr__(self, "reason", clean_identifier(self.reason, "reason"))


@dataclass(frozen=True, slots=True)
class ElectionResult:
    """Explainable outcome of one election attempt."""

    term: int
    candidate: str
    votes: frozenset[str]
    quorum_description: str
    won: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Explainable outcome of one replication attempt."""

    entry: LogEntry
    acknowledgements: frozenset[str]
    quorum_description: str
    committed: bool


@dataclass(frozen=True, slots=True)
class Membership:
    """A stable or joint old/new configuration."""

    members: frozenset[str]
    joint_old: frozenset[str] | None = None
    joint_new: frozenset[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", normalized_members(self.members))
        if (self.joint_old is None) != (self.joint_new is None):
            raise ValidationError("joint_old and joint_new must be provided together")
        if self.joint_old is not None and self.joint_new is not None:
            object.__setattr__(self, "joint_old", normalized_members(self.joint_old))
            object.__setattr__(self, "joint_new", normalized_members(self.joint_new))
            if self.members != self.joint_old | self.joint_new:
                raise ValidationError("joint membership must contain old and new nodes")

    @classmethod
    def from_members(cls, members: Iterable[str]) -> "Membership":
        """Construct a stable membership."""
        normalized = normalized_members(members)
        return cls(normalized)

    @property
    def in_joint_phase(self) -> bool:
        """Whether both old and new configurations are active."""
        return self.joint_old is not None

    @property
    def voters(self) -> frozenset[str]:
        """Return the nodes eligible to acknowledge an entry."""
        if self.in_joint_phase:
            return self.members
        return self.members

    @property
    def quorum_description(self) -> str:
        """Describe the quorum rule used for a decision."""
        if not self.in_joint_phase:
            return f"majority of {len(self.members)} ({self.quorum_size(self.members)})"
        return (
            f"joint majority old={self.quorum_size(self.joint_old)} "
            f"and new={self.quorum_size(self.joint_new)}"
        )

    @staticmethod
    def quorum_size(configuration: Iterable[str]) -> int:
        """Return floor(N / 2) + 1 for one configuration."""
        members = normalized_members(configuration)
        return len(members) // 2 + 1

    def has_quorum(self, votes: Iterable[str]) -> bool:
        """Check a stable majority or a majority in both joint configs."""
        vote_set = set(votes)
        if not vote_set.issubset(self.voters):
            raise ValidationError("votes contain a node outside the membership")
        if not self.in_joint_phase:
            return len(vote_set & self.members) >= self.quorum_size(self.members)
        assert self.joint_old is not None and self.joint_new is not None
        old_votes = len(vote_set & self.joint_old)
        new_votes = len(vote_set & self.joint_new)
        return (
            old_votes >= self.quorum_size(self.joint_old)
            and new_votes >= self.quorum_size(self.joint_new)
        )

    def start_change(self, new_members: Iterable[str]) -> "Membership":
        """Enter a joint old/new configuration."""
        if self.in_joint_phase:
            raise ValidationError("finish the existing membership change first")
        normalized = normalized_members(new_members)
        if normalized == self.members:
            raise ValidationError("new membership must differ from the current one")
        return Membership(
            self.members | normalized,
            joint_old=self.members,
            joint_new=normalized,
        )

    def finish_change(self) -> "Membership":
        """Finalize a previously committed joint configuration."""
        if not self.in_joint_phase:
            raise ValidationError("there is no membership change to finalize")
        assert self.joint_new is not None
        return Membership(self.joint_new)


class ConsensusCluster:
    """A deterministic, in-memory cluster for teaching consensus invariants."""

    def __init__(self, members: Iterable[str]) -> None:
        normalized = normalized_members(members)
        self.nodes: dict[str, Node] = {node_id: Node(node_id) for node_id in normalized}
        self.membership = Membership.from_members(normalized)
        self.term = 0
        self.leader_id: str | None = None
        self._committed_entries: list[LogEntry] = []

    @property
    def committed_entries(self) -> tuple[LogEntry, ...]:
        """Return the immutable view of entries known to be committed."""
        return tuple(self._committed_entries)

    @property
    def committed_commands(self) -> tuple[str, ...]:
        """Return committed command names for concise assertions."""
        return tuple(entry.command for entry in self._committed_entries)

    def add_node(self, node_id: str) -> None:
        """Register a stopped-out node before proposing its membership."""
        node_id = clean_identifier(node_id, "node_id")
        if node_id in self.nodes:
            raise ValidationError(f"node already exists: {node_id}")
        self.nodes[node_id] = Node(node_id)

    def crash(self, node_id: str) -> None:
        """Make a node unavailable to messages and acknowledgements."""
        node = self._node(node_id)
        node.alive = False
        if self.leader_id == node_id:
            self.leader_id = None
            node.role = Role.FOLLOWER

    def recover(self, node_id: str) -> None:
        """Bring an existing node back; it may need log catch-up."""
        node = self._node(node_id)
        node.alive = True

    def _node(self, node_id: str) -> Node:
        """Look up a known node with normalized errors."""
        node_id = clean_identifier(node_id, "node_id")
        try:
            return self.nodes[node_id]
        except KeyError as error:
            raise ValidationError(f"unknown node: {node_id}") from error

    def _require_leader(self) -> Node:
        """Return a live leader or explain why a proposal cannot proceed."""
        if self.leader_id is None:
            raise NotLeader("there is no elected leader")
        leader = self._node(self.leader_id)
        if not leader.alive or leader.role is not Role.LEADER:
            raise NotLeader("the recorded leader is not alive and leading")
        return leader

    def start_election(self, candidate_id: str) -> ElectionResult:
        """Run one deterministic election round for a live member."""
        candidate = self._node(candidate_id)
        if candidate_id not in self.membership.members:
            raise ElectionRejected("candidate is outside the active membership")
        if not candidate.alive:
            raise ElectionRejected("offline candidate cannot start an election")

        self.term = max(self.term, max(node.current_term for node in self.nodes.values())) + 1
        candidate.current_term = self.term
        candidate.role = Role.CANDIDATE
        candidate.voted_for = candidate.node_id
        request = VoteRequest(
            self.term,
            candidate.node_id,
            candidate.last_log_index,
            candidate.last_log_term,
        )

        votes = {candidate.node_id}
        reasons = ["candidate voted for itself"]
        for node_id in sorted(self.membership.voters):
            if node_id == candidate.node_id:
                continue
            response = self.nodes[node_id].handle_vote_request(request)
            if response.term > self.term:
                self.term = response.term
                candidate.current_term = response.term
            if response.vote_granted:
                votes.add(node_id)
            else:
                reasons.append(f"{node_id}: {response.reason}")

        won = self.membership.has_quorum(votes)
        if won:
            for node in self.nodes.values():
                if node.node_id != candidate.node_id and node.current_term <= self.term:
                    node.role = Role.FOLLOWER
            candidate.role = Role.LEADER
            self.leader_id = candidate.node_id
            reasons.append("quorum reached; candidate is leader")
        else:
            candidate.role = Role.FOLLOWER
            self.leader_id = None
            reasons.append("quorum not reached; no leader was installed")
        return ElectionResult(
            self.term,
            candidate.node_id,
            frozenset(votes),
            self.membership.quorum_description,
            won,
            tuple(reasons),
        )

    def _replicate_entry(self, entry: LogEntry, voters: Iterable[str]) -> frozenset[str]:
        """Copy the leader prefix and entry to every live voter.

        Full-log copying is intentionally simple. It demonstrates that a
        follower can catch up before acknowledging, while avoiding a network
        implementation in a single-file lesson.
        """
        leader = self._require_leader()
        leader_prefix = list(leader.log)
        acknowledgements: set[str] = set()
        for node_id in sorted(set(voters)):
            node = self._node(node_id)
            if not node.alive:
                continue
            node.log = leader_prefix.copy()
            node.log.append(entry)
            node.current_term = max(node.current_term, entry.term)
            node.role = Role.FOLLOWER if node.node_id != leader.node_id else Role.LEADER
            acknowledgements.add(node.node_id)
        return frozenset(acknowledgements)

    def _discard_uncommitted_entry(self, entry: LogEntry) -> None:
        """Remove a failed proposal that was appended but never committed."""
        for node in self.nodes.values():
            if node.log and node.log[-1] == entry and node.commit_index < entry.index:
                node.log.pop()

    def _record_commit(self, entry: LogEntry) -> None:
        """Mark one sequential entry committed on every replica with it."""
        expected_index = len(self._committed_entries) + 1
        if entry.index != expected_index:
            raise ValidationError("committed entries must be sequential")
        self._committed_entries.append(entry)
        for node in self.nodes.values():
            if len(node.log) >= entry.index and node.log[entry.index - 1] == entry:
                node.commit_index = entry.index

    def append(self, command: str) -> CommitResult:
        """Replicate a user command and commit it only after quorum."""
        leader = self._require_leader()
        command = clean_identifier(command, "command")
        entry = LogEntry(self.term, leader.last_log_index + 1, command)
        acknowledgements = self._replicate_entry(entry, self.membership.voters)
        committed = self.membership.has_quorum(acknowledgements)
        if committed:
            self._record_commit(entry)
        return CommitResult(
            entry,
            acknowledgements,
            self.membership.quorum_description,
            committed,
        )

    def begin_membership_change(self, new_members: Iterable[str]) -> CommitResult:
        """Commit a joint old/new configuration before finalization."""
        leader = self._require_leader()
        target = normalized_members(new_members)
        unknown = target - self.nodes.keys()
        if unknown:
            raise ValidationError(
                "register new nodes with add_node before proposing them: "
                + ", ".join(sorted(unknown))
            )
        joint = self.membership.start_change(target)
        entry = LogEntry(
            self.term,
            leader.last_log_index + 1,
            "CONFIG_JOINT:" + ",".join(sorted(target)),
        )
        acknowledgements = self._replicate_entry(entry, joint.voters)
        if not joint.has_quorum(acknowledgements):
            self._discard_uncommitted_entry(entry)
            raise CommitRejected(
                "joint configuration did not reach a majority of both configurations"
            )
        self._record_commit(entry)
        self.membership = joint
        return CommitResult(entry, acknowledgements, joint.quorum_description, True)

    def finalize_membership_change(self) -> CommitResult:
        """Commit a final configuration entry, then install the new membership."""
        leader = self._require_leader()
        if not self.membership.in_joint_phase:
            raise ValidationError("there is no joint configuration to finalize")
        assert self.membership.joint_new is not None
        final_members = self.membership.joint_new
        entry = LogEntry(
            self.term,
            leader.last_log_index + 1,
            "CONFIG_FINAL:" + ",".join(sorted(final_members)),
        )
        acknowledgements = self._replicate_entry(entry, self.membership.voters)
        if not self.membership.has_quorum(acknowledgements):
            self._discard_uncommitted_entry(entry)
            raise CommitRejected(
                "final configuration still needs a joint quorum"
            )
        if leader.node_id not in final_members:
            raise ValidationError(
                "this teaching model requires leadership transfer before removal"
            )
        self._record_commit(entry)
        self.membership = self.membership.finish_change()
        return CommitResult(
            entry,
            acknowledgements,
            self.membership.quorum_description,
            True,
        )


def build_demo_cluster() -> ConsensusCluster:
    """Build a three-node cluster with no external dependencies."""
    return ConsensusCluster({"node-a", "node-b", "node-c"})


def run_self_checks() -> None:
    """Exercise safety, quorum, replication, and membership invariants."""
    cluster = build_demo_cluster()
    election = cluster.start_election("node-a")
    assert election.won
    assert election.votes == frozenset({"node-a", "node-b", "node-c"})
    assert cluster.leader_id == "node-a"

    first = cluster.append("set:mode=training")
    assert first.committed
    assert cluster.committed_commands == ("set:mode=training",)

    # An offline majority cannot elect a replacement in a three-node cluster.
    cluster.crash("node-b")
    cluster.crash("node-c")
    blocked = cluster.start_election("node-a")
    assert not blocked.won
    assert "quorum not reached" in blocked.reasons[-1]
    cluster.recover("node-b")
    cluster.recover("node-c")
    assert cluster.start_election("node-a").won

    # A voter grants one candidate per term and rejects a lagging candidate log.
    voter = Node("voter", current_term=4)
    voter.log = [LogEntry(4, 1, "committed")]
    stale_request = VoteRequest(5, "short-log", 0, 0)
    stale_response = voter.handle_vote_request(stale_request)
    assert not stale_response.vote_granted
    assert stale_response.reason == "candidate log is stale"
    valid_request = VoteRequest(5, "valid-candidate", 1, 4)
    valid_response = voter.handle_vote_request(valid_request)
    assert valid_response.vote_granted
    second_request = VoteRequest(5, "other-candidate", 1, 4)
    second_response = voter.handle_vote_request(second_request)
    assert not second_response.vote_granted
    assert second_response.reason == "already voted this term"

    # Losing the old/new joint quorum blocks a removal that would be unsafe.
    cluster = build_demo_cluster()
    assert cluster.start_election("node-a").won
    cluster.add_node("node-d")
    cluster.crash("node-c")
    cluster.crash("node-d")
    try:
        cluster.begin_membership_change({"node-a", "node-c", "node-d"})
    except CommitRejected:
        pass
    else:
        raise AssertionError("joint quorum must fail when the new side has one vote")
    cluster.recover("node-c")
    cluster.recover("node-d")

    joint = cluster.begin_membership_change({"node-a", "node-c", "node-d"})
    assert joint.committed
    assert cluster.membership.in_joint_phase
    final = cluster.finalize_membership_change()
    assert final.committed
    assert cluster.membership.members == frozenset({"node-a", "node-c", "node-d"})

    # Joint quorum requires both sides; two old-only votes are insufficient
    # when the new configuration has no overlap with them.
    disjoint = Membership.from_members({"a", "b", "c"}).start_change({"d", "e", "f"})
    assert not disjoint.has_quorum({"a", "b"})
    assert disjoint.has_quorum({"a", "b", "d", "e"})

    invalid_calls = (
        lambda: LogEntry(0, 1, "x"),
        lambda: LogEntry(1, 0, "x"),
        lambda: normalized_members("abc"),
        lambda: Membership.from_members(set()),
        lambda: cluster.start_election("unknown"),
        lambda: cluster.add_node("node-a"),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, ConsensusError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def demo() -> None:
    """Show a committed command and a safe two-phase membership change."""
    cluster = build_demo_cluster()
    election = cluster.start_election("node-a")
    print(f"Election term {election.term}: {sorted(election.votes)} -> {election.won}")

    command = cluster.append("write:hello")
    print(
        f"Command {command.entry.command!r}: "
        f"{len(command.acknowledgements)} acknowledgements, "
        f"committed={command.committed}"
    )

    cluster.add_node("node-d")
    joint = cluster.begin_membership_change({"node-a", "node-b", "node-d"})
    print(f"Joint configuration committed: {joint.committed}")
    final = cluster.finalize_membership_change()
    print(f"Final members: {sorted(cluster.membership.members)}")
    print(f"Final entry committed: {final.committed}")


# Practice exercises
# ------------------
# 1. Add an election timeout policy. Use a deterministic clock and a seeded
#    timeout sequence; start an election only when the deadline expires. Explain
#    why randomized timeouts reduce ties but cannot grant a quorum.
# 2. Replace full-log copying with AppendEntries-style conflict repair. Include
#    previous-index and previous-term checks, retry a follower with a lower next
#    index, and keep committed prefixes immutable.
# 3. Persist a node's term, vote, and log in a small SQLite repository. Simulate
#    a crash between writing the log and the commit marker and recover safely.
# 4. Add read-only requests. Compare a leader lease, a quorum read, and a
#    ReadIndex-like round; document the clock and partition assumptions for each.
# 5. Add snapshot installation. Compact entries through a committed index, then
#    restore a lagging node from a verified snapshot before it can acknowledge.

# Solutions / clear guidance
# --------------------------
# 1. Store monotonic deadlines, never wall-clock strings. A seeded sequence keeps
#    tests reproducible. Reset a candidate's timer when it receives a valid
#    heartbeat or a higher-term vote response; still require a majority.
# 2. Keep next_index per follower. Send prev_log_index and prev_log_term, reject
#    mismatches, decrement next_index, and retry. Never overwrite an entry that
#    is at or below the local committed index; truncate only an uncommitted
#    conflicting suffix.
# 3. Use a durable transaction for term/vote/log metadata and a separate commit
#    marker only after quorum evidence. On recovery, replay only contiguous
#    entries and discard or quarantine an uncommitted suffix. Test torn writes
#    and duplicate recovery.
# 4. A lease can be fast but is safe only with bounded clock skew and valid
#    authority. A quorum read trades latency for fresh evidence and remains
#    available only with a quorum. ReadIndex separates the read from log writes
#    but still needs a current-term leader confirmation.
# 5. Include the last included index and term in the snapshot. Verify a digest
#    before installing, reject a snapshot older than committed state, and keep
#    the snapshot plus suffix atomically so a restart cannot expose a gap.

# Expert challenge / mini-project
# --------------------------------
# Build a deterministic multi-node consensus simulator around this lesson:
#
# * Add an event queue with delivery delay, duplication, reordering, and
#   partition controls. Keep all randomness seeded and expose the event trace.
# * Implement RequestVote, AppendEntries, heartbeats, elections, conflict
#   repair, commit advancement, and snapshots. Model a crash/restart that
#   reloads durable term, vote, and log state.
# * Exercise a five-node cluster under partitions. Prove that two leaders in
#   one term cannot both commit different commands at one index. Also show a
#   minority partition can remain unavailable without violating safety.
# * Implement joint-consensus membership changes, including adding a node,
#   removing a node, leadership transfer, and a blocked finalization when either
#   side lacks a majority.
# * Emit an audit timeline for elections, votes, rejected messages, commits,
#   configuration phases, and recovery. Add property-based or state-machine
#   tests for monotonic terms, one vote per term, quorum intersection, and
#   committed-prefix immutability.
#
# Keep the simulator local and synthetic. A strong solution separates transport,
# durable state, quorum policy, log replication, membership, and observability
# behind narrow interfaces. State the safety invariants in documentation before
# optimizing the implementation or adding real networking.

if __name__ == "__main__":
    run_self_checks()
    demo()
