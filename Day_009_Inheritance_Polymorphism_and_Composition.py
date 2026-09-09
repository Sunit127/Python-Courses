"""Day 9: Inheritance, polymorphism, and composition trade-offs.

Learning goals
--------------
1. Use inheritance for a genuine "is-a" relationship with a stable contract.
2. Define abstract base classes that require subclasses to implement behavior.
3. Write polymorphic code that works with many concrete implementations.
4. Use super() to extend cooperative parent behavior safely.
5. Prefer composition when behavior should be wrapped, replaced, or combined.

Teaching notes
--------------
- Inheritance lets a subclass reuse and specialize a parent class. It is most
  useful when every subclass can honor the parent's promises.
- An abstract base class (ABC) documents a contract. @abstractmethod prevents
  incomplete subclasses from being instantiated.
- Polymorphism means callers depend on the shared contract, not a chain of
  type checks. NotificationService calls deliver() without knowing whether the
  channel represents email, SMS, or an in-memory inbox.
- Overridden methods must preserve the parent contract: accepted inputs,
  meaningful return values, and documented exceptions should remain compatible.
- super() delegates to the next implementation in the method-resolution order.
  Use it consistently in cooperative class hierarchies.
- Composition models a "has-a" relationship. RetryingChannel has another
  DeliveryChannel and adds retries without creating EmailWithRetry,
  SmsWithRetry, and many other subclasses.
- Deep inheritance trees make behavior difficult to predict. Favor small
  hierarchies, immutable value objects, and collaborators passed explicitly.
- Catch only failures a layer can handle. RetryingChannel retries transient
  failures but lets permanent DeliveryError failures surface immediately.

Run this file with Python 3.10+ to execute the examples and self-checks.

Practice exercises
------------------
1. Implement InboxChannel. It should satisfy DeliveryChannel, keep delivered
   messages in memory, and return defensive snapshots rather than its list.
2. Write deliver_with_fallback(channels, recipient, message). Try channels in
   order, return the first successful receipt, and raise DeliveryError only
   after every channel fails.
3. Write unique_channel_names(channels). Return names in order and reject
   duplicate names case-insensitively.

Solutions appear below the main example.

Expert challenge: reliable notification hub
-------------------------------------------
Build a notification hub with channel priority, recipient preferences,
idempotency keys, and an audit trail. Distinguish transient failures from
permanent address failures; retry only transient failures. Inject a clock and
storage interface so tests stay deterministic. Add tests for duplicate
idempotency keys, fallback order, exhausted retries, partial fan-out, and an
audit-store failure.

Solution guidance:
1. Keep Message and Recipient as immutable value objects.
2. Define narrow interfaces for delivery, idempotency storage, and audit output.
3. Compose retry and fallback policies around channels instead of subclassing
   every concrete channel-policy combination.
4. Record an idempotency key before delivery with an explicit pending state;
   define how interrupted pending work is recovered.
5. Return a result for every attempted channel, but never expose credentials or
   full private addresses in logs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass


def _clean_text(value: str, field_name: str) -> str:
    """Return stripped, non-empty text with a useful error."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


@dataclass(frozen=True, slots=True)
class Recipient:
    """Validated destinations for one person.

    A production system would use dedicated email and phone value objects with
    stricter validation. These checks intentionally remain readable.
    """

    name: str
    email: str | None = None
    phone: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_text(self.name, "name"))

        if self.email is not None:
            email = _clean_text(self.email, "email").casefold()
            if email.count("@") != 1 or email.startswith("@") or email.endswith("@"):
                raise ValueError("email must contain text on both sides of @")
            object.__setattr__(self, "email", email)

        if self.phone is not None:
            phone = self.phone.replace(" ", "").replace("-", "")
            if phone.startswith("+"):
                digits = phone[1:]
            else:
                digits = phone
            if not digits.isdigit() or not 7 <= len(digits) <= 15:
                raise ValueError("phone must contain 7 to 15 digits")
            object.__setattr__(self, "phone", phone)

        if self.email is None and self.phone is None:
            raise ValueError("recipient needs an email address or phone number")


@dataclass(frozen=True, slots=True)
class Message:
    """Immutable message content shared by every delivery channel."""

    subject: str
    body: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject", _clean_text(self.subject, "subject"))
        object.__setattr__(self, "body", _clean_text(self.body, "body"))


@dataclass(frozen=True, slots=True)
class DeliveryReceipt:
    """Presentation-neutral evidence of a successful delivery."""

    channel: str
    recipient_name: str
    reference: str


class DeliveryError(Exception):
    """A channel cannot deliver the message."""


class TransientDeliveryError(DeliveryError):
    """Delivery may succeed if attempted again."""


class DeliveryChannel(ABC):
    """Contract implemented by every notification channel."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a stable, human-readable channel name."""

    @abstractmethod
    def deliver(
        self,
        recipient: Recipient,
        message: Message,
    ) -> DeliveryReceipt:
        """Deliver one message or raise DeliveryError."""


class EmailChannel(DeliveryChannel):
    """Safe demonstration channel that records no private message content."""

    def __init__(self, sender: str) -> None:
        self._sender = _clean_text(sender, "sender")
        if self._sender.count("@") != 1:
            raise ValueError("sender must be an email-like address")
        self._delivery_count = 0

    @property
    def name(self) -> str:
        return "email"

    @property
    def delivery_count(self) -> int:
        return self._delivery_count

    def deliver(
        self,
        recipient: Recipient,
        message: Message,
    ) -> DeliveryReceipt:
        if recipient.email is None:
            raise DeliveryError("recipient has no email address")

        self._delivery_count += 1
        return DeliveryReceipt(
            channel=self.name,
            recipient_name=recipient.name,
            reference=f"email-{self._delivery_count:04d}",
        )


class SmsChannel(DeliveryChannel):
    """Safe SMS-like channel with deterministic in-memory behavior."""

    def __init__(self) -> None:
        self._delivery_count = 0

    @property
    def name(self) -> str:
        return "sms"

    def deliver(
        self,
        recipient: Recipient,
        message: Message,
    ) -> DeliveryReceipt:
        if recipient.phone is None:
            raise DeliveryError("recipient has no phone number")

        self._delivery_count += 1
        return DeliveryReceipt(
            channel=self.name,
            recipient_name=recipient.name,
            reference=f"sms-{self._delivery_count:04d}",
        )


class RetryingChannel(DeliveryChannel):
    """Add bounded retry behavior by composing another channel."""

    def __init__(self, channel: DeliveryChannel, max_attempts: int = 3) -> None:
        if not isinstance(channel, DeliveryChannel):
            raise TypeError("channel must implement DeliveryChannel")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise TypeError("max_attempts must be an integer")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self._channel = channel
        self._max_attempts = max_attempts

    @property
    def name(self) -> str:
        return f"retrying-{self._channel.name}"

    def deliver(
        self,
        recipient: Recipient,
        message: Message,
    ) -> DeliveryReceipt:
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._channel.deliver(recipient, message)
            except TransientDeliveryError as error:
                if attempt == self._max_attempts:
                    raise DeliveryError(
                        f"{self._channel.name} failed after {attempt} attempts"
                    ) from error

        raise AssertionError("retry loop must return or raise")


class NotificationService:
    """Fan out messages by composing independent channel objects."""

    def __init__(self, channels: Iterable[DeliveryChannel]) -> None:
        materialized = tuple(channels)
        if not materialized:
            raise ValueError("provide at least one delivery channel")
        if not all(isinstance(channel, DeliveryChannel) for channel in materialized):
            raise TypeError("every channel must implement DeliveryChannel")

        normalized_names = [channel.name.casefold() for channel in materialized]
        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError("channel names must be unique")
        self._channels = materialized

    @property
    def channels(self) -> tuple[DeliveryChannel, ...]:
        """Return an immutable snapshot of configured channels."""
        return self._channels

    def notify(
        self,
        recipient: Recipient,
        message: Message,
    ) -> tuple[DeliveryReceipt, ...]:
        """Deliver through every channel using the common polymorphic contract."""
        return tuple(
            channel.deliver(recipient, message) for channel in self._channels
        )


# Practice exercise solutions


class InboxChannel(DeliveryChannel):
    """Solution 1: an in-memory channel useful for tests and local tools."""

    def __init__(self) -> None:
        self._items: list[tuple[Recipient, Message]] = []

    @property
    def name(self) -> str:
        return "inbox"

    @property
    def items(self) -> tuple[tuple[Recipient, Message], ...]:
        return tuple(self._items)

    def deliver(
        self,
        recipient: Recipient,
        message: Message,
    ) -> DeliveryReceipt:
        self._items.append((recipient, message))
        return DeliveryReceipt(
            channel=self.name,
            recipient_name=recipient.name,
            reference=f"inbox-{len(self._items):04d}",
        )


def deliver_with_fallback(
    channels: Iterable[DeliveryChannel],
    recipient: Recipient,
    message: Message,
) -> DeliveryReceipt:
    """Solution 2: return the first successful channel receipt."""
    materialized = tuple(channels)
    if not materialized:
        raise ValueError("provide at least one delivery channel")

    failures: list[DeliveryError] = []
    for channel in materialized:
        if not isinstance(channel, DeliveryChannel):
            raise TypeError("every channel must implement DeliveryChannel")
        try:
            return channel.deliver(recipient, message)
        except DeliveryError as error:
            failures.append(error)

    raise DeliveryError(
        f"all {len(failures)} delivery channels failed"
    ) from failures[-1]


def unique_channel_names(channels: Iterable[DeliveryChannel]) -> tuple[str, ...]:
    """Solution 3: preserve order while rejecting case-insensitive duplicates."""
    names: list[str] = []
    seen: set[str] = set()
    for channel in channels:
        if not isinstance(channel, DeliveryChannel):
            raise TypeError("every channel must implement DeliveryChannel")
        name = _clean_text(channel.name, "channel name")
        key = name.casefold()
        if key in seen:
            raise ValueError(f"duplicate channel name: {name!r}")
        seen.add(key)
        names.append(name)
    return tuple(names)


class _AlwaysFailsChannel(DeliveryChannel):
    """Test helper demonstrating a permanent failure."""

    @property
    def name(self) -> str:
        return "always-fails"

    def deliver(
        self,
        recipient: Recipient,
        message: Message,
    ) -> DeliveryReceipt:
        raise DeliveryError("permanent demonstration failure")


class _FailOnceChannel(DeliveryChannel):
    """Test helper demonstrating a transient failure."""

    def __init__(self) -> None:
        self.attempts = 0

    @property
    def name(self) -> str:
        return "fail-once"

    def deliver(
        self,
        recipient: Recipient,
        message: Message,
    ) -> DeliveryReceipt:
        self.attempts += 1
        if self.attempts == 1:
            raise TransientDeliveryError("temporary demonstration failure")
        return DeliveryReceipt(self.name, recipient.name, "recovered-0001")


def run_self_checks() -> None:
    """Check contracts, polymorphism, retries, and important edge cases."""
    recipient = Recipient(
        " Asha ",
        email="ASHA@example.com",
        phone="+977 980-000-0000",
    )
    message = Message(" Course update ", " Day 9 is ready. ")
    assert recipient.name == "Asha"
    assert recipient.email == "asha@example.com"
    assert recipient.phone == "+9779800000000"
    assert message.subject == "Course update"

    email = EmailChannel("course@example.com")
    sms = SmsChannel()
    service = NotificationService([email, sms])
    receipts = service.notify(recipient, message)
    assert [receipt.channel for receipt in receipts] == ["email", "sms"]
    assert email.delivery_count == 1
    assert unique_channel_names(service.channels) == ("email", "sms")

    inbox = InboxChannel()
    inbox_receipt = inbox.deliver(recipient, message)
    assert inbox_receipt.reference == "inbox-0001"
    assert inbox.items == ((recipient, message),)

    flaky = _FailOnceChannel()
    recovered = RetryingChannel(flaky, max_attempts=2).deliver(recipient, message)
    assert recovered.reference == "recovered-0001"
    assert flaky.attempts == 2

    fallback_receipt = deliver_with_fallback(
        [_AlwaysFailsChannel(), inbox],
        recipient,
        message,
    )
    assert fallback_receipt.channel == "inbox"

    invalid_calls = (
        lambda: Recipient("No destination"),
        lambda: Recipient("Bad email", email="@example.com"),
        lambda: Recipient("Bad phone", phone="12"),
        lambda: Message("", "body"),
        lambda: EmailChannel("invalid-sender"),
        lambda: RetryingChannel(email, 0),
        lambda: RetryingChannel(email, True),
        lambda: NotificationService([]),
        lambda: NotificationService([email, EmailChannel("other@example.com")]),
        lambda: deliver_with_fallback([], recipient, message),
        lambda: deliver_with_fallback(
            [_AlwaysFailsChannel()],
            recipient,
            message,
        ),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (DeliveryError, TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run safe, deterministic demonstrations without external services."""
    run_self_checks()

    recipient = Recipient(
        "Bikash",
        email="bikash@example.com",
        phone="+977-981-234-5678",
    )
    message = Message(
        "Practice reminder",
        "Review inheritance, then refactor one hierarchy using composition.",
    )
    service = NotificationService(
        [EmailChannel("course@example.com"), SmsChannel(), InboxChannel()]
    )

    for receipt in service.notify(recipient, message):
        print(
            f"{receipt.channel}: delivered to {receipt.recipient_name} "
            f"({receipt.reference})"
        )
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
