"""Day 36: Caching strategies, invalidation, and consistency.

Learning goals
--------------
1. Decide what to cache by measuring latency, reuse, staleness tolerance, and cost.
2. Build a bounded time-to-live cache with least-recently-used eviction.
3. Apply cache-aside loading while preserving cached None values.
4. Prevent stale fills from winning a race with writes by versioning namespaces.
5. Test expiration and invalidation deterministically with an injected clock.

Teaching notes
--------------
- A cache is a disposable copy, never the source of truth. Correctness must survive
  an empty cache, eviction, process restart, and a temporarily unavailable cache.
- Cache keys are part of the data model. Include every input that changes a result:
  tenant, locale, permissions, query options, and a schema or namespace version.
- TTL limits how long stale data may remain; it does not guarantee freshness.
  Choose it from a business staleness budget, then measure hit rate and load cost.
- Cache-aside reads the cache first, loads the source on a miss, and stores the
  result. Writes normally update the source first and then invalidate cached data.
- Deleting one key can lose a race: an earlier slow reader may refill stale data
  after the delete. A generation token makes that stale fill unreachable.
- Negative caching stores "not found" briefly. It protects the source from repeated
  misses, but its TTL should usually be shorter so newly created data appears soon.
- Capacity is a correctness concern as well as a memory concern. An unbounded cache
  can become a memory leak. LRU is useful, but real workloads may need other rules.
- Thread safety prevents corrupted local state; it does not prevent a cache
  stampede. Production systems may add per-key single-flight locks, TTL jitter,
  request coalescing, or carefully bounded stale-while-revalidate behavior.
- Cache metrics should separate hits, misses, expirations, evictions, load latency,
  and source errors. Never put secrets or personal data in keys or diagnostic logs.
- Distributed caches add serialization, network failure, partial availability, and
  cross-service invalidation. Prefer simple policies with explicit failure modes.

Run this file with Python 3.10+.
It uses only the standard library, opens no network connections, and writes no files.

Practice exercises
------------------
1. Implement hit_ratio(stats). Return zero before any lookup and otherwise return
   hits divided by hits plus misses.
2. Implement warm_catalog(service, product_ids). Load each distinct identifier once,
   preserve first-seen order, and return only products that exist.
3. Implement VersionedCache.get_or_load(). Retry when a namespace changes during a
   source load so stale data is never published under the current generation.

Solutions appear in the executable code below.

Expert challenge: a production cache layer
------------------------------------------
Build a two-level cache for a read-heavy catalog: a small in-process L1 and a
shared L2 adapter. Add per-key stampede protection, bounded TTL jitter, negative
caching, namespace invalidation, and an option to bypass caches for repair tools.
Emit metrics for each layer and prove that source failures never overwrite a good
entry. Document the consistency promise and the behavior when L2 is unavailable.

Solution guidance
-----------------
1. Define a Cache protocol so the L2 can be replaced by an in-memory fake in tests.
2. Put canonical, bounded, non-secret fields in keys and version serialized values.
3. Read L1, then L2, then the source. Promote an L2 hit to L1 with a shorter TTL.
4. On writes, commit to the source before invalidating. Use an outbox or change-data
   capture when database and distributed-cache updates cannot be one transaction.
5. Allow only one loader per hot key. Waiting callers should share its result, but
   cancellation and exceptions must release the single-flight slot.
6. Add random TTL jitter from an injected random source to avoid mass expiration.
7. Decide whether stale-while-revalidate is acceptable per data class. Never use it
   for authorization decisions or other data that requires immediate revocation.
8. Test an empty cache, cached None, LRU order, expiration boundaries, concurrent
   invalidation, repeated invalidation, source exceptions, and cache unavailability.

"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
import math
from threading import RLock
import time
from typing import Generic, Protocol, TypeVar


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class CacheError(Exception):
    """Base class for expected cache failures."""


class ValidationError(CacheError, ValueError):
    """A cache policy or domain value is invalid."""


class CacheMiss(CacheError, KeyError):
    """A key has no live cached value."""


class ConcurrentInvalidationError(CacheError):
    """A namespace changed too often to publish a loaded value safely."""


class ConflictError(CacheError):
    """A source write used an unexpected revision."""


def _clean_name(value: str, *, field: str) -> str:
    """Return bounded text suitable for identifiers and cache-key components."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be text")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 100:
        raise ValidationError(f"{field} must be bounded non-empty text")
    return cleaned


def _positive_float(value: int | float, *, field: str) -> float:
    """Return a finite positive float while rejecting Boolean integers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValidationError(f"{field} must be finite and positive")
    return converted


def _positive_int(value: int, *, field: str) -> int:
    """Return a positive integer while rejecting Boolean integers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value <= 0:
        raise ValidationError(f"{field} must be positive")
    return value


def _require_hashable(value: object, *, field: str) -> None:
    """Raise a clear error before a value reaches an internal dictionary."""
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(f"{field} must be hashable") from error


class ManualClock:
    """A deterministic monotonic clock for examples and tests."""

    def __init__(self, start: int | float = 0.0) -> None:
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            raise TypeError("clock start must be a number")
        converted = float(start)
        if not math.isfinite(converted):
            raise ValidationError("clock start must be finite")
        self._value = converted
        self._lock = RLock()

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def advance(self, seconds: int | float) -> None:
        amount = _positive_float(seconds, field="clock advance")
        with self._lock:
            self._value += amount


@dataclass(frozen=True, slots=True)
class CacheStats:
    """A snapshot of cache lookup and capacity behavior."""

    hits: int
    misses: int
    expirations: int
    evictions: int

    @property
    def lookups(self) -> int:
        return self.hits + self.misses


@dataclass(frozen=True, slots=True)
class _Entry(Generic[V]):
    value: V
    expires_at: float


class TtlLruCache(Generic[K, V]):
    """A thread-safe, bounded TTL cache with LRU capacity eviction."""

    def __init__(
        self,
        capacity: int,
        ttl_seconds: int | float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._capacity = _positive_int(capacity, field="capacity")
        self._ttl_seconds = _positive_float(ttl_seconds, field="ttl_seconds")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._entries: OrderedDict[K, _Entry[V]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._expirations = 0
        self._evictions = 0
        self._lock = RLock()

    def _now(self) -> float:
        now = self._clock()
        if isinstance(now, bool) or not isinstance(now, (int, float)):
            raise TypeError("clock must return a number")
        converted = float(now)
        if not math.isfinite(converted):
            raise ValidationError("clock must return a finite value")
        return converted

    def get(self, key: K) -> V:
        """Return a live value and update LRU order, or raise CacheMiss."""
        _require_hashable(key, field="cache key")
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                raise CacheMiss(key)
            if self._now() >= entry.expires_at:
                del self._entries[key]
                self._misses += 1
                self._expirations += 1
                raise CacheMiss(key)
            self._entries.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(
        self,
        key: K,
        value: V,
        *,
        ttl_seconds: int | float | None = None,
    ) -> None:
        """Insert or replace a value, evicting the least-recently-used key."""
        _require_hashable(key, field="cache key")
        ttl = (
            self._ttl_seconds
            if ttl_seconds is None
            else _positive_float(ttl_seconds, field="ttl_seconds")
        )
        expires_at = self._now() + ttl
        with self._lock:
            self._entries[key] = _Entry(value, expires_at)
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
                self._evictions += 1

    def delete(self, key: K) -> bool:
        """Delete a key if present and report whether anything was removed."""
        _require_hashable(key, field="cache key")
        with self._lock:
            return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        """Remove all entries without rewriting historical metrics."""
        with self._lock:
            self._entries.clear()

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                self._hits,
                self._misses,
                self._expirations,
                self._evictions,
            )

    def live_keys(self) -> tuple[K, ...]:
        """Return unexpired keys in LRU-to-MRU order."""
        with self._lock:
            now = self._now()
            expired = [
                key
                for key, entry in self._entries.items()
                if now >= entry.expires_at
            ]
            for key in expired:
                del self._entries[key]
                self._expirations += 1
            return tuple(self._entries)

    def get_or_load(
        self,
        key: K,
        loader: Callable[[], V],
        *,
        ttl_seconds: int | float | None = None,
    ) -> V:
        """Apply cache-aside; loader exceptions are never cached."""
        if not callable(loader):
            raise TypeError("loader must be callable")
        try:
            return self.get(key)
        except CacheMiss:
            value = loader()
            self.set(key, value, ttl_seconds=ttl_seconds)
            return value


def hit_ratio(stats: CacheStats) -> float:
    """Solution 1: calculate cache hits divided by all lookups."""
    if not isinstance(stats, CacheStats):
        raise TypeError("stats must be CacheStats")
    return 0.0 if stats.lookups == 0 else stats.hits / stats.lookups


class NamespaceVersions:
    """Thread-safe generation counters used for coarse invalidation."""

    def __init__(self) -> None:
        self._versions: dict[str, int] = {}
        self._lock = RLock()

    def current(self, namespace: str) -> int:
        cleaned = _clean_name(namespace, field="namespace")
        with self._lock:
            return self._versions.get(cleaned, 0)

    def bump(self, namespace: str) -> int:
        cleaned = _clean_name(namespace, field="namespace")
        with self._lock:
            version = self._versions.get(cleaned, 0) + 1
            self._versions[cleaned] = version
            return version


@dataclass(frozen=True, slots=True)
class VersionedKey:
    """A logical key made unreachable when its namespace generation changes."""

    namespace: str
    generation: int
    identity: Hashable

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "namespace", _clean_name(self.namespace, field="namespace")
        )
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ValidationError("generation must be a non-negative integer")
        _require_hashable(self.identity, field="identity")


@dataclass(frozen=True, slots=True)
class LoadResult(Generic[V]):
    """A loaded value and the TTL selected for its data class."""

    value: V
    ttl_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ttl_seconds",
            _positive_float(self.ttl_seconds, field="load TTL"),
        )


class VersionedCache(Generic[V]):
    """Cache-aside reads protected from stale-fill invalidation races."""

    def __init__(
        self,
        cache: TtlLruCache[VersionedKey, V],
        versions: NamespaceVersions,
        *,
        max_load_attempts: int = 3,
    ) -> None:
        if not isinstance(cache, TtlLruCache):
            raise TypeError("cache must be TtlLruCache")
        if not isinstance(versions, NamespaceVersions):
            raise TypeError("versions must be NamespaceVersions")
        self._cache = cache
        self._versions = versions
        self._max_load_attempts = _positive_int(
            max_load_attempts, field="max_load_attempts"
        )

    def get_or_load(
        self,
        namespace: str,
        identity: Hashable,
        loader: Callable[[], LoadResult[V]],
    ) -> V:
        """Solution 3: retry if invalidation overlaps a source load."""
        cleaned = _clean_name(namespace, field="namespace")
        _require_hashable(identity, field="identity")
        if not callable(loader):
            raise TypeError("loader must be callable")

        for _ in range(self._max_load_attempts):
            generation = self._versions.current(cleaned)
            key = VersionedKey(cleaned, generation, identity)
            try:
                return self._cache.get(key)
            except CacheMiss:
                loaded = loader()
                if not isinstance(loaded, LoadResult):
                    raise TypeError("loader must return LoadResult")
                if self._versions.current(cleaned) == generation:
                    self._cache.set(
                        key,
                        loaded.value,
                        ttl_seconds=loaded.ttl_seconds,
                    )
                    return loaded.value
        raise ConcurrentInvalidationError(
            f"namespace {cleaned!r} changed during every load attempt"
        )

    def invalidate_namespace(self, namespace: str) -> int:
        """Advance a namespace generation after committing a source write."""
        return self._versions.bump(namespace)


@dataclass(frozen=True, slots=True)
class Product:
    """An immutable source record safe to share through this local cache."""

    product_id: str
    name: str
    price_cents: int
    revision: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "product_id", _clean_name(self.product_id, field="product_id")
        )
        object.__setattr__(self, "name", _clean_name(self.name, field="name"))
        if (
            isinstance(self.price_cents, bool)
            or not isinstance(self.price_cents, int)
            or self.price_cents < 0
        ):
            raise ValidationError("price_cents must be a non-negative integer")
        _positive_int(self.revision, field="revision")


class ProductSource(Protocol):
    """The source-of-truth boundary used by CatalogService."""

    def get(self, product_id: str) -> Product | None:
        ...

    def save(self, product: Product) -> None:
        ...


class MemoryProductSource:
    """A deterministic source fake with optimistic revision checks."""

    def __init__(self, products: Iterable[Product] = ()) -> None:
        self._products: dict[str, Product] = {}
        self._reads = 0
        self._lock = RLock()
        for product in products:
            if not isinstance(product, Product):
                raise TypeError("products must contain Product values")
            if product.product_id in self._products:
                raise ValidationError("product identifiers must be unique")
            self._products[product.product_id] = product

    @property
    def reads(self) -> int:
        with self._lock:
            return self._reads

    def get(self, product_id: str) -> Product | None:
        cleaned = _clean_name(product_id, field="product_id")
        with self._lock:
            self._reads += 1
            return self._products.get(cleaned)

    def save(self, product: Product) -> None:
        if not isinstance(product, Product):
            raise TypeError("product must be Product")
        with self._lock:
            current = self._products.get(product.product_id)
            expected_revision = 1 if current is None else current.revision + 1
            if product.revision != expected_revision:
                raise ConflictError(
                    f"expected revision {expected_revision}, "
                    f"received {product.revision}"
                )
            self._products[product.product_id] = product


class CatalogService:
    """Cache-aside catalog with short negative caching and version invalidation."""

    _NAMESPACE = "products"

    def __init__(
        self,
        source: ProductSource,
        cache: VersionedCache[Product | None],
        *,
        positive_ttl: int | float = 30.0,
        negative_ttl: int | float = 3.0,
    ) -> None:
        if not callable(getattr(source, "get", None)) or not callable(
            getattr(source, "save", None)
        ):
            raise TypeError("source must provide get and save")
        if not isinstance(cache, VersionedCache):
            raise TypeError("cache must be VersionedCache")
        self._source = source
        self._cache = cache
        self._positive_ttl = _positive_float(
            positive_ttl, field="positive_ttl"
        )
        self._negative_ttl = _positive_float(
            negative_ttl, field="negative_ttl"
        )

    def get(self, product_id: str) -> Product | None:
        cleaned = _clean_name(product_id, field="product_id")

        def load() -> LoadResult[Product | None]:
            product = self._source.get(cleaned)
            ttl = (
                self._positive_ttl
                if product is not None
                else self._negative_ttl
            )
            return LoadResult(product, ttl)

        return self._cache.get_or_load(self._NAMESPACE, cleaned, load)

    def save(self, product: Product) -> int:
        """Commit to the source first, then make older cache keys unreachable."""
        self._source.save(product)
        return self._cache.invalidate_namespace(self._NAMESPACE)


def warm_catalog(
    service: CatalogService,
    product_ids: Iterable[str],
) -> tuple[Product, ...]:
    """Solution 2: warm each first-seen identifier and return found products."""
    if not isinstance(service, CatalogService):
        raise TypeError("service must be CatalogService")
    seen: set[str] = set()
    found: list[Product] = []
    for raw_id in product_ids:
        product_id = _clean_name(raw_id, field="product_id")
        if product_id in seen:
            continue
        seen.add(product_id)
        product = service.get(product_id)
        if product is not None:
            found.append(product)
    return tuple(found)


def run_self_checks() -> None:
    """Exercise cache hits, expiry, eviction, negative values, and races."""
    clock = ManualClock()
    cache: TtlLruCache[str, object] = TtlLruCache(
        capacity=2,
        ttl_seconds=10,
        clock=clock,
    )
    assert hit_ratio(cache.stats) == 0.0

    cache.set("none", None)
    assert cache.get("none") is None
    assert cache.stats == CacheStats(1, 0, 0, 0)

    cache.set("a", 1)
    assert cache.get("none") is None
    cache.set("b", 2)
    assert cache.live_keys() == ("none", "b")
    try:
        cache.get("a")
    except CacheMiss:
        pass
    else:
        raise AssertionError("least-recently-used key must be evicted")
    assert cache.stats.evictions == 1

    clock.advance(10)
    try:
        cache.get("none")
    except CacheMiss:
        pass
    else:
        raise AssertionError("value must expire at the TTL boundary")
    assert cache.stats.expirations == 1
    assert 0.0 < hit_ratio(cache.stats) < 1.0

    load_calls = 0

    def load_once() -> str:
        nonlocal load_calls
        load_calls += 1
        return "loaded"

    assert cache.get_or_load("load", load_once) == "loaded"
    assert cache.get_or_load("load", load_once) == "loaded"
    assert load_calls == 1

    source = MemoryProductSource(
        (Product("book", "Python Systems", 4500, 1),)
    )
    versions = NamespaceVersions()
    product_cache: TtlLruCache[VersionedKey, Product | None] = TtlLruCache(
        capacity=8,
        ttl_seconds=60,
        clock=clock,
    )
    catalog = CatalogService(
        source,
        VersionedCache(product_cache, versions),
        positive_ttl=30,
        negative_ttl=2,
    )

    assert catalog.get("book") == Product("book", "Python Systems", 4500, 1)
    assert catalog.get("book") == Product("book", "Python Systems", 4500, 1)
    assert source.reads == 1

    assert catalog.get("missing") is None
    assert catalog.get("missing") is None
    assert source.reads == 2
    clock.advance(2)
    assert catalog.get("missing") is None
    assert source.reads == 3

    new_generation = catalog.save(
        Product("book", "Python Systems, Second Edition", 5200, 2)
    )
    assert new_generation == 1
    assert catalog.get("book") == Product(
        "book", "Python Systems, Second Edition", 5200, 2
    )

    warmed = warm_catalog(catalog, ("book", "missing", "book"))
    assert tuple(product.product_id for product in warmed) == ("book",)

    race_source = MemoryProductSource(
        (Product("pen", "Old Pen", 100, 1),)
    )
    race_versions = NamespaceVersions()
    race_cache: TtlLruCache[VersionedKey, Product | None] = TtlLruCache(
        4, 60, clock=clock
    )
    protected = VersionedCache(race_cache, race_versions)
    attempts = 0

    def racing_loader() -> LoadResult[Product | None]:
        nonlocal attempts
        attempts += 1
        product = race_source.get("pen")
        if attempts == 1:
            race_source.save(Product("pen", "New Pen", 125, 2))
            race_versions.bump("products")
        return LoadResult(product, 30)

    loaded = protected.get_or_load("products", "pen", racing_loader)
    assert loaded == Product("pen", "New Pen", 125, 2)
    assert attempts == 2

    failing_cache: TtlLruCache[str, str] = TtlLruCache(1, 1, clock=clock)

    def fail_load() -> str:
        raise RuntimeError("source unavailable")

    for _ in range(2):
        try:
            failing_cache.get_or_load("x", fail_load)
        except RuntimeError:
            pass
        else:
            raise AssertionError("loader failures must propagate")
    assert failing_cache.stats.misses == 2

    invalid_calls: tuple[Callable[[], object], ...] = (
        lambda: TtlLruCache(0, 1),
        lambda: TtlLruCache(1, float("inf")),
        lambda: cache.set("x", 1, ttl_seconds=0),
        lambda: cache.get([]),  # type: ignore[arg-type]
        lambda: ManualClock(float("nan")),
        lambda: VersionedKey("products", -1, "x"),
        lambda: LoadResult("x", True),
        lambda: Product("x", "Example", -1, 1),
        lambda: source.save(Product("book", "Wrong revision", 1, 7)),
        lambda: warm_catalog(catalog, (" ",)),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, ValueError, CacheError):
            pass
        else:
            raise AssertionError("invalid input must raise a specific error")


def main() -> None:
    """Run a deterministic cache-aside catalog demonstration."""
    run_self_checks()

    clock = ManualClock()
    raw_cache: TtlLruCache[VersionedKey, Product | None] = TtlLruCache(
        capacity=16,
        ttl_seconds=60,
        clock=clock,
    )
    source = MemoryProductSource(
        (Product("course", "Production Python", 7900, 1),)
    )
    catalog = CatalogService(
        source,
        VersionedCache(raw_cache, NamespaceVersions()),
    )

    print("First read:", catalog.get("course"))
    print("Cached read:", catalog.get("course"))
    print("Source reads:", source.reads)
    print("Cache stats:", raw_cache.stats)
    print("Hit ratio:", f"{hit_ratio(raw_cache.stats):.0%}")
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
