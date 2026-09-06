"""
Phase 3: make the model usable — reliable, parallel, and honest about failing.

Wraps a raw `adjudicate(item, candidates) -> verdict` in the things a single
call does not give you. Composable and each independently optional, so a caller
takes only what it needs:

    call = retrying(adjudicate)                  # flaky output
    call = concurrent(call, workers=8)           # latency
    call = self_consistent(call, agree=2)        # accuracy, at double cost

Three measured problems this exists to solve
--------------------------------------------
**1. Structured output is flaky and fails silently.** Three identical texture
calls to `glm-4-7-flash` returned 8, then 0, then 7 usable entries. The zero was
a well-formed `tool_calls` block whose contents failed validation — nothing
raised. A wrapper that only retries on exceptions would not have retried, so
`retrying` validates the *shape* of the reply and treats a malformed one as a
failure worth another attempt.

**2. Latency makes it unusable at volume.** Measured at ~13 seconds per call
against the live endpoint, sequential. Stage 4 sees ~44% of payments, so a
1,500-payment ledger is ~2.4 hours of wall clock. That is not a tuning problem,
it is the difference between a component that can run in a close cycle and one
that cannot. `concurrent` bounds the parallelism rather than removing it —
an unbounded fan-out at an endpoint with unknown rate limits trades one
unusable failure mode for another.

**3. A broken stage looks like a cautious one.** Both produce no matches and a
full review queue. `CircuitBreaker` makes the difference visible and stops
paying for calls that are all failing.

What this does NOT do
---------------------
It does not touch the verdict's content, and it does not retry a model that
answered clearly. An `UNSURE` is a *successful* call — abstention is the
behaviour this project wants, and a harness that retried until it got a claim
would be manufacturing confidence, which is the exact failure the arithmetic
gate exists to catch downstream. Only malformed replies are retried.
"""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Sequence

VALID_VERDICTS = ("MATCH", "MATCH_GROUP", "NO_MATCH", "UNSURE")

# Sized against a measured 60 requests/minute limit and ~13s per call.
# Throughput with W workers is W x 60/13 requests per minute, so W=13 sits
# exactly on the limit and W=10 leaves headroom for the retries in `retrying`,
# which consume the same budget and are not free.
#
# Stated as arithmetic rather than a tuned number so it can be re-derived when
# either input changes: RATE_LIMIT_PER_MIN * SECONDS_PER_CALL / 60, minus slack.
RATE_LIMIT_PER_MIN = 60
SECONDS_PER_CALL = 13.0          # measured, live endpoint, sequential
DEFAULT_WORKERS = 10             # ~46 req/min, leaving room for retries
DEFAULT_ATTEMPTS = 3


def workers_for(rate_limit_per_min: int = RATE_LIMIT_PER_MIN,
                seconds_per_call: float = SECONDS_PER_CALL,
                headroom: float = 0.8) -> int:
    """Largest safe worker count for a given rate limit.

    Retries share the same budget, so `headroom` reserves part of it rather than
    running at exactly the ceiling and discovering the limit the hard way.
    """
    return max(1, int(rate_limit_per_min * seconds_per_call / 60 * headroom))


class MalformedVerdict(ValueError):
    """The call succeeded but the reply is not a usable verdict."""


def validate_verdict(verdict: object) -> dict:
    """Check the shape of a reply, and raise if it cannot be acted on.

    Shape only — never whether the answer is *right*, which is the arithmetic
    gate's job in `pipeline.verify_arithmetic` and not something a wrapper
    should start second-guessing.
    """
    if not isinstance(verdict, dict):
        raise MalformedVerdict(f"expected dict, got {type(verdict).__name__}")
    kind = verdict.get("verdict")
    if kind not in VALID_VERDICTS:
        raise MalformedVerdict(f"unknown verdict {kind!r}")

    indices = verdict.get("candidate_indices", [])
    if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
        raise MalformedVerdict("candidate_indices must be a list of ints")
    if kind in ("MATCH", "MATCH_GROUP") and not indices:
        raise MalformedVerdict(f"{kind} named no candidates")
    return verdict


@dataclass
class Stats:
    """What the harness observed. Reported, never silently discarded."""

    calls: int = 0
    retries: int = 0
    failures: Counter = field(default_factory=Counter)
    seconds: float = 0.0

    @property
    def per_call(self) -> float:
        return self.seconds / self.calls if self.calls else 0.0

    def render(self) -> str:
        out = [f"{self.calls} calls, {self.per_call:.1f}s/call, {self.retries} retries"]
        for reason, n in self.failures.most_common():
            out.append(f"  {n:>4}  {reason}")
        return "\n".join(out)


class CircuitBreaker:
    """Stop paying for calls that are all failing, and say so.

    Opens after `threshold` consecutive failures. Once open every further call
    fails immediately, which turns a slow, expensive, invisible outage into a
    fast and legible one. Any success closes it — a transient blip should not
    disable a working stage for the rest of a run.
    """

    def __init__(self, threshold: int = 5):
        self.threshold = threshold
        self.consecutive = 0
        self.tripped = False

    def record(self, ok: bool) -> None:
        self.consecutive = 0 if ok else self.consecutive + 1
        if self.consecutive >= self.threshold:
            self.tripped = True

    def check(self) -> None:
        if self.tripped:
            raise RuntimeError(
                f"circuit open after {self.consecutive} consecutive failures — "
                f"stage is down, not abstaining")


def retrying(adjudicate: Callable, attempts: int = DEFAULT_ATTEMPTS,
             stats: Stats | None = None, breaker: CircuitBreaker | None = None,
             backoff: float = 0.5) -> Callable:
    """Retry malformed or failed replies, up to `attempts`.

    Retries a *malformed* reply, not an unwelcome one: `UNSURE` and `NO_MATCH`
    are successful calls and are returned untouched. Retrying until the model
    produces a claim would manufacture confidence, which is precisely what this
    codebase is built to avoid.
    """
    stats = stats if stats is not None else Stats()

    def call(item, candidates):
        if breaker:
            breaker.check()
        last = None
        for attempt in range(attempts):
            started = time.time()
            try:
                verdict = validate_verdict(adjudicate(item, candidates))
                stats.calls += 1
                stats.seconds += time.time() - started
                if breaker:
                    breaker.record(True)
                return verdict
            except Exception as exc:
                stats.calls += 1
                stats.seconds += time.time() - started
                stats.failures[f"{type(exc).__name__}: {str(exc)[:60]}"] += 1
                last = exc
                if attempt + 1 < attempts:
                    stats.retries += 1
                    time.sleep(backoff * (attempt + 1))
        if breaker:
            breaker.record(False)
        raise last if last else RuntimeError("no attempts made")

    call.stats = stats
    return call


def concurrent(adjudicate: Callable, workers: int = DEFAULT_WORKERS) -> Callable:
    """Run many adjudications in parallel, preserving input order.

    Returns a function over a *list* of `(item, candidates)` pairs rather than a
    drop-in single-item call, because the parallelism has to be visible to the
    caller: a stage that silently fans out is a stage whose cost and rate-limit
    exposure cannot be reasoned about.

    Exceptions are returned in place rather than raised, so one bad item cannot
    discard a batch of expensive successful calls.
    """
    def call_batch(pairs: Sequence[tuple]) -> list:
        if not pairs:
            return []

        def one(pair):
            try:
                return adjudicate(*pair)
            except Exception as exc:
                return exc

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, pairs))

    return call_batch


def self_consistent(adjudicate: Callable, agree: int = 2,
                    attempts: int = 3) -> Callable:
    """Accept a claim only when independent calls agree on it.

    Costs `agree` calls minimum and up to `attempts`. Disagreement is resolved
    as `UNSURE` rather than by majority: if the model gives different answers to
    the same question, that *is* the finding, and picking the more popular one
    would discard exactly the signal worth keeping.

    Whether this earns its cost is an empirical question — stage 4 sees ~44% of
    payments, so doubling its calls is a real budget decision. It is built here
    so the trade-off can be measured rather than argued.
    """
    def call(item, candidates):
        seen: list[dict] = []
        for _ in range(attempts):
            verdict = adjudicate(item, candidates)
            seen.append(verdict)
            key = _claim_key(verdict)
            if sum(_claim_key(v) == key for v in seen) >= agree:
                if key is None:      # agreed on abstaining
                    return verdict
                return {**verdict, "self_consistent_runs": len(seen)}
        return {"verdict": "UNSURE", "candidate_indices": [], "confidence": 0.0,
                "reason": f"the model gave {len({_claim_key(v) for v in seen})} "
                          f"different answers across {len(seen)} attempts",
                "key_facts": [], "self_consistent_runs": len(seen)}

    return call


def _claim_key(verdict: dict) -> tuple | None:
    """What a verdict actually claims, for comparison. None means abstained."""
    if verdict.get("verdict") not in ("MATCH", "MATCH_GROUP"):
        return None
    return tuple(sorted(verdict.get("candidate_indices", [])))
