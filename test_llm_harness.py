"""
Tests for the reliability harness.

The one that matters most is `test_an_abstention_is_never_retried`: retrying
until the model produces a claim would manufacture confidence, which is the
failure this entire codebase is built to prevent. Everything else is about
failing loudly instead of quietly.

Runs under pytest, or standalone (`python test_llm_harness.py`).
"""

from llm_harness import (CircuitBreaker, MalformedVerdict, Stats, concurrent,
                         retrying, self_consistent, validate_verdict,
                         workers_for)


def claim(indices=(0,), kind="MATCH"):
    return {"verdict": kind, "candidate_indices": list(indices),
            "confidence": 0.7, "reason": "", "key_facts": []}


def abstain(kind="UNSURE"):
    return {"verdict": kind, "candidate_indices": [], "confidence": 0.2,
            "reason": "", "key_facts": []}


# --- shape validation --------------------------------------------------------

def test_a_well_formed_claim_passes():
    assert validate_verdict(claim())["verdict"] == "MATCH"


def test_abstentions_are_well_formed_without_candidates():
    for kind in ("UNSURE", "NO_MATCH"):
        assert validate_verdict(abstain(kind))["verdict"] == kind


def test_malformed_replies_are_rejected():
    for bad in (None, "MATCH", [], {"verdict": "MAYBE", "candidate_indices": []},
                {"verdict": "MATCH", "candidate_indices": []},
                {"verdict": "MATCH", "candidate_indices": "0"},
                {"verdict": "MATCH", "candidate_indices": [0, "x"]}):
        try:
            validate_verdict(bad)
        except MalformedVerdict:
            continue
        raise AssertionError(f"accepted malformed reply: {bad!r}")


def test_validation_checks_shape_not_correctness():
    """A claim naming candidate 999 is well-formed; whether it is *right* is the
    arithmetic gate's job, and a wrapper second-guessing it would duplicate the
    decision in two places."""
    assert validate_verdict(claim([999]))


# --- retrying ----------------------------------------------------------------

def test_a_malformed_reply_is_retried_then_succeeds():
    """The measured failure: a well-formed call whose contents are unusable.
    Nothing raises, so a wrapper that only caught exceptions would not retry."""
    replies = [{"verdict": "MATCH", "candidate_indices": []}, claim()]
    call = retrying(lambda i, c: replies.pop(0), attempts=3)
    assert call(None, None)["verdict"] == "MATCH"
    assert call.stats.retries == 1


def test_an_abstention_is_never_retried():
    """UNSURE is a successful call. Retrying for a claim manufactures confidence."""
    calls = {"n": 0}

    def model(i, c):
        calls["n"] += 1
        return abstain()

    assert retrying(model, attempts=5)(None, None)["verdict"] == "UNSURE"
    assert calls["n"] == 1, "an abstention was retried"


def test_exhausted_retries_raise_rather_than_return_a_guess():
    def broken(i, c):
        return {"verdict": "NONSENSE"}

    call = retrying(broken, attempts=2, backoff=0)
    try:
        call(None, None)
    except MalformedVerdict:
        assert call.stats.retries == 1
        return
    raise AssertionError("a broken model returned something usable")


def test_stats_record_what_happened():
    replies = [{"verdict": "?"}, claim()]
    stats = Stats()
    retrying(lambda i, c: replies.pop(0), stats=stats, backoff=0)(None, None)
    assert stats.calls == 2 and stats.retries == 1
    assert "MalformedVerdict" in stats.render()


# --- circuit breaker ---------------------------------------------------------

def test_the_breaker_opens_after_repeated_failure():
    """A stage failing on everything must become loud, not just unproductive."""
    breaker = CircuitBreaker(threshold=2)
    call = retrying(lambda i, c: (_ for _ in ()).throw(RuntimeError("down")),
                    attempts=1, breaker=breaker, backoff=0)
    for _ in range(2):
        try:
            call(None, None)
        except Exception:
            pass
    assert breaker.tripped
    try:
        call(None, None)
    except RuntimeError as exc:
        assert "stage is down" in str(exc)
        return
    raise AssertionError("open circuit still made calls")


def test_a_success_closes_the_breaker():
    """A transient blip must not disable a working stage for the whole run."""
    breaker = CircuitBreaker(threshold=3)
    breaker.record(False)
    breaker.record(False)
    breaker.record(True)
    assert breaker.consecutive == 0 and not breaker.tripped


# --- concurrency -------------------------------------------------------------

def test_concurrent_preserves_order():
    """Results must line up with inputs, or every score is attributed wrongly."""
    batch = concurrent(lambda item, c: {"id": item}, workers=4)
    assert [r["id"] for r in batch([(i, None) for i in range(20)])] == list(range(20))


def test_one_failure_does_not_discard_the_batch():
    """Expensive successful calls must survive a single bad item."""
    def flaky(item, c):
        if item == 3:
            raise ValueError("bad item")
        return {"id": item}

    results = concurrent(flaky, workers=4)([(i, None) for i in range(6)])
    assert isinstance(results[3], ValueError)
    assert [r["id"] for i, r in enumerate(results) if i != 3] == [0, 1, 2, 4, 5]


def test_empty_batch_is_fine():
    assert concurrent(lambda i, c: 1)([]) == []


def test_worker_count_is_derived_from_the_rate_limit():
    """Sized by arithmetic, not by taste: 60/min at 13s per call is ~13 in
    flight, minus headroom for retries."""
    assert workers_for(60, 13.0, headroom=1.0) == 13
    assert workers_for(60, 13.0) == 10
    assert workers_for(6, 13.0) == 1
    assert workers_for(1, 0.1) >= 1


# --- self-consistency --------------------------------------------------------

def test_agreement_returns_the_claim():
    call = self_consistent(lambda i, c: claim([1, 2]), agree=2)
    result = call(None, None)
    assert result["verdict"] == "MATCH" and result["self_consistent_runs"] == 2


def test_disagreement_becomes_unsure_not_a_majority_vote():
    """If the model answers differently each time, that IS the finding. Taking
    the most popular answer would discard exactly the signal worth keeping."""
    replies = [claim([0]), claim([1]), claim([2])]
    call = self_consistent(lambda i, c: replies.pop(0), agree=2, attempts=3)
    result = call(None, None)
    assert result["verdict"] == "UNSURE"
    assert "different answers" in result["reason"]


def test_agreeing_to_abstain_is_still_an_abstention():
    call = self_consistent(lambda i, c: abstain(), agree=2)
    assert call(None, None)["verdict"] == "UNSURE"


def test_self_consistency_stops_as_soon_as_it_agrees():
    """It costs calls; it must not spend more than it needs."""
    calls = {"n": 0}

    def model(i, c):
        calls["n"] += 1
        return claim([0])

    self_consistent(model, agree=2, attempts=5)(None, None)
    assert calls["n"] == 2


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as exc:
            failed.append(name)
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
