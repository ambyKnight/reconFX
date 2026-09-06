"""
Confidence, derived from outcomes instead of asserted.

This exists because of a specific incident. The group-allocation tier shipped
with `confidence = 0.85` and measured 35% precision (42.6% on the main variant,
0% on the date-window variant, n=148). Nothing was wrong with the arithmetic;
the number was simply typed by a person who hoped it was right, against a bar of
99.8%. PIPELINE.md §1.5 lists the others still in the codebase — 0.95, 0.80,
0.35, DATE_WINDOW_DAYS, COLLISION_LIMIT — and says every one should be fit, not
chosen. This module is how a tier gets a number.

Why a lower bound, not the observed rate
----------------------------------------
The naive fix for 0.85-vs-35% is to write 0.35 instead. That repeats the mistake
in the other direction, because a rate measured on 148 items is not known to
±0.002 — and 99.8% is a bar you can only clear by knowing the rate that well.
So a tier is credited with the **Wilson score lower bound** at 95%: the worst
precision consistent with what was actually observed.

This makes the sample size do the work. A tier with 148/148 correct — a perfect
record — earns a lower bound near 0.975, which does *not* clear 0.998, so it
cannot go AUTO no matter how clean it looks. Clearing 0.998 on a flawless record
takes roughly 1,900 observations. That is not a policy choice to argue about; it
is what the evidence supports, and it means a new tier is REVIEW by construction
until it has earned its way out. The 0.85 incident could not have happened with
this in the loop: the tier had n=148, so the ceiling on any claim it could make
was ~0.975 before a single error was counted.

Wilson rather than the textbook normal interval because the normal interval is
useless in exactly the region we care about — near p=1 it produces bounds above
1.0 and degenerate widths at small n, which is the whole 0.95–1.00 band that the
AUTO decision reads (PIPELINE.md §Stage 5).

Where this sits
---------------
Stage 5 of PIPELINE.md, in the shape that is honest with the data we have. The
document specifies isotonic regression over a learned continuous score; that is
the right destination and this is not a substitute for it. But isotonic needs a
scorer (stage 4) that does not exist yet, and the tiers are discrete today. Per
bucket, with a bound that respects n, is the same idea at the resolution
currently available — and it is strictly better than a literal, which is the
thing actually in the repo right now.

Deliberately stdlib-only, so calibration never becomes the reason the evaluation
loop needs a dependency.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

# PIPELINE.md's contract: precision among accepted items.
REQUIRED_PRECISION = 0.998

# 95% one-sided. Not tuned — moving it to make a tier pass would be the 0.85
# mistake wearing a statistics costume.
Z = 1.959963984540054


def wilson_lower_bound(correct: int, n: int, z: float = Z) -> float:
    """Lower end of the Wilson score interval for `correct`/`n`.

    Returns 0.0 for n == 0: a tier with no observations has earned no claim.
    That is the correct answer for a newly added tier, and it is why adding one
    cannot silently start auto-posting.
    """
    if n <= 0:
        return 0.0
    p = correct / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def observations_needed(target: float = REQUIRED_PRECISION, z: float = Z) -> int:
    """How many consecutive correct calls a tier needs to clear `target`.

    Answers "when could this tier be trusted?" with a number instead of a
    conversation. Reported by `Calibration.report()` for tiers that fall short
    on sample size rather than on accuracy — the two failures look identical in
    a precision column and want opposite responses.
    """
    n = 1
    while wilson_lower_bound(n, n, z) < target and n < 10_000_000:
        n *= 2
    lo, hi = n // 2, n
    while lo < hi:
        mid = (lo + hi) // 2
        if wilson_lower_bound(mid, mid, z) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


@dataclass(frozen=True)
class Bucket:
    """What one tier has actually earned."""

    key: str
    n: int
    correct: int
    lower_bound: float

    @property
    def observed(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def auto_eligible(self) -> bool:
        return self.lower_bound >= REQUIRED_PRECISION


@dataclass
class Calibration:
    """Fitted per-tier confidence. Serialisable so a run can record what it used.

    PIPELINE.md §Stage 0 wants a config hash in the run manifest; a calibration
    fitted on one split and applied to another is exactly the kind of thing that
    has to be pinned for a result to mean anything, so `to_json`/`from_json`
    exist to be written into the manifest rather than reconstructed later.
    """

    buckets: dict[str, Bucket]
    target: float = REQUIRED_PRECISION

    @classmethod
    def fit(cls, records: Iterable[tuple[str, bool]], target: float = REQUIRED_PRECISION) -> "Calibration":
        """Fit from `(tier_key, was_correct)` pairs.

        Must be fit on a split disjoint from the one it is reported on
        (PIPELINE.md §Stage 5-6). Fitting on eval and reporting on eval
        reproduces §1.1's leakage caveat, where an 81.1% match rate was
        knowingly not a result.
        """
        tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for key, ok in records:
            tally[key][0] += 1
            tally[key][1] += bool(ok)
        return cls(
            {k: Bucket(k, n, c, wilson_lower_bound(c, n)) for k, (n, c) in tally.items()},
            target,
        )

    @classmethod
    def fit_folds(cls, records: Iterable[tuple[object, str, bool]],
                  target: float = REQUIRED_PRECISION) -> "Calibration":
        """Fit across folds, crediting each tier the **worst** fold it saw.

        Wilson answers "how much evidence is there?" but not "does this tier
        behave the same everywhere?", and those come apart badly in practice.
        Measured on BenchRec train, the collision band k=8-11 was clean on all
        1,682 observations — a perfect record, earning a lower bound of 0.99772,
        which ranked it *above* the k=1 band and its 44,267 observations. On
        eval that same band scored 93.47%. The record was real; the stability
        was not, and a single pooled bound cannot tell those apart because
        sampling error is not the thing that went wrong.

        Splitting train into folds and crediting the minimum makes a tier prove
        itself repeatedly rather than once. A band that is clean because it
        happens to cover one well-behaved cluster shows its variance across
        folds; a genuinely uniform band barely moves. This is ordinary
        cross-fitting, and it is the cheapest available guard against the
        failure that actually occurred.

        `records` is `(fold_id, tier_key, was_correct)`. Folds must be assigned
        by something stable and content-derived — a hash of the match group, not
        row order — or the split leaks structure it was meant to test.
        """
        per_fold: dict[object, list[tuple[str, bool]]] = defaultdict(list)
        keys: set[str] = set()
        for fold, key, ok in records:
            per_fold[fold].append((key, ok))
            keys.add(key)

        fits = {fold: cls.fit(rows, target) for fold, rows in per_fold.items()}
        buckets: dict[str, Bucket] = {}
        for key in keys:
            seen = [f.buckets[key] for f in fits.values() if key in f.buckets]
            worst = min(b.lower_bound for b in seen)
            buckets[key] = Bucket(key, sum(b.n for b in seen),
                                  sum(b.correct for b in seen), worst)
        return cls(buckets, target)

    def confidence(self, key: str) -> float:
        """Calibrated confidence for a tier — 0.0 if it has never been measured.

        An unmeasured tier scoring 0.0 is the intended behaviour, not a gap: it
        routes to REVIEW. Silence about a tier is not evidence for it.
        """
        bucket = self.buckets.get(key)
        return bucket.lower_bound if bucket else 0.0

    def verdict(self, key: str) -> str:
        """AUTO only where the evidence carries it; REVIEW otherwise."""
        return "AUTO" if self.confidence(key) >= self.target else "REVIEW"

    def report(self) -> str:
        """Per-tier table: observed rate, what it supports, and what it needs."""
        need = observations_needed(self.target)
        rows = [f"{'tier':<44} {'n':>6} {'observed':>9} {'lower':>8} {'verdict':>8}  note"]
        for b in sorted(self.buckets.values(), key=lambda b: -b.n):
            note = ""
            if not b.auto_eligible:
                note = (f"clean but n<{need:,}" if b.correct == b.n
                        else f"{b.n - b.correct} error(s)")
            rows.append(
                f"{b.key:<44} {b.n:>6} {100 * b.observed:>8.2f}% "
                f"{100 * b.lower_bound:>7.2f}% {('AUTO' if b.auto_eligible else 'REVIEW'):>8}  {note}"
            )
        return "\n".join(rows)

    def to_json(self) -> str:
        return json.dumps(
            {"target": self.target,
             "buckets": {k: {"n": b.n, "correct": b.correct, "lower_bound": b.lower_bound}
                         for k, b in self.buckets.items()}},
            indent=2, sort_keys=True,
        )

    @classmethod
    def from_json(cls, blob: str) -> "Calibration":
        data = json.loads(blob)
        return cls(
            {k: Bucket(k, v["n"], v["correct"], v["lower_bound"])
             for k, v in data["buckets"].items()},
            data["target"],
        )


def tier_key(assembly, variant: str = "exact") -> str:
    """The bucket an `Assembly` belongs to.

    Deliberately fine-grained. The incident report separates the main variant
    (42.6%) from the date-window variant (0%) because they behave nothing alike,
    and averaging them into one 35% number would have hidden a tier that is
    wrong every single time. Anything that changes the error rate belongs in the
    key, so it gets its own bound and can be retired on its own evidence.
    """
    f = getattr(assembly, "features", {}) or {}
    parts = [f"group:{assembly.verdict.lower()}", variant]
    if assembly.verdict in ("UNIQUE", "CORROBORATED", "AMBIGUOUS"):
        size = f.get("subset_size") or 0
        parts.append(f"size{min(size, 4)}{'+' if size > 4 else ''}")
        if not f.get("search_exhaustive", False):
            # A truncated search is a different animal from a completed one and
            # must never borrow the completed one's track record.
            parts.append("truncated")
    return "/".join(parts)


def summarise(records: Sequence[tuple[str, bool]]) -> str:
    return Calibration.fit(records).report()
