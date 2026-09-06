"""
The cascade: ingest -> arithmetic -> group assembly -> AI -> triage -> report.

    1. take in dataset
    2. basic maths to match what it can
    3. group assembly (subset-sum) on the rest
    4. what's still left goes to the model
    5. label everything; anything not safely automatic goes to a human
    6. final accuracy list

Each stage consumes what it can settle and hands the remainder on, which is
ARCHITECTURE.md §1's "passes, not a pipeline": a stage never sees an item an
earlier stage already claimed, so the expensive stages only ever run on the
genuinely hard residue. Ordering is by cost, cheapest first — the model is last
because it is the only stage that costs money per item.

Two rules hold across every stage, and they are the reason this file exists
rather than one function per stage calling the next:

**No stage assigns its own confidence.** Stages emit *evidence* — what they
found, how many rivals existed, what corroborated it. Stage 5 attaches a number,
and only from `calibrate.py`, fitted on outcomes. This is the discipline that
was missing when a group tier shipped at a hand-typed 0.85 against a measured
35% precision, and it is enforced structurally: `Resolution.confidence` starts
as None and only `TriageStage` may set it.

**The model proposes, arithmetic disposes.** A verdict from stage 4 is re-checked
in Python before it counts (ARCHITECTURE.md §3.3): if the model says three
invoices settle a payment, we add them up ourselves. An LLM that cannot be
checked cannot be trusted with a number that reaches a regulator, and every claim
it makes here is one we can check.

Dataset-agnostic on purpose. Stage 1 is an adapter boundary: the cascade sees
`Item` and `Candidate`, never a CSV column name. BenchRec is one adapter;
`adapters.py`-shaped code for a real bank export is another, and nothing
downstream changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Iterable, Protocol, Sequence

from assemble import (CORROBORATED, Line, Payment, UNIQUE, _tokens_of, assemble,
                      explain)
from calibrate import REQUIRED_PRECISION, Calibration

# Verdicts the cascade routes on. AUTO posts without a human; REVIEW goes to the
# queue with a reason and a ranked shortlist; NO_MATCH is a first-class answer,
# not a failure — PIPELINE.md §Stage 6 is explicit that a system which cannot say
# "nothing matches" is the system this project replaces.
AUTO, REVIEW, NO_MATCH = "AUTO", "REVIEW", "NO_MATCH"


@dataclass(frozen=True)
class Item:
    """One entry needing resolution — a bank line, or an open intercompany item."""

    id: str
    amount_cents: int
    date: date | None = None
    references: str = ""
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    """One possible counterpart, already blocked in by stage 1."""

    id: str
    allocation: str
    amount_cents: int
    date: date | None = None
    references: str = ""

    def as_line(self) -> Line:
        return Line(self.id, self.allocation, self.amount_cents, self.date,
                    _tokens_of(self.references))


@dataclass
class Resolution:
    """What one stage concluded about one item.

    `confidence` and `decision` are deliberately None until stage 5. A stage that
    sets them itself is the bug this pipeline is shaped to prevent.
    """

    item_id: str
    stage: str
    allocations: tuple[str, ...]
    reason: str
    features: dict = field(default_factory=dict)
    confidence: float | None = None
    decision: str | None = None

    @property
    def tier(self) -> str:
        """The calibration bucket. Fine-grained on purpose: anything that changes
        the error rate needs its own track record, or a tier that is wrong every
        time hides inside a tier that is usually right."""
        parts = [self.stage, self.features.get("verdict", "?")]
        size = len(self.allocations)
        if size:
            parts.append(f"size{min(size, 4)}{'+' if size > 4 else ''}")
        if self.features.get("search_exhaustive") is False:
            parts.append("truncated")
        return "/".join(parts)


class Stage(Protocol):
    """Consume what you can settle; return the rest untouched."""

    name: str

    def run(self, items: Sequence[Item],
            candidates: Callable[[Item], Sequence[Candidate]]
            ) -> tuple[list[Resolution], list[Item]]:
        ...


# --- stage 2: basic maths ----------------------------------------------------

class ExactMatchStage:
    """Exact amount tie, collapsed by allocation.

    match.py's insight, kept: several ledger rows routinely share one allocation,
    so a 17-way tie on amount is not ambiguous if all 17 rows are the same
    allocation. Ambiguity is counted in allocations, never in rows.
    """

    name = "exact"

    def __init__(self, tolerance_cents: int = 0):
        self.tolerance_cents = tolerance_cents

    def run(self, items, candidates):
        resolved, remaining = [], []
        for item in items:
            ties = [c for c in candidates(item)
                    if abs(c.amount_cents - item.amount_cents) <= self.tolerance_cents]
            allocations = {c.allocation for c in ties}
            if len(allocations) == 1:
                resolved.append(Resolution(
                    item.id, self.name, (next(iter(allocations)),),
                    f"Exact amount tie on {len(ties)} ledger row(s), one allocation.",
                    {"verdict": UNIQUE, "n_rivals": 1, "n_rows": len(ties),
                     "search_exhaustive": True}))
            else:
                remaining.append(item)
        return resolved, remaining


# --- stage 3: group assembly -------------------------------------------------

class GroupAssemblyStage:
    """One payment covering several invoices (assemble.py).

    Only claims an item on UNIQUE or CORROBORATED. AMBIGUOUS and ABSTAIN are
    passed on deliberately — a stage that grabbed those would be tie-breaking
    silently, which is the failure mode the whole project exists to avoid. They
    are better served by the model, or by a human.
    """

    name = "group"
    CLAIMS = (UNIQUE, CORROBORATED)

    def __init__(self, tolerance_cents: int = 0):
        self.tolerance_cents = tolerance_cents

    def run(self, items, candidates):
        resolved, remaining = [], []
        for item in items:
            pool = [c.as_line() for c in candidates(item)]
            payment = Payment(item.id, item.amount_cents, item.date,
                              _tokens_of(item.references))
            result = assemble(pool, payment, tolerance_cents=self.tolerance_cents)

            if result.verdict in self.CLAIMS and result.chosen:
                resolved.append(Resolution(
                    item.id, self.name, result.chosen.allocations,
                    explain(result), {**result.features, "verdict": result.verdict}))
            else:
                remaining.append(item)
        return resolved, remaining


# --- stage 4: the model ------------------------------------------------------

def verify_arithmetic(allocations: Sequence[str], item: Item,
                      candidates: Sequence[Candidate], tolerance_cents: int) -> bool:
    """Gate 1 of ARCHITECTURE.md §3.3: re-check the model's claim in Python.

    The model selects candidates; it never computes. So whatever it selected must
    still add up, and we are the ones who add it up. This catches the failure that
    matters most — a fluent, well-argued verdict whose numbers do not tie — and it
    is cheap precisely because the model was never asked to do arithmetic.
    """
    chosen = [c for c in candidates if c.allocation in set(allocations)]
    if not chosen or len(set(allocations)) != len({c.allocation for c in chosen}):
        return False
    total = sum(c.amount_cents for c in chosen)
    return abs(total - item.amount_cents) <= tolerance_cents


class AdjudicatorStage:
    """LLM adjudication, gated (ARCHITECTURE.md §3.3).

    `adjudicate` is injected rather than imported so the cascade runs without an
    API key, without a network, and under test. Absent one, this stage claims
    nothing and everything flows to the human queue — which is the correct
    degradation: no model means more review, never more guessing.

    The model's own `confidence` is recorded as a *feature* and never used as a
    number. It is the model's opinion of itself, and this codebase has already
    paid once for trusting a self-reported confidence.
    """

    name = "ai"
    CLAIMS = ("MATCH", "MATCH_GROUP")

    def __init__(self, adjudicate: Callable | None = None, tolerance_cents: int = 0):
        self.adjudicate = adjudicate
        self.tolerance_cents = tolerance_cents
        # Errors are counted, not swallowed. Failing safe is right; failing
        # *silently* is not — a stage broken in every call looks identical to a
        # stage that considered every item and abstained, and the queue fills up
        # either way. This shipped broken once (the dossier field mismatch) and
        # every test passed, because they all injected a stub model.
        self.errors: list[tuple[str, str]] = []
        self.attempted = 0

    @property
    def healthy(self) -> bool:
        """False when the model errored on everything it was given."""
        return not (self.errors and len(self.errors) == self.attempted)

    def run(self, items, candidates):
        self.errors, self.attempted = [], 0
        if self.adjudicate is None:
            return [], list(items)

        resolved, remaining = [], []
        for item in items:
            # Shortlist HERE, not inside the model call. The reply names indices
            # into whatever list it was shown, so the list used to build the
            # prompt and the list used to resolve those indices must be the same
            # object -- shortlisting in two places silently maps index 3 to two
            # different candidates.
            pool = shortlist(item, list(candidates(item)))
            self.attempted += 1
            try:
                verdict = self.adjudicate(item, pool)
            except Exception as exc:
                # A model failure is an abstention, never a match: a timeout or a
                # malformed reply must not become a guess. But record it, so the
                # difference between "abstained" and "never worked" is visible.
                self.errors.append((item.id, f"{type(exc).__name__}: {exc}"))
                remaining.append(item)
                continue

            resolution = self._gate(verdict, item, pool)
            if resolution is None:
                remaining.append(item)
            else:
                resolved.append(resolution)
        return resolved, remaining

    def _gate(self, verdict: dict, item: Item,
              pool: Sequence[Candidate]) -> Resolution | None:
        """Apply gate 1 and shape the result. Gates 2 and 3 are stage 5's."""
        if verdict.get("verdict") not in self.CLAIMS:
            return None

        picked = [pool[i] for i in verdict.get("candidate_indices", [])
                  if 0 <= i < len(pool)]
        allocations = tuple(sorted({c.allocation for c in picked}))
        if not allocations:
            return None

        arithmetic_ok = verify_arithmetic(allocations, item, pool, self.tolerance_cents)
        if not arithmetic_ok:
            # The model made a claim the numbers do not support. Not a match, and
            # worth surfacing: a model that does this often is a model to retire.
            return None

        return Resolution(
            item.id, self.name, allocations,
            verdict.get("reason", "(model gave no reason)"),
            {"verdict": verdict["verdict"],
             "model_confidence": verdict.get("confidence"),  # a feature, not a number we use
             "key_facts": verdict.get("key_facts", []),
             "arithmetic_verified": True,
             "search_exhaustive": True,
             "n_candidates": len(pool)},
        )


def build_facts(item: Item, candidate: Candidate) -> dict:
    """Deterministic facts about one candidate, computed here so the model doesn't.

    ARCHITECTURE.md §3.2: the model reads facts, it never derives them. Every
    number in a dossier is produced by this function, which means every number
    the model reasons over is one we can reproduce and re-check. It also keeps
    the model's job to the part it is actually good at — weighing messy evidence
    — rather than arithmetic, which it is bad at and which we can do exactly.
    """
    tokens = _tokens_of(item.references) & _tokens_of(candidate.references)
    gap = (abs((candidate.date - item.date).days)
           if candidate.date and item.date else None)
    return {
        "allocation": candidate.allocation,
        "amount": f"{candidate.amount_cents / 100:,.2f}",
        "amount_delta": f"{(candidate.amount_cents - item.amount_cents) / 100:,.2f}",
        "date_gap_days": gap,
        "shared_reference_tokens": sorted(tokens),
    }


SHORTLIST_SIZE = 8


def shortlist(item: Item, candidates: Sequence[Candidate],
              k: int = SHORTLIST_SIZE) -> list[Candidate]:
    """The k most plausible candidates, best first.

    Stage 1 blocks for recall and routinely hands over 14-40 candidates. That is
    the right size for exhaustive subset-sum, which does not care, and the wrong
    size for a model that reasons in proportion to its input: measured, a full
    dossier ran 362 seconds before the endpoint gave up, while the same call on
    a short one returns in seconds.

    Ranked by the signals that are cheap and already computed — amount
    proximity first, because a fee or short payment sits close to the invoice it
    settles; then shared reference tokens; then date. Truncating costs recall,
    so it is a real trade and not a free optimisation: if the true counterpart
    is ranked below k it is lost, and stage 4 cannot recover it. It is worth it
    because the alternative measured out as a call that never returns at all.

    Ordering is total and deterministic — allocation breaks remaining ties — so
    the same dossier is produced on every run.
    """
    memo = _tokens_of(item.references)

    def rank(c: Candidate) -> tuple:
        return (abs(c.amount_cents - item.amount_cents),
                -len(memo & _tokens_of(c.references)),
                abs((c.date - item.date).days) if c.date and item.date else 9999,
                c.allocation)

    return sorted(candidates, key=rank)[:k]


def llm_adjudicator_callable(tracker, adjudicate=None):
    """Adapt `llm_adjudicator.adjudicate` to `AdjudicatorStage`'s contract.

    Imported lazily so the cascade runs — and its tests pass — on a machine with
    no `litellm`, no `neatlogs` and no API key. The model is an optional
    component of this system, not a dependency of it.
    """
    if adjudicate is None:
        from llm_adjudicator import adjudicate  # noqa: PLC0415 - deliberate

    def call(item: Item, candidates: Sequence[Candidate]) -> dict:
        entry = {"entity": item.meta.get("entity", "-"),
                 "amount": f"{item.amount_cents / 100:,.2f}",
                 "currency": item.meta.get("currency", ""),
                 "period": str(item.date or ""),
                 "reference": item.references}
        return adjudicate(tracker, entry, [build_facts(item, c) for c in candidates])

    return call


# --- stage 5: label and route ------------------------------------------------

class TriageStage:
    """Attach a calibrated confidence and route. The only stage that may.

    Gate 2 (recalibrate) and gate 3 (one threshold for every tier) both live
    here, so the model's output is held to exactly the same bar as an exact
    amount tie. A tier the calibration has never seen scores 0.0 and goes to
    review: silence about a tier is not evidence for it.
    """

    name = "triage"

    def __init__(self, calibration: Calibration | None = None,
                 target: float = REQUIRED_PRECISION):
        self.calibration = calibration
        self.target = target

    def label(self, resolutions: Iterable[Resolution]) -> list[Resolution]:
        for r in resolutions:
            r.confidence = (self.calibration.confidence(r.tier)
                            if self.calibration else 0.0)
            r.decision = AUTO if r.confidence >= self.target else REVIEW
        return list(resolutions)

    def queue(self, unresolved: Sequence[Item]) -> list[Resolution]:
        """Everything no stage could settle. NO_MATCH is an answer, not a gap."""
        return [Resolution(item.id, "unresolved", (), "No stage could settle this.",
                           {"verdict": "NONE"}, confidence=0.0, decision=NO_MATCH)
                for item in unresolved]


# --- the cascade -------------------------------------------------------------

@dataclass
class Run:
    """One pass of the cascade over a dataset."""

    resolutions: list[Resolution]
    consumed: dict[str, int]
    n_items: int
    # Stages that errored. Empty is the normal case; a populated list means a
    # stage was broken rather than cautious, which the decision counts alone
    # cannot show — everything lands in REVIEW either way.
    warnings: list[str] = field(default_factory=list)

    def by_decision(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.resolutions:
            counts[r.decision or "?"] = counts.get(r.decision or "?", 0) + 1
        return counts


def run_cascade(items: Sequence[Item],
                candidates: Callable[[Item], Sequence[Candidate]],
                stages: Sequence[Stage],
                triage: TriageStage) -> Run:
    """Run stages in order, each on what the previous could not settle."""
    remaining, resolutions, consumed = list(items), [], {}
    for stage in stages:
        settled, remaining = stage.run(remaining, candidates)
        consumed[stage.name] = len(settled)
        resolutions.extend(settled)

    consumed["unresolved"] = len(remaining)
    return Run(triage.label(resolutions) + triage.queue(remaining),
               consumed, len(items), _stage_warnings(stages))


def _stage_warnings(stages: Sequence[Stage]) -> list[str]:
    """Ask each stage whether it actually worked.

    A stage that errors on every item still returns a clean-looking result: it
    simply claims nothing. Without this the only symptom is a slightly longer
    review queue, which is indistinguishable from a hard day's data.
    """
    warnings = []
    for stage in stages:
        errors = getattr(stage, "errors", [])
        if errors:
            first = errors[0][1]
            warnings.append(
                f"{stage.name}: {len(errors)} error(s) of {getattr(stage, 'attempted', '?')} "
                f"attempted — first was {first}"
                + ("  [STAGE IS DOWN, not abstaining]"
                   if not getattr(stage, "healthy", True) else ""))
    return warnings


# --- stage 6: the accuracy list ----------------------------------------------

def accuracy_report(run: Run, truth: dict[str, tuple[str, ...]] | None = None) -> str:
    """Per-stage accuracy, plus the auto/review split.

    Reported per stage rather than pooled because stages fail in different ways
    and a pooled number hides the one that is wrong every time — the same reason
    calibration buckets are fine-grained. Without `truth` this still reports
    coverage, which is the half of the picture that needs no labels.
    """
    rows: dict[str, list[int]] = {}
    for r in run.resolutions:
        row = rows.setdefault(r.stage, [0, 0, 0])
        row[0] += 1
        row[1] += r.decision == AUTO
        if truth is not None and r.item_id in truth:
            row[2] += tuple(sorted(r.allocations)) == tuple(sorted(truth[r.item_id]))

    out = [f"items: {run.n_items}",
           f"decisions: {run.by_decision()}"]
    out.extend(f"WARNING  {w}" for w in run.warnings)
    out.extend(["",
           f"{'stage':<12} {'n':>7} {'share':>8} {'AUTO':>7} {'correct':>9}"])
    for stage, (n, auto, correct) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
        accuracy = f"{100 * correct / n:>8.1f}%" if truth is not None else "        -"
        out.append(f"{stage:<12} {n:>7} {100 * n / run.n_items:>7.1f}% "
                   f"{auto:>7} {accuracy}")
    return "\n".join(out)


def demo_case() -> tuple[list[Item], Callable, dict]:
    """A four-item cascade exercising every stage and every exit.

    Small and synthetic on purpose: this is a smoke test of the wiring, not a
    measurement of anything. p3 is shaped so arithmetic genuinely cannot settle
    it — two candidates tie the amount, so stage 3 must refuse and hand it to
    the model rather than pick one.
    """
    items = [
        Item("p1", 420_000, None, "INV88213", {"currency": "GBP"}),
        Item("p2", 100_000, None, "ACME", {"currency": "GBP"}),
        Item("p3", 50_000, None, "REMIT 7781", {"currency": "GBP"}),
        Item("p4", 7_777, None, "", {"currency": "GBP"}),
        # Two groups tie the amount and NOTHING deterministic separates them:
        # no single line matches, and no reference token discriminates. Stage 3
        # must refuse, so this is the only item that reaches the model.
        Item("p5", 50_000, None, "PAYMENT", {"currency": "GBP"}),
    ]
    pools = {
        "p1": [Candidate("l1", "INV_A", 420_000, None, "INV88213")],
        "p2": [Candidate("l2", "INV_B", 60_000), Candidate("l3", "INV_C", 40_000)],
        # Two rival groups both tie 50,000 and no single line does, so stage 2
        # cannot claim it and stage 3 must refuse to choose. Only the memo
        # ("REMIT 7781") separates them, which is judgement, not arithmetic —
        # exactly the residue stage 4 exists for.
        "p3": [Candidate("l4", "INV_D", 30_000, None, "REMIT 7781"),
               Candidate("l5", "INV_E", 20_000, None, "REMIT 7781"),
               Candidate("l6", "INV_F", 35_000, None, "UNRELATED"),
               Candidate("l7", "INV_G", 15_000, None, "UNRELATED")],
        "p4": [Candidate("l8", "INV_H", 12)],
        "p5": [Candidate("m1", "INV_J", 30_000), Candidate("m2", "INV_K", 20_000),
               Candidate("m3", "INV_L", 35_000), Candidate("m4", "INV_M", 15_000)],
    }
    truth = {"p1": ("INV_A",), "p2": ("INV_B", "INV_C"), "p3": ("INV_D", "INV_E"),
             "p5": ("INV_J", "INV_K")}
    return items, (lambda i: pools[i.id]), truth


def main(live: bool = False) -> Run:
    """Run the cascade. `--live` calls the real model for stage 4."""
    adjudicate = None
    if live:
        import env
        env.load_env()
        env.require("TENSORMUX_API_KEY")
        print(env.describe("TENSORMUX_API_KEY", "NEATLOGS_API_KEY"))

        import neatlogs
        tracker = neatlogs.init(api_key=os.environ.get("NEATLOGS_API_KEY", ""),
                                tags=["reconfx", "cascade"], debug=False)
        adjudicate = llm_adjudicator_callable(tracker)

    items, candidates, truth = demo_case()
    stages = [ExactMatchStage(), GroupAssemblyStage(), AdjudicatorStage(adjudicate)]
    run = run_cascade(items, candidates, stages, TriageStage())

    print(f"\n=== cascade ({'live model' if live else 'no model'}) ===")
    print(accuracy_report(run, truth))
    print("\n=== per item ===")
    for r in sorted(run.resolutions, key=lambda r: r.item_id):
        print(f"  {r.item_id}  {r.stage:<10} {str(r.decision):<9} "
              f"conf={r.confidence}  {r.reason[:70]}")
    return run


def fit_from_run(run: Run, truth: dict[str, tuple[str, ...]]) -> Calibration:
    """Fit calibration from a labelled run, so a second pass can route on it.

    Must be fit on a split that is not the one reported on, or this reproduces
    PIPELINE.md §1.1's leakage caveat with extra steps.
    """
    return Calibration.fit(
        (r.tier, tuple(sorted(r.allocations)) == tuple(sorted(truth[r.item_id])))
        for r in run.resolutions if r.item_id in truth)


if __name__ == "__main__":
    import sys
    main(live="--live" in sys.argv)
