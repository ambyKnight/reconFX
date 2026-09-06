"""
Phase 0: measure what stage 4 actually does.

    python eval_llm.py --n 60 --seed 7          # against the live model
    python eval_llm.py --n 60 --dry             # against a stub, no API calls

Nothing about the adjudicator has ever been measured. It has been called
correctly, and once observed to abstain sensibly, but its accuracy, its
abstention rate and — the number that matters — how often it is *confidently
wrong* are all unknown. Every later change to the prompt, the dossier or the
gates is a guess until this exists.

What it isolates
----------------
Stage 4 only ever sees what stages 2 and 3 could not settle, so evaluating it on
a random sample of payments would measure the wrong population and flatter it
enormously. `build_inbox` runs the real cascade and hands back exactly the
residue, which is a deliberately hostile set: fees, short payments, partials,
genuine ambiguities and unmatchable cash.

The outcomes, and why they are not one number
---------------------------------------------
Accuracy alone is meaningless for a component whose job includes declining. A
model that abstains on everything scores the same as one that is useless, and a
model that guesses scores better than one that is honest. So every item lands in
exactly one of:

    CORRECT_CLAIM    claimed, and right
    WRONG_CLAIM      claimed, passed the arithmetic gate, and still wrong
                     -- the only genuinely dangerous outcome, because it is what
                     reaches the ledger
    GATE_REJECTED    claimed, but the arithmetic did not hold, so we caught it
                     -- a model error that cost nothing
    CORRECT_ABSTAIN  declined, and there was nothing to find
    MISSED           declined, but the answer was there and reachable
                     -- costs match rate, costs no correctness
    ERROR            the call failed or returned nothing usable

WRONG_CLAIM and MISSED are both "wrong" and must never be pooled: one puts a bad
number in the accounts, the other puts an item in a human queue. Treating them
as the same failure is how a system ends up optimised toward confident guessing.

The ceiling is reported alongside the score
-------------------------------------------
`verify_arithmetic` currently requires an exact tie, so for any case with a
deliberate residual — a bank fee, a short payment — the gate rejects the *true*
answer and no model can win. Measured on 1,500 synthetic payments, 375 of the
664 items reaching stage 4 are in that position. So the report states the
reachable ceiling next to the achieved score; a model scoring 35% against a 35%
ceiling is perfect, and reporting only the 35% would be a libel.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from pipeline import (AdjudicatorStage, Candidate, ExactMatchStage,
                      GroupAssemblyStage, Item, shortlist, verify_arithmetic)

OUTCOMES = ("CORRECT_CLAIM", "WRONG_CLAIM", "GATE_REJECTED", "CORRECT_ABSTAIN",
            "MISSED", "ERROR")

# Outcomes that put a wrong answer somewhere it can do harm. Kept separate from
# "wrong" in general; see module docstring.
HARMFUL = ("WRONG_CLAIM",)


@dataclass
class Sample:
    """One evaluated item: what was asked, what came back, what it was worth."""

    item_id: str
    kind: str
    outcome: str
    verdict: str = ""
    model_confidence: float | None = None
    reachable: bool = True
    reason: str = ""


@dataclass
class Report:
    samples: list[Sample] = field(default_factory=list)

    def counts(self) -> Counter:
        return Counter(s.outcome for s in self.samples)

    def claimed(self) -> list[Sample]:
        return [s for s in self.samples if s.outcome in ("CORRECT_CLAIM", "WRONG_CLAIM")]

    def precision(self) -> float | None:
        """Of the answers that got past the gate, how many were right.

        The headline number for safety: this is the rate at which stage 4 would
        put something wrong into the ledger if its output were trusted.
        """
        claimed = self.claimed()
        if not claimed:
            return None
        return sum(s.outcome == "CORRECT_CLAIM" for s in claimed) / len(claimed)

    def ceiling(self) -> float:
        """Share of items whose true answer the gate could accept at all."""
        return sum(s.reachable for s in self.samples) / max(len(self.samples), 1)

    def render(self) -> str:
        n = len(self.samples)
        counts = self.counts()
        lines = [f"evaluated: {n} items reaching stage 4", ""]
        for outcome in OUTCOMES:
            c = counts.get(outcome, 0)
            lines.append(f"  {outcome:<16} {c:>5}  ({100 * c / max(n, 1):>5.1f}%)")

        precision = self.precision()
        solved = counts.get("CORRECT_CLAIM", 0)
        lines += [
            "",
            f"claim precision      : "
            + (f"{100 * precision:.1f}%  ({len(self.claimed())} claims)"
               if precision is not None else "no claims made"),
            f"resolved             : {solved}/{n} ({100 * solved / max(n, 1):.1f}%)",
            f"reachable ceiling    : {100 * self.ceiling():.1f}%  "
            f"(gate can accept the true answer)",
            f"harmful errors       : {sum(counts.get(o, 0) for o in HARMFUL)}"
            f"   <- the number that matters",
            "",
            f"{'kind':<18} {'n':>4}  outcomes",
        ]
        by_kind: dict[str, Counter] = defaultdict(Counter)
        for s in self.samples:
            by_kind[s.kind][s.outcome] += 1
        # Short labels must stay distinguishable: "correct" for both
        # CORRECT_CLAIM and CORRECT_ABSTAIN would read as one thing and hide
        # whether a kind was solved or merely declined safely.
        short = {"CORRECT_CLAIM": "solved", "WRONG_CLAIM": "WRONG",
                 "GATE_REJECTED": "gated", "CORRECT_ABSTAIN": "abstain",
                 "MISSED": "missed", "ERROR": "error"}
        for kind, outcomes in sorted(by_kind.items(), key=lambda kv: -sum(kv[1].values())):
            detail = ", ".join(f"{short.get(o, o)}:{c}"
                               for o, c in outcomes.most_common())
            lines.append(f"{kind:<18} {sum(outcomes.values()):>4}  {detail}")
        return "\n".join(lines)


def build_inbox(dataset: dict, limit: int | None = None):
    """Run stages 2-3 and return exactly what stage 4 would see.

    Returns `(items, candidates_fn, truth, kinds)`. Sampling happens *after* the
    cascade, so the sample is drawn from the real residue rather than from all
    payments — the distinction that keeps this from measuring the easy cases.
    """
    import adapters

    items = adapters.to_items(dataset)
    candidates = adapters.candidates_for(dataset)
    truth, kinds = adapters.truth_map(dataset), adapters.kind_map(dataset)

    remaining = items
    for stage in (ExactMatchStage(), GroupAssemblyStage()):
        _, remaining = stage.run(remaining, candidates)
    return (remaining[:limit] if limit else remaining), candidates, truth, kinds


def classify(verdict: dict | None, item: Item, pool: list[Candidate],
             want: tuple[str, ...], tolerance_cents: int = 0) -> tuple[str, str]:
    """Map one model reply to an outcome. Returns `(outcome, note)`.

    Mirrors `AdjudicatorStage._gate` deliberately rather than calling it, so the
    evaluation can distinguish *why* a claim failed — a claim killed by the
    arithmetic gate is a different event from a claim that passed and was still
    wrong, and the stage itself collapses both into "not resolved".
    """
    if verdict is None:
        return "ERROR", "no usable reply"

    reachable = bool(want) and verify_arithmetic(want, item, pool, tolerance_cents)

    if verdict.get("verdict") not in ("MATCH", "MATCH_GROUP"):
        if not want:
            return "CORRECT_ABSTAIN", "nothing to find"
        return ("MISSED", "answer was reachable") if reachable else (
            "CORRECT_ABSTAIN", "answer was not reachable through the gate")

    picked = [pool[i] for i in verdict.get("candidate_indices", [])
              if 0 <= i < len(pool)]
    got = tuple(sorted({c.allocation for c in picked}))
    if not got:
        return "ERROR", "claimed a match but named no valid candidate"
    if not verify_arithmetic(got, item, pool, tolerance_cents):
        return "GATE_REJECTED", "arithmetic did not hold"
    return ("CORRECT_CLAIM", "") if got == want else ("WRONG_CLAIM", f"claimed {got}")


def evaluate(dataset: dict, adjudicate, limit: int = 60,
             tolerance_cents: int = 0, progress: bool = True,
             workers: int = 1) -> Report:
    """Run the adjudicator over the stage-4 inbox and score every reply.

    Prints progress as it goes and flushes each line. Measured at ~13s per call
    against the live endpoint, a 60-item run takes ~13 minutes — long enough
    that a silent process is indistinguishable from a hung one, and long enough
    that a partial result is worth seeing before the end.
    """
    import time

    items, candidates, truth, kinds = build_inbox(dataset, limit)
    report = Report()
    started = time.time()

    if workers > 1:
        return _evaluate_concurrently(items, candidates, truth, kinds, adjudicate,
                                      workers, tolerance_cents, progress, started)

    for n, item in enumerate(items, 1):
        # Same shortlist the real stage applies, or the evaluation measures
        # a different system than the one that ships.
        pool = shortlist(item, list(candidates(item)))
        want = truth.get(item.id, ())
        try:
            verdict = adjudicate(item, pool)
        except Exception as exc:
            report.samples.append(Sample(item.id, kinds.get(item.id, "?"), "ERROR",
                                         reason=f"{type(exc).__name__}: {exc}"))
            continue

        outcome, note = classify(verdict, item, pool, want, tolerance_cents)
        report.samples.append(Sample(
            item_id=item.id, kind=kinds.get(item.id, "?"), outcome=outcome,
            verdict=str(verdict.get("verdict", "")),
            model_confidence=verdict.get("confidence"),
            reachable=bool(want) and verify_arithmetic(want, item, pool, tolerance_cents),
            reason=note))

        if progress:
            elapsed = time.time() - started
            print(f"  [{n}/{len(items)}] {item.id} {kinds.get(item.id, '?'):<18} "
                  f"-> {outcome:<16} {elapsed / n:.1f}s/call "
                  f"eta {(len(items) - n) * elapsed / n / 60:.1f}m", flush=True)
    return report


def _evaluate_concurrently(items, candidates, truth, kinds, adjudicate, workers,
                           tolerance_cents, progress, started):
    """Same scoring, run in parallel.

    Latency here is entirely the endpoint's: ~13s per call, so a 60-item run is
    13 minutes sequentially and about 90 seconds at 10 workers. Worker count is
    derived from the measured rate limit in llm_harness.workers_for rather than
    picked, since exceeding it trades a slow stage for a throttled one.
    """
    import time

    from llm_harness import concurrent

    pools = {item.id: shortlist(item, list(candidates(item))) for item in items}
    results = concurrent(adjudicate, workers=workers)(
        [(item, pools[item.id]) for item in items])

    report = Report()
    for item, result in zip(items, results):
        pool, want = pools[item.id], truth.get(item.id, ())
        if isinstance(result, Exception):
            report.samples.append(Sample(item.id, kinds.get(item.id, "?"), "ERROR",
                                         reason=f"{type(result).__name__}: {result}"))
            continue
        outcome, note = classify(result, item, pool, want, tolerance_cents)
        report.samples.append(Sample(
            item_id=item.id, kind=kinds.get(item.id, "?"), outcome=outcome,
            verdict=str(result.get("verdict", "")),
            model_confidence=result.get("confidence"),
            reachable=bool(want) and verify_arithmetic(want, item, pool, tolerance_cents),
            reason=note))

    if progress:
        elapsed = time.time() - started
        print(f"  {len(items)} items in {elapsed:.0f}s "
              f"({elapsed / max(len(items), 1):.1f}s/item at {workers} workers)",
              flush=True)
    return report


def stub_adjudicator(item: Item, pool: list[Candidate]) -> dict:
    """A deterministic stand-in so the harness is testable without spending money.

    Deliberately naive — it takes the first subset it finds that ties, which is
    the early-exit behaviour this whole project exists to argue against. As a
    baseline that is useful: it shows what "plausible-looking but unprincipled"
    scores, which is the bar a real model has to beat.
    """
    for i, c in enumerate(pool):
        if c.amount_cents == item.amount_cents:
            return {"verdict": "MATCH", "candidate_indices": [i], "confidence": 0.9,
                    "reason": "amount ties exactly", "key_facts": []}
    return {"verdict": "UNSURE", "candidate_indices": [], "confidence": 0.2,
            "reason": "no single candidate ties", "key_facts": []}


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure stage 4.")
    parser.add_argument("--n", type=int, default=60, help="items to evaluate")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--payments", type=int, default=800)
    parser.add_argument("--parties", type=int, default=80)
    parser.add_argument("--dry", action="store_true", help="stub model, no API calls")
    parser.add_argument("--trace", action="store_true",
                        help="enable neatlogs tracing (noisy)")
    parser.add_argument("--workers", type=int, default=0,
                        help="parallel calls; 0 = derive from the rate limit")
    parser.add_argument("--out", default="", help="write per-item results as JSON")
    args = parser.parse_args()

    import synth
    dataset = synth.generate(seed=args.seed, n_payments=args.payments,
                             n_parties=args.parties)

    if args.dry:
        adjudicate = stub_adjudicator
        label = "stub (no model)"
    else:
        import env
        env.load_env()
        env.require("TENSORMUX_API_KEY")
        from pipeline import llm_adjudicator_callable
        # No tracer by default: neatlogs dumps its whole payload plus stack
        # traces to stdout when export fails, which it always does without a
        # key, drowning the report this command exists to print.
        tracker = None
        if args.trace:
            import neatlogs
            tracker = neatlogs.init(api_key=os.environ.get("NEATLOGS_API_KEY", ""),
                                    tags=["reconfx", "eval"], debug=False)
        adjudicate = llm_adjudicator_callable(tracker)
        from llm_adjudicator import MODEL
        label = MODEL

    from llm_harness import workers_for
    workers = args.workers or (1 if args.dry else workers_for())
    report = evaluate(dataset, adjudicate, limit=args.n, workers=workers)
    print(f"\n=== stage 4 baseline: {label} ===")
    print(report.render())

    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(
            [s.__dict__ for s in report.samples], indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
