"""
Dataset adapters: turn a stored dataset into what the cascade consumes.

    python adapters.py data/synth_v1.json

pipeline.py deliberately knows nothing about file formats — it sees `Item` and
`Candidate` and never a column name. This is where a specific dataset is taught
to speak that contract, and it is the boundary a real bank export would plug
into unchanged.

The blocking key is the point
-----------------------------
`candidates_for` blocks on **remitter account**, which is how cash application
actually works: the bank line says who paid, and you look at that customer's
open invoices. AGENTS.md §3b records what happens without it — on BenchRec that
field is a single constant, candidate pools ran to a median of 4,470 lines, and
group assembly abstained on 99.9% of items because nothing that size can be
searched exhaustively. Here the same code gets pools of a handful, and the
subset-sum in `assemble.py` can actually run and count rivals.

So this file is small on purpose. It is not doing clever matching; it is
supplying the one field that makes the rest of the system tractable.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

from pipeline import Candidate, Item

# How far before the payment an invoice may have been issued and still be a
# candidate. Generous on purpose: this is stage 1, where recall is the job and
# a missed candidate can never be recovered downstream (PIPELINE.md §Stage 1).
LOOKBACK_DAYS = 180


def _as_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def load(path: str | Path) -> dict:
    """Read a dataset, and refuse to proceed if it fails its own oracle.

    Validating on load as well as on write is deliberate: a file can be edited,
    truncated or hand-patched between the two, and measuring a matcher against a
    corrupted answer key produces numbers that look real and are not.
    """
    from synth_schema import validate

    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate(dataset)
    if problems:
        raise ValueError(f"{path} failed validation:\n  " + "\n  ".join(problems[:10]))
    return dataset


def to_items(dataset: dict) -> list[Item]:
    """Payments become the things needing resolution."""
    return [
        Item(id=p["id"],
             amount_cents=p["amount_cents"],
             date=_as_date(p.get("value_date")),
             references=p.get("memo", ""),
             meta={"currency": p["currency"],
                   "remitter_account": p["remitter_account"],
                   "remitter_name": p.get("remitter_name", "")})
        for p in dataset["payments"]
    ]


def candidates_for(dataset: dict,
                   lookback_days: int = LOOKBACK_DAYS) -> Callable[[Item], list[Candidate]]:
    """Build the blocking function: payment -> that payer's open invoices.

    Indexed once up front rather than scanned per payment; the whole point of a
    blocking key is that it is a lookup, and rebuilding the index per item would
    reintroduce the cost the key exists to remove.
    """
    account_of = {c["id"]: c["account"] for c in dataset["counterparties"]}
    by_account: dict[str, list[Candidate]] = {}
    issued: dict[str, date | None] = {}

    for invoice in dataset["invoices"]:
        account = account_of.get(invoice["counterparty_id"])
        candidate = Candidate(
            id=invoice["id"],
            allocation=invoice["allocation"],
            amount_cents=invoice["amount_cents"],
            date=_as_date(invoice.get("issue_date")),
            references=invoice.get("reference", ""),
        )
        by_account.setdefault(account, []).append(candidate)
        issued[invoice["id"]] = candidate.date

    def candidates(item: Item) -> list[Candidate]:
        pool = by_account.get(item.meta.get("remitter_account"), [])
        if item.date is None:
            return list(pool)
        earliest = item.date - timedelta(days=lookback_days)
        # An invoice issued after the payment cannot be what the payment settles.
        return [c for c in pool
                if c.date is None or earliest <= c.date <= item.date]

    return candidates


def truth_map(dataset: dict) -> dict[str, tuple[str, ...]]:
    """`payment_id -> the allocations that settle it`, in the cascade's terms.

    Allocations rather than invoice ids, because that is what a `Resolution`
    carries — the matcher never sees an invoice id, and scoring it on one would
    compare two different things.
    """
    allocation_of = {i["id"]: i["allocation"] for i in dataset["invoices"]}
    return {
        row["payment_id"]: tuple(sorted(allocation_of[i] for i in row["invoice_ids"]))
        for row in dataset["truth"] if row["invoice_ids"]
    }


def kind_map(dataset: dict) -> dict[str, str]:
    """`payment_id -> case kind`, so accuracy can be broken out by difficulty.

    A pooled accuracy number over a mixed dataset says almost nothing: exact
    singles and decoy groups are different problems, and averaging them hides
    whichever one is broken.
    """
    return {row["payment_id"]: row["kind"] for row in dataset["truth"]}


def from_dataset(path: str | Path):
    """Everything the cascade needs from one file."""
    dataset = load(path)
    return (to_items(dataset), candidates_for(dataset), truth_map(dataset),
            kind_map(dataset), dataset)


def _accuracy_by_kind(run, truth: dict, kinds: dict) -> str:
    """Per-case-kind scoreboard: settled, correct, and where it was settled."""
    from collections import defaultdict
    rows: dict[str, list] = defaultdict(lambda: [0, 0, defaultdict(int)])
    for r in run.resolutions:
        kind = kinds.get(r.item_id, "?")
        row = rows[kind]
        row[0] += 1
        want = truth.get(r.item_id, ())
        got = tuple(sorted(r.allocations))
        # For a NO_MATCH case the correct answer is claiming nothing at all.
        row[1] += (got == want) if want else (got == ())
        row[2][r.stage] += 1

    out = [f"{'kind':<18} {'n':>5} {'correct':>9}  settled by"]
    for kind, (n, correct, stages) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
        where = ", ".join(f"{s}:{c}" for s, c in sorted(stages.items(),
                                                        key=lambda kv: -kv[1]))
        out.append(f"{kind:<18} {n:>5} {100 * correct / n:>8.1f}%  {where}")
    return "\n".join(out)


def main(path: str) -> None:
    from pipeline import (ExactMatchStage, GroupAssemblyStage, TriageStage,
                          accuracy_report, run_cascade)

    items, candidates, truth, kinds, dataset = from_dataset(path)
    pools = [len(candidates(i)) for i in items]
    pools.sort()
    print(f"{len(items)} payments, {len(dataset['invoices'])} invoices")
    print(f"candidate pool per payment: median {pools[len(pools) // 2]}, "
          f"max {pools[-1]}  (BenchRec median was 4,470 — see AGENTS.md 3b)\n")

    run = run_cascade(items, candidates,
                      [ExactMatchStage(), GroupAssemblyStage()], TriageStage())
    print(accuracy_report(run, truth))
    print()
    print(_accuracy_by_kind(run, truth, kinds))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "data/synth_v1.json")
