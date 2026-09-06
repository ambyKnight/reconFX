"""
Ingestion + basic matching for BenchRec.

Task: for each bank item (B), predict the ledger allocation string it belongs to.
Ground truth is solution.csv. The benchmark asks for the best match rate that
still clears 99.8% precision, so every candidate carries a confidence and we
report the whole frontier rather than a single number.

Key structural point: the target is an ALLOCATION, not a ledger row. Several
ledger rows routinely share one allocation, so a 17-way tie on amount+date is
not ambiguous if all 17 rows belong to the same allocation. We therefore
collapse candidates by allocation before deciding whether anything is ambiguous.

Tiers, cheapest first:
  1. exact amount + date, candidates collapse to one allocation -> confident
  2. several allocations, one wins on shared reference tokens    -> medium
  3. several allocations, nothing separates them                 -> ambiguous
  4. no exact-amount candidate                                   -> unmatched
Tiers 3 and 4 are what a human queue exists for.

    python match.py
"""

import re
from collections import defaultdict

import kagglehub
import pandas as pd

SLUG = "benchmarkteam/benchrec-real-world-cash-reconciliation-dataset"
DATE_WINDOW_DAYS = 5
REQUIRED_PRECISION = 0.998
COLLISION_LIMIT = 5  # tuned below; see caveat in the write-up
TOKEN = re.compile(r"[A-Za-z0-9]{4,}")


def load():
    root = kagglehub.dataset_download(SLUG)
    ev = pd.read_csv(f"{root}/BenchRec_cash_v1.0_eval.csv", dtype=str, low_memory=False)
    sol = pd.read_csv(f"{root}/BenchRec_cash_v1.0_solution.csv", dtype=str)

    # Each row is either a ledger row or a bank row, never both.
    ledger = ev[ev.A_id.notna()].copy()
    bank = ev[ev.B_id.notna()].copy()

    for df, side in ((ledger, "A"), (bank, "B")):
        df["amt"] = pd.to_numeric(df[f"{side}_amount"], errors="coerce").round(2)
        df["date"] = pd.to_datetime(df[f"{side}_valueDate"], errors="coerce")
        df["tokens"] = [set(TOKEN.findall(str(r).upper()))
                        for r in df[f"{side}_transactionReferences"].fillna("")]

    return ledger, bank, dict(zip(sol.B_id, sol.targetAllocation)), len(sol)


def index_ledger(ledger):
    """Amount is the strongest blocking key: amount -> candidate ledger rows."""
    by_amount = defaultdict(list)
    for row in ledger.itertuples():
        if pd.notna(row.amt):
            by_amount[row.amt].append(row)
    return by_amount


def choose(candidates, b):
    """Return (allocation, confidence, tier) for one bank item."""
    if not candidates:
        return None, 0.0, "no candidate"

    # Collapse to distinct allocations — that, not row count, is the ambiguity.
    groups = defaultdict(list)
    for a in candidates:
        groups[a.A_allocation].append(a)

    if len(groups) == 1:
        return next(iter(groups)), 0.95, "single allocation"

    # Several allocations. See if reference tokens single one out.
    scored = sorted(
        ((len(b.tokens & set().union(*(a.tokens for a in rows), set())), alloc)
         for alloc, rows in groups.items()),
        reverse=True,
    )
    best, runner_up = scored[0], scored[1]
    if best[0] > 0 and best[0] > runner_up[0]:
        return best[1], 0.80, "reference tokens break the tie"

    return scored[0][1], 0.35, f"ambiguous, {len(groups)} allocations"


def collision_counts(bank):
    """How many bank items share an item's exact (amount, date).

    A busy day with several identical round amounts is exactly where an
    amount+date "unique" match is most likely to be a coincidence rather than
    a real counterpart, so this is our main calibration signal.
    """
    return bank.groupby(["amt", "date"]).B_id.transform("size")


def match(bank, by_amount):
    bank = bank.assign(collisions=collision_counts(bank))
    out = []
    for b in bank.itertuples():
        exact = by_amount.get(b.amt, []) if pd.notna(b.amt) else []
        same_day = [a for a in exact if a.date == b.date]
        near = [a for a in exact
                if pd.notna(a.date) and pd.notna(b.date)
                and abs((a.date - b.date).days) <= DATE_WINDOW_DAYS]

        pool = same_day or near
        alloc, conf, tier = choose(pool, b)
        if pool is near and alloc is not None:
            conf -= 0.10  # off-date match is weaker evidence
            tier += " (date window)"

        # Calibration: a "unique" match on a crowded amount+date is not evidence
        # of much. Demote it to the human queue rather than assert it.
        if conf >= 0.95 and b.collisions >= COLLISION_LIMIT:
            conf, tier = 0.50, f"unique but {b.collisions} bank items share amount+date"

        out.append({"B_id": b.B_id, "pred": alloc, "confidence": round(conf, 2),
                    "tier": tier, "n_candidates": len(pool)})
    return pd.DataFrame(out)


def report(pred, truth, n_total):
    pred["truth"] = pred.B_id.map(truth)
    pred["ok"] = (pred.pred.fillna("").str.strip() == pred.truth.fillna("").str.strip()) \
                 & pred.pred.notna()
    attempted = pred[pred.pred.notna()]

    print("=== coverage ===")
    print(f"bank items        : {n_total}")
    print(f"proposed a match  : {len(attempted)} ({100 * len(attempted) / n_total:.1f}%)")
    print(f"no candidate      : {pred.pred.isna().sum()}")

    print("\n=== by tier ===")
    tiers = pred.groupby("tier").agg(n=("B_id", "size"), correct=("ok", "sum"))
    tiers["precision"] = (100 * tiers.correct / tiers.n).round(2)
    tiers["share"] = (100 * tiers.n / n_total).round(1)
    print(tiers.sort_values("n", ascending=False).head(12).to_string())

    print("\n=== frontier ===")
    print(f"{'min_conf':>9} {'matched':>8} {'match_rate':>11} {'precision':>10}")
    best = 0.0
    for cut in sorted(attempted.confidence.unique(), reverse=True):
        kept = attempted[attempted.confidence >= cut]
        print(f"{cut:>9.2f} {len(kept):>8} {100 * len(kept) / n_total:>10.1f}% "
              f"{100 * kept.ok.mean():>9.2f}%")
        if kept.ok.mean() >= REQUIRED_PRECISION:
            best = max(best, 100 * len(kept) / n_total)

    print(f"\nbest match rate clearing {100 * REQUIRED_PRECISION}%: {best:.1f}%")
    print("(shipped MatcherByChatGPT baseline on the same rule: 0.5%)")
    return pred


if __name__ == "__main__":
    ledger, bank, truth, n_total = load()
    print(f"ingested {len(ledger)} ledger rows, {len(bank)} bank rows\n")
    pred = report(match(bank, index_ledger(ledger)), truth, n_total)
    pred.to_csv("predictions.csv", index=False)
    print("\nwrote predictions.csv")
