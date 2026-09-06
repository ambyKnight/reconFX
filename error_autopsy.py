"""
Error autopsy — open every mistake the confident tier makes.

The question we are answering: are these OUR errors, or the dataset's?
BenchRec ships with a ~0.2% label error rate (that is why the bar is 99.8%
and not 100%), and we assert ~20k matches, so ~40 bad labels are expected
among them. Our error count is the same order of magnitude. If a large
share turn out to be defensible, the real precision ceiling is higher than
the scoreboard says and we should be spending surplus on match rate.

For each error we ask, cheapest question first:
  1. Was the true allocation even in the candidate pool?   -> ranking error
  2. Does the true allocation exist anywhere in the ledger? -> data gap
  3. Does our answer tie the bank amount exactly?           -> defensible?
  4. Does the true answer tie the bank amount at all?       -> label suspect

    python -X utf8 error_autopsy.py
"""

from collections import defaultdict

import pandas as pd

from match import load, index_ledger, match

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 40)

CONFIDENT = 0.95


def allocation_index(ledger):
    """allocation -> its ledger rows, so we can look up the truth's own shape."""
    by_alloc = defaultdict(list)
    for row in ledger.itertuples():
        by_alloc[row.A_allocation].append(row)
    return by_alloc


def pool_for(b, by_amount):
    """Recreate the candidate pool match() saw for this bank item."""
    exact = by_amount.get(b.amt, []) if pd.notna(b.amt) else []
    same_day = [a for a in exact if a.date == b.date]
    if same_day:
        return same_day
    return [a for a in exact
            if pd.notna(a.date) and pd.notna(b.date)
            and abs((a.date - b.date).days) <= 5]


def autopsy(errors, bank, by_amount, by_alloc):
    rows = []
    bank_by_id = {b.B_id: b for b in bank.itertuples()}

    for e in errors.itertuples():
        b = bank_by_id[e.B_id]
        pool = pool_for(b, by_amount)
        pool_allocs = {a.A_allocation for a in pool}

        true_rows = by_alloc.get(e.truth, [])
        true_sum = round(sum(a.amt for a in true_rows if pd.notna(a.amt)), 2)
        true_dates = sorted({a.date for a in true_rows if pd.notna(a.date)})
        pred_rows = by_alloc.get(e.pred, [])
        pred_sum = round(sum(a.amt for a in pred_rows if pd.notna(a.amt)), 2)

        # how far off is the true allocation, in the only two dims we blocked on
        date_gap = None
        if true_dates and pd.notna(b.date):
            date_gap = min(abs((d - b.date).days) for d in true_dates)

        rows.append({
            "B_id": e.B_id,
            "bank_amt": b.amt,
            "bank_date": b.date.date() if pd.notna(b.date) else None,
            "pool_size": len(pool),
            "pool_allocs": len(pool_allocs),
            "true_in_pool": e.truth in pool_allocs,
            "true_exists": len(true_rows) > 0,
            "true_n_rows": len(true_rows),
            "true_sum": true_sum if true_rows else None,
            "true_ties_amt": bool(true_rows) and abs(true_sum - b.amt) < 0.01,
            "true_date_gap": date_gap,
            "pred_n_rows": len(pred_rows),
            "pred_ties_amt": bool(pred_rows) and abs(pred_sum - b.amt) < 0.01,
            "collisions": b.collisions if hasattr(b, "collisions") else None,
            "tier": e.tier,
        })
    return pd.DataFrame(rows)


def verdict(a):
    """Bucket each error into something actionable."""
    def label(r):
        if not r.true_exists:
            return "A. true allocation absent from ledger entirely"
        if not r.true_in_pool and not r.true_ties_amt:
            return "B. true allocation exists but amount does not tie"
        if not r.true_in_pool and r.true_ties_amt:
            return "C. amount ties but fell outside the date window"
        return "D. true was in the pool, we ranked it below our answer"
    return a.assign(verdict=a.apply(label, axis=1))


if __name__ == "__main__":
    ledger, bank, truth, n_total = load()
    by_amount = index_ledger(ledger)
    by_alloc = allocation_index(ledger)

    bank = bank.assign(collisions=bank.groupby(["amt", "date"]).B_id.transform("size"))
    pred = match(bank, by_amount)
    pred["truth"] = pred.B_id.map(truth)
    pred["ok"] = (pred.pred.fillna("").str.strip() == pred.truth.fillna("").str.strip()) & pred.pred.notna()

    confident = pred[(pred.confidence >= CONFIDENT) & pred.pred.notna()]
    errors = confident[~confident.ok]
    print(f"confident tier: {len(confident)} asserted, {len(errors)} wrong "
          f"({100 * (1 - len(errors) / len(confident)):.3f}% precision)\n")

    a = verdict(autopsy(errors, bank, by_amount, by_alloc))
    a.to_csv("error_autopsy.csv", index=False)

    print("=== why each one went wrong ===\n")
    v = a.verdict.value_counts()
    for k, n in v.items():
        print(f"{n:>4}  ({100 * n / len(a):>5.1f}%)  {k}")

    print("\n\n=== is OUR answer defensible? ===\n")
    print(f"our predicted allocation ties the bank amount exactly : "
          f"{a.pred_ties_amt.sum()} of {len(a)}")
    print(f"the TRUE allocation ties the bank amount exactly       : "
          f"{a.true_ties_amt.sum()} of {len(a)}")
    print("\nWhen ours ties and the truth's does not, the label is at least suspect:")
    suspect = a[a.pred_ties_amt & ~a.true_ties_amt]
    print(f"  {len(suspect)} of {len(a)} errors ({100 * len(suspect) / len(a):.0f}%)")

    print("\n\n=== every error, in full ===\n")
    cols = ["B_id", "bank_amt", "bank_date", "pool_size", "pool_allocs",
            "true_in_pool", "true_exists", "true_n_rows", "true_sum",
            "true_ties_amt", "true_date_gap", "pred_ties_amt", "collisions"]
    print(a[cols].to_string(index=False))

    print("\n\nwrote error_autopsy.csv")
