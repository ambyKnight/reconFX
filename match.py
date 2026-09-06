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

from calibrate import REQUIRED_PRECISION, Calibration

SLUG = "benchmarkteam/benchrec-real-world-cash-reconciliation-dataset"
DATE_WINDOW_DAYS = 5
TOKEN = re.compile(r"[A-Za-z0-9]{4,}")

# Collision bands. PIPELINE.md §1.1 measured precision as *non-monotone* in the
# number of bank items sharing an (amount, date): 99.4% at 5-7, 91.8% at 8-11,
# 100.0% at 12+. The original rule demoted everything at k >= 5, which is the
# wrong shape entirely — a 76-way block collapsing to one allocation is a bulk
# sweep, structurally unambiguous, and was being sent to a human. The risk is a
# middle band, so the policy has to be a band and not a threshold.
#
# Which band is decided below by measurement, not by these boundaries; the bands
# only set the resolution at which precision is estimated.
COLLISION_BANDS = ((1, 1), (2, 2), (3, 4), (5, 7), (8, 11), (12, None))


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


def load_train():
    """Train rows, with truth recovered from `matchId`.

    Train has no `targetAllocation`; instead A and B rows that belong together
    share a `matchId`. Restricted to match groups holding exactly one distinct
    allocation, because that is the population the confident tier serves — group
    targets are assemble.py's problem and mixing them in would charge this tier
    for failures that are not its own.
    """
    root = kagglehub.dataset_download(SLUG)
    tr = pd.read_csv(f"{root}/BenchRec_cash_v1.0_train.csv", dtype=str, low_memory=False)

    ledger = tr[tr.A_id.notna()].copy()
    bank = tr[tr.B_id.notna()].copy()
    for df, side in ((ledger, "A"), (bank, "B")):
        df["amt"] = pd.to_numeric(df[f"{side}_amount"], errors="coerce").round(2)
        df["date"] = pd.to_datetime(df[f"{side}_valueDate"], errors="coerce")
        df["tokens"] = [set(TOKEN.findall(str(r).upper()))
                        for r in df[f"{side}_transactionReferences"].fillna("")]

    allocs = defaultdict(set)
    for r in ledger.itertuples():
        if isinstance(r.A_allocation, str) and r.A_allocation.strip():
            allocs[r.matchId].add(r.A_allocation)
    truth = {r.B_id: next(iter(allocs[r.matchId]))
             for r in bank.itertuples() if len(allocs.get(r.matchId, ())) == 1}

    return ledger, bank, truth


def collision_band(k):
    """Label for the band `k` falls in — the key precision is estimated per."""
    for lo, hi in COLLISION_BANDS:
        if k >= lo and (hi is None or k <= hi):
            return f"k={lo}+" if hi is None else (f"k={lo}" if lo == hi else f"k={lo}-{hi}")
    return "k=0"


def fit_collision_policy(bank, by_amount, truth, folds=5):
    """Fit, on train, which collision bands the confident tier can be trusted in.

    Runs the matcher with no collision policy, then asks of each band: what did
    the confident tier actually achieve here? A band is trusted only if its
    **Wilson lower bound** clears 99.8% — so a band that merely looks clean on
    thin evidence does not qualify, and the promote/demote boundary stops being
    a number someone picked.

    This is the fix for §1.1's stated caveat. Choosing `k >= 12` by reading eval
    labels was leakage and the document says so; fitting the same decision here
    and applying it to eval once is the difference between a result and a number
    that happened to hold.
    """
    pred = match(bank, by_amount, policy=None)
    pred["truth"] = pred.B_id.map(truth)
    confident = pred[(pred.tier.str.startswith("single allocation")) & pred.truth.notna()]

    # Folds keyed on a hash of B_id: stable across runs, independent of row
    # order, and reproducible from the manifest. Crediting the worst fold is
    # what stops a small, locally-clean band from outranking a large one — see
    # Calibration.fit_folds.
    records = [(hash(r.B_id) % folds, collision_band(r.collisions), r.pred == r.truth)
               for r in confident.itertuples()]
    return Calibration.fit_folds(records)


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


def match(bank, by_amount, policy=None):
    """Match every bank item. `policy` is a fitted Calibration over collision
    bands; None runs without any collision rule, which is what the fit needs."""
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

        # Collision policy, fitted on train. A confident match is demoted only in
        # bands that did not earn 99.8% there — which, per §1.1, is a middle band
        # and emphatically not "everything above k=5". Bands the fit never saw
        # score 0.0 and demote, so an unmeasured band fails safe.
        if policy is not None and conf >= 0.95:
            band = collision_band(b.collisions)
            conf = policy.confidence(band)
            gate = policy.verdict(band)
            tier = f"{tier} [{band}, train {100 * conf:.2f}%, {gate}]"

        out.append({"B_id": b.B_id, "pred": alloc, "confidence": round(conf, 4),
                    "tier": tier, "n_candidates": len(pool),
                    "collisions": b.collisions})
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
    print(f"{'min_conf':>10} {'matched':>8} {'match_rate':>11} {'precision':>10}")
    best = 0.0
    for cut in sorted(attempted.confidence.unique(), reverse=True):
        kept = attempted[attempted.confidence >= cut]
        print(f"{cut:>10.4f} {len(kept):>8} {100 * len(kept) / n_total:>10.1f}% "
              f"{100 * kept.ok.mean():>9.2f}%")
        if kept.ok.mean() >= REQUIRED_PRECISION:
            best = max(best, 100 * len(kept) / n_total)

    print(f"\nbest match rate clearing {100 * REQUIRED_PRECISION}%: {best:.1f}%")
    print("(shipped MatcherByChatGPT baseline on the same rule: 0.5%)")
    return pred


if __name__ == "__main__":
    tr_ledger, tr_bank, tr_truth = load_train()
    print(f"train: {len(tr_ledger)} ledger rows, {len(tr_bank)} bank rows, "
          f"{len(tr_truth)} single-allocation labels")
    policy = fit_collision_policy(tr_bank, index_ledger(tr_ledger), tr_truth)
    print("\n=== collision policy, fitted on train ===")
    print(policy.report())

    ledger, bank, truth, n_total = load()
    print(f"\neval: {len(ledger)} ledger rows, {len(bank)} bank rows\n")
    pred = report(match(bank, index_ledger(ledger), policy=policy), truth, n_total)
    pred.to_csv("predictions.csv", index=False)
    print("\nwrote predictions.csv")
