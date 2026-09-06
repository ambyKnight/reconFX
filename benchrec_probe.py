"""
BenchRec probe — what's in the dataset and how good is the shipped baseline?

Dataset: benchmarkteam/benchrec-real-world-cash-reconciliation-dataset (Kaggle, CC BY 4.0)
Obfuscated production GL<->bank reconciliation data from a Tier 1 financial institution.
Published by Operartis; first shown at ACM AI in Finance 2023.

Benchmark rule (per Operartis): maximise MATCH RATE subject to PRECISION >= 99.8%.
The 99.8% (rather than 100%) allows for a ~0.2% label error rate they identified.

    python benchrec_probe.py
"""

import kagglehub
import pandas as pd
import numpy as np

SLUG = "benchmarkteam/benchrec-real-world-cash-reconciliation-dataset"
REQUIRED_PRECISION = 0.998


def load():
    root = kagglehub.dataset_download(SLUG)
    read = lambda n, **kw: pd.read_csv(f"{root}/BenchRec_cash_v1.0_{n}.csv", **kw)
    train = read("train", dtype=str, low_memory=False)
    evaluation = read("eval", dtype=str, low_memory=False)
    solution = read("solution", dtype=str)
    baseline = pd.read_csv(f"{root}/MatcherByChatGPT_submission.csv")
    baseline["B_id"] = baseline.B_id.astype(str)
    return train, evaluation, solution, baseline


def describe(train, evaluation, solution):
    print("=== shape ===")
    print(f"train {train.shape}  eval {evaluation.shape}  solution {solution.shape}")
    print(f"train: {train.A_id.notna().sum()} ledger (A) rows, "
          f"{train.B_id.notna().sum()} bank (B) rows, "
          f"{train.matchId.nunique()} match groups")
    print(f"dates: train {train.B_valueDate.min()}..{train.B_valueDate.max()}  "
          f"eval {evaluation.B_valueDate.min()}..{evaluation.B_valueDate.max()}")

    print("\n=== how were matches made in production? ===")
    print(train.matchRule.fillna("(blank)").value_counts().to_string())
    manual_groups = train.groupby("matchId").matchRule.apply(lambda s: (s == "MANUAL").any())
    print(f"\nMANUAL: {100 * (train.matchRule == 'MANUAL').mean():.1f}% of rows, "
          f"{100 * manual_groups.mean():.1f}% of match groups")

    print("\n=== match cardinality (A-side count, B-side count) ===")
    shape = train.groupby("matchId").apply(
        lambda d: (d.A_id.notna().sum(), d.B_id.notna().sum()), include_groups=False)
    print(shape.value_counts().head(10).to_string())


def score(solution, baseline):
    """Score a submission the way the benchmark asks: match rate at >=99.8% precision."""
    truth = dict(zip(solution.B_id, solution.targetAllocation))
    pred = baseline[baseline.A_allocation.notna()].copy()
    pred["ok"] = pred.A_allocation.str.strip() == pred.B_id.map(truth).fillna("").str.strip()
    n = len(solution)

    print("\n=== shipped baseline (MatcherByChatGPT) ===")
    print(f"strategy: {pred.explanation.iloc[0]}")
    print(f"attempted {len(pred)} of {n}  ->  {100 * len(pred) / n:.1f}% match rate "
          f"at {100 * pred.ok.mean():.2f}% precision")

    print("\n--- its confidence scores ---")
    print(f"{100 * (pred.confidence == pred.confidence.mode()[0]).mean():.1f}% of predictions "
          f"share one identical confidence value ({pred.confidence.mode()[0]:.6f})")
    print("i.e. the confidence column is very nearly a constant — it carries almost no signal.")

    # Frontier: sweep the threshold, find the best match rate that still clears the bar.
    ranked = pred.sort_values("confidence", ascending=False)
    best = 0.0
    for thresh in np.unique(ranked.confidence.values):
        kept = ranked[ranked.confidence >= thresh]
        if len(kept) and kept.ok.mean() >= REQUIRED_PRECISION:
            best = max(best, 100 * len(kept) / n)

    ceiling = 100 * pred.ok.sum() / n
    print(f"\nscored under the benchmark rule (>= {100 * REQUIRED_PRECISION}% precision):")
    print(f"  actual   {best:.1f}% match rate")
    print(f"  oracle   {ceiling:.1f}% match rate  (if its OWN predictions were ranked perfectly)")
    print(f"\n  the {ceiling - best:.1f} point gap is pure calibration, not matching:")
    print("  it finds the right answer ~95% of the time and cannot tell which times those are.")


if __name__ == "__main__":
    train, evaluation, solution, baseline = load()
    describe(train, evaluation, solution)
    score(solution, baseline)
