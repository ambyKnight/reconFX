"""
Stage 9: Evaluation and reporting.

Evaluates reconciliation predictions against ground truth labels, reporting:
  - Candidate recall ceiling (Stage 1 instrumentation)
  - Coverage summary
  - Per-tier breakdown
  - Full confidence frontier
  - Best match rate clearing the 99.8% precision benchmark
"""

import pandas as pd

from recon.config import REQUIRED_PRECISION


def evaluate(pred, truth, n_total, candidate_recall_stats=None, print_report=True):
    """Evaluate predictions against ground truth labels and output benchmark report.

    Parameters:
      pred: DataFrame containing B_id, pred, confidence, tier, n_candidates.
      truth: Dict mapping B_id -> targetAllocation.
      n_total: Total number of bank items.
      candidate_recall_stats: Optional dict from measure_candidate_recall.
      print_report: Whether to print formatted report to stdout.

    Returns:
      (pred, metrics_dict)
    """
    pred = pred.copy()
    pred["truth"] = pred.B_id.map(truth)
    pred["ok"] = (
        (pred.pred.fillna("").str.strip() == pred.truth.fillna("").str.strip())
        & pred.pred.notna()
    )
    attempted = pred[pred.pred.notna()]

    best_match_rate = 0.0
    frontier_rows = []
    for cut in sorted(attempted.confidence.unique(), reverse=True):
        kept = attempted[attempted.confidence >= cut]
        rate = 100.0 * len(kept) / n_total
        prec = 100.0 * kept.ok.mean() if len(kept) else 0.0
        frontier_rows.append({
            "min_conf": cut,
            "matched": len(kept),
            "match_rate": rate,
            "precision": prec,
        })
        if kept.ok.mean() >= REQUIRED_PRECISION:
            best_match_rate = max(best_match_rate, rate)

    metrics = {
        "n_total": n_total,
        "n_attempted": len(attempted),
        "coverage_pct": 100.0 * len(attempted) / n_total,
        "no_candidate": int(pred.pred.isna().sum()),
        "candidate_recall_stats": candidate_recall_stats,
        "best_match_rate_at_required_precision": best_match_rate,
        "frontier": frontier_rows,
    }

    if print_report:
        print("=== coverage ===")
        print(f"bank items        : {n_total}")
        print(f"proposed a match  : {len(attempted)} ({100.0 * len(attempted) / n_total:.1f}%)")
        print(f"no candidate      : {pred.pred.isna().sum()}")

        if candidate_recall_stats:
            c_hits = candidate_recall_stats["pool_hits"]
            c_target = candidate_recall_stats["n_with_target"]
            c_recall_tgt = 100.0 * candidate_recall_stats["pool_recall_target"]
            c_recall_tot = 100.0 * candidate_recall_stats["pool_recall_total"]
            print(
                f"candidate recall  : {c_hits}/{c_target} "
                f"({c_recall_tgt:.1f}% of items with target, {c_recall_tot:.1f}% of total [ceiling])"
            )

        print("\n=== by tier ===")
        tiers = pred.groupby("tier").agg(n=("B_id", "size"), correct=("ok", "sum"))
        tiers["precision"] = (100.0 * tiers.correct / tiers.n).round(2)
        tiers["share"] = (100.0 * tiers.n / n_total).round(1)
        print(tiers.sort_values("n", ascending=False).head(12).to_string())

        print("\n=== frontier ===")
        print(f"{'min_conf':>9} {'matched':>8} {'match_rate':>11} {'precision':>10}")
        for r in frontier_rows:
            print(
                f"{r['min_conf']:>9.2f} {r['matched']:>8} {r['match_rate']:>10.1f}% "
                f"{r['precision']:>9.2f}%"
            )

        print(f"\nbest match rate clearing {100 * REQUIRED_PRECISION}%: {best_match_rate:.1f}%")
        print("(shipped MatcherByChatGPT baseline on the same rule: 0.5%)")

    return pred, metrics
