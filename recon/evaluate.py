"""
Stage 9: Evaluation and reporting.

Evaluates reconciliation predictions against ground truth labels, reporting:
  - Candidate recall ceiling (Stage 1 instrumentation)
  - Coverage summary
  - Per-tier breakdown
  - Full confidence frontier
  - Best match rate clearing the 99.8% precision benchmark
"""

import sys
import pandas as pd

from recon.config import CONF_AUTO_THRESHOLD, REQUIRED_PRECISION, WRITE_OFF_THRESHOLD
from recon.suspense import build_suspense_register, create_case_file

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def evaluate(
    pred,
    truth,
    n_total,
    candidate_recall_stats=None,
    bank=None,
    as_of_date=None,
    write_off_threshold=None,
    by_amount=None,
    print_report=True,
):
    """Evaluate predictions against ground truth labels and output benchmark report.

    Parameters:
      pred: DataFrame containing B_id, pred, confidence, tier, n_candidates.
      truth: Dict mapping B_id -> targetAllocation.
      n_total: Total number of bank items.
      candidate_recall_stats: Optional dict from measure_candidate_recall.
      bank: Optional bank lines DataFrame for building suspense case files.
      as_of_date: Optional as-of date for suspense aging (defaults to max date in run).
      write_off_threshold: Immaterial write-off threshold (defaults to config).
      by_amount: Optional ledger index by amount for fee/FX residual detection.
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

    wo_thresh = write_off_threshold if write_off_threshold is not None else WRITE_OFF_THRESHOLD

    # Build Suspense Subsystem Case Files & Exceptions Register (PIPELINE §5.2, ARCHITECTURE §4)
    suspense_register = None
    bank_df = bank
    if bank_df is None and "amt" in pred.columns and "date" in pred.columns:
        bank_df = pred

    if bank_df is not None:
        run_as_of = as_of_date
        if run_as_of is None:
            if "date" in bank_df.columns:
                run_as_of = bank_df["date"].dropna().max()

        bank_dict = {str(r.B_id): r for r in bank_df.itertuples()}
        case_files = []

        for p_row in pred.itertuples():
            b_id_str = str(p_row.B_id)
            b_item = bank_dict.get(b_id_str)
            if b_item is None:
                continue

            # Suspense items: any item that is not auto-posted (confidence < CONF_AUTO_THRESHOLD or unmatched)
            is_auto = (p_row.confidence >= CONF_AUTO_THRESHOLD) and (p_row.pred is not None and pd.notna(p_row.pred))
            if not is_auto:
                cf = create_case_file(
                    bank_item=b_item,
                    candidates=[],
                    as_of_date=run_as_of,
                    write_off_threshold=wo_thresh,
                    by_amount=by_amount,
                    tier=p_row.tier,
                    confidence=p_row.confidence,
                    pred_alloc=p_row.pred,
                )
                case_files.append(cf)

        if case_files:
            suspense_register = build_suspense_register(case_files)

    metrics = {
        "n_total": n_total,
        "n_attempted": len(attempted),
        "coverage_pct": 100.0 * len(attempted) / n_total,
        "no_candidate": int(pred.pred.isna().sum()),
        "candidate_recall_stats": candidate_recall_stats,
        "best_match_rate_at_required_precision": best_match_rate,
        "frontier": frontier_rows,
        "suspense": suspense_register,
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

        if suspense_register:
            s_count = suspense_register["total_count"]
            s_val = suspense_register["total_value"]
            s_pct = 100.0 * s_count / n_total if n_total else 0.0
            wo_info = suspense_register["write_off_candidates"]
            print("\n=== exceptions register (suspense) ===")
            print(f"suspense items    : {s_count} ({s_pct:.1f}% of bank lines, £{s_val:,.2f} gross value)")
            print(
                f"write-off pool    : {wo_info['count']} items ({wo_info['share_pct']:.1f}% of suspense, "
                f"£{wo_info['value']:,.2f} value <= £{wo_thresh:.2f})"
            )

            print("\n--- by category ---")
            print(f"{'category':<25} {'items':>8} {'total value (£)':>18} {'share %':>10}")
            for cat in suspense_register["by_category"]:
                print(f"{cat['category']:<25} {cat['count']:>8} {cat['value']:>18,.2f} {cat['share_pct']:>9.1f}%")

            print("\n--- aging buckets ---")
            print(f"{'bucket':<25} {'items':>8} {'total value (£)':>18} {'share %':>10}")
            for ab in suspense_register["by_aging"]:
                bucket_label = f"{ab['bucket']} days" if not str(ab['bucket']).endswith("days") else ab['bucket']
                print(f"{bucket_label:<25} {ab['count']:>8} {ab['value']:>18,.2f} {ab['share_pct']:>9.1f}%")

    return pred, metrics
