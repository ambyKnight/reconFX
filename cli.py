"""
Command-line interface for the reconciliation pipeline.

Commands:
  python cli.py fit       Fit collision thresholds on train split (train.csv only)
  python cli.py run       Run reconciliation on eval split and save predictions.csv
  python cli.py evaluate  Run full evaluation with candidate recall instrumentation
  python cli.py explain   Explain decision for a given bank line ID (B_id)
"""

import argparse
import sys
from pathlib import Path

from recon.blocking import index_ledger, measure_candidate_recall
from recon.config import ARTIFACTS_DIR, load_fitted_config
from recon.decide import match
from recon.evaluate import evaluate as evaluate_predictions
from recon.explain import explain_decision
from recon.fit import fit_collision_thresholds
from recon.ingest import load_eval, load_train


def cmd_fit(args):
    """Fit collision thresholds on the training split only."""
    print("Loading training data (train split only)...")
    ledger, bank, truth, n_total, manifest = load_train(data_dir=args.data_dir)
    print(f"Ingested {len(ledger)} ledger rows, {len(bank)} bank rows (manifest config {manifest['config_hash']})\n")

    fitted_params, stats = fit_collision_thresholds(
        bank,
        ledger,
        truth,
        save_to_config=True,
        verbose=True,
    )
    return 0


def cmd_run(args):
    """Run reconciliation on evaluation split and export predictions."""
    print("Loading evaluation data...")
    ledger, bank, truth, n_total, manifest = load_eval(data_dir=args.data_dir)
    print(f"Ingested {len(ledger)} ledger rows, {len(bank)} bank rows\n")

    by_amount = index_ledger(ledger)
    fitted = load_fitted_config() or {}
    demote_min = args.demote_min if args.demote_min is not None else fitted.get("collision_demote_min")
    promote_min = args.promote_min if args.promote_min is not None else fitted.get("collision_promote_min")

    print(f"Running matcher (demote middle band: [{demote_min}, {promote_min}))...")
    pred = match(bank, by_amount, demote_min=demote_min, promote_min=promote_min)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_path, index=False)
    print(f"\nWrote predictions to {out_path} ({len(pred)} items)")

    # Also mirror to artifacts directory
    artifact_pred_path = ARTIFACTS_DIR / "predictions.csv"
    artifact_pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(artifact_pred_path, index=False)
    return 0


def cmd_evaluate(args):
    """Run evaluation with candidate recall instrumentation."""
    print("Loading evaluation data...")
    ledger, bank, truth, n_total, manifest = load_eval(data_dir=args.data_dir)
    print(f"Ingested {len(ledger)} ledger rows, {len(bank)} bank rows\n")

    by_amount = index_ledger(ledger)

    print("Measuring Stage 1 candidate recall...")
    candidate_recall_stats = measure_candidate_recall(bank, by_amount, truth)

    fitted = load_fitted_config() or {}
    demote_min = args.demote_min if args.demote_min is not None else fitted.get("collision_demote_min")
    promote_min = args.promote_min if args.promote_min is not None else fitted.get("collision_promote_min")

    print(f"Running matcher (collision band: demote [{demote_min}, {promote_min}), promote [{promote_min}+))...\n")
    pred = match(bank, by_amount, demote_min=demote_min, promote_min=promote_min)

    evaluated_df, metrics = evaluate_predictions(
        pred,
        truth,
        n_total,
        candidate_recall_stats=candidate_recall_stats,
        print_report=True,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    evaluated_df.to_csv(out_path, index=False)
    print(f"\nWrote predictions to {out_path}")
    return 0


def cmd_explain(args):
    """Explain the decision for a specific bank line ID."""
    print(f"Looking up explanation for B_id: {args.b_id}...")
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"Predictions file not found at {pred_path}. Run 'python cli.py run' first.")
        return 1

    import pandas as pd
    df = pd.read_csv(pred_path, dtype=str)
    row = df[df.B_id == str(args.b_id)]
    if row.empty:
        print(f"No prediction found for B_id {args.b_id}")
        return 1

    r = row.iloc[0]
    print(f"B_id       : {r.B_id}")
    print(f"Prediction : {r.pred}")
    print(f"Confidence : {r.confidence}")
    print(f"Tier       : {r.tier}")
    if "truth" in r and pd.notna(r.truth):
        print(f"Truth      : {r.truth}")
        print(f"Correct    : {r.get('ok', 'unknown')}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="recon",
        description="End-to-end cash reconciliation system (BenchRec)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # fit
    p_fit = subparsers.add_parser("fit", help="Fit collision thresholds on train.csv split")
    p_fit.add_argument("--data-dir", default=None, help="Path to raw dataset directory")

    # run
    p_run = subparsers.add_parser("run", help="Run reconciliation on eval split")
    p_run.add_argument("--data-dir", default=None, help="Path to raw dataset directory")
    p_run.add_argument("--output", default="predictions.csv", help="Output predictions path")
    p_run.add_argument("--demote-min", type=int, default=None, help="Override demote min threshold")
    p_run.add_argument("--promote-min", type=int, default=None, help="Override promote min threshold")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run full evaluation and report frontier")
    p_eval.add_argument("--data-dir", default=None, help="Path to raw dataset directory")
    p_eval.add_argument("--output", default="predictions.csv", help="Output predictions path")
    p_eval.add_argument("--demote-min", type=int, default=None, help="Override demote min threshold")
    p_eval.add_argument("--promote-min", type=int, default=None, help="Override promote min threshold")

    # explain
    p_explain = subparsers.add_parser("explain", help="Explain prediction for a bank line")
    p_explain.add_argument("b_id", help="Bank item B_id to explain")
    p_explain.add_argument("--predictions", default="predictions.csv", help="Predictions CSV file")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "fit": cmd_fit,
        "run": cmd_run,
        "evaluate": cmd_evaluate,
        "explain": cmd_explain,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
