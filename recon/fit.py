"""
Fitting routine for collision-band thresholds on the training split.

Per PIPELINE.md §1.1 and Part 3 Step 1:
Fits the collision-band thresholds on train.csv ONLY (never peeking at eval labels).
Identifies the safe high-collision band (bulk sweeps / batch postings) and the risky
middle band.
"""

from collections import defaultdict
import pandas as pd

from recon.blocking import collision_counts, get_candidate_pool, index_ledger
from recon.config import (
    COLLISION_DEMOTE_MIN,
    COLLISION_PROMOTE_MIN,
    CONF_DATE_WINDOW_PENALTY,
    CONF_SINGLE_ALLOCATION,
    DATE_WINDOW_DAYS,
    save_fitted_config,
)
from recon.decide import choose


def fit_collision_thresholds(
    bank_train,
    ledger_train,
    truth_train,
    demote_min_candidates=(COLLISION_DEMOTE_MIN,),
    promote_min_candidates=(12, 14, 16, 18, 20),
    target_promoted_precision=0.997,
    save_to_config=True,
    verbose=True,
):
    """Fit collision promotion and demotion thresholds on the training dataset.

    Parameters:
      bank_train: Bank lines DataFrame from train.csv.
      ledger_train: Ledger DataFrame from train.csv.
      truth_train: Dict of B_id -> targetAllocation for training bank items.
      demote_min_candidates: Candidates for the lower boundary of the demote band.
      promote_min_candidates: Candidates for the upper boundary (promotion threshold).
      target_promoted_precision: Minimum precision required for the promoted batch band.
      save_to_config: Whether to persist fitted parameters to artifacts/fitted_config.json.
      verbose: Whether to print fitting diagnostic tables.

    Returns:
      (fitted_params, summary_stats)
    """
    by_amount = index_ledger(ledger_train)
    if "collisions" not in bank_train.columns:
        bank_train = bank_train.assign(collisions=collision_counts(bank_train))

    # Precompute raw candidates without collision demotion
    records = []
    for b in bank_train.itertuples():
        pool, pool_type = get_candidate_pool(b, by_amount, date_window_days=DATE_WINDOW_DAYS)
        alloc, conf, tier = choose(pool, b)
        if pool_type == "near" and alloc is not None:
            conf -= CONF_DATE_WINDOW_PENALTY

        t = truth_train.get(b.B_id)
        # Check correctness against train label
        ok = (
            isinstance(alloc, str)
            and isinstance(t, str)
            and alloc.strip() == t.strip()
        ) if alloc else False

        records.append({
            "B_id": b.B_id,
            "pred": alloc,
            "raw_conf": conf,
            "collisions": b.collisions,
            "ok": ok,
        })

    df_train = pd.DataFrame(records)
    n_total = len(df_train)
    single = df_train[df_train.raw_conf >= CONF_SINGLE_ALLOCATION].copy()

    # Baseline performance (original rule: demote all k >= 5)
    baseline_demoted = single[single.collisions >= 5]
    baseline_kept = single[single.collisions < 5]
    baseline_prec = baseline_kept.ok.mean()
    baseline_rate = len(baseline_kept) / n_total

    if verbose:
        print("=== Training Set Collision Analysis ===")
        print(f"Total training bank items: {n_total}")
        print(f"Candidates with raw_conf >= {CONF_SINGLE_ALLOCATION}: {len(single)}")
        print(
            f"Baseline rule (k >= 5 demoted): matched={len(baseline_kept)} "
            f"({100*baseline_rate:.2f}%), precision={100*baseline_prec:.3f}%"
        )
        print("\nCollision count breakdown on training split (k >= 5):")
        for k in range(5, 17):
            sub = single[single.collisions == k]
            if len(sub) > 0:
                print(
                    f"  k={k:2d}: n={len(sub):4d}, correct={sub.ok.sum():4d}, "
                    f"errors={(~sub.ok).sum():2d}, prec={100*sub.ok.mean():.2f}%"
                )
        tail = single[single.collisions >= 17]
        if len(tail) > 0:
            print(
                f"  k>=17: n={len(tail):4d}, correct={tail.ok.sum():4d}, "
                f"errors={(~tail.ok).sum():2d}, prec={100*tail.ok.mean():.2f}%"
            )

    best_demote_min = COLLISION_DEMOTE_MIN
    best_promote_min = COLLISION_PROMOTE_MIN
    best_rate = -1.0
    best_stats = None

    # Search candidate threshold pairs on train split
    for d in demote_min_candidates:
        for p in promote_min_candidates:
            if p <= d:
                continue

            promoted = single[single.collisions >= p]
            demoted = single[(single.collisions >= d) & (single.collisions < p)]
            kept = single[(single.collisions < d) | (single.collisions >= p)]

            prom_prec = promoted.ok.mean() if len(promoted) else 0.0
            dem_prec = demoted.ok.mean() if len(demoted) else 0.0
            kept_prec = kept.ok.mean() if len(kept) else 0.0
            rate = len(kept) / n_total

            # Objective:
            # 1. Promoted high-collision band must meet target precision
            # 2. Promoted band must be more precise than or equal to demoted band
            # 3. Overall kept precision must not fall below baseline
            # 4. Maximizes match rate on train
            if (
                prom_prec >= target_promoted_precision
                and prom_prec >= dem_prec
                and kept_prec >= baseline_prec
            ):
                if rate > best_rate:
                    best_rate = rate
                    best_demote_min = d
                    best_promote_min = p
                    best_stats = {
                        "demote_min": d,
                        "promote_min": p,
                        "train_match_rate": rate,
                        "train_kept_precision": kept_prec,
                        "train_promoted_precision": prom_prec,
                        "train_demoted_precision": dem_prec,
                        "n_promoted": len(promoted),
                        "n_demoted": len(demoted),
                        "n_kept": len(kept),
                    }

    # Fallback to defaults if no candidate strictly satisfied the constraints
    if best_stats is None:
        best_demote_min = COLLISION_DEMOTE_MIN
        best_promote_min = COLLISION_PROMOTE_MIN
        promoted = single[single.collisions >= best_promote_min]
        demoted = single[(single.collisions >= best_demote_min) & (single.collisions < best_promote_min)]
        kept = single[(single.collisions < best_demote_min) | (single.collisions >= best_promote_min)]
        best_stats = {
            "demote_min": best_demote_min,
            "promote_min": best_promote_min,
            "train_match_rate": len(kept) / n_total,
            "train_kept_precision": kept.ok.mean() if len(kept) else 0.0,
            "train_promoted_precision": promoted.ok.mean() if len(promoted) else 0.0,
            "train_demoted_precision": demoted.ok.mean() if len(demoted) else 0.0,
            "n_promoted": len(promoted),
            "n_demoted": len(demoted),
            "n_kept": len(kept),
        }

    fitted_params = {
        "collision_demote_min": int(best_demote_min),
        "collision_promote_min": int(best_promote_min),
        "date_window_days": DATE_WINDOW_DAYS,
    }

    if verbose:
        print("\n=== Fitted Collision Parameters (Train Split) ===")
        print(f"Fitted Demote Min    : {best_demote_min}")
        print(f"Fitted Promote Min   : {best_promote_min}")
        print(f"Demoted Band         : [{best_demote_min}, {best_promote_min})")
        print(f"Promoted Band        : [{best_promote_min}+)")
        print(f"Train Match Rate     : {100*best_stats['train_match_rate']:.2f}%")
        print(f"Train Kept Precision : {100*best_stats['train_kept_precision']:.3f}%")
        print(f"Promoted Precision   : {100*best_stats['train_promoted_precision']:.3f}% (n={best_stats['n_promoted']})")
        print(f"Demoted Precision    : {100*best_stats['train_demoted_precision']:.3f}% (n={best_stats['n_demoted']})")

    if save_to_config:
        save_fitted_config(fitted_params)
        if verbose:
            print("Saved fitted parameters to artifacts/fitted_config.json")

    return fitted_params, best_stats
