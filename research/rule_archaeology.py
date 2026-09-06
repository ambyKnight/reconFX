"""
Rule archaeology — reverse-engineer the bank's production matching rules.

BenchRec labels every match group with the rule that made it (RULE 1..9) or MANUAL.
That is a free specification: whatever RULE 1 does, it did 63k times and was trusted.
And whatever MANUAL cases look like, those are the ones no rule could handle.

    python rule_archaeology.py
"""

import re

import kagglehub
import pandas as pd

SLUG = "benchmarkteam/benchrec-real-world-cash-reconciliation-dataset"
pd.set_option("display.width", 200)


def load_train():
    root = kagglehub.dataset_download(SLUG)
    df = pd.read_csv(f"{root}/BenchRec_cash_v1.0_train.csv", dtype=str, low_memory=False)
    for side in "AB":
        df[f"{side}_amt"] = pd.to_numeric(df[f"{side}_amount"], errors="coerce")
        df[f"{side}_vd"] = pd.to_datetime(df[f"{side}_valueDate"], errors="coerce")
    return df


def tokens(s):
    """Alphanumeric runs of length >= 4 — the bits that look like references."""
    return set(re.findall(r"[A-Za-z0-9]{4,}", str(s or "").upper()))


def profile(df):
    """One row per match group, describing its shape."""
    rows = []
    for mid, g in df.groupby("matchId"):
        a, b = g[g.A_id.notna()], g[g.B_id.notna()]
        rule = "MANUAL" if (g.matchRule == "MANUAL").any() else \
               (g.matchRule.dropna().iloc[0] if g.matchRule.notna().any() else "(blank)")

        a_sum, b_sum = a.A_amt.sum(), b.B_amt.sum()
        a_tok = set().union(*[tokens(x) for x in a.A_transactionReferences], set()) if len(a) else set()
        b_tok = set().union(*[tokens(x) for x in b.B_transactionReferences], set()) if len(b) else set()

        date_gap = None
        if len(a) and len(b) and a.A_vd.notna().any() and b.B_vd.notna().any():
            date_gap = abs((a.A_vd.max() - b.B_vd.max()).days)

        rows.append({
            "matchId": mid, "rule": rule, "n_a": len(a), "n_b": len(b),
            "cardinality": f"{min(len(a),9)}:{min(len(b),9)}",
            # signs are opposite across the two sides, so a clean match sums to ~0
            "nets_to_zero": abs(a_sum + b_sum) < 0.01 if len(a) and len(b) else None,
            "abs_equal": abs(abs(a_sum) - abs(b_sum)) < 0.01 if len(a) and len(b) else None,
            "amt_delta": abs(abs(a_sum) - abs(b_sum)) if len(a) and len(b) else None,
            "date_gap": date_gap,
            "ref_overlap": len(a_tok & b_tok),
            "shares_ref": bool(a_tok & b_tok),
        })
    return pd.DataFrame(rows)


def report(p):
    order = p.rule.value_counts()
    order = order[order >= 50].index

    print("=== what each production rule does ===\n")
    summary = p[p.rule.isin(order)].groupby("rule").agg(
        groups=("matchId", "size"),
        one_to_one=("cardinality", lambda s: (s == "1:1").mean()),
        amount_ties=("abs_equal", "mean"),
        shares_a_ref=("shares_ref", "mean"),
        median_date_gap=("date_gap", "median"),
        p90_date_gap=("date_gap", lambda s: s.quantile(0.9)),
    ).sort_values("groups", ascending=False)
    summary["one_to_one"] = (100 * summary.one_to_one).round(1)
    summary["amount_ties"] = (100 * summary.amount_ties).round(1)
    summary["shares_a_ref"] = (100 * summary.shares_a_ref).round(1)
    print(summary.to_string())
    print("\n(one_to_one / amount_ties / shares_a_ref are percentages)")

    print("\n\n=== cardinality mix per rule ===\n")
    mix = pd.crosstab(p[p.rule.isin(order)].rule, p[p.rule.isin(order)].cardinality)
    keep = mix.sum().sort_values(ascending=False).head(8).index
    print(mix[keep].to_string())

    print("\n\n=== MANUAL vs automated: what makes a match hard? ===\n")
    p2 = p[p.rule.isin(order)].copy()
    p2["is_manual"] = p2.rule == "MANUAL"
    cmp = p2.groupby("is_manual").agg(
        groups=("matchId", "size"),
        pct_one_to_one=("cardinality", lambda s: 100 * (s == "1:1").mean()),
        pct_amount_ties=("abs_equal", lambda s: 100 * s.mean()),
        pct_shares_ref=("shares_ref", lambda s: 100 * s.mean()),
        median_date_gap=("date_gap", "median"),
        median_group_size=("n_a", lambda s: s.median()),
    )
    print(cmp.round(1).to_string())

    print("\n\n=== where the amount does NOT tie exactly (residual sizes) ===\n")
    off = p[(p.abs_equal == False) & p.amt_delta.notna()]
    print(f"{len(off)} of {p.abs_equal.notna().sum()} groups have a residual "
          f"({100 * len(off) / max(p.abs_equal.notna().sum(), 1):.1f}%)")
    print(off.amt_delta.describe(percentiles=[.25, .5, .75, .9]).to_string())
    print("\nby rule:")
    print(off.rule.value_counts().head(8).to_string())


if __name__ == "__main__":
    df = load_train()
    p = profile(df)
    p.to_csv("match_group_profile.csv", index=False)
    report(p)
    print(f"\n\nwrote match_group_profile.csv ({len(p)} match groups)")
