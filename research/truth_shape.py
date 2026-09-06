"""
The solution file has TWO target formats, not one:
  single   'USD_2023-03-05_ACC#00001_...'          -> one allocation
  list     '[USD_...,USD_...,USD_...]'             -> a GROUP of allocations

Our matcher only ever emits a single allocation, so every list-shaped target is
an automatic miss. This quantifies how much of the benchmark that costs us and
whether we are at least finding one member of the right group.
"""
import sys
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = RESEARCH_DIR.parent
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from match import load, index_ledger, match
from error_autopsy import allocation_index

def split_targets(v):
    """Return the list of allocations a target names."""
    if not isinstance(v, str):
        return []
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        return [p.strip() for p in v[1:-1].split(",") if p.strip()]
    return [v]

ledger, bank, truth, n_total = load()
by_alloc = allocation_index(ledger)

t = pd.DataFrame({"B_id": list(truth), "target": list(truth.values())})
t["members"] = t.target.map(split_targets)
t["n_members"] = t.members.map(len)

print("=== target shape across the whole eval set ===")
print(f"total bank items      : {len(t)}")
print(f"null / no target      : {(t.n_members == 0).sum()}")
print(f"single allocation     : {(t.n_members == 1).sum()} ({100*(t.n_members==1).mean():.1f}%)")
print(f"list of allocations   : {(t.n_members > 1).sum()} ({100*(t.n_members>1).mean():.1f}%)")
print("\ngroup size distribution (list targets only):")
print(t[t.n_members > 1].n_members.value_counts().sort_index().head(15).to_string())

print("\n=== are list members real ledger allocations? ===")
exploded = t[t.n_members > 1].explode("members")
hit = exploded.members.isin(by_alloc)
print(f"{hit.sum()} of {len(exploded)} members found in the eval ledger ({100*hit.mean():.1f}%)")

print("\n\n=== what this costs us ===")
bank = bank.assign(collisions=bank.groupby(["amt", "date"]).B_id.transform("size"))
pred = match(bank, index_ledger(ledger))
pred = pred.merge(t, on="B_id", how="left")
pred["exact_ok"] = pred.pred.fillna("").str.strip() == pred.target.fillna("").str.strip()
pred["member_ok"] = [p in m if isinstance(p, str) else False
                     for p, m in zip(pred.pred, pred.members)]

conf = pred[(pred.confidence >= 0.95) & pred.pred.notna()]
print(f"confident tier: {len(conf)} asserted, {(~conf.exact_ok).sum()} wrong")
wrong = conf[~conf.exact_ok]
print(f"  of those wrong, target was a LIST : {(wrong.n_members > 1).sum()}")
print(f"  of those wrong, target was SINGLE : {(wrong.n_members == 1).sum()}")
print(f"  of those wrong, target was NULL   : {(wrong.n_members == 0).sum()}")
print(f"\n  we picked a MEMBER of the right group: {wrong.member_ok.sum()} of {len(wrong)}")

print("\n=== how much match rate is locked behind group targets? ===")
lists = pred[pred.n_members > 1]
print(f"list-target bank items          : {len(lists)} ({100*len(lists)/n_total:.1f}% of eval)")
print(f"  we currently score 0 on all of them (we can only emit one allocation)")
print(f"  but we already name a member of the right group in "
      f"{lists.member_ok.sum()} ({100*lists.member_ok.mean():.1f}%)")
print(f"\nceiling if group assembly were solved: "
      f"{100*(pred.n_members>=1).mean():.1f}% of items have a real target to hit")
