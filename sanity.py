"""Is 'true allocation absent from ledger' real, or an artefact of my index?"""
import pandas as pd
from match import load, index_ledger
from error_autopsy import allocation_index

ledger, bank, truth, n_total = load()
by_alloc = allocation_index(ledger)
known = set(by_alloc)

t = pd.Series(truth)
present = t.isin(known)
print(f"solution rows                       : {len(t)}")
print(f"targetAllocation found in eval ledger: {present.sum()} ({100*present.mean():.1f}%)")
print(f"NOT found                            : {(~present).sum()} ({100*(~present).mean():.1f}%)")

print("\n--- sample of allocations that are NOT in the ledger ---")
print(t[~present].head(10).to_string())

print("\n--- sample of allocations that ARE ---")
print(t[present].head(5).to_string())

print("\n--- are the missing ones a sentinel / null-ish value? ---")
miss = t[~present]
print(miss.value_counts().head(10).to_string())
print(f"\ndistinct missing allocations: {miss.nunique()} across {len(miss)} bank items")

print("\n--- do they appear in the TRAIN ledger instead? ---")
import kagglehub
root = kagglehub.dataset_download("benchmarkteam/benchrec-real-world-cash-reconciliation-dataset")
tr = pd.read_csv(f"{root}/BenchRec_cash_v1.0_train.csv", dtype=str, low_memory=False)
train_alloc = set(tr.A_allocation.dropna())
in_train = miss.isin(train_alloc)
print(f"{in_train.sum()} of {len(miss)} missing allocations appear in the train ledger")
