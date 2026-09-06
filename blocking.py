"""
Stage 1: candidate generation for group assembly.

The problem this solves
-----------------------
assemble.py can only count rival subsets honestly if it can search the pool
exhaustively, and exhaustive search is only affordable on a small pool. Measured
on BenchRec's 1,779 real group targets, the obvious pool — same account, same
currency, ±5 days — holds a **median of 4,470 ledger lines**. The true group is
in there 98.9% of the time, so recall is not the problem; size is. Only 0.1% of
those pools come in under assemble.MAX_POOL, so stage 2 abstained on 99.9% of
real items and the 5.6% group-target pool in PIPELINE.md §1.3 stayed forfeited.

Two things that did not work, recorded so they are not retried
--------------------------------------------------------------
- **Account/currency.** BenchRec has exactly **one** account. Blocking on it
  filters nothing at all, which is most of why the pool is 4,470 wide.
- **Reference-token overlap.** The inverted-index approach that works for single
  matches collapses here: median pool 0 lines, true group present 0.1% of the
  time. Bank memos and ledger references in this dataset share almost no tokens
  (the text is pseudonymised — "MORIBUND ROTENONES S SMILAX"), so there is no
  signal to index. This is worth knowing before anyone reaches for an LLM to
  narrow the pool: the shortage is signal in the data, not cleverness in the
  matcher.

What does work: allocation prefix
---------------------------------
Members of a real group have near-identical allocation strings, differing only
in a few digits deep inside:

    ...66912     0673506735842289267089OKET216  MORIBUND ROTENONES...
    ...66912     0673506735842289567089OKET216  MORIBUND ROTENONES...

The shared head encodes currency, value date, account and customer, so grouping
ledger rows by a fixed-length allocation prefix reconstructs the cluster the
group was drawn from. Measured across prefix lengths:

    prefix   groups whose members     median      clusters
    length   share one prefix         cluster     <= 40
    ------   --------------------     -------     --------
      40           87.9%                 30         58.0%
      50           40.2%                 12         80.7%
      60            4.0%                  8         97.2%

Longer prefixes cut the cluster down but slice real groups apart, which costs
recall that no downstream stage can recover. 40 keeps 87.9% of groups intact at
a median of 30 lines — small enough to search exhaustively, which is the whole
point. `PREFIX_LEN` is stated here as a measured trade-off, not a tuned constant;
`profile_prefix_lengths` regenerates the table above so the choice can be
re-checked when the data changes.

This is a blocking key, so its job is recall (PIPELINE.md §Stage 1). Precision is
assemble.py's problem and confidence is calibrate.py's; nothing here decides
anything.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Callable, Iterable, Sequence

# Measured above. Shorter keeps more groups whole but grows the cluster past what
# can be searched exhaustively; longer splits real groups apart irrecoverably.
PREFIX_LEN = 40

# A cluster this size is not a cluster — it usually means a prefix shared by an
# entire day's postings. Passing it on would only produce an ABSTAIN downstream,
# so it is dropped here where the reason is visible.
MAX_CLUSTER = 400


def allocation_prefix(allocation: str, prefix_len: int = PREFIX_LEN) -> str:
    """The blocking key: the shared head of an allocation string."""
    return (allocation or "")[:prefix_len]


class PrefixIndex:
    """Ledger rows grouped by allocation prefix.

    Built once per run and queried per payment. Kept as an object rather than a
    dict so the prefix length that built it travels with it — an index queried at
    a different length than it was built at returns silently wrong pools, and
    that is the kind of bug that shows up as a precision drop three stages later.
    """

    def __init__(self, rows: Iterable, allocation_of: Callable[[object], str],
                 prefix_len: int = PREFIX_LEN):
        self.prefix_len = prefix_len
        self.clusters: dict[str, list] = defaultdict(list)
        for row in rows:
            allocation = allocation_of(row)
            if allocation:
                self.clusters[allocation_prefix(allocation, prefix_len)].append(row)

    def cluster_for(self, prefix: str) -> list:
        return self.clusters.get(prefix, [])

    def sizes(self) -> list[int]:
        return [len(v) for v in self.clusters.values()]

    def describe(self) -> str:
        sizes = sorted(self.sizes())
        if not sizes:
            return "empty index"
        return (f"{len(sizes)} clusters, median {statistics.median(sizes):.0f}, "
                f"p90 {sizes[int(0.9 * len(sizes))]}, max {sizes[-1]}")


def candidate_pool(index: PrefixIndex, prefixes: Iterable[str],
                   max_cluster: int = MAX_CLUSTER) -> list:
    """Pool for one payment: the union of the named clusters, deduplicated.

    Takes several prefixes because a payment is not tied to one cluster a priori
    — the caller supplies whichever are plausible (see `prefixes_near`). Union
    rather than intersection: this stage optimises recall, and a member missing
    here cannot be recovered by any later stage.
    """
    pool, seen = [], set()
    for prefix in prefixes:
        cluster = index.cluster_for(prefix)
        if len(cluster) > max_cluster:
            continue
        for row in cluster:
            key = id(row)
            if key not in seen:
                seen.add(key)
                pool.append(row)
    return pool


def prefixes_near(index: PrefixIndex, contains: str) -> list[str]:
    """Clusters whose prefix contains a marker — e.g. a currency + value date.

    Deliberately a substring test rather than similarity: the leading segment of
    an allocation is structured (`USD_2023-03-05_ACC#00001_...`), so an exact
    substring is both cheap and exactly as selective as the structure allows.
    """
    return [p for p in index.clusters if contains in p]


def profile_prefix_lengths(
    allocations: Sequence[str],
    groups: Sequence[Sequence[str]],
    lengths: Sequence[int] = (40, 50, 60, 70),
) -> str:
    """Regenerate the trade-off table in this module's docstring.

    Exists so `PREFIX_LEN` stays a measurement rather than a constant someone
    once chose. Run it against a new dataset before trusting 40 there — the
    prefix layout is a property of how that ledger writes allocation strings,
    and nothing guarantees it carries over.
    """
    rows = [f"{'prefix':>6} {'groups intact':>14} {'median':>8} {'<=40':>7}"]
    for length in lengths:
        clusters: dict[str, set[str]] = defaultdict(set)
        for allocation in set(allocations):
            clusters[allocation_prefix(allocation, length)].add(allocation)

        intact, sizes = 0, []
        for members in groups:
            prefixes = {allocation_prefix(m, length) for m in members}
            if len(prefixes) == 1:
                intact += 1
                sizes.append(len(clusters[next(iter(prefixes))]))
        median = statistics.median(sizes) if sizes else 0
        small = 100 * sum(s <= 40 for s in sizes) / len(sizes) if sizes else 0
        rows.append(f"{length:>6} {100 * intact / len(groups):>13.1f}% "
                    f"{median:>8.0f} {small:>6.1f}%")
    return "\n".join(rows)
