"""
Stage 2: group assembly — one payment against many ledger lines.

PIPELINE.md §1.3: 1,779 eval items (5.6%) have a *list* target, and the matcher
forfeits every one because it emits a single allocation. We already name a
member of the correct group in 80.2% of them, so candidate generation is fine
and only the assembly step is missing.

The design turns on one sentence from PIPELINE.md §Stage 2:

    Uniqueness is the signal. Exactly one subset summing to the bank amount is
    strong evidence. Several is ambiguity — emit them all and let stage 6
    abstain. Do not tie-break silently.

which is easy to write and easy to get wrong, because the arithmetic is exact
and the *inference* is not. Three invoices summing to £4,200.00 and a different
five summing to £4,200.00 are both correct sums; at most one is the transaction.
The question this module answers is therefore not "does a subset exist" — cheap
and nearly meaningless — but "how many distinct answers exist", which is the
only thing that makes uniqueness evidence rather than an assumption.

The failure mode being designed against
---------------------------------------
A solver that stops at the first (or first two) valid subsets cannot distinguish
"unique" from "gave up early", yet reports the former. Not hypothetical: it is
what the published state of the art does. Wu et al., "The Subset Sum Matching
Problem" (ECAI 2025, arXiv:2508.19218), formalises this tolerance-matching
problem and offers a MILP, a cached search and a DP — all three terminating at
the first valid match ("backtracking until a valid match is found"). None counts
feasible subsets. Deciding uniqueness is genuinely hard in general (uniqueness
of an optimal knapsack solution is Δ₂P-complete), so an exhaustive count is not
free and cannot always be had.

The resolution is a budget with a direction. Search is bounded — pool size,
subset size, distinct-solution cap, node budget — but every bound is built so
that exhausting it makes the answer *less* confident:

    exhausting any budget ⇒ `search_exhaustive` is False ⇒ verdict is never
    UNIQUE.

Running out of compute degrades to "I found several, there may be more", never
to "this is the only one". `Assembly.assert_never_false_unique` states the
invariant executably and runs on every return path; test_assemble.py holds it.

Two independent signals, not one signal twice
---------------------------------------------
Sum agreement is one vote. match.py's tier 2 ("reference tokens break the tie")
established the second on single matches and it extends to groups unchanged:
when several subsets tie on arithmetic, one whose members share reference tokens
with the payment memo is corroborated by evidence that is not arithmetic at all.
Two independent signals agreeing is a different claim from one signal agreeing
twice. Cash-application vendors lean on the same thing — remittance text as a
second vote.

What this module deliberately does not do
-----------------------------------------
It does not produce a confidence number. `0.85` was a guess and reality was 35%;
per PIPELINE.md §Stage 4-5 the fix is that "was it unique", "how many rivals"
and "did references corroborate" are *features*, and a calibration step fit on
real outcomes assigns the probability. So `Assembly.features` is the output that
matters, `verdict` is a coarse deterministic summary of it, and nothing here
invents a float. calibrate.py owns that, measured against labels.

Layout
------
    Line / Payment / Solution / Assembly   the data contract
    _SubsetSearch                          the bounded enumeration, one job per method
    enumerate_solutions                    thin functional wrapper over it
    _pool_features / _result_features      feature construction, no decisions
    _choose_verdict                        decisions, no feature construction
    assemble                               orchestration only
    explain                                one small formatter per verdict

Pure stdlib on purpose: no pandas, no kagglehub, no network at import, so the
subset-sum logic is unit-testable without the dataset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Callable, Iterable, Sequence

# Shared with match.py; PIPELINE.md §Stage 0 wants exactly one definition of
# this regex in the codebase, and this is a step toward that rather than a
# fourth copy of it.
TOKEN = re.compile(r"[A-Za-z0-9]{4,}")

# --- search bounds -----------------------------------------------------------
# Every one of these, when hit, sets search_exhaustive=False and therefore
# forbids a UNIQUE verdict. They trade recall for honesty, never the reverse.

MAX_POOL = 40          # PIPELINE.md §Stage 2: beyond this, abstain, don't burn compute
MAX_SUBSET_SIZE = 8    # covers ~97% of observed groups
MAX_SOLUTIONS = 64     # distinct answers kept before we stop counting
NODE_BUDGET = 200_000  # DFS nodes; bounds pools that slip past MAX_POOL

# Must exceed 1: a cap of 1 could not tell "only one exists" from "stopped at
# one", which is the bug this module exists not to have. (The original defect
# was the same error one notch out: a cap of 2 that still reported uniqueness.)
assert MAX_SOLUTIONS >= 2

UNIQUE, CORROBORATED, AMBIGUOUS, NONE, ABSTAIN = (
    "UNIQUE", "CORROBORATED", "AMBIGUOUS", "NONE", "ABSTAIN")


def to_cents(amount) -> int:
    """Money as integers. PIPELINE.md §Stage 0: floats invite 1-cent tie failures.

    Subset-sum on floats is worse than imprecise, it is unstable — whether a
    subset "ties" the target can depend on summation order. Integers make the
    tolerance comparison exact and reproducible.
    """
    return int((Decimal(str(amount)) * 100).to_integral_value())


def _tokens_of(references) -> frozenset[str]:
    return frozenset(TOKEN.findall(str(references).upper()))


# --- data contract -----------------------------------------------------------

@dataclass(frozen=True)
class Line:
    """One ledger row in the candidate pool.

    `allocation` is the unit of truth, not `id` — see `_SubsetSearch._record`.
    """

    id: str
    allocation: str
    amount_cents: int
    date: date | None = None
    tokens: frozenset[str] = frozenset()

    @classmethod
    def of(cls, id, allocation, amount, date=None, references=""):
        return cls(id, allocation, to_cents(amount), date, _tokens_of(references))


@dataclass(frozen=True)
class Payment:
    """The bank item we are trying to explain."""

    id: str
    amount_cents: int
    date: date | None = None
    tokens: frozenset[str] = frozenset()

    @classmethod
    def of(cls, id, amount, date=None, references=""):
        return cls(id, to_cents(amount), date, _tokens_of(references))


@dataclass(frozen=True)
class Solution:
    """One arithmetically valid explanation of the payment.

    Identified by `allocations`, not by `line_ids`: two solutions differing only
    in which ledger row carried an allocation are the same answer.
    """

    allocations: tuple[str, ...]
    line_ids: tuple[str, ...]
    residual_cents: int
    token_overlap: int

    @property
    def size(self) -> int:
        return len(self.allocations)


@dataclass
class Assembly:
    """The result. `features` is the payload; `verdict` summarises it."""

    verdict: str
    chosen: Solution | None
    solutions: list[Solution]
    features: dict = field(default_factory=dict)

    def assert_never_false_unique(self) -> None:
        """The invariant the whole module exists to preserve.

        UNIQUE may only be claimed when the search actually completed and found
        exactly one distinct answer. Called on every return path so the
        guarantee is enforced at runtime rather than merely documented: a future
        edit reintroducing an early exit fails here instead of shipping a
        quietly overconfident number.
        """
        if self.verdict == UNIQUE:
            assert self.features.get("search_exhaustive") is True, \
                "UNIQUE claimed from a truncated search — this is the max_solutions bug"
            assert self.features.get("n_solutions") == 1, \
                "UNIQUE claimed with rivals present"


# --- bounded enumeration -----------------------------------------------------

def _reachable(amounts: Sequence[int]) -> tuple[list[int], list[int]]:
    """Suffix bounds on what the remaining tail can still contribute.

    Positives and negatives are tracked separately because credit notes and
    refunds make amounts non-monotone: with a negative in the pool, "current sum
    already exceeds target" is not a valid prune. These bounds stay admissible
    either way, so pruning can never discard a real solution — which matters
    more than usual here, since a discarded solution surfaces as false
    uniqueness rather than as a miss.
    """
    n = len(amounts)
    up, down = [0] * (n + 1), [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        up[i] = up[i + 1] + max(amounts[i], 0)
        down[i] = down[i + 1] + min(amounts[i], 0)
    return up, down


class _SubsetSearch:
    """Depth-first enumeration of every distinct subset that ties the target.

    A class rather than nested closures so each concern is separately readable
    and separately testable: `_over_budget` owns the bounds, `_record` owns what
    counts as a distinct answer, `_prunable` owns admissibility, `_dfs` owns
    traversal. The invariant to preserve when editing any of them is that
    nothing may set `self.truncated` to False, and nothing may return early on a
    hit — those are the two ways false uniqueness gets reintroduced.
    """

    def __init__(self, pool, payment, tolerance_cents, max_subset_size,
                 max_solutions, node_budget):
        # Descending order makes the reachability prune bite early.
        self.lines = sorted(pool, key=lambda l: -l.amount_cents)
        self.amounts = [l.amount_cents for l in self.lines]
        self.up, self.down = _reachable(self.amounts)
        self.payment = payment
        self.target = payment.amount_cents
        self.tolerance = tolerance_cents
        self.max_subset_size = max_subset_size
        self.max_solutions = max_solutions
        self.node_budget = node_budget

        self.found: dict[tuple[str, ...], Solution] = {}
        self.truncated = False
        self.nodes = 0

    def run(self) -> tuple[list[Solution], bool]:
        """Enumerate, then rank. Returns `(solutions, exhaustive)`."""
        self._dfs(0, 0, [])
        return self._ranked(), not self.truncated

    def _over_budget(self) -> bool:
        """Every budget check lives here, and every one only ever *sets*
        `truncated` — there is no path where a budget makes an answer look
        better than the evidence supports."""
        self.nodes += 1
        if self.nodes > self.node_budget:
            self.truncated = True
        return self.truncated

    def _is_hit(self, total: int) -> bool:
        return abs(total - self.target) <= self.tolerance

    def _prunable(self, start: int, total: int) -> bool:
        """Admissible bound: an unreachable tail cannot contain a solution."""
        need = self.target - total
        return (need > self.up[start] + self.tolerance
                or need < self.down[start] - self.tolerance)

    def _record(self, chosen: list[int], total: int) -> None:
        """Keep one representative per allocation set.

        Distinctness is by **allocation set**, which is match.py's point for
        single matches ("a 17-way tie on amount+date is not ambiguous if all 17
        rows belong to the same allocation") applied to groups. Counting raw row
        subsets would manufacture ambiguity out of bookkeeping: three rows of one
        allocation would report as rivals and suppress a match that is unique in
        the only sense that matters.

        The representative kept is the one with the strongest reference
        evidence, so the tie-break downstream compares each rival at its best
        rather than at whichever row was enumerated first.
        """
        allocations = tuple(sorted(self.lines[i].allocation for i in chosen))
        tokens: set[str] = set()
        for i in chosen:
            tokens |= self.lines[i].tokens

        candidate = Solution(
            allocations=allocations,
            line_ids=tuple(self.lines[i].id for i in chosen),
            residual_cents=total - self.target,
            token_overlap=len(self.payment.tokens & tokens),
        )
        prior = self.found.get(allocations)
        if prior is None or candidate.token_overlap > prior.token_overlap:
            self.found[allocations] = candidate

    def _dfs(self, start: int, total: int, chosen: list[int]) -> None:
        if self.truncated or self._over_budget():
            return

        if chosen and self._is_hit(total):
            self._record(chosen, total)
            if len(self.found) >= self.max_solutions:
                self.truncated = True
                return
            # Deliberately no `return` on success: a valid subset may extend
            # into another valid subset (a +£100 and a -£100 line downstream),
            # and stopping here is exactly the early exit that produces false
            # uniqueness. Cost stays bounded by the budgets above.

        if len(chosen) == self.max_subset_size or start == len(self.lines):
            return
        if self._prunable(start, total):
            return

        for i in range(start, len(self.lines)):
            chosen.append(i)
            self._dfs(i + 1, total + self.amounts[i], chosen)
            chosen.pop()
            if self.truncated:
                return

    def _ranked(self) -> list[Solution]:
        """Best first: strongest reference evidence, then tightest residual,
        then smallest group. Ordering is total and input-independent, which is
        what makes the whole assembly deterministic."""
        return sorted(
            self.found.values(),
            key=lambda s: (-s.token_overlap, abs(s.residual_cents), s.size, s.allocations),
        )


def enumerate_solutions(
    pool: Sequence[Line],
    payment: Payment,
    tolerance_cents: int = 0,
    max_subset_size: int = MAX_SUBSET_SIZE,
    max_solutions: int = MAX_SOLUTIONS,
    node_budget: int = NODE_BUDGET,
) -> tuple[list[Solution], bool]:
    """Every distinct subset of `pool` summing to the payment ± tolerance.

    Returns `(solutions, exhaustive)`. `exhaustive` is True only if the search
    completed without hitting `max_solutions` or `node_budget` — only then is
    `len(solutions)` a true count rather than a lower bound. Callers must not
    infer uniqueness when it is False.

    Subsets of size 1 are included. A lone £4,200 invoice and a trio summing to
    £4,200 are rivals for the same payment, so scoring groups in a separate
    universe from singles would hide exactly the collisions being counted.
    """
    return _SubsetSearch(pool, payment, tolerance_cents, max_subset_size,
                         max_solutions, node_budget).run()


# --- features (no decisions) -------------------------------------------------

def _pool_features(pool: Sequence[Line], payment: Payment, tolerance_cents: int) -> dict:
    return {
        "pool_size": len(pool),
        "pool_allocations": len({l.allocation for l in pool}),
        "tolerance_cents": tolerance_cents,
        "payment_amount_cents": payment.amount_cents,
    }


def _date_span(solution: Solution, pool: Iterable[Line], payment: Payment):
    """Widest gap in days between the payment and any member of the group."""
    if payment.date is None:
        return None
    chosen = set(solution.line_ids)
    gaps = [abs((l.date - payment.date).days)
            for l in pool if l.id in chosen and l.date is not None]
    return max(gaps) if gaps else None


def _result_features(solutions, exhaustive, pool, payment) -> dict:
    """Everything a calibrator needs to learn what a verdict is worth.

    `n_solutions` is a true count when `search_exhaustive`, else a lower bound.
    Downstream must read the two together — which is why both are features and
    neither is meaningful alone.
    """
    best = solutions[0] if solutions else None
    runner_up = solutions[1] if len(solutions) > 1 else None
    return {
        "n_solutions": len(solutions),
        "search_exhaustive": exhaustive,
        "solution_unique": exhaustive and len(solutions) == 1,
        "abstained_on_pool_size": False,
        "subset_size": best.size if best else 0,
        "residual_cents": best.residual_cents if best else None,
        "residual_pct": (abs(best.residual_cents) / payment.amount_cents
                         if best and payment.amount_cents else None),
        "token_overlap_best": best.token_overlap if best else 0,
        "token_overlap_runner_up": runner_up.token_overlap if runner_up else 0,
        # The margin, not the raw overlap, is what discriminates: two subsets
        # both sharing a token say nothing about which one is real.
        "token_margin": ((best.token_overlap - runner_up.token_overlap)
                         if best and runner_up else (best.token_overlap if best else 0)),
        "date_span_days": _date_span(best, pool, payment) if best else None,
    }


# --- decision (no feature construction) --------------------------------------

def _choose_verdict(solutions: Sequence[Solution], features: dict) -> str:
    """Map evidence to one of five labels. Deliberately tiny and total."""
    if not solutions:
        return NONE
    if features["solution_unique"]:
        return UNIQUE

    # A single solution from a *truncated* search lands here, not above: one
    # found is not one existing. With no rival to compare against there is
    # nothing for references to corroborate, so it stays ambiguous.
    best = solutions[0]
    if len(solutions) == 1:
        return AMBIGUOUS

    runner_up = solutions[1]
    if best.token_overlap > 0 and best.token_overlap > runner_up.token_overlap:
        # Two independent signals agree. Available whether or not the search was
        # exhaustive: an unfound rival would still have to beat this one on
        # references to overturn it, so corroboration degrades gracefully where
        # a uniqueness claim would not.
        return CORROBORATED
    return AMBIGUOUS


def _abstained(pool, payment, tolerance_cents) -> Assembly:
    """Pool too large to search honestly, so we say so rather than search it
    dishonestly. PIPELINE.md §Stage 2."""
    return Assembly(ABSTAIN, None, [], {
        **_pool_features(pool, payment, tolerance_cents),
        "n_solutions": 0, "search_exhaustive": False,
        "solution_unique": False, "abstained_on_pool_size": True,
    })


def assemble(
    pool: Sequence[Line],
    payment: Payment,
    tolerance_cents: int = 0,
    max_pool: int = MAX_POOL,
    **kwargs,
) -> Assembly:
    """Assemble candidate groups for one payment and report what was found.

    Verdicts:
      NONE          no subset ties the amount
      UNIQUE        search completed; exactly one distinct answer exists
      CORROBORATED  several tie on arithmetic; one is singled out by shared
                    reference tokens — a second, non-arithmetic signal
      AMBIGUOUS     several answers, nothing independent separates them
      ABSTAIN       pool too large to search honestly

    AMBIGUOUS and ABSTAIN still carry `chosen` (the best-ranked solution) so a
    reviewer gets a starting point, but the verdict says plainly that it is a
    shortlist head and not a finding. Stage 6 decides what to do with each;
    this module describes the evidence and does not act on it.
    """
    pool = list(pool)
    if len(pool) > max_pool:
        result = _abstained(pool, payment, tolerance_cents)
        result.assert_never_false_unique()
        return result

    solutions, exhaustive = enumerate_solutions(
        pool, payment, tolerance_cents=tolerance_cents, **kwargs)

    features = {**_pool_features(pool, payment, tolerance_cents),
                **_result_features(solutions, exhaustive, pool, payment)}
    result = Assembly(_choose_verdict(solutions, features),
                      solutions[0] if solutions else None, solutions, features)
    result.assert_never_false_unique()
    return result


# --- explanation -------------------------------------------------------------
# PIPELINE.md §Stage 7 wants a template built from the top features, not free
# text. One formatter per verdict, chosen by table, so adding a verdict cannot
# silently fall through to a wrong sentence.

def _money(cents: int) -> str:
    return f"{cents / 100:,.2f}"


def _count_phrase(features: dict) -> str:
    """"3" vs "at least 3" — a completed count and a truncated one are different
    claims, and the reviewer is the one who has to tell them apart."""
    n = features["n_solutions"]
    return str(n) if features["search_exhaustive"] else f"at least {n}"


def _explain_abstain(a: Assembly) -> str:
    return (f"{a.features['pool_size']} candidate lines — too many to search "
            f"exhaustively; not guessing.")


def _explain_none(a: Assembly) -> str:
    return (f"No combination of {a.features['pool_size']} candidate lines ties "
            f"the amount.")


def _explain_unique(a: Assembly) -> str:
    return (f"{' + '.join(a.chosen.allocations)} sum to "
            f"{_money(a.features['payment_amount_cents'])} — the only "
            f"combination that does, out of {a.features['pool_size']} "
            f"candidate lines.")


def _explain_corroborated(a: Assembly) -> str:
    amount = _money(a.features["payment_amount_cents"])
    return (f"{' + '.join(a.chosen.allocations)} sum to {amount}, and share "
            f"{a.features['token_overlap_best']} reference token(s) with the "
            f"payment — but {_count_phrase(a.features)} other combinations also "
            f"sum to {amount}.")


def _explain_ambiguous(a: Assembly) -> str:
    return (f"{_count_phrase(a.features)} different combinations sum to "
            f"{_money(a.features['payment_amount_cents'])} and nothing "
            f"separates them — needs a human.")


_EXPLAINERS: dict[str, Callable[[Assembly], str]] = {
    ABSTAIN: _explain_abstain,
    NONE: _explain_none,
    UNIQUE: _explain_unique,
    CORROBORATED: _explain_corroborated,
    AMBIGUOUS: _explain_ambiguous,
}


def explain(assembly: Assembly) -> str:
    """One reviewer-facing line stating the count and its status."""
    return _EXPLAINERS[assembly.verdict](assembly)
