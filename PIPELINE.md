# Pipeline design — end-to-end reconciliation

Companion to [PROJECT.md](PROJECT.md). What the system should be, why, and in
what order to build it.

---

## Part 1 — What the research says

### 1.1 The collision rule is inverted (largest single finding)

`match.py` demotes a confident match to the human queue when many bank items
share its `(amount, date)`:

```python
if conf >= 0.95 and b.collisions >= COLLISION_LIMIT:   # COLLISION_LIMIT = 5
```

Measured on `predictions.csv`, precision **rises** with collision count:

| bank items sharing amount+date | n | precision |
|---|---|---|
| 5–7 | 715 | 99.4% |
| 8–11 | 582 | **91.8%** |
| 12+ | 5,644 | **100.0%** |

The reasoning was that a crowded amount+date makes a unique match a coincidence.
The data says the opposite: a 76-way amount+date block that still collapses to
*one* allocation is a bulk sweep or batch posting — structurally unambiguous. The
danger zone is the middle (8–11), where there are enough lookalikes to be wrong
but not enough to be a batch.

Cost of the current rule: **6,941 items (21.7% of eval) demoted at 99.41% actual
precision.**

Promoting the high-collision band:

| policy | matched | match rate | precision | errors |
|---|---|---|---|---|
| today (`k >= 5` demote) | 20,096 | 62.7% | 99.821% | 36 |
| promote `k >= 12` | 25,740 | 80.3% | 99.860% | 36 |
| promote `k >= 10` | 25,993 | **81.1%** | **99.819%** | 47 |

> **Caveat, stated plainly:** I picked that threshold by looking at eval labels.
> That is test-set leakage and is *not* a result — it is evidence that ~18 points
> of match rate are sitting behind one bad constant. The pipeline below fits the
> threshold on `train` and reports it on `eval` once. Expect the honest number to
> land below 81.1%.

### 1.2 Most "errors" are not errors

`error_autopsy.csv`, all 36 confident-tier misses:

| verdict | n |
|---|---|
| A. true allocation absent from the eval ledger entirely | 24 |
| B. true allocation exists but its amount does not tie the bank line | 10 |
| C. amount ties, fell outside the 5-day window | **2** |

Only **2** are ranking failures we could have avoided. In 24 cases the labelled
answer is not in the ledger we were given, so no matcher could find it. In 21 of
36 our own answer ties the bank amount exactly while the truth's does not.

BenchRec ships a ~0.2% label error rate — the reason the bar is 99.8% not 100%.
Across 20k assertions that predicts ~40 bad labels; we see 36 misses of which 34
are defensible. **The precision budget is being spent on label noise, not on our
mistakes.** That is the argument for spending the surplus on match rate.

### 1.3 Group targets are 5.6% of the benchmark, and we forfeit all of it

`solution.csv` has two target shapes. `truth_shape.py` found it; nothing acts on
it yet.

- 30,057 items — a single allocation
- **1,779 items (5.6%) — a list**, `[USD_...,USD_...,USD_...]`
- 212 items — null (correct answer: no match)

The matcher emits one allocation, so every list target is an automatic zero. But
we already name **a member of the correct group in 80.2%** of them. The candidate
generation is fine; there is no assembly step. This is one payment covering six
invoices — the case PROJECT.md opens with, and the one currently unimplemented.

This is a bounded subset-sum problem: find the subset of ledger rows summing to
the bank amount. Exponential in general, tractable here because group sizes are
small (930 of 1,779 groups are size 2; ~90% are ≤ 5) and amount + counterparty +
date prune the space hard.

### 1.4 Where the rest of the work is

| tier | n | share | precision |
|---|---|---|---|
| single allocation | 20,245 | 63.2% | 99.71% |
| collision-demoted | 6,941 | 21.7% | 99.41% |
| ambiguous (2+ allocations) | 2,890 | 9.0% | 18.10% |
| no candidate at all | 1,634 | 5.1% | — |
| reference-token tie-break | 338 | 1.1% | 98.52% |

Two untouched pools:

- **Ambiguous, 2,890 items at 18% precision.** Genuinely hard — this is the human
  queue and should stay one. But nothing ranks the shortlist, so the reviewer
  gets an unordered set. Ranking here is worth reviewer-seconds, not match rate.
- **No candidate, 1,634 items.** Blocked only on exact amount. Anything with a
  bank fee, an FX residual, or a partial payment never enters the funnel.
  `rule_archaeology.py` already measures residual sizes per production rule —
  that distribution is the tolerance spec, and it is unused.

### 1.5 Free supervision nobody is using

`train` labels every match group with the rule that made it (`RULE 1..9`) or
`MANUAL`. `rule_archaeology.py` profiles them but the output feeds nothing.

That file is a specification the bank wrote by doing the job 63,000 times:
whatever RULE 1 does, it was trusted at scale. And `MANUAL` groups are a labelled
set of *what no rule could handle* — a ready-made training target for "should
this go to a human?", which is the actual product.

Current code hand-tunes `DATE_WINDOW_DAYS = 5`, `COLLISION_LIMIT = 5`, and the
confidence literals `0.95 / 0.80 / 0.35`. Every one of those should be fit on
train, not chosen.

### 1.6 External state of the art

- **BenchRec** (Operartis, launched at ACM AI in Finance 2023) scores on match
  rate, match precision, *and confidence calibration*. Calibration is a
  first-class metric, not a nicety — which is exactly the thesis of this project.
- **Selective prediction / conformal risk control** is the right formal frame for
  "I'm sure / I'm not sure". It gives a distribution-free, finite-sample
  guarantee on error rate *among the cases you chose to answer* — precisely the
  99.8%-precision-subject-to-abstention contract. Replaces hand-picked cutoffs
  with a threshold carrying a proof.
- **Subset-sum matching** is the standard formulation for one-payment-many-invoices,
  and the literature agrees it is tractable at the 5–15 candidate scale we have.
- **Graph representation learning** for bank reconciliation is the current
  research frontier (bipartite bank↔ledger graph, learned edge scores). Worth
  knowing; not worth building before stages 1–6 below are done.

---

## Part 2 — The pipeline

Ten stages. Each is independently testable, each has one job, data flows one way.

```
  raw CSV
     │
 [0] ingest ──────────► canonical frames + run manifest
     │
 [1] block ───────────► candidate pairs      (recall-oriented, cheap)
     │
 [2] assemble ────────► candidate groups     (subset-sum, one↔many)
     │
 [3] featurize ───────► feature matrix       (pairwise + group + context)
     │
 [4] score ───────────► raw score per candidate   (learned, fit on train)
     │
 [5] calibrate ───────► P(correct)           (isotonic, held-out)
     │
 [6] decide ──────────► AUTO / REVIEW / NO-MATCH  (conformal threshold)
     │
 [7] explain ─────────► reason string per decision
     │
 [8] journal ─────────► append-only audit log, reversible
     │
 [9] evaluate ────────► frontier, calibration curve, drift
```

### Stage 0 — Ingest and canonicalise

**Job:** one trusted in-memory shape, so no later stage re-parses anything.

- Split rows into `ledger` (A) and `bank` (B); they never co-occur.
- Typed columns: `amt` (Decimal, 2dp — *not* float; money in float invites
  1-cent tie failures), `date`, `counterparty`, `currency`, `tokens`.
- Reference tokenisation lifted out of `match.py` into one shared function —
  today `match.py` and `rule_archaeology.py` each define their own
  `[A-Za-z0-9]{4,}` regex and they must not be allowed to drift.
- Emit a **run manifest**: dataset hash, code git SHA, config hash, timestamp.
  Nothing downstream is reproducible without it.

**Boundary:** everything after this reads canonical frames only. No stage reaches
back to raw CSV.

### Stage 1 — Blocking (candidate generation)

**Job:** maximise recall of the true counterpart, cheaply. Precision is stage 6's
problem, not this one's.

Today: one key, exact amount. That is why 1,634 items have no candidate.

Union of several keys, each generating candidates independently:

| key | catches |
|---|---|
| exact amount | the current 95% |
| amount within tolerance τ | bank fees, FX residuals |
| reference-token overlap (inverted index) | amount-mismatched but referenced |
| counterparty + date window | payments with no usable reference |
| rounded-amount bucket | the partial-payment tail |

τ comes from `rule_archaeology.py`'s residual distribution per rule — measured,
not guessed.

**Instrument this stage on its own.** Report *candidate recall*: in what fraction
of items is the true allocation in the pool at all? That number is the ceiling on
everything downstream, and right now nobody knows it. Measure it before tuning
anything else.

### Stage 2 — Group assembly

**Job:** the 5.6% currently forfeited. Emit allocation *sets*, not just single
allocations.

Bounded subset-sum over the stage-1 pool:

- Restrict to same counterparty and currency, date within window.
- Cap subset size at 8 (covers ~97% of observed groups) and pool size at ~40;
  beyond that, abstain to REVIEW rather than burn compute.
- Meet-in-the-middle or a DP over cent-quantised amounts; both are fast at this
  scale.
- **Uniqueness is the signal.** Exactly one subset summing to the bank amount is
  strong evidence. Several is ambiguity — emit them all and let stage 6 abstain.
  Do not tie-break silently; that is the failure mode this whole project exists
  to avoid.

Also handle many-to-one (several bank lines, one ledger row) — `rule_archaeology.py`
already reports the cardinality mix per production rule; build for the
cardinalities that actually occur.

**Output contract change:** a candidate is now a *set* of allocations. Stage 6's
verdict, the journal, and the scorer all need to accept sets from day one.
Retrofitting this later means touching every stage.

### Stage 3 — Features

**Job:** turn each (bank item, candidate set) pair into a row of numbers. No
decisions here.

- **Amount:** exact tie / signed residual / residual as % of amount / does the
  residual look like a fee (small, positive, round)?
- **Date:** signed day gap, gap vs. that counterparty's historical median gap.
- **Reference:** token overlap count, Jaccard, longest shared token, whether a
  shared token looks like an invoice number (long, digit-heavy).
- **Counterparty:** exact / normalised / fuzzy string similarity.
- **Competition (the ones that carry the calibration):** pool size, number of
  distinct allocations, **margin between best and second-best score**,
  amount+date collision count `k`, and — per §1.1 — `k` as a *non-monotone*
  feature. A tree model will find the middle-band risk on its own; a linear
  model will not. Prefer trees.
- **Group features:** subset size, whether the subset is the unique solution, how
  many alternative subsets tie.

### Stage 4 — Scoring

**Job:** one score per candidate. Fit on `train`, never on `eval`.

Start with gradient-boosted trees (LightGBM/XGBoost) on pairwise labels derived
from train `matchId` groups. Reasons: handles the non-monotone collision feature,
needs no scaling, trains in seconds at this scale, and the feature importances
are themselves an audit artefact.

Keep the current tier logic as a **baseline the model must beat**, and keep it in
the repo. It is the interpretable fallback if the model ever behaves oddly in
production, and its per-tier precision is a regression test.

Optionally add `MANUAL` vs. `RULE n` as an auxiliary target (§1.5) — predicting
"a human had to do this" directly is close to the product's actual question.

### Stage 5 — Calibration

**Job:** convert scores to probabilities that *mean* what they say. BenchRec
scores calibration explicitly; this stage is the project's thesis.

- Isotonic regression (monotone, non-parametric, no shape assumption) fit on a
  held-out slice of train — never on the data the scorer saw.
- Ship a **reliability diagram** as a build artefact: bucket by predicted
  probability, plot observed accuracy. A diagonal is the claim; the plot is the
  evidence.
- Report **ECE** and, more usefully here, per-bucket precision in the 0.95–1.00
  range, since that is the only region the auto-post decision reads.

### Stage 6 — Decide (the part that matters)

**Job:** three buckets, with a guarantee rather than a hunch.

Given the calibrated probability, choose threshold `t` on a **calibration split
disjoint from both training and eval** such that precision among accepted items
is ≥ 99.8% — using a conformal / selective-risk bound so `t` carries a
finite-sample guarantee, not just a point estimate that happened to hold once.

```
p >= t                          → AUTO      post it, no human
alternatives exist, p < t       → REVIEW    ranked shortlist + reasons
no candidate survives blocking  → NO-MATCH  say so out loud
```

NO-MATCH is a first-class verdict, not a failure. 212 eval items have a null
target — "nothing matches" is the *correct* answer there and must be scoreable as
such. A system that cannot say "I don't know" is the system this project exists
to replace.

This stage replaces every hand-tuned constant in `match.py`. `COLLISION_LIMIT`,
the 0.95/0.80/0.35 literals, and the 5-day window all become either learned
features or a fitted threshold.

### Stage 7 — Explain

**Job:** every REVIEW item arrives with a ranked shortlist and a plain-English
reason per candidate, so the decision takes seconds.

Template from the top features, not free text:
`"£4,200.00 exact, same day, shares reference INV88213 — but 2 other customers
paid £4,200.00 on 2023-04-12."`

For AUTO items, store the same string. That is what the accountant signing off on
the month reads.

### Stage 8 — Journal

**Job:** PROJECT.md promises "everything it decides is written down… any of it can
be undone." Nothing in the repo does this yet.

Append-only table, one row per decision: `run_id`, `B_id`, verdict, chosen
allocation set, calibrated probability, reason, model version, config hash,
timestamp, and `reversed_by` (nullable). Reversal is a new row, never an update —
the history must be immutable to be auditable.

SQLite is sufficient and keeps the whole thing a single portable file.

### Stage 9 — Evaluate

**Job:** one command, one report, comparable across runs.

- Candidate recall (stage 1 ceiling) — the number currently missing.
- Full precision/match-rate frontier, as `match.py` already prints.
- Headline: best match rate clearing 99.8%, plus the shipped baseline (0.5%).
- Reliability diagram + ECE.
- Per-tier breakdown, kept from today's `report()` — it is genuinely good.
- **Error autopsy on every run**, auto-bucketed A/B/C/D as `error_autopsy.py`
  already does. Regressions should arrive pre-diagnosed.
- Group-target scoring: exact-set match *and* member-recall, so §1.3 progress is
  visible.

---

## Part 3 — Build order

Ordered by measured value per unit of work.

| # | Work | Expected |
|---|---|---|
| 1 | **Fix the collision rule.** Fit the band on train, apply to eval once. | 62.7% → ~75–81% |
| 2 | **Candidate recall instrumentation.** Learn the real ceiling. | unblocks everything |
| 3 | **Group assembly** (stage 2). | up to +5.6% |
| 4 | **Widen blocking** (stage 1, tolerance keys). | attacks the 5.1% no-candidate pool |
| 5 | **Learned scorer + isotonic calibration** (4–5). | replaces all hand-tuned constants |
| 6 | **Conformal threshold** (6). | turns 99.8% into a guarantee |
| 7 | **Journal + explanations** (7–8). | delivers PROJECT.md's audit promise |
| 8 | Graph model (§1.6). | research; only after 1–7 |

Steps 1–2 are days of work and are worth more than everything below them. Do not
start at step 5.

---

## Part 4 — Repo shape

Currently seven flat scripts that import from each other (`sanity.py` imports
`match.py` *and* `error_autopsy.py`), plus a 12 MB `predictions.csv` and a 3 MB
profile CSV sitting next to the source, and no git repository.

```
recon/
  __init__.py
  ingest.py        # stage 0
  blocking.py      # stage 1
  assemble.py      # stage 2
  features.py      # stage 3
  model.py         # stages 4-5
  decide.py        # stage 6
  explain.py       # stage 7
  journal.py       # stage 8
  evaluate.py      # stage 9
  config.py        # every constant, one file, hashed into the manifest
cli.py             # recon fit | recon run | recon evaluate | recon explain <B_id>
research/          # benchrec_probe, rule_archaeology, truth_shape, sanity, error_autopsy
tests/
artifacts/         # predictions, journals, plots — gitignored
```

Move the five investigation scripts to `research/` rather than deleting them.
They are the evidence for Part 1 and each one earned a finding.

Housekeeping worth doing on day one:

- **`git init`.** No history exists. The 62.7% number is currently unattributable
  to any specific state of the code.
- **`.gitignore`** the CSV outputs; they are build products.
- **Cache the Kaggle download.** Every script calls `kagglehub.dataset_download`
  at import; the evaluation loop should not depend on the network.
- **Pin the split.** Fitting on train and reporting on eval only works if the
  calibration slice is fixed and stored in the manifest.

## Part 5 — Tests that would have caught real bugs

- **Monotonicity:** precision must not decrease as the confidence threshold
  rises. Today it does — §1.1 is exactly this test failing, and nothing was
  checking.
- **Golden set:** ~50 hand-verified items, including the 2 genuine type-C errors,
  asserted per run.
- **Group assembly:** synthetic one-to-many cases with known answers, plus an
  explicit case with two valid subsets asserting the verdict is REVIEW, not a
  silent pick.
- **Calibration:** predicted probability in bucket b must be within tolerance of
  observed accuracy in bucket b.
- **Journal:** every AUTO decision has exactly one journal row; reversal appends
  and never mutates.
- **Determinism:** same input + same config hash ⇒ byte-identical predictions.
