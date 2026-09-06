# Glossary

Every term used in [PIPELINE.md](PIPELINE.md) and [PROJECT.md](PROJECT.md),
explained from scratch. Examples come from this project's real numbers, not
textbooks.

---

## Start here — the six that unlock everything else

If you read nothing else, read these. The rest of the document leans on them.

**Precision** — Of the answers the system *chose to give*, what fraction were
right? Yours is 99.82%: of 20,096 matches it asserted, 36 were wrong.

Precision only counts the questions you answered. That's the whole point — a
system allowed to stay silent can have very high precision by only speaking when
sure. Which is exactly the design.

**Match rate** — Of *all* the work, what fraction did the system handle by
itself? Yours is 62.7%: 20,096 out of 32,048 bank lines.

Precision and match rate pull against each other. Answer only the dead-obvious
cases → precision near 100%, match rate tiny (that's the 0.5% baseline). Answer
everything → match rate 100%, precision collapses. **The entire project is about
where you sit on that trade-off, and being honest about it.**

**Ground truth / label** — The known-correct answer, used for grading. Here it's
`solution.csv`: for each bank line, which ledger allocation it really belonged
to. You only have this because the dataset ships it. In real production nobody
hands you the answers — which is why the confidence machinery matters.

**Allocation** — The thing you're matching *to*. Not a single ledger row — a
named bucket that one or more ledger rows belong to (`USD_2023-03-05_ACC#00001_...`).

This distinction is the smartest thing already in your code. Seventeen ledger
rows tied for the same bank line looks like hopeless ambiguity — but if all
seventeen belong to *one allocation*, there's no ambiguity at all. Same answer
either way. `match.py` collapses candidates by allocation before asking "is this
ambiguous?", and that's why it works.

**Confidence** — The number the system attaches to each answer saying how sure it
is (your `0.95 / 0.80 / 0.35`). Right now these are hand-picked constants that
*rank* answers but don't literally mean anything.

**Calibration** — Whether those numbers are *true*. If you take every answer
labelled 90% confident, are about 90% of them actually right? If yes, calibrated.
If 60% are right, your 90% is a lie — and in accounting, a confident lie is worse
than an admitted unknown.

This is your project's actual thesis. Everyone can match. Nobody can tell you
which matches to trust.

---

## The finance side

**Ledger / books / GL** — The company's own record of what money was *supposed*
to move. "General Ledger." Side **A** in your data.

**Bank statement** — The bank's record of what money *actually* moved. Side **B**.

**Reconciliation** — Making A and B agree, line by line. The monthly job this
project automates.

**Value date** — The date the money actually counts as moved, which is not always
the date it was entered. Your `A_valueDate` / `B_valueDate`, and why matching
allows a date *window* instead of demanding same-day.

**Counterparty** — The other party. Who paid you, or who you paid.

**Transaction reference** — Free-text scribbled on a payment: an invoice number,
a customer code, a description. Sometimes it identifies the payment precisely,
often it's junk. Your `transactionReferences` column, and the source of the
"reference tokens break the tie" tier.

**Match group / `matchId`** — A set of rows from both sides declared to be the
same real-world event. One bank line + one ledger row, or one bank line + six
ledger rows, and so on.

**Cardinality** — The *shape* of a match group, written `A:B`.
- `1:1` — one bank line, one ledger row. The easy case.
- `1:many` — one payment covering six invoices. **This is the 5.6% you currently
  forfeit** (see §1.3 of PIPELINE.md).
- `many:1` — several bank lines settling one ledger entry.

**Posting** — Committing an entry to the books for real. "Post it" = the system
acted without a human. Reversing a wrong posting is real, visible, annoying work
— which is why a wrong answer costs more than no answer.

**Suspense / holding account** — Where unmatched money is parked until someone
figures it out. Anything sitting here means the company doesn't fully know its own
cash position.

**Month-end close** — The deadline. Books must balance before the month can be
closed, and suspense must be cleared.

**Residual** — The leftover when two amounts nearly-but-not-quite tie. You're owed
£4,200, £4,187.50 lands, residual £12.50 — probably a bank fee. Your matcher
requires *exact* amounts, so every residual case falls out of the funnel
entirely. That's a large part of the 1,634 "no candidate" items.

**FX** — Foreign exchange. Currency conversion, another common source of residuals.

**Sweep / batch posting** — One instruction moving many identical amounts at once
(payroll, a standing settlement run). This is *why* the collision finding works:
76 bank lines sharing an amount and date that all collapse to one allocation
aren't 76 coincidences, they're one batch.

**Nets to zero** — The two sides of a real match have opposite signs, so a correct
group sums to ~0. A cheap correctness check, used in `rule_archaeology.py`.

**Audit trail** — The written record of what was decided, by what, when, and why —
such that an accountant signing off can trace any number back to its origin.
PROJECT.md promises this; nothing in the code does it yet (that's stage 8).

**Reversal** — Undoing a posting. Critically, in an audit trail you undo by
*adding a correcting record*, never by editing the old one. History has to be
immutable to be trustworthy.

---

## Measuring things

**Recall** — Precision's mirror. Of all the answers that *existed to be found*,
what fraction did you find? Precision asks "were you right when you spoke";
recall asks "did you miss things".

**Candidate recall** — Recall for one specific step: how often is the correct
answer *anywhere in the shortlist* the system generates, before any ranking?

This is the number PIPELINE.md keeps insisting nobody has measured. It's a
**ceiling**: if the right answer isn't in the shortlist, no amount of clever
ranking can find it. If candidate recall is 85%, then 85% is the best match rate
you could ever reach, no matter what you build. Worth knowing before optimising
anything.

**Ceiling** — The best score achievable given a constraint you haven't lifted
yet. Your ceilings: candidate recall (unmeasured), and the label noise in §1.2.

**Baseline** — The thing you have to beat, so a number means something. Yours is
`MatcherByChatGPT` at 0.5%.

**Frontier** — The whole precision/match-rate trade-off curve, rather than one
point. Your `report()` already prints it: at confidence ≥0.95 you get 62.7% @
99.82%; at ≥0.50 you get 85.9% @ 99.62%. Showing the curve is more honest than
quoting one number, because it lets a reader pick their own risk appetite.

**Threshold / cutoff** — The confidence line above which you act. Everything
above it → auto-post; below → human. Choosing this line *is* the product
decision.

**Label noise** — Errors in the ground truth itself. BenchRec admits ~0.2% of its
answers are wrong, which is why the bar is 99.8% and not 100%.

This matters enormously to you: across 20k assertions, ~40 bad labels are
*expected*. You have 36 misses, of which 34 look defensible. **You're being graded
down for being right.** Hence: spend the leftover precision on match rate.

**Ambiguity** — Multiple candidates that the available evidence genuinely cannot
separate. Two customers, both £4,200, both Tuesday, no useful reference. The
correct response is to abstain — not to guess and sound confident.

---

## Training, and how to not fool yourself

**Training set / train split** — Data with answers attached, used to *fit* the
system. Your `train.csv`, ~63k match groups.

**Evaluation set / eval / test set** — Held back, used to *grade*. Your
`eval.csv` + `solution.csv`.

**Holdout / calibration split** — A third slice, carved from train, never used
for fitting. You need it because a model's confidence on data it trained on is
always inflated. Calibrate on data the model has never seen.

**Leakage** — Letting information from the grading set influence your design.
This is the cardinal sin of ML, and easy to commit by accident.

**I committed it deliberately in §1.1** so you could see the size of the prize:
I chose the collision threshold by looking at eval answers. That's why the doc
calls the 81.1% "not a result." Do it properly and you fit that threshold on
train, then look at eval exactly once. The honest number will be lower — which is
the point of doing it honestly.

**Overfitting** — Learning quirks of your specific data rather than real patterns.
Memorising the answer key instead of the subject. Shows up as great numbers in
development and bad ones in production.

**Feature** — One measurable fact about a candidate, expressed as a number.
"Amounts match exactly: yes/no." "Days apart: 3." "Shared reference tokens: 2."

**Feature matrix** — All those numbers as a table: one row per candidate, one
column per feature. What a model actually consumes.

**Model / scorer** — The thing that turns a feature row into a score. You don't
have one; you have hand-written rules, which is a fine place to start.

**Gradient-boosted trees / LightGBM / XGBoost** — A specific family of models. A
"tree" is a flowchart of yes/no questions ("amount exact? → date within 2 days? →
…"). Boosting builds hundreds of small ones, each correcting the last.

Recommended here for concrete reasons: they handle **non-monotone** relationships
(see below), they're fast on data this size, and you can read out which features
mattered — which is itself an audit artefact.

**Monotone / non-monotone** — Monotone: more of X always means more of Y.
Non-monotone: the relationship bends.

Your collision count is beautifully non-monotone. Precision goes 99.4% (5–7
collisions) → **91.8%** (8–11) → **100%** (12+). A simple rule assuming "more
collisions = worse" — which is exactly what `COLLISION_LIMIT = 5` assumes — gets
it backwards. A tree model discovers the bend on its own.

**Pairwise labels** — Turning "these rows form a match group" into training rows
of the form (bank line, candidate) → correct/incorrect. How you convert your
`matchId` groups into something a model can learn from.

**Auxiliary target** — A second thing to predict alongside the main one, because
it teaches something useful. Here: your train data marks each group `RULE 1..9`
or `MANUAL`. Predicting "would a human have had to do this?" is almost exactly the
product's real question.

---

## Confidence, properly

This cluster is the technical heart of the project.

**Selective prediction** — A model allowed to say "I'm not answering this one."
The formal name for your three-bucket design. The field's insight: you're
measured on the answers you *give*, so declining to answer is a legitimate move,
not a failure.

**Abstention** — Declining to answer. Your REVIEW and NO-MATCH buckets.

**Isotonic regression** — The specific technique for fixing miscalibrated
confidence. Feed it raw scores and known outcomes; it learns a correction curve
turning "0.8" into "actually right 94% of the time." "Isotonic" = the correction
only ever goes one way (higher score never maps to lower probability), which
keeps the ranking intact while fixing the meaning.

**Reliability diagram** — The picture that proves calibration. X-axis: predicted
confidence. Y-axis: how often those were actually right. Perfect calibration is
a diagonal line. Sagging below = overconfident. One glance tells you whether the
numbers mean anything.

**ECE (Expected Calibration Error)** — The reliability diagram as a single number:
average distance from the diagonal. Lower is better. Useful for tracking across
runs; the diagram is better for understanding.

**Conformal prediction** — A method for producing confidence statements that come
with a *mathematical guarantee* rather than a hope.

The difference matters. Today you can say: "I tried threshold 0.95 and got 99.82%
precision on this dataset." Conformal lets you say: "with 95% certainty, this
threshold holds ≥99.8% precision on new data too." One is an observation about
the past; the other is a promise about the future. For a bank sign-off, only the
second is worth much.

**Finite-sample guarantee** — The guarantee holds with the data you actually
have, not "as your sample size approaches infinity." Practically important —
you have 32k rows, not infinite rows.

**Distribution-free** — Requires no assumptions about the shape of your data.
Robust, because financial data reliably violates every tidy assumption.

---

## The algorithms

**Blocking** — Cheaply narrowing 32,000 × 37,000 possible pairs down to a
plausible shortlist, so you only score candidates worth scoring. Comparing
everything to everything is ~1.2 billion pairs; blocking makes it tractable.

**Blocking key** — The field you narrow *on*. Yours is exact amount, and only
that — which is precisely why fee-adjusted and partial payments never enter the
funnel at all.

**Inverted index** — A lookup table pointing from a value back to everything
containing it: `"INV88213" → [rows 4, 92, 1043]`. Same structure a search engine
uses. Suggested for reference tokens so you can block on shared references, not
just amounts.

**Tokenisation** — Chopping free text into searchable pieces. Yours:
`[A-Za-z0-9]{4,}` — every alphanumeric run of 4+ characters, so `"PAYMENT REF
INV88213"` → `{PAYMENT, INV88213}`. The 4-character floor drops noise like "TO"
and "REF".

**Jaccard similarity** — How much two sets overlap, scaled 0 to 1: shared items ÷
total distinct items. Better than a raw count, because 2 shared tokens out of 3
is much stronger evidence than 2 out of 50.

**Fuzzy matching** — Comparing things that are close but not identical. "ACME LTD"
vs "Acme Limited". As opposed to exact matching, which is all you do today.

**Subset sum** — Given a list of numbers and a target, find which subset adds up
to it. Formally hard in general (the number of subsets doubles with each extra
item), but genuinely easy at your scale: ~90% of your groups have ≤5 members, and
counterparty/date filtering shrinks the pool first.

**This is the missing 5.6%.** "One payment covering six invoices" is literally a
subset-sum problem, and you have no code for it.

**Combinatorial explosion** — Why subset sum is scary in the abstract: 40 items
have over a trillion subsets. Handled by capping subset size (≤8) and pool size
(≤40), and abstaining beyond that rather than burning compute.

**Meet-in-the-middle** — A trick that makes subset sum much faster: split the
items in half, enumerate all sums of each half separately, then look for pairs
that add to the target. Turns a problem of size 2ⁿ into roughly 2^(n/2).

**Dynamic programming (DP)** — Solving a big problem by solving small overlapping
pieces once and reusing the answers. The other standard subset-sum approach: build
a table of "which totals are reachable using the first *k* items."

**Bipartite graph** — A network with two kinds of node and edges only *between*
the kinds — bank lines on one side, ledger rows on the other, edges as possible
matches. A natural way to picture reconciliation.

**Graph representation learning** — Machine learning that reads the whole network
structure rather than each pair in isolation, so evidence about one match informs
its neighbours (if bank line 5 certainly takes ledger row 12, row 12 is no longer
available to anyone else). Current research frontier; deliberately last on your
build order.

---

## Engineering

**Pipeline** — Work split into ordered stages, each with one job, data flowing one
direction. The value is that you can test, fix, or replace any single stage
without disturbing the others. Right now `match.py` does blocking, scoring, and
deciding in one function, so you can't improve one without risking the rest.

**Canonical form** — One agreed, cleaned-up shape for the data that every later
stage trusts. Prevents the bug where two files parse the same column slightly
differently — which you're at real risk of, since `match.py` and
`rule_archaeology.py` each define their own copy of the token regex today.

**Decimal vs float** — How the computer stores numbers. Floats are approximate:
`0.1 + 0.2` genuinely doesn't equal `0.3`. Fine for physics, dangerous for money,
because an exact-amount comparison can fail by a hundredth of a penny that
doesn't really exist. `Decimal` is exact. Always use it for currency.

**Manifest** — A small record written with each run: which data, which code
version, which settings, what time. Without it, "we got 62.7%" is unreproducible
folklore.

**Config hash** — A short fingerprint of all your settings, so you can tell at a
glance whether two runs were configured identically.

**Git / git SHA** — Version control; the SHA is a unique ID for one exact state of
the code. You have no repository, so today's 62.7% can't be tied to any specific
version of anything.

**`.gitignore`** — A list of files version control should ignore. Your
`predictions.csv` (12 MB) and `match_group_profile.csv` (3 MB) belong here —
they're outputs, regenerable from code, not source.

**Append-only** — A store you only ever add to, never edit or delete. The
foundation of an audit trail: corrections arrive as new entries, so the full
history stays visible.

**SQLite** — A complete database that lives in one ordinary file. No server, no
setup. Ideal for the journal.

**CLI** — Command-line interface. One entry point (`recon run`, `recon evaluate`)
instead of running seven scripts in a remembered order.

**Cache** — Keeping a local copy so you don't re-fetch. Every script currently
calls `kagglehub.dataset_download` on import, so your evaluation loop needs
network access to run at all.

**Determinism** — Same inputs always produce identical outputs. Non-negotiable
for auditability: "the system posted this last Tuesday" must be reproducible
today.

**Regression test** — A test that catches you *breaking* something that used to
work. Named for "regressing" — going backwards.

**Golden set** — A small batch of hand-verified examples with known-correct
answers, checked on every run. Cheap, and catches most catastrophes.

**Test coverage of a real bug** — Worth noticing: a single test asserting
"precision must not go *down* as confidence goes *up*" would have caught the
collision bug in §1.1 immediately. It's not a subtle failure. Nothing was
checking.
