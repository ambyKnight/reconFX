# Architecture — reconciliation that works like an accountant

Successor to [PIPELINE.md](PIPELINE.md). That document described a *pipeline*:
data enters one end, decisions come out the other. This one describes something
closer to how the job is actually done — repeated passes, a suspense account,
and a close report at the end.

Terms are defined in [GLOSSARY.md](GLOSSARY.md).

---

## 1. The organizing idea: passes, not a pipeline

A real accountant does not process 32,000 bank lines in one sweep. They do the
obvious ones first, and then — this is the important part — **the obvious ones
make the hard ones easier.**

If bank line #4,001 certainly belongs to allocation X, then allocation X is spent.
It can no longer compete for bank line #7,882. A tie that looked unbreakable at the
start of the day resolves itself for free once the rival has been claimed by
somebody else.

Your current matcher scores every bank line **independently**, in one pass, and
never revisits. Two different bank lines can both be told "you match allocation
X" and nothing notices. That is not a small bug — it means the 2,890 ambiguous
items include an unknown number that are not actually ambiguous, just
prematurely judged.

So the architecture is a **loop**, and each turn of the loop is cheaper and
smarter than the last:

**Pass 1 — Certainties.** Deterministic rules only, no LLM. Unique amount+date
collapsing to one allocation, and (per PIPELINE.md §1.1) the high-collision batch
postings. These are consumed and removed from the pool.

**Pass 2 — Re-block after consumption.** Rebuild the candidate pools now that
pass 1's allocations are spent. Free. Requires no new logic and no model. Some
former ties are now single-candidate. Re-run pass 1's rules on the survivors.

**Pass 3 — Group assembly.** Subset-sum over what remains: one payment covering
six invoices, one invoice settled by three payments. Half the "ambiguous" pool is
actually this (see §3.4). Consume, then loop back to pass 2 — assembled groups
free up allocations too.

**Pass 4 — LLM adjudication.** Only now, on what deterministic logic could not
settle, and only on a pool that earlier passes have already thinned and
disambiguated. Roughly 4,500 items today; materially fewer after passes 1–3.

**Pass 5 — LLM group proposal.** The exotic tail: partial payments, fee-adjusted
amounts, cross-period timing, "this looks like the same customer under a
different name." The LLM proposes a *structure*, deterministic code verifies the
arithmetic.

**Stop** when a pass resolves nothing new. Everything still unresolved goes to
suspense, with a record of what was tried.

Two things follow from this design and are worth saying explicitly:

**Matching is a global assignment problem, not a per-row lookup.** Each pass
should resolve mutually-exclusive claims rather than let two bank lines claim one
allocation. Greedy-by-confidence (highest-confidence claims first, consume, repeat)
is the accountant's own algorithm and is good enough to start. Min-cost bipartite
matching is the formal version if greedy proves lossy.

**Every pass makes the next one cheaper.** Passes 1–3 cost nothing but CPU, and
each one removes items from the LLM's bill. Ordering the loop this way is not
just tidier — it is the cost control.

---

## 2. The components

Nine, each with one job. Passes are orchestrated over these; the components
themselves are stateless except where noted.

**2.1 Ingest.** Canonical, typed representation of both sides. Money as `Decimal`,
never float. Emits a run manifest (data hash, code version, config hash, model
version) so any close can be reproduced.

**2.2 Candidate engine (blocking).** Cheap, recall-oriented shortlisting: exact
amount, amount within tolerance, shared reference tokens, counterparty + date
window. Its job is to never lose the right answer; precision is somebody else's
problem. It reruns each pass against the shrinking pool.

**2.3 Deterministic matcher.** The rules that need no model. Fast, free,
explainable, and the fallback if anything downstream misbehaves. This is your
existing `match.py`, corrected and kept.

**2.4 Group assembler.** Bounded subset-sum, with uniqueness as the signal: one
subset summing to the target is evidence; several is ambiguity, and it says so
rather than picking.

**2.5 LLM adjudicator.** Covered in §3.

**2.6 Decision and risk control.** Takes a calibrated probability and returns
AUTO / REVIEW / SUSPENSE, against a threshold that is fitted rather than chosen.
**Crucially, this component is the only thing that can post.** The LLM proposes;
this decides. That separation is what keeps the audit story coherent.

**2.7 Suspense subsystem.** §4. This is the part most systems treat as a
leftovers bin and it deserves better.

**2.8 Journal.** Append-only record of every decision: what, why, how sure, by
which model version, when, and by whose authority. Reversals are new rows, never
edits.

**2.9 Reporting.** §5. The close pack.

---

## 3. The LLM's job, precisely

The value of an LLM here is real but narrow. It is very good at weighing
scruffy human-written evidence — "PYMT RE INV 88213 & 88219 LESS CREDIT" — and
saying what that probably means. It is bad at arithmetic, and it is a security
surface. So the job description is deliberately tight.

### 3.1 What it receives

A **dossier**, assembled deterministically: the bank line, the shortlist of
candidates (numbered), and for each candidate a set of **pre-computed facts** —
amount delta, date gap, shared reference tokens, counterparty similarity, whether
this counterparty historically pays late or short, whether the candidate is
already claimed.

The model never computes any of these. It reads them.

### 3.2 What it returns

A structured verdict, schema-enforced:

- **MATCH** — candidate index *n* (an index into the list it was given, never an
  identifier it composed itself)
- **MATCH_GROUP** — a set of indices
- **NO_MATCH** — nothing here is the counterpart; this belongs in suspense
- **UNSURE** — I cannot tell

plus a confidence, a reason in one plain sentence, and which facts drove it.

**UNSURE and NO_MATCH are the point.** A model that must choose between MATCH and
NO_MATCH will confabulate a winner, and you have rebuilt the exact problem this
project exists to solve — software that sounds equally certain whether it is
right or wrong. The abstention options are load-bearing, and the prompt should
make abstaining explicitly respectable.

### 3.3 What happens to the verdict

Three gates, in order:

1. **Arithmetic re-verification.** If the model says these five invoices sum to
   the payment, Python adds them up. Disagreement voids the verdict. The model is
   never trusted on a number.
2. **Calibration.** The model's stated confidence is meaningless raw — a
   self-reported "90%" is a linguistic habit, not a probability. Note that the
   Anthropic API does not expose token logprobs, so there is no free confidence
   signal to read out. You get calibrated numbers one of two ways: fit isotonic
   regression on train-set outcomes to map stated confidence onto observed
   accuracy, or sample the model *k* times and use agreement rate as the
   confidence (self-consistency). Agreement is usually better calibrated and
   costs *k*×.
3. **The same threshold as everything else.** The calibrated LLM verdict competes
   on identical terms with the deterministic tiers. It gets no special authority
   because it produced prose.

This is the architectural point worth holding onto: **the LLM is one more scorer
feeding the existing calibration and decision stages.** It does not get a side
door to the ledger. Everything that made the rule-based system trustworthy
applies to it unchanged.

### 3.4 Why this pool suits an LLM particularly well

Measured on the current 2,890 ambiguous items: **1,420 of them (49%) have a
list-shaped truth.** So half the "ambiguity" is not a contest between rivals —
it is a group waiting to be assembled. The matcher sees nine candidates and asks
"which one?", when the correct answer is "six of these, together."

That is a reading-comprehension problem over payment references, which is exactly
what a language model is for, and exactly what amount-and-date rules cannot do.

Meanwhile in the no-candidate pool, 181 of 1,634 items have a genuinely null
truth — **NO_MATCH is the correct answer about 11% of the time there**, and a
system without that verdict scores zero on all of them.

### 3.5 Cost

At current pool size (~4,524 items, ~16 candidates each), one full pass on Claude
Opus 5 runs roughly **$106**, or **~$53 through the Batch API** — this is
inherently an overnight batch job, so the 50% discount is free money. About
**$0.02 per item.** Three-sample self-consistency triples that to ~$160 batched.
Sonnet 5 would be ~$21 batched.

Cache the instruction block and schema (identical on every call); only the
dossier varies. And note passes 1–3 shrink this bill before it is ever incurred.

At these numbers, **cost is not a design constraint** — a bank spends
dramatically more on the analyst hours this replaces. Spend the tokens on
reasoning quality and on self-consistency, not on trimming prompts.

### 3.6 Where the LLM does not belong

- **Not in blocking.** 32,000 × 37,000 is 1.2 billion pairs. Deterministic keys
  do this in seconds.
- **Not doing arithmetic.** Ever.
- **Not making the post decision.** That is a calibrated threshold with a
  guarantee behind it.
- **Not as the system of record.** The journal is. Store the verdict the model
  gave; never re-derive it later by asking again. Models are non-deterministic and
  they get upgraded — last March's close must still reproduce exactly, and the
  only way to guarantee that is to have written the answer down.

---

## 4. Suspense, properly

Suspense is where the honesty lives. It is not a failure bucket; it is the
system's considered statement that it does not know, and it should be as
structured as any other output.

### 4.1 Every suspense item carries a case file

Not just "unmatched" — the record should say: which passes ran, what was tried,
the best candidates considered and why each was rejected, the model's reasoning if
it got that far, the amount and its materiality, the age, and the assigned
category.

An accountant picking up a suspense item should never have to redo the
investigation the system already did.

### 4.2 Categories, because they route differently

- **Timing difference** — the counterpart probably exists but is not booked yet.
  Expected to clear on its own next period.
- **Missing ledger entry** — the money moved and nobody recorded it. Someone must
  create the entry.
- **Fee or FX residual** — matched except for a small deduction. Resolvable by a
  tolerance rule or a write-off.
- **Genuine ambiguity** — several plausible counterparts, none separable. Needs a
  human who knows the business.
- **Unidentified receipt** — money arrived, nobody knows from whom. Needs the
  customer contacted.
- **Suspected duplicate** — looks like the same payment twice.
- **Data error** — malformed, wrong sign, impossible date.

The category determines *who* gets it and *what they do*, which is the difference
between a queue and a list.

### 4.3 Aging

Bucket by days outstanding: 0–7, 8–30, 31–60, 60+. Aging is the health metric of
the whole system. A growing 60+ bucket means either the matcher is degrading or
the business changed, and either way somebody needs to know before the auditors
find it.

### 4.4 Suspense is re-attempted, not archived

**Every run re-feeds open suspense items through the full pass loop.** This is not
an optimisation — it is the single most valuable property of the subsystem.

A payment that arrived on the 28th before its invoice was booked on the 2nd is
unmatchable in month one and trivially matchable in month two. Timing differences
are the largest category of real-world suspense, and they resolve themselves *if
you keep looking*. A system that files unmatched items away permanently guarantees
that a human will eventually do work the machine could have done for free.

Auto-clearing a suspense item on a later run is a first-class event: journalled,
reported, and reversible.

### 4.5 Materiality

Accountants do not spend forty minutes chasing eleven pence. Below a configurable
threshold, the system should propose a **write-off** rather than an
investigation, batched for a single approval. Above a high-value threshold, the
opposite: escalate immediately, and demand higher confidence before ever
auto-posting (see §6.3).

---

## 5. The close report

After each pass — and at period end — the system produces a pack. This is the
deliverable an accountant actually signs, so it should be built as a first-class
output rather than scraped out of logs.

**5.1 Reconciliation summary.** Volume and value: lines in, lines matched, %
matched by count *and by value* (they differ, often sharply — a 90% count match
rate can leave 60% of the money unresolved). Auto-posted vs. human-reviewed vs.
suspense. Comparison against prior periods.

**5.2 Exceptions register.** The suspense book: every open item by category, aged,
sorted by value, each with its case file. This is the working document for the
close, not an appendix.

**5.3 Control and assurance report.** The part that makes an auditor comfortable:
how many items were auto-posted, at what calibrated confidence, the measured
precision, the calibration evidence, and the results of the sampling programme
(§6.4). This is where "62.7% at 99.8%" becomes an attestable statement rather
than a slide.

**5.4 Override log.** Every case where a human disagreed with the system —
overrode a match, rejected a proposal, matched something it called ambiguous.
This is both a control signal (are overrides rising? in which category?) and the
highest-quality training data you will ever get, since each one is a labelled
example produced by an expert for free.

**5.5 Drift and trend.** Match rate, suspense aging, category mix, and per-rule
performance over time. Reconciliation quality degrades quietly — a new payment
provider, a changed reference format, a migrated ERP — and the numbers move before
anyone notices the cause.

**5.6 The narrative.** A good use of the LLM: write the plain-English summary
that goes on top. *"Match rate held at 81% this period. Suspense value rose 22%,
driven almost entirely by 41 items from one counterparty whose reference format
changed on the 14th."* The model writes prose over numbers that were computed
deterministically — it never produces a figure of its own.

---

## 6. Further features, roughly by value

### 6.1 Counterparty behavioural memory

Learn each counterparty's habits: pays 4 days late on average, always £15 short
(their bank's charge), references invoices as `INV-#####`, settles in batches on
month-end. Store it, feed it to both the feature model and the LLM dossier.

This is what makes a human reconciler fast — they *know* their counterparties —
and it converts a large class of fuzzy matches into confident ones. Probably the
highest-value item on this list.

### 6.2 Learning from overrides

Each human correction is a labelled example. Feed them back into the scorer and
the calibration, and track whether the categories humans keep correcting are
shrinking. A system that makes the same mistake every month is not learning, and
the override log is the only place that shows up.

### 6.3 Risk-weighted thresholds

A £3 line and a £3,000,000 line should not face the same confidence bar. Make the
threshold a function of exposure: auto-post small items freely, demand near-
certainty on large ones, and let expected cost of error — not a single global
number — drive the decision.

This mirrors how accountants actually allocate attention, and it usually buys
match rate *and* reduces risk simultaneously, because most volume is small and
most risk is concentrated in a few large lines.

### 6.4 Continuous assurance sampling

In production there is no `solution.csv`. So how do you know the 99.8% still
holds?

Randomly sample ~1% of auto-posted items for human verification, continuously.
That gives a live, unbiased precision estimate with no ground truth — and it is
the honest answer to the question an auditor will certainly ask. Pair it with a
**circuit breaker**: if sampled precision drops below the bar, stop auto-posting
and route everything to humans until someone looks.

### 6.5 Duplicate payment detection

Same counterparty, same amount, near dates, no matching second invoice. Real
money — organisations pay suppliers twice more often than they would like to
admit, and reconciliation is exactly where it surfaces. This feature can pay for
the entire project.

### 6.6 Anomaly and fraud flags

Reconciliation is a natural fraud detection point. Worth flagging: a counterparty
never seen before, an amount just below an approval threshold, a round-number
payment to a new account, out-of-hours activity, a reference that mimics a known
one. Not accusations — flags for a human.

### 6.7 What-if / shadow mode

Run a proposed config or model change against historical closes and report what
would have changed: which items would newly auto-post, which would stop, which
past decisions would have flipped. Nothing that touches the books should ever be
deployed without this. It also makes the "is this change safe?" conversation
quantitative.

### 6.8 Batch review by pattern

Cluster the review queue by shape rather than presenting it as a flat list.
"These 40 items are all the same counterparty, all short by exactly £15 — accept
all?" One decision instead of forty. This is where reviewer hours actually go, and
it is a bigger practical win than a few points of match rate.

### 6.9 Queue ordering by value × uncertainty

Order the human queue so the most consequential and least certain items come
first. If the close runs out of time — it always does — the work that got done
should be the work that mattered.

### 6.10 Run continuously, not monthly

Nightly rather than at month-end. Suspense never accumulates into a wall, timing
differences resolve within days, and the close becomes a report rather than a
scramble.

### 6.11 Period lock and correct-forward

Once a period closes, its journal is immutable. Later corrections post to the
current period with a reference to the original. This is standard accounting
practice and it constrains the data model, so it is worth designing in early
rather than discovering later.

### 6.12 Segregation of duties

The system proposes; a named human approves; both identities are recorded against
the entry. Auditors require this for anything material, and it is much easier to
build in from the start than to retrofit into a journal that never had an actor
column.

---

## 7. Guardrails specific to LLM-plus-money

**Reference fields are attacker-writable.** Whoever sends a payment types its
reference. That text reaches a prompt in a system that can post money. This is a
live prompt-injection surface, not a theoretical one. Three mitigations, all
structural rather than prompt-based: the model may only select from a
pre-computed candidate list **by index**; it can never emit an identifier of its
own; and its output schema admits no free-form action, only a verdict. Treat
reference text as data in the dossier, clearly delimited, and never as
instruction.

**Nothing posts on the model's say-so.** Arithmetic is re-verified, confidence is
recalibrated, and the decision stage owns the final call.

**Reproducibility beats re-derivation.** Store the verdict and the model version.
Never reconstruct a past close by asking the model again.

**Cost ceiling per run**, with the run degrading to "everything unresolved goes to
suspense" rather than silently truncating.

**A deterministic fallback path.** If the LLM layer is unavailable, the system
still reconciles — worse, but correctly, with a smaller auto-post rate and a
larger queue. Never a hard dependency.

---

## 8. What to build first

The pass loop and suspense are the structural pieces; the LLM is comparatively
easy to add once the structure holds it correctly.

1. **Mutual exclusivity + the pass loop** (passes 1–2). No model, no new
   algorithms, resolves an unknown but real slice of the ambiguous pool for free.
2. **Fix the collision rule** (PIPELINE.md §1.1). Days of work, ~18 points.
3. **Suspense subsystem with case files and re-attempt.** Delivers the "it's
   honest about what it doesn't know" promise, which is the actual product.
4. **Group assembly** (pass 3). Attacks the 49% of the ambiguous pool that is
   really a grouping problem.
5. **LLM adjudicator** (pass 4) on what survives, with calibration and the three
   gates.
6. **Close report.** Makes all of the above legible to the person who signs.
7. **Counterparty memory, override learning, sampling assurance.**

The ordering is deliberate: every step before the LLM makes the LLM's job smaller,
cheaper, and more accurate.
