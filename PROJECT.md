# Knowing When You're Sure

An assistant for the monthly job of getting two subsidiaries of the same
company to agree on what they owe each other.

---

## In one line

Software can already guess which intercompany entries match which. It just
can't tell you which guesses to trust — so a person re-checks all of it anyway.
We built the part that knows when it's sure.

---

## The job

A company with more than one legal entity — a UK parent and a US subsidiary, say
— is constantly trading with itself. The UK entity recharges the US entity for
shared engineering headcount. The US entity resells stock it bought from the UK
entity at a markup. One books a loan to the other to fund working capital.

Each side records its own half of every transaction, in its own ledger, often
in its own currency, sometimes on its own close calendar. **Both halves should
net to zero when the group consolidates.** They never quite do. So every month,
before the group can close its accounts, somebody sits down and reconciles every
intercompany pair — entity by entity, currency by currency — until every
mismatch is explained.

This is not a smaller version of bank reconciliation. It's the same shape of
problem — two records of the same money that should agree and don't — played
between two departments of the same company instead of a company and its bank.
That should make it easier. In practice it's its own mess, for its own reasons.

## Why it's harder than it sounds

If every intercompany entry booked itself identically on both sides, this would
be a solved problem. They don't.

- The UK entity books in GBP, the US entity books the same transaction in USD —
  and the **FX rate each side used to translate it** is rarely identical, so the
  two halves land a few pounds apart for no real reason at all.
- One entity's month closes on the 28th, the other's on the 31st — so a
  transaction booked by one hasn't been booked by the other yet, and looks like
  a mismatch that will resolve itself in three days if nobody touches it.
- A recharge carries a **transfer-pricing markup** applied on one side and not
  the other, so the two amounts are supposed to differ, by a specific, known
  percentage — and a naive matcher flags every single one as broken.
- Several small cross-charges get **netted into one settlement payment**, so
  one side records a lump sum against fifteen entries the other side kept
  separate.
- Some entries have **no counterpart at all** — booked on one side and simply
  never raised on the other — and the right answer is "one entity owes the
  other an entry, go raise it," not "keep looking."

So it's not a lookup. It's a judgement call, applied to related-party
transactions that end up in a regulatory filing, thousands of times a month.

## What happens when nobody can do it fast enough

Anything that doesn't tie gets parked as an open intercompany item and waits.
Until it's resolved, the group's consolidated numbers carry an unexplained
imbalance, and the close can't finish. At period end, unreconciled intercompany
balances are exactly the kind of thing an external auditor pulls a sample of and
asks pointed questions about — this isn't a back-office nicety, it sits directly
under the numbers that go in the filing.

## Why existing software hasn't fixed it

It's not that the software can't propose matches. Most reconciliation platforms
can pair up two amounts that are close enough.

The problem is the same one bank reconciliation has: it can't tell you **which
of its pairings to believe**. A £4,200 recharge and a $5,304 receipt look like a
match or a coincidence depending on that day's FX rate, a markup you may or may
not know about, and whether the US side has even closed the period yet. Software
that always sounds equally confident either way forces the accountant to
re-verify every single pairing by hand — which is the job they had before.

## What we're building

A system that sorts every intercompany entry into one of three buckets, and is
honest about which is which:

**1. "I'm sure."** Post it, or clear it, with no human involved.

**2. "I'm not sure."** Send it to a person — with the shortlist of likely
counterpart entries, a plain-English reason for each (including the FX rate,
markup, or timing gap it's weighing), so the decision takes seconds rather than
minutes.

**3. "One side owes the other an entry."** Say so, out loud — and say which
entity, roughly how much, and why it looks that way — instead of quietly
forcing a match that doesn't hold up under audit.

Everything it decides is written down: what it chose, why, how sure it was, and
when. Any of it can be undone. A controller signing off on the group's
consolidated close can see exactly how every number got there.

## Where the evidence for this comes from — and where it doesn't

There is no public, labelled intercompany-reconciliation dataset. Intercompany
data reveals a group's internal transfer pricing and consolidation structure, so
no company publishes it and no benchmarking consortium anonymises it the way
banks have started to for cash reconciliation. We looked; nothing usable exists
yet, and building or licensing a real one is future work, not something we
should fake our way around.

What we do have, and what the quantified numbers elsewhere in this project
(match rate, precision, the error autopsy, the calibration argument) actually
measure, is
[**BenchRec**](https://www.kaggle.com/datasets/benchmarkteam/benchrec-real-world-cash-reconciliation-dataset) —
real, anonymised bank-vs-ledger cash reconciliation data from a Tier 1 bank,
released at ACM AI in Finance 2023: about 32,000 bank lines and 37,000 ledger
entries over three months.

That is a different domain from intercompany reconciliation, and we say so every
time we cite a number from it. But it is the same *underlying problem*: two
records of the same money, matched under ambiguity, where a system has to know
when to trust its own answer. The things we learned there — that collapsing
candidates before judging ambiguity matters, that a confidence score is worthless
until it's calibrated against real outcomes, that "no match" has to be a first-
class answer, that most "errors" turn out to be label noise rather than real
mistakes — are properties of *matching under uncertainty*, not properties of
bank statements specifically. That is the argument for why the evidence still
counts, and it is also the limit of what we're claiming: we have evidence that
the *method* works on one two-sided matching problem, not a measurement of it on
intercompany data.

See [PIPELINE.md](PIPELINE.md) for the numbers themselves, and
[ARCHITECTURE.md](ARCHITECTURE.md) for the system designed for intercompany
specifically.

## Where it stands

On the cash-reconciliation evidence base, the method currently handles **62.7%
of the work on its own** at 99.8% precision — versus 0.5% for the best published
attempt on the same data, held to the same standard of care. Of the roughly
20,000 matches it approved by itself, 12 were genuine mistakes; nearly all the
rest of the 36 "errors" turned out to be dataset label noise, not our mistakes
(see PIPELINE.md §1.2).

We do not yet have an equivalent number for intercompany data, because we do not
yet have intercompany data. That gap is stated plainly rather than papered over.

## Why this is the right way round

The instinct is to build something that matches everything and hope it's right.

That's backwards, and it's worse for intercompany than for bank reconciliation:
a confident wrong match here doesn't just need reversing, it can misstate a
number that ends up disclosed to regulators and auditors. A system that resolves
60% of the work and is honest about the other 40% is worth far more than one
that claims 100% and quietly gets some of it wrong.

So the goal was never "match more." It was **be trustworthy about what you
match, and clear about what you don't** — and be honest, in the same breath,
about exactly what evidence that claim currently rests on.

That's the whole idea.
