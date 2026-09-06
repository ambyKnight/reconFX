# Knowing When You're Sure

An assistant for the monthly job of checking a company's bank statement
against its own books.

---

## In one line

Software can already guess which payments match which records. It just can't
tell you which guesses to trust — so a person re-checks all of it anyway.
We built the part that knows when it's sure.

---

## The job

Every company keeps two records of its own money.

The **bank** says what actually happened. Money came in, money went out, here
are the dates and amounts.

The **books** say what was supposed to happen. This customer was going to pay
us. We were going to pay that supplier.

These two should agree. They never quite do. So every month, before the company
can close its accounts, somebody sits down and matches them up — line by line,
by hand, until every pound is accounted for.

It is the financial equivalent of matching a shoebox of receipts to your bank
statement. Except there are tens of thousands of them, and there is a deadline.

## Why it's harder than it sounds

If every payment arrived with a note saying what it was for, this would be a
solved problem. They don't.

- A customer sends one payment covering **six** invoices.
- Two different customers pay the **exact same amount on the same day**.
- A payment arrives a few days **late**, or on a different date than the book
  expects.
- The bank takes a fee, so the amount that lands is slightly **less** than the
  amount owed.
- Some money arrives that **doesn't match anything at all** — and the right
  answer is "we don't know yet."

So it's not a lookup. It's a judgement call, thousands of times a month.

## What happens when nobody can do it fast enough

Anything unmatched gets parked in a holding account and waits. Until it's
resolved, the company doesn't actually know its own cash position, and the
month can't be closed.

We can put a number on how bad this is. We're working with real, anonymised
records from a large bank — around 32,000 bank lines and 37,000 book entries
over three months. In that bank's own data, **about a third of all matches were
made by hand, by a person.**

That's not an estimate from a sales brochure. It's what the bank's records say
they actually did.

## Why existing software hasn't fixed it

It's not that the software can't find matches. It usually can.

The problem is it can't tell you **which of its answers to believe**. It hands
over a pile of suggestions with no reliable sense of which ones are solid and
which ones are coincidence. Two customers paid £4,200 on the same Tuesday —
the software picks one, sounds equally confident either way, and is wrong half
the time.

So the accountant checks every single one. Which is the job they had before.

The best published attempt on our dataset finds the right answer roughly two
thirds of the time. But once you insist it only auto-approve matches it's
genuinely sure about, it can safely handle **0.5%** of the work. Almost all of
its value evaporates, not because it can't match, but because it can't tell
when it's right.

## What we're building

A system that sorts every line into one of three buckets, and is honest about
which is which:

**1. "I'm sure."** Post it. No human involved.

**2. "I'm not sure."** Send it to a person — with the shortlist of likely
answers and a plain-English reason for each, so the decision takes seconds
rather than minutes.

**3. "Nothing here matches."** Say so, out loud, instead of quietly forcing a
wrong answer. This turns out to matter more than we expected.

Everything it decides is written down: what it chose, why, how sure it was, and
when. Any of it can be undone. An accountant signing off on the month can see
exactly how every number got there.

## Where it stands

On the real bank data, it currently handles **62.7% of the work on its own** —
versus 0.5% for the best published attempt on the same data, held to the same
standard of care.

Of the roughly 20,000 lines it approved by itself, **12 were genuine mistakes**.

The remaining ~37% goes to a person, which is the point. That's the part that
actually needs a human.

## Why this is the right way round

The instinct is to build something that matches everything and hope it's right.

That's backwards. In finance, a confident wrong answer is worse than no answer,
because it goes into the books and someone has to find it and reverse it later.
A system that does 60% of the work and is honest about the other 40% is worth
far more than one that claims 100% and quietly gets some of it wrong.

So the goal was never "match more." It was **be trustworthy about what you
match, and clear about what you don't.**

That's the whole idea.
