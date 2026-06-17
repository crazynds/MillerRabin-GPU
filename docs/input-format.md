# Input Format

The driver reads a plain text file where each non-blank, non-comment line
describes one integer candidate to test for primality.

---

## Line syntax

```
[group_id:] equation
```

| Part       | Required | Description                                                                                    |
| ---------- | -------- | ---------------------------------------------------------------------------------------------- |
| `group_id` | No       | Any string that does not contain `:`. Used to link related candidates (see [Groups](#groups)). |
| `equation` | Yes      | An arithmetic expression that evaluates to the candidate integer.                              |

Lines that start with `#` and blank lines are ignored.

---

## Equation grammar

All arithmetic is performed in arbitrary precision using GMP (`mpz_t`) — there
is no overflow at any step, including intermediate values. A single number
literal of any size is read directly by
`mpz_set_str` and handled exactly.

```
expr    = term   (('+' | '-') term)*
term    = factor (('*' | '/' | '%') factor)*
power   = unary  ('^' power)?          # right-associative
unary   = ('+' | '-') unary | primary
primary = NUMBER | '(' expr ')'
NUMBER  = one or more decimal digits   # arbitrarily large
```

### Operators

| Operator | Meaning                                   | Notes                                                                                     |
| -------- | ----------------------------------------- | ----------------------------------------------------------------------------------------- |
| `+`      | Addition                                  |                                                                                           |
| `-`      | Subtraction / unary negation              |                                                                                           |
| `*`      | Multiplication                            |                                                                                           |
| `/`      | Integer division (truncated toward zero)  |                                                                                           |
| `%`      | Integer remainder (truncated toward zero) |                                                                                           |
| `^`      | Exponentiation                            | Right-associative; exponent must be ≥ 0 and fit in `unsigned long` (~4.3 × 10⁹ on 64-bit) |

### Examples

```
# A Mersenne candidate
2^74207281 - 1

# A sparse decimal form
10^18001 - 25*10^1334 - 91*10^249 - 1

# A direct large literal (GMP handles it exactly)
123456789012345678901234567890...

# Composed expression
(10^5000 - 1) / 9

# Factorial-like construction
2^8192 - 2^4096 + 2^2048 - 1
```

---

## Groups

### What is a group?

A group is a set of related equations that must **all be prime** for the result
to be meaningful. All equations in a group share the same `group_id` prefix.

```
1: 10^18001 - 25*10^1334 - 91*10^249 - 1
1: 10^18001 - 52*10^16665 - 19*10^17750 - 1
```

Here group `1` has two equations.

### Why groups?

The original use-case for this project was hunting for primes of a special
decimal form `N = 10^a - k·10^b - j·10^c - 1` where, for the candidate to be
interesting, **both** N and its digit-reversed companion `revN` must be prime.
Testing `revN` when N is already composite is a waste of GPU time.

Groups generalize this to any number of related candidates: define as many
equations as you like under one group ID, and the driver skips the remaining
ones the moment any fails.

### How testing works

The driver processes candidates **round by round**:

- **Round 1** — every group's first equation is batched and tested on the GPU simultaneously.
- **Round 2** — only the groups that survived round 1 proceed; their second equation is batched.
- **…and so on** until all rounds are exhausted or all groups are eliminated.

```
Round 1:  [group 1 eq 1]  [group 2 eq 1]  [group 3 eq 1]  ...
              ↓ passes         ↓ FAILS          ↓ passes
Round 2:  [group 1 eq 2]  (group 2 gone)   [group 3 eq 2]  ...
```

This is maximally efficient: the GPU always receives the largest possible batch
for each round, and groups that fail early never occupy slots in later rounds.

### Singleton groups

A line without a `:` is treated as a standalone group with one equation. It
participates in round 1 only and is reported as a winner if it passes.

---

## Large number literals

You can place an arbitrarily large decimal number directly on a line:

```
# 2048-digit literal — parsed exactly by GMP, no truncation
12345678901234567890....(2048 digits)....
```

The parser accumulates the digit characters into a `std::string` and hands it
to `mpz_set_str(out, str, 10)`. GMP reads the entire string in one call — the
size is limited only by available RAM.

The same applies to sub-expressions: `(10^50000 - 1) / 7` first computes
`10^50000 - 1` as a ~50 000-digit GMP integer, then divides — all exact.

---

## File format notes

- Encoding: UTF-8 or ASCII. No BOM required.
- Line endings: `LF` or `CRLF` (both stripped).
- The `group_id` is the text **before the first `:`** on the line. Group IDs are
  matched by exact string equality — `1` and `01` are different groups.
- A line that contains `:` but has nothing before it (i.e., starts with `:`) is
  treated as a group with an empty-string ID.
