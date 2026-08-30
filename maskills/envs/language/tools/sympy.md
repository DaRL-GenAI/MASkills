# Tool: sympy

A Python sandbox with the `sympy` library pre-imported.  Use it to
evaluate symbolic expressions, solve equations, simplify, factor, and
generally avoid hand-arithmetic mistakes on MATH-style problems.

## Underlying behaviour

```
run_sympy(code) -> str
```

* A single expression is `eval`'d and `str(value)` is returned.
* A multi-line snippet is `exec`'d with stdout captured; you must
  `print(...)` whatever you want back.
* The namespace is the full `sympy` namespace (so `Rational`, `solve`,
  `simplify`, `Symbol`, `Eq`, `pi`, `oo`, `Matrix`, ... are all in
  scope).  Python builtins are restricted to a small safe set.
* Errors are returned as `Sympy error: <Type>: <msg>` — the agent
  should read the message and retry with a fixed expression.
* Results are truncated at ~1500 characters.

## Invocation protocol

```
<sympy>EXPRESSION OR SNIPPET</sympy>
```

Only the first `<sympy>` tag in a response is executed.  Other tags in
the same response are ignored.  Per-turn budget is shared with the
other tools (`search`, `grep`).

## When to use sympy

* **Algebra.**  `solve(Eq(2*a + 3, -3), a)` instead of mental algebra.
* **Continuity / piecewise problems.**  Pose the equality at the
  boundary as `Eq(...)` and `solve` for the unknowns.
* **Combinatorics.**  `binomial(10, 3)`, `factorial(7)`.
* **Number theory.**  `gcd(...)`, `divisors(...)`, `isprime(...)`.
* **Simplification of expressions before reading the answer**, e.g.
  `simplify((x**2 - 1)/(x - 1))`.
* **Series / limits / derivatives / integrals** for calculus problems.

## When NOT to use sympy

* For pure word counting or bookkeeping — easier in Python without the
  sympy round-trip.
* When the problem has a one-line closed-form answer that is faster to
  reason about directly (don't burn a tool call to compute `1 + 1`).
* When a previous sympy call already returned the exact value — reuse
  it instead of recomputing.

## Examples

**Solving for `a + b`** in a piecewise-continuity problem:

```
<sympy>
from sympy import symbols, solve, Eq
a, b = symbols('a b')
sol = solve([Eq(2*a + 3, 2 - 5), Eq(-2 - 5, 2*(-2) - b)], [a, b])
print(sol[a] + sol[b])
</sympy>
```

→ returns `0`.

**Simplifying a final expression** before submitting:

```
<sympy>simplify((x**3 - 1)/(x - 1))</sympy>
```

→ returns `x**2 + x + 1`.

**Counting divisors:**

```
<sympy>len(divisors(360))</sympy>
```

→ returns `24`.
