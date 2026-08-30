# Tool: grep

A line-oriented filter (Python `re.search`, case-insensitive) for
extracting just the relevant sentences from a long search result.  It
runs locally — no network — and is much cheaper than reissuing a search.

## Underlying behaviour

```
run_grep(payload, fallback_text=last_search_result, max_lines=20)
```

* The pattern is a Python regular expression.  Plain literal substrings
  also work (e.g. `Oscar`, `1923`).
* Matching is case-insensitive.
* Up to 20 matching lines are returned, each prefixed with its
  1-based line number.
* If no line matches, the result is `No lines match /<pattern>/.`

## Invocation protocol

Two forms are accepted; both use the `<grep>...</grep>` tag.  The first
form is the one you will use most often.

**Form 1 — filter the most recent search result.**

```
<grep>PATTERN</grep>
```

The haystack is the most recent `<search>` result for this agent.  If
the agent has not run any search yet, the call returns an error and the
agent should issue a search first.

**Form 2 — filter ad-hoc text.**

```
<grep>PATTERN ||| TEXT</grep>
```

The separator `|||` (three pipes) splits the payload into pattern and
arbitrary text.  Useful when the relevant text is in the question
itself or in another agent's pool message.

## When to use grep

* **Long search returns.**  If a search hit is hundreds of words and
  you only need the year, the office, or the spouse's name, grep for
  the keyword instead of asking the LLM to re-read the whole extract.
* **Multiple bridge entities.**  After resolving "director of
  *Inception*", grep for `Nolan` in subsequent extracts to confirm the
  bridge entity is mentioned.
* **Disambiguation.**  When a search returns several pages with the
  same name, grep for a discriminator (year, occupation) to pick the
  right one.

## When NOT to use grep

* On *empty* or *No results.* search outputs — re-search instead.
* For numeric reasoning or arithmetic — the line is not the answer.
* As a substitute for a real search — grep can only see what was
  already retrieved.

## Examples

After `<search>Christopher Nolan filmography</search>`:

```
<grep>Inception</grep>
```

→ returns only the lines that mention `Inception` from the search hit.

Filtering the question itself:

```
<grep>year ||| What year did the director of Inception win his first Oscar?</grep>
```

→ returns the line containing `year`, useful when you want to confirm
the question's target attribute before searching.
