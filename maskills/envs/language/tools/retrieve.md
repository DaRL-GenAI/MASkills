# Tool: retrieve

`<retrieve>` delegates a self-contained sub-task to the **retrieval
sub-agent**, which has access to the full task context (including all
passages) and runs in an **isolated context** — it does not see your
prior reasoning, only the original task plus your specific query.  The
sub-agent's response comes back to you as a `<retrieve_result>` block.

This tool is only available in the centralized orchestration setting,
where you (the main agent) act as the reasoner and the sub-agent acts
as the retriever.

## Invocation

```
<retrieve>QUERY</retrieve>
```

* `QUERY` is a short natural-language hint to the sub-agent — e.g. the
  bridge entity you want grounded, the attribute you need, or the
  passage-level facts to surface.  An empty payload is allowed (the
  sub-agent will run on the original task without extra guidance), but
  a focused query usually returns better evidence.
* Each call runs the sub-agent end to end and returns its full
  response as a `<retrieve_result>` block.

## When to use retrieve

* You do not have the context passages — the original task you see
  shows only the question.  The sub-agent has the passages and is
  trained to filter them.
* You need a specific bridge value or attribute and would rather get
  cited evidence than guess from prior knowledge.
* You want a second pass after a previous `<retrieve_result>` was
  insufficient (e.g. the wrong bridge, missing the attribute year);
  re-issue with a refined query.

## When NOT to use retrieve

* The question is fully answered by the question text itself
  (definitional, arithmetic, simple facts you can verify).
* You already received `<retrieve_result>` evidence that resolves both
  hops — re-querying just to confirm wastes your budget and dilutes
  context.

## Reading the result

The sub-agent's response follows the `retrieval-passage-filter` (or
`retrieval-passage-filter-grep`) protocol:

```
Bridge entity: <phrase>
Relevant passages: [Passage X], [Passage Y]
Evidence:
- [Passage X] "<quote>"
- [Passage Y] "<quote>"
Notes: ...
```

Treat the verbatim quotes as your ground truth.  If the bridge looks
wrong or a hop is missing, issue another `<retrieve>` with a tighter
query rather than guessing.

## Hard rules

* Do **not** put your final answer inside `<retrieve>...</retrieve>` —
  the tag is read by the sub-agent, not by the grader.
* Do **not** re-issue the same query after a usable evidence pack is
  returned.
* Do **not** wrap the query in code fences or extra markup; the
  payload is the literal text between the tags.
