"""Per-category LOCOMO skill-evolution run.

For one of the four LOCOMO question categories (1 multi-hop, 2 temporal,
3 open-domain, 4 single-hop), a decentralized retriever + reasoner pair is
trained through the MASkills evolution loop.

Two things are specific to this benchmark and live here rather than in the
generic trainer.

**Sharpened operator prompts.** The library's default induce / refine /
consolidate prompts ask for an "abstract goal, when it applies, expected
coordination behavior", which the optimizer takes literally and answers with
600-1000 word abstractions carrying no tool syntax, no grader-format rules and
no worked examples. The replacements below demand the missing pieces and ban
invented LOCOMO data.

**Per-category task hints.** The failure-analysis bullets for the category
being trained are injected into the optimizer's prompt, so the coach knows the
high-leverage targets from the first iteration. The agent never sees them; it
only sees what the optimizer writes into its skills.

The grep tool is enabled from the start while the skills begin empty: the agent
*can* call it but does not know how, which is what the first inductions have to
teach it from the tool library the optimizer is shown.

Nothing here is applied at import. :func:`install_runtime_patches` and
:func:`install_ablation` do the patching, and :func:`run` calls them.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# ── temperature 0, for per-task evals that reproduce ──────────────────────
import maskills.llm.client as _llm_client_mod  # noqa: E402


def _chat_messages_temp0(self, messages, max_tokens=None):
    max_tokens = max_tokens or self.config.max_tokens
    params = {"model": self.model, "messages": messages, "temperature": 0.0}
    if any(k in self.model.lower() for k in ("o1", "o3", "gpt-5")):
        params["max_completion_tokens"] = max_tokens
        params.pop("temperature", None)
    else:
        params["max_tokens"] = max_tokens
    try:
        resp = _llm_client_mod._create_with_retry(
            self._client.chat.completions.create, **params,
        )
    except TypeError as e:
        # OpenAI/OpenRouter content_filter / refusal: returns finish_reason=
        # 'content_filter' and content=None.  After retries exhausted, the
        # client raises TypeError("LLM response has no usable content ...").
        # Treat this as an empty-output task instead of killing the whole
        # iteration — one filtered task should not crash 200 sibling rollouts.
        if "no usable content" in str(e) or "content_filter" in str(e):
            return "", {"input": 0, "output": 0}
        raise
    text = (resp.choices[0].message.content or "").strip()
    if hasattr(resp, "usage") and resp.usage:
        tokens = {"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens}
    else:
        joined = " ".join(m.get("content", "") for m in messages)
        tokens = {"input": len(joined.split()) * 2, "output": len(text.split()) * 2}
    return text, tokens



import maskills.core.optimizer as _opt_mod  # noqa: E402

_orig_block = _opt_mod.PolicyGradientOptimizer._tool_library_block


def _patched_block(self) -> str:
    blk = _orig_block(self)
    hints = getattr(self, "task_hints", "").strip()
    if hints:
        blk += (
            "\n## Coaching Hints for THIS Task (high-leverage guidance from\n"
            "## prior failure analysis — visible to you, the coach, not the\n"
            "## agent.  Use these when writing skills so the agent's library\n"
            "## bakes the patterns in.)\n"
            + hints + "\n"
        )
    return blk



# ── replace optimizer prompts: tool-use, evidence-first, concrete examples ─
# The original library prompts ask for "abstract goal / when applies / expected
# coordination behavior" — which the optimizer LLM (esp. gpt-5.1) takes literally
# and produces 600–1000-word abstract skill bodies with no tool syntax, no
# grader-format rules, no worked examples.  Replace with prompts that DEMAND
# the missing pieces.
import maskills.core.skill_credit as _sc_mod  # noqa: E402
import maskills.core.skill_operators as _so_mod  # noqa: E402

_NEW_INDUCE_PROMPT = """\
You are inducing ONE new skill for an LLM agent based on the residual
failures below.  The skill body you produce will be injected verbatim into
the agent's system prompt at every rollout — the agent reads it and acts.

## Existing skills (do NOT duplicate these)
{existing}

## Hard-case evidence (residual failures with no responsible skill)
{evidence}

## SPECIFICITY CONTRACT — every rule in the new skill body must satisfy:

> "Could a generic LLM coach have written this rule WITHOUT ever reading
>  the hard cases or the task hints above?"
> If yes, the rule is REJECTED.  Generic guidance is what the failing
> agent already knows; what it needs are concrete patterns it can pattern-
> match on at runtime.

### Requirement 1 — every rule references a concrete pattern

Every sentence in the body MUST mention at least one of:
- a specific question form (verbatim or near-verbatim from a hard case or
  the task hints),
- a specific cue inside a turn (an exact string the agent will see, like
  ``last week``, ``ten years ago``, ``for 4 years``),
- a specific gold-answer shape (an exact form the grader will compare
  against, like ``The week before 9 June 2023``, ``Likely no``, ``Yes``),
- the exact tool tag syntax (``<grep>PATTERN</grep>``) and a concrete
  pattern that matched in a hard case.

BANNED filler phrases (any appearance = rejected skill):
``in general``, ``typically``, ``as needed``, ``when applicable``,
``ensure that``, ``be sure to``, ``draws on background knowledge``,
``operates under uncertainty``, ``best-effort``, ``communicate uncertainty``,
``transparently``, ``where relevant``, ``if appropriate``.

### Requirement 2 — NO INVENTED EXAMPLES (most critical)

Every worked example (``Q / Evidence / Answer`` triple, ``Cue → Output``
mapping, tool-call demo) in the body MUST use strings copied verbatim or
near-verbatim from one of these two sources only:
  (a) the ``## Hard-case evidence`` block above,
  (b) the task-specific Coaching Hints in the prompt.

Inventing plausible-looking examples is FORBIDDEN.  In particular,
NEVER write fake LOCOMO data like:
  - ``[D0:0]`` (real dia ids look like ``[D7:3]``),
  - ``Session: 2023-05-14 – Discussion about Jolene's move``
    (real session headers look like ``--- session_3 (8 May 2023) ---``),
  - ``Current date: 2026-05-24``  (LOCOMO has no such field),
  - ``Q: On what date did this conversation take place?``
    (no real LOCOMO QA looks like this).

If neither the hard cases nor the task hints contain a pattern you would
have liked to illustrate, OMIT the example.  Stick to a verbatim rule.

### Requirement 3 — ROLE-AWARE tool rules

Identify the skill's role from (a) the existing-skills list, (b) the
hard-case agent labels (``agent_1``/``retriever`` vs ``agent_2``/
``reasoner``), and (c) the description you are about to write:

- **Retriever** (sees the full conversation + has a tool, hands evidence
  to a downstream reasoner):
  MUST include the exact tool tag syntax (e.g. ``<grep>PATTERN</grep>``)
  plus at least one worked invocation with a real pattern from the hard
  cases.
- **Reasoner** (NO tool, sees only the retriever's evidence excerpts and
  emits the FINAL graded answer):
  MUST NOT include ``<grep>`` tags or any tool-call syntax.  A reasoner
  skill that contains ``<grep>`` is a failed skill and is REJECTED.
  Instead, include ``Given retriever evidence of form X → emit answer of
  form Y`` rules.

### Requirement 4 — mandatory "Do NOT emit" list

The body MUST end with a ``## Do NOT emit`` section enumerating concrete
strings/patterns the agent must NEVER output, drawn from observed
failures and from the task hints' banned-output list (e.g. for cat 1-4:
``not mentioned``; for cat 2 ``2023-10-14``-style ISO dates fabricated
from thin air; for cat 3 ``since …`` / ``because …`` clauses; for a
retriever the ``Answer:`` prefix line).

### Requirement 4b — REFUSAL LOOPHOLE BAN (reasoner skills only)

If you are inducing a REASONER skill (no tools) and the failure cases
show the agent over-refusing (``not mentioned`` answers), DO NOT teach
the agent a *different* way to refuse.  Specifically the new skill is
REJECTED if its body contains, recommends, or treats as acceptable ANY
of these refusal alternatives:
- ``cannot be determined`` / ``cannot be determined from the provided excerpts``
- ``no information`` / ``insufficient evidence`` / ``insufficient information``
- ``unknown`` / ``unclear`` / ``not specified``
- ``based on the provided evidence ...`` (preamble that signals refusal)
- ``please refer to ...`` or ``consult the source ...``
- Any sentence of the form ``the X cannot be ...`` / ``no Y is mentioned``
- Any meta-language about ``minimal context`` / ``sparse evidence`` /
  ``weak retriever evidence`` (this teaches the agent to TALK about
  uncertainty instead of committing to an answer).

A reasoner skill that tries to dress up refusal as ``a more elegant
non-answer'' is the failure mode this requirement is fighting.  The
correct teaching is: pick a concrete answer from the evidence
(or from a fallback like the session date), period.  Add the above
refusal phrases to the ``## Do NOT emit`` section explicitly.

### Requirement 5 — banned section headers and template language

These section headers and any of their variants are BANNED — replace
with concrete rules + examples:
- ``Abstract goal``
- ``When this skill applies``
- ``Expected coordination behavior``
- ``How this skill interacts with teammates``
- ``General principles``

### Requirement 6 — keep it tight

Total body ≤ 350 words.  Every word should give the agent a rule, an
example token, or a banned output.

## Output
Output ONLY a skill in Anthropic SKILL.md format (``---`` frontmatter
with ``name:`` and ``description:``, then a markdown body that satisfies
all 6 requirements).  Output ``NONE`` if no new skill is warranted."""


_NEW_REFINE_PROMPT = """\
You are refining ONE skill in an agent's library via small anchor-based
edits.  Do NOT rewrite from scratch.

## Current skill
name: {name}
description: {description}
body:
{body}

## Aggregated improvement direction
{summary}

Suggested edit: {suggested_edit}

## EDIT REQUIREMENTS — each edit must INCREASE specificity

The same Specificity Contract from skill induction applies (see Req 1-5):
- Every rule must reference a concrete question form, cue, answer shape,
  or tool tag — generic filler ("typically", "ensure that", "communicate
  uncertainty", etc.) is REJECTED.
- All worked examples must use strings copied verbatim or near-verbatim
  from the failure case above OR the task hints; NEVER invent fake
  LOCOMO data (no ``[D0:0]``, no ``Current date: ...``, no ``Session:
  2023-05-14`` headers, no made-up questions).
- ROLE-AWARE: if this is a reasoner skill (no tool), DELETE any
  ``<grep>`` tag content; if this is a retriever skill, ENSURE the body
  has at least one ``<grep>PATTERN</grep>`` example with a real
  pattern from the failure case.

Prefer these edit shapes:
- INSERT a verbatim ``Q / Evidence / Answer`` triple from the failure
  case after a relevant anchor.
- REPLACE an abstract paragraph with a concrete rule keyed on a real
  question form or cue.
- INSERT/UPDATE the trailing ``## Do NOT emit`` list with the observed
  bad output from the failure case.
- DELETE any section whose header is ``Abstract goal`` / ``When this
  skill applies`` / ``Expected coordination behavior``.

Every edit must make the skill MORE specific, MORE evidence-grounded,
or MORE role-appropriate — never more abstract, never adding made-up
data.

## Output
Output ONLY a JSON list of edit instructions, fenced in ```json. Each item:
  {{"op": "replace"|"insert_after"|"insert_before"|"delete",
    "anchor": "<verbatim unique substring of the body to locate the edit>",
    "old": "<exact snippet to replace/delete; null for inserts>",
    "new": "<replacement or inserted text; empty string for delete>"}}
An empty list [] means no change is warranted."""


_NEW_CONSOLIDATE_PROMPT = """\
The skills below are functionally overlapping.  Merge them into ONE
consolidated skill that subsumes every capability without redundancy.

## Skills to merge
{skills}

## REQUIREMENTS for the consolidated body

- **KEEP every concrete asset** from the input skills: every verbatim
  ``Q / Evidence / Answer`` example, every cue → output mapping row,
  every exact tool-call (``<grep>PATTERN</grep>``) example, every
  ``Do NOT emit`` entry.  The whole point of consolidation is to
  preserve concreteness while removing redundancy.
- **DROP** abstract template headers (``Abstract goal``, ``When this
  skill applies``, ``Expected coordination behavior``) and the BANNED
  filler phrases listed in skill induction.
- **NO INVENTED EXAMPLES** — if a worked example from an input skill
  used fake LOCOMO data (``[D0:0]``, fabricated session headers, made-up
  questions), DROP it.  Only keep examples whose strings appear in real
  LOCOMO data (the task hints + the hard-case evidence the inputs were
  trained on).
- **ROLE-AWARE** — if all inputs are reasoner skills, the consolidated
  body MUST NOT contain ``<grep>`` tags; if all inputs are retriever
  skills, MUST contain the ``<grep>`` tag syntax + at least one real
  worked invocation.
- The body MUST end with a ``## Do NOT emit`` section union'ing the
  banned outputs of the inputs.
- Body ≤ 400 words.

## Output
Output ONLY the consolidated skill in Anthropic SKILL.md format (``---``
frontmatter with ``name:`` and ``description:``, then the markdown body)."""


# Credit-side push toward specific suggested_edits — vague edits flow into
# refine/induce and produce vague skills.
_NEW_SKILL_CREDIT_PROMPT = _sc_mod._SKILL_CREDIT_PROMPT.replace(
    'Use these EXACT strings (verbatim, case-sensitive) as the "agent" field',
    """## How to write the JSON fields (READ CAREFULLY)

### ``evidence`` field
Quote or near-verbatim paraphrase the ACTUAL turns from the trajectory
above — give a specific span (the agent's actual output, the actual
grep result, the actual gold answer).  Never write generic claims like
"agent retrieved too much" or "answer was wrong".

### ``suggested_edit`` field — must be DROP-IN READY

``suggested_edit`` will be fed verbatim into the next skill refinement
step.  Write it as a SHORT MARKDOWN SNIPPET (≤ 80 words) ready to drop
into the skill body — NOT abstract advice.

GOOD examples of ``suggested_edit`` (drop-in ready text):
- ``Add rule: When the cited turn says ``last week`` and the session is
  ``--- session_3 (9 June 2023) ---``, the reasoner outputs ``The week
  before 9 June 2023`` — NEVER compute a calendar date.``
- ``Add to ## Do NOT emit:  ``2023-10-14`` or any ISO-style date the
  agent fabricated from thin air; cat-2 dates must echo the speaker's
  format (e.g. ``25 May 2023``).``
- ``Insert a worked Q/E/A triple after the ## Examples header:
    Q: When did Melanie run a charity race?
    Session: --- session_2 (25 May 2023) --- ; cue in turn: "last Saturday"
    Evidence: [D2:1] (25 May 2023) Melanie said: "I ran a charity race ... last Saturday"
    Answer: The sunday before 25 May 2023``
- ``Delete the section "Expected coordination behavior" and replace with
  the rule: retriever output MUST start each excerpt with
  ``[D<s>:<t>] (SESSION_DATE)`` — no excerpt without the date.``

BAD examples of ``suggested_edit`` (REJECTED — they produce abstract skills):
- ``reason more carefully``
- ``communicate uncertainty transparently``
- ``draw on background knowledge``
- ``handle the case where evidence is sparse``
- ``be sure to use tools when appropriate``
- ``output 'cannot be determined' when evidence is insufficient`` — REFUSAL LOOPHOLE; the grader scores this as zero just like ``not mentioned``.  The correct edit when the agent over-refuses is to add a CONCRETE FALLBACK rule (e.g. "if cue is `last week`, output `The week before <SESSION_DATE>`"), NOT to suggest a different refusal phrase.
- ``teach the agent to acknowledge sparse evidence`` — same loophole.
- Any edit that adds ``based on the provided evidence`` or ``cannot``-style preambles.

### Other fields
- ``contribution`` is one of: helped, redundant, harmful, neutral
- ``utility_delta`` is a float in [-1.0, 1.0] — negative if the skill
  encouraged the exact failure observed; positive if it caused a good
  output.

## Agent identifier convention
Use these EXACT strings (verbatim, case-sensitive) as the "agent" field""",
)


# ── imports that depend on the patched modules ───────────────────────────
import maskills.trainer.skill_evolution as _se_mod  # noqa: E402
from maskills.core.policy import AgentPolicy  # noqa: E402
from maskills.core.skills import SkillLibrary  # noqa: E402
from maskills.experiments.locomo import LocomoExperiment  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────
# Per-cat task hints — LOCOMO-specific format contracts + real example tuples.
#
# Layered so the optimizer can drop the example tuples DIRECTLY into the
# induced skill body verbatim (satisfying Requirement 2 of _NEW_INDUCE_PROMPT).
# ─────────────────────────────────────────────────────────────────────────

# ── Grep tag syntax — retriever (agent_1) only ───────────────────────────
_GREP_TAG_SYNTAX_FOR_RETRIEVER = """\
### Grep tool — exact tag syntax (retriever / agent_1 ONLY)

```
<grep>PATTERN</grep>
```

- ``PATTERN`` is a Python case-insensitive regex matched line-by-line
  against the transcript.
- Env replies with ``<grep_result>...</grep_result>`` containing up to
  20 matching turn lines like ``[D7:1] Nate said, "I won my first ..."``;
  or ``No lines match /<PATTERN>/.`` when nothing hits.
- ONE strong keyword per call (named entity, noun, topic).  Up to 3
  calls per rollout.
- BANNED grep patterns (observed to fail or distract):
  - compound alternation: ``<grep>writing|career|art|counseling</grep>``
  - ``A.*B`` style regexes (turn wording varies; almost always 0 hits)
  - ``<grep>not mentioned</grep>``, ``<grep>date</grep>``, ``<grep>2020|2021|2022|2023</grep>``
- If the first grep returns 0 hits, retry with (a) a different speaker's
  name, (b) a synonym, (c) the rarest single noun from the question.
- After greps, write evidence excerpts.  The reasoner (agent_2) is the
  ONLY consumer of your output — agent_2 does NOT see the conversation
  and CANNOT call grep.

### Retriever output contract (every cat-1-4 retriever skill MUST follow)

```
Relevant excerpts:
- [D<s>:<t>] (SESSION_DATE) SPEAKER said: "<verbatim turn text>"
- [D<s>:<t>] (SESSION_DATE) SPEAKER said: "<verbatim turn text>"
```

- ``(SESSION_DATE)`` is MANDATORY on every excerpt line — read it off
  the ``--- session_<s> (DATE) ---`` header (grep returns ONLY turn
  lines, never the header; the retriever must copy the date manually).
  Strip the time-of-day: ``--- session_2 (1:14 pm on 25 May, 2023) ---``
  → ``25 May 2023``.
- Preserve every time cue inside the quoted turn verbatim: ``yesterday``,
  ``last week``, ``last Saturday``, ``last year``, ``X years ago``,
  ``for X years``, ``since YYYY``, ``last summer``.
- ``NO_EVIDENCE`` is FORBIDDEN for cat 1-4 (it cascades into the
  reasoner saying ``not mentioned`` which scores 0).  If 3 greps all
  miss, scan the transcript yourself and return any tangentially-related
  turn with its session date.
- Do NOT answer the question yourself.  Do NOT append ``Answer:`` lines.
  The reasoner does the inference."""


# ─────────────────────────────────────────────────────────────────────────
# CAT 1 multi-hop
# ─────────────────────────────────────────────────────────────────────────

RETRIEVER_HINTS = {}
REASONER_HINTS = {}

RETRIEVER_HINTS[1] = """\
agent_1 RETRIEVER for **category 1 (multi-hop)** — the answer chains 2+
facts across different turns/sessions.

### Concrete grep strategy
Spend ONE grep per hop (up to 3 greps):
1. Most specific named entity in the question (a person, a unique object).
2. The abstract noun the question targets (``hobby``, ``interest``,
   ``career``, ``screenplay``, ``hike``, ``allergy``).
3. If question contains ``and`` (e.g. *"interests Joanna and Nate share"*),
   grep each party separately and intersect the topics.

### Real LOCOMO cat-1 hard cases (lift these into your skill body)

```
Q: What are Joanna's hobbies?
Gold: Writing, watchingmovies, exploringnature, hanging withfriends.
Grep plan:  <grep>Joanna</grep>  then  <grep>hobby</grep>
Evidence:
- [D1:10] (21 Jan 2022) Joanna said: "Writing is my passion ..."
- [D2:25] (23 Jan 2022) Joanna said: "I love watching movies and exploring nature with my friends ..."
```

```
Q: What is Joanna allergic to?
Gold: Most reptiles,animals with fur,cockroaches, dairy
Grep plan:  <grep>allergic</grep>  then  <grep>allergy</grep>
Evidence:
- [D4:4] (4 Feb 2022) Joanna said: "I'm allergic to most reptiles ..."
- [D5:11] (...) Joanna said: "... cockroaches ..."
- [D2:23] (...) Joanna said: "... dairy ..."
```

```
Q: How many times has Joanna found new hiking trails?
Gold: twice
Grep plan:  <grep>hiking trail</grep>  →  2 hits  →  Answer = twice
```"""

REASONER_HINTS[1] = """\
agent_2 REASONER for **category 1 (multi-hop)** — gold is a comma-separated
list of short noun-phrases (sometimes a single phrase).

### Output contract — exact grader format
- Plural-form question (``hobbies``, ``allergies``, ``interests``,
  ``what kinds of …``, ``how many``) → ``item1, item2, item3`` —
  comma-separated, NO ``and``, drop articles, drop filler (``etc.``,
  ``and more``, ``as well``).
- Singular-form question (``what career``, ``what name``, ``where did``)
  → ONE short noun-phrase, drop articles.
- Counting question (``how many times``) → a short word: ``twice``,
  ``three``, ``four``.

### Real LOCOMO cat-1 (Q → Gold) pairs (lift verbatim into your skill)

```
Q: What kind of interests do Joanna and Nate share?
Gold: Watching movies, making desserts
```

```
Q: What are Joanna's hobbies?
Gold: Writing, watchingmovies, exploringnature, hanging withfriends.
```

```
Q: What is Joanna allergic to?
Gold: Most reptiles,animals with fur,cockroaches, dairy
```

```
Q: What emotions is Joanna feeling about the screenplay she submitted?
Gold: Relief, excitement,worry, hope,anxiety.
```

```
Q: How many times has Joanna found new hiking trails?
Gold: twice
```

### Banned output for cat 1
- ``not mentioned`` — REJECTED.
- ``Based on the excerpts ...`` / ``The answer is ...`` preambles — REJECTED.
- Trailing period.  Trailing ``and``.  ``D2:3`` style citations in the answer.
- Quoting items: ``"running", "painting"`` — REJECTED."""


# ─────────────────────────────────────────────────────────────────────────
# CAT 2 temporal
# ─────────────────────────────────────────────────────────────────────────

RETRIEVER_HINTS[2] = """\
agent_1 RETRIEVER for **category 2 (temporal)** — answer is a date /
year / duration / relative phrase.  The reasoner CANNOT see the
conversation; it depends entirely on what you write.

### Concrete grep strategy
1. The rarest event noun/verb (``charity race``, ``screenplay``,
   ``audition``, ``tournament``, ``camping``, ``birthday``, ``adopted``).
2. If 0 hits: try a synonym or shorter root.
3. NEVER grep time words alone (``date``, ``2022``, ``year``) — they
   match every session.

### Real LOCOMO cat-2 hard cases (lift into your skill body verbatim)

```
Q: When did Joanna finish her first screenplay?
Session: --- session_2 (2:01 pm on 23 January, 2022) ---  →  use ``23 January, 2022``
Cited: [D2:3] Joanna said: "Woo! I finally finished my first full screenplay and printed it last Friday."
Cue: ``last Friday``
Gold: The Friday before 23January, 2022
```

```
Q: When did Nate get purple hair?
Session: --- session_7 (15 April, 2022) ---
Cited: [D7:1] Nate said: "Dyed my hair last week - come see!"
Cue: ``last week``
Gold: The week before 15April, 2022.
```

```
Q: How long has Nate had his first two turtles?
Cited: [D2:12] Nate said: "I've had them for 3 years now and they bring me tons of joy!"
Cue: ``for 3 years``
Gold: three years
```

```
Q: When did Nate get his first two turtles?
Session: --- session_2 (23 January, 2022) ---
Cited: [D2:12] Nate said: "I've had them for 3 years now ..."
Cue: ``for 3 years`` (subtract from session year 2022)
Gold: 2019
```

```
Q: Where did Joanna travel to in July 2022?  ← this is a "Where", NOT "When"!
Cited: [D17:4] Joanna said: "I went to Woodhaven ..."
Gold: Woodhaven
```

### Mandatory retriever output for cat 2 — the reasoner needs all three:
1. ``[D<s>:<t>]`` id with the verbatim turn quote.
2. ``(SESSION_DATE)`` in parentheses on EVERY excerpt line.
3. The time cue (``last week``, ``ten years ago``, ``for 3 years``,
   ``since 2020``, etc.) preserved verbatim inside the quote.

Missing ANY of these tanks cat 2 — the reasoner WILL fabricate an
ISO-style date like ``2023-10-14`` if you omit the session date."""

REASONER_HINTS[2] = """\
agent_2 REASONER for **category 2 (temporal)** — answer shape depends
on the question word, NOT on the cue alone.

### STEP 1 — question-form decision (overrides Step 2)
| Question opens with                   | Answer shape                                            |
|---------------------------------------|---------------------------------------------------------|
| ``Who …``                             | a person/relation: ``her mother``, ``Max``, ``Anna``   |
| ``What movie / book / song / event …``| the noun-phrase: ``Lord of the Rings``, ``Valorant``    |
| ``What did X do/attend/accomplish …`` | activity noun-phrase: ``finished her screenplay``, ``a tech-for-good convention``, ``dyed his hair purple`` |
| ``Where …``                           | place name: ``Woodhaven``, ``UK``, ``France``           |
| ``Which X did Y win/visit …``         | named entity: ``Valorant``, ``Whispering Falls waterfall`` |
| ``How long has X / How long did it take`` | duration: ``three years``, ``four months`` (drop "for") |
| ``How long ago …``                    | echo ``X years ago`` / ``X months ago`` verbatim       |
| ``How many <unit>s passed …``         | ``<n> <unit>s`` (compute from dates if needed)         |
| ``Which week did …``                  | ``The week before <SESSION_DATE>`` (relative phrase)    |
| ``Which month was X in …``            | ``<Month> <YEAR>`` from cited turn (``December 2023``)  |
| ``When did / When is / When was …``   | → Step 2                                                |

### STEP 2 — "When" cue → output mapping (LOCOMO conventions)

| Cue in cited turn                                | Output                                                |
|--------------------------------------------------|-------------------------------------------------------|
| explicit date (``on 7 May 2023``, ``in March 2022``, ``back in 2020``, ``on February 24, 2023``) | echo speaker's exact format (preserve commas/word order) |
| explicit year only (``back in 2022``)            | ``2022``                                              |
| ``yesterday`` / ``last night``                   | ``The day before <SESSION_DATE>``                     |
| ``the other day`` / ``a few days ago``           | ``A few days before <SESSION_DATE>``                  |
| ``last week`` / ``earlier this week``            | ``The week before <SESSION_DATE>``                    |
| ``last sunday`` / ``last monday`` / ... / ``last friday`` | ``The <day> before <SESSION_DATE>`` (literal phrase) |
| ``next sunday`` / ``next week`` / ...            | ``The <day/week> after <SESSION_DATE>``               |
| ``last month``                                   | ``The month before <SESSION_DATE>``                   |
| ``next month``                                   | ``The month after <SESSION_DATE>``                    |
| ``last year``                                    | ``<SESSION_YEAR − 1>`` (bare year)                    |
| ``last summer / winter / spring / fall``         | ``<season> of <SESSION_YEAR − 1>``                    |
| ``X years/months/weeks/days ago``                | echo verbatim (``ten years ago``); NEVER convert      |
| ``for X years/months``                           | ``X years`` / ``X months`` (drop "for")                |
| ``since YYYY``                                   | ``Since YYYY``                                        |
| ``two weekends ago``                             | ``two weekends before <SESSION_DATE>``                |

### Real LOCOMO cat-2 (Cue → Gold) pairs (lift verbatim)

```
Cue: "last Friday" in session_2 (23 January, 2022)  →  Gold: The Friday before 23January, 2022
Cue: "last week" in session_7 (15 April, 2022)      →  Gold: The week before 15April, 2022.
Cue: "for 3 years" in session_2 (23 January, 2022)  →  Gold: three years
Cue: "for 3 years" + question "When did X happen"   →  Gold: 2019  (session_year − 3)
Cue: "yesterday" in session_6 (24 March, 2022)      →  Gold: 23 March, 2022.
Cue: "last summer" in session_4 (4 February, 2023)  →  Gold: in summer 2022
Cue: "in 2010" (explicit year)                       →  Gold: in 2010
```

### Banned output for cat 2
- ``not mentioned`` — REJECTED (fall back to session year before refusing).
- ISO-style dates fabricated from thin air: ``2023-10-14``, ``2023-11-01``,
  ``2025-...`` — REJECTED.  The agent must NOT invent dates not in the
  cited turn or session header.
- ``[D2:1]`` citations in the answer string — REJECTED.
- ``The day before 25 August 2023`` when speaker said ``yesterday`` and
  the gold uses the literal computed date — apply the cue table strictly,
  not a paraphrase."""


# ─────────────────────────────────────────────────────────────────────────
# CAT 3 open-domain reasoning
# ─────────────────────────────────────────────────────────────────────────

RETRIEVER_HINTS[3] = """\
agent_1 RETRIEVER for **category 3 (open-domain reasoning)** —
inference questions (``would/could/likely/how old/what would X do``).
Gather 3–6 turns that bear on EITHER side of the inference; the
reasoner picks a verdict from your evidence.

### Concrete grep strategy
- 3 greps on DIFFERENT facets of the topic (the activity name, the
  speaker's values/history, the alternative).
- Example: question *"Would Caroline pursue writing as a career?"*
  → ``<grep>writing</grep>`` ··· ``<grep>career</grep>`` ···
    ``<grep>counselor</grep>`` (the alternative she already chose).
- BANNED: compound alternation ``<grep>writing|career|art|counseling</grep>``
  (observed to return 0 hits in cat-3 trajectories).

### Real LOCOMO cat-3 hard cases (lift verbatim)

```
Q: What pets wouldn't cause any discomfort to Joanna?
Gold: Hairless cats or pigs,since they don't have fur, which is one of the main causes of Joanna's allergy.
Grep plan: <grep>allergic</grep>  <grep>fur</grep>  <grep>pets</grep>
Evidence:
- [D2:23] Joanna said: "... allergic to ... animals with fur ..."
- [D5:11] Joanna said: "... also some food and cockroaches ..."
```

```
Q: What underlying condition might Joanna have based on her allergies?
Gold: asthma
Grep plan: <grep>asthma</grep>  <grep>breathing</grep>
```

```
Q: What alternative career might Nate consider after gaming?
Gold: an animalkeeper at a local zoo and working with turtles; ...
Grep plan: <grep>turtles</grep>  <grep>career</grep>  <grep>zoo</grep>
```

### What to NEVER do
- ``NO_EVIDENCE``: BANNED for cat 3.  Even tangential turns help the
  reasoner pick a side.
- Returning ``Answer: ...`` lines — that's the reasoner's job."""

REASONER_HINTS[3] = """\
agent_2 REASONER for **category 3 (open-domain reasoning)** — gold is a
SHORT reasoned claim (2–6 tokens).

### CRITICAL grader rule
The LOCOMO grader takes ONLY the FIRST ``;``-segment of the gold answer
for cat 3:

```
gold:   "Likely no; though she likes reading, she wants to be a counselor"
graded against: "Likely no"
```

Therefore: any ``since/because/though …`` clause you append ONLY ADDS
tokens that the gold lacks → lowers precision → tanks F1.  **Output
ONLY the short claim — DROP the justification.**

### Output shape decision tree
- Yes/No inference (``Would …``, ``Is …``, ``Did they …``, ``Could …``)
  → ``Yes`` / ``No`` / ``Likely yes`` / ``Likely no`` (hedge when evidence
  is mixed).
- ``How old / how long`` question → a short phrase
  (``Likely no more than 30``, ``four years``).
- ``What fields / kinds of / things …`` → tiny comma-separated noun-phrase
  list (``Psychology, counseling``).
- ``What would they likely become / do`` → noun-phrase predicting the
  role/action (``become a basketball coach``).
- ``What pets … wouldn't / which alternative …`` → noun-phrase only.

### Real LOCOMO cat-3 (Q → Gold first-segment) pairs (lift verbatim)

```
Q: Is it likely that Nate has friends besides Joanna?
Full gold: Yes teammates on his video game team.
Graded gold: Yes
Correct answer to emit: Yes
```

```
Q: Would Caroline pursue writing as a career option?
Full gold: LIkely no; though she likes reading, she wants to be a counselor
Graded gold: LIkely no
Correct answer to emit: Likely no
```

```
Q: What pets wouldn't cause any discomfort to Joanna?
Full gold: Hairless cats or pigs,since they don't have fur, ...
Graded gold: Hairless cats or pigs
Correct answer to emit: Hairless cats or pigs
```

```
Q: What underlying condition might Joanna have?
Gold: asthma
Correct answer to emit: asthma
```

```
Q: How many hikes has Joanna been on?
Gold: Four
Correct answer to emit: Four
```

### Banned output for cat 3
- ``not mentioned`` — REJECTED (commit to the best-supported short claim).
- Any verbose answer ≥ 8 tokens — REJECTED.
- Any sentence containing ``since``, ``because``, ``though``, ``given that``,
  ``based on the evidence``, ``it is likely that`` — strip these clauses.
- ``Caroline is unlikely to pursue writing as a career option since …``
  — exemplifies what NOT to emit; the correct emit is ``Likely no``."""


# ─────────────────────────────────────────────────────────────────────────
# CAT 4 single-hop
# ─────────────────────────────────────────────────────────────────────────

RETRIEVER_HINTS[4] = """\
agent_1 RETRIEVER for **category 4 (single-hop)** — the answer lives in
ONE specific turn.

### Concrete grep strategy
- ONE grep on the rarest noun/verb in the question.  If 0 hits, retry
  with a synonym or speaker name.  Up to 2 calls total — more dilutes.
- Use bigrams when single keywords are generic (``<grep>charity race</grep>``
  beats ``<grep>charity</grep>``).

### Real LOCOMO cat-4 hard cases (lift verbatim)

```
Q: What is one of Joanna's favorite movies?
Grep: <grep>Eternal Sunshine</grep>  OR  <grep>favorite movie</grep>
Cited: [D1:18] Joanna said: "I first watched 'Eternal Sunshine of the Spotless Mind' around 3 years ago. I even went out and got a physical copy!"
Gold: "Eternal Sunshineof the Spotless Mind"
```

```
Q: What color did Nate choose for his hair?
Grep: <grep>hair</grep>
Cited: [D7:1] Nate said: "Dyed my hair last week ..."  and  [D7:3] "purple ..."
Gold: purple
```

```
Q: What is Nate's favorite video game?
Grep: <grep>Xenoblade</grep>  OR  <grep>favorite</grep>
Gold: Xenoblade Chronicles
```

```
Q: What kind of lighting does Nate's gaming room have?
Grep: <grep>lighting</grep>
Cited: [D10:2] Nate said: "My gaming room has red and purple lighting ..."
Gold: red and purple lighting
```

### Retriever output for cat 4 — 1–2 excerpts ONLY (more = noise)"""

REASONER_HINTS[4] = """\
agent_2 REASONER for **category 4 (single-hop)** — gold is the SHORTEST
faithful noun-phrase from the cited turn.

### Output shape rules
- 1–6 tokens.  Drop articles (``the`` / ``a`` / ``an``).  Drop trailing
  punctuation.  Drop surrounding sentence structure ("Nate's gaming room
  has ...").
- Lift the phrase that DIRECTLY answers the question (usually right
  after the speaker's verb).
- One value.  Never ``X or Y`` / ``X, Y`` for cat 4 (that's cat 1).
- Preserve speaker formatting if the gold has quotes (e.g.
  ``"Eternal Sunshineof the Spotless Mind"``).

### Real LOCOMO cat-4 (Q → Gold) pairs (lift verbatim)

```
Q: What is one of Joanna's favorite movies?
Gold: "Eternal Sunshineof the Spotless Mind"
```

```
Q: What color did Nate choose for his hair?
Gold: purple
```

```
Q: What is Nate's favorite movie trilogy?
Gold: Lord of the Rings
```

```
Q: What is Nate's favorite video game?
Gold: Xenoblade Chronicles
```

```
Q: What kind of lighting does Nate's gaming room have?
Gold: red and purple lighting
```

```
Q: What is Joanna's third screenplay about?
Gold: loss, identity, and connection
```

```
Q: What game was the second tournament that Nate won based on?
Gold: Street Fighter
```

### Banned output for cat 4
- ``not mentioned`` — REJECTED.
- ``Based on the cited turn, the answer is …`` preambles — REJECTED.
- Pulling 2 sentences when 1 suffices (e.g. for "What did Melanie realize",
  emit ``self-care is important``, NOT ``taking care of our minds``).
- ``D7:1`` style citations or quotes around the answer."""


def task_hints_for(category: int) -> str:
    """Compose retriever + reasoner hints + tool/format contracts for one cat.

    Layout:
      1. Architecture + role boundaries (who is who, who sees what)
      2. The RETRIEVER section (with grep syntax, output contract, real
         cat-specific examples, banned outputs) — used by induce/refine
         when writing skills for agent_1.
      3. The REASONER section (with the cat-specific output shape, real
         examples, banned outputs) — used by induce/refine when writing
         skills for agent_2.
      4. A meta note that reasoner skills MUST NOT include <grep>.
    """
    parts = [
        "## LOCOMO task — category " + str(category),
        "",
        "Architecture: decentralized 2-agent team.",
        "- ``agent_1`` (RETRIEVER) sees the full conversation, has the "
        "``<grep>`` tool, hands evidence excerpts to ``agent_2``.",
        "- ``agent_2`` (REASONER) sees ONLY the question + the retriever's "
        "excerpts.  ``agent_2`` has NO tools.  Its text IS the FINAL answer "
        "scored by the LOCOMO grader (token-level F1 + BLEU).",
        "",
        "### Skill-writing rule based on agent role",
        "- A skill whose ``name`` / ``description`` mentions ``retriever`` "
        "/ ``agent_1`` / ``grep`` / ``evidence extraction`` is a RETRIEVER "
        "skill.  It MUST include the ``<grep>PATTERN</grep>`` tag syntax "
        "and at least one worked invocation.",
        "- A skill whose ``name`` / ``description`` mentions ``reasoner`` "
        "/ ``agent_2`` / ``final answer`` / ``answer synthesis`` is a "
        "REASONER skill.  It MUST NOT include ``<grep>`` tags or any "
        "tool-call syntax (agent_2 has no tools).  Tags in the final "
        "answer corrupt the graded string.",
        "",
        "─────────────────────────────────────────────────────────────────",
        "## RETRIEVER (agent_1) hints — for cat " + str(category),
        "─────────────────────────────────────────────────────────────────",
        "",
        _GREP_TAG_SYNTAX_FOR_RETRIEVER,
        "",
        RETRIEVER_HINTS[category],
        "",
        "─────────────────────────────────────────────────────────────────",
        "## REASONER (agent_2) hints — for cat " + str(category),
        "─────────────────────────────────────────────────────────────────",
        "",
        REASONER_HINTS[category],
        "",
        "─────────────────────────────────────────────────────────────────",
        "## Skill library bookkeeping",
        "─────────────────────────────────────────────────────────────────",
        "- Empty starting library.  Your first induct should establish the "
        "role-specific output contract above.",
        "- Keep each skill body ≤ 300 words; cumulative library ≤ 3-4 "
        "skills per agent.",
        "- When proposing a skill for a specific agent, lift Q/E/A example "
        "tuples DIRECTLY from the ``Real LOCOMO cat-N hard cases`` blocks "
        "above — do not invent fake LOCOMO data.",
    ]
    return "\n".join(parts)



# ── runtime patches ──────────────────────────────────────────────────────

def _dedup_immediate_repeats(text: str) -> str:
    """Collapse immediately-repeated substrings of at least 8 characters.

    Catches the LLM-edit artifact where an ``insert_after`` lands content that
    already follows the anchor, producing ``**Header**1. \\n**Header**``.
    """
    import re

    pattern = re.compile(r"(.{8,400}?)(\s*\d*\.?\s*)\1", re.DOTALL)
    previous, out = None, text
    for _ in range(5):
        if out == previous:
            break
        previous = out
        out = pattern.sub(lambda m: m.group(1) + m.group(2), out)
    return out


_PATCHED = False


def install_runtime_patches() -> None:
    """Swap in this benchmark's prompts and output cleanups.

    Idempotent, and never run at import: importing this module must not change
    how the rest of the package behaves.
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    _llm_client_mod.LLMClient.chat_messages_with_usage = _chat_messages_temp0
    _opt_mod.PolicyGradientOptimizer._tool_library_block = _patched_block
    _so_mod._INDUCE_PROMPT = _NEW_INDUCE_PROMPT
    _so_mod._REFINE_PROMPT = _NEW_REFINE_PROMPT
    _so_mod._CONSOLIDATE_PROMPT = _NEW_CONSOLIDATE_PROMPT
    _sc_mod._SKILL_CREDIT_PROMPT = _NEW_SKILL_CREDIT_PROMPT

    # Post-edit dedup, on both the anchor edits and the bodies that induction
    # and consolidation produce, before either enters the library.
    original_apply_edits = _opt_mod.PolicyGradientOptimizer._apply_edits

    def apply_edits(text, instructions):
        return _dedup_immediate_repeats(original_apply_edits(text, instructions))

    _opt_mod.PolicyGradientOptimizer._apply_edits = staticmethod(apply_edits)

    def dedup_body(fn):
        def wrapped(self, *a, **kw):
            skill = fn(self, *a, **kw)
            if skill is not None and skill.body:
                skill.body = _dedup_immediate_repeats(skill.body)
            return skill
        return wrapped

    for name in ("induce_skill", "consolidate_skills", "refine_skill"):
        setattr(_so_mod.SkillEvolutionOptimizer, name,
                dedup_body(getattr(_so_mod.SkillEvolutionOptimizer, name)))


def install_ablation(name: str) -> None:
    """Disable one mechanism, leaving the rest of the loop intact."""
    if name == "rollback":
        # Keep computing validation metrics for comparison, but commit the
        # candidate library every iteration instead of gating on them.
        def always_accept(self, base_policies, candidate_policies):
            per_agent = {a: {"changed": True, "accepted": True}
                         for a in candidate_policies}
            return dict(candidate_policies), {"status": "ablated", "per_agent": per_agent}

        _se_mod.SkillEvolutionTrainer._validation_gate = always_accept
        print("[ablation:rollback] validation gate replaced with always-accept "
              "(candidate library committed every iteration)", flush=True)

    elif name == "credit":
        # Every invoked skill gets the same neutral credit derived from the
        # trajectory reward, so the operators see no per-skill differentiation
        # and no LLM-suggested edits.
        def degenerate_credit(self, trajectory, policies):
            trace = getattr(trajectory, "skill_trace", {}) or {}
            reward = float(getattr(trajectory, "reward", 0.0) or 0.0)
            utility = max(-1.0, min(1.0, 2.0 * reward - 1.0))
            summary = f"ablation-credit: traj.reward={reward:.3f}"

            skill_credits, residuals, seen = [], {}, set()
            for agent, entries in trace.items():
                for entry in entries:
                    for skill_id in entry.get("skills", []):
                        if (agent, skill_id) in seen:
                            continue
                        seen.add((agent, skill_id))
                        skill_credits.append(_sc_mod.SkillCredit(
                            agent=agent, skill_id=skill_id,
                            contribution="neutral", evidence=summary,
                            suggested_edit="", redundant_with=[],
                            conflict_with=[], utility_delta=utility,
                        ))
                residuals[agent] = _sc_mod.ResidualCredit(
                    agent=agent, summary=summary,
                    needs_new_skill=(reward <= 0.5), missing_capability="")
            # An agent with no trace still needs a residual.
            for agent in policies:
                residuals.setdefault(agent, _sc_mod.ResidualCredit(
                    agent=agent, summary=summary,
                    needs_new_skill=(reward <= 0.5), missing_capability=""))
            return _sc_mod.TrajectorySkillCredit(
                skill_credits=skill_credits, residuals=residuals,
                reward=reward, raw_response="(ablation: credit disabled)")

        _sc_mod.SkillCreditCritic.evaluate_skill_credit = degenerate_credit
        print("[ablation:credit] per-skill LLM credit replaced with a neutral "
              "utility_delta of 2r-1", flush=True)

    elif name == "momentum":
        # The trainer normally keeps an EMA, skill.utility = 0.6*old + 0.4*mean.
        # Overwrite with the batch mean instead.
        def no_momentum(self, policies, trajectories, credits):
            invocations: dict = {}
            for trajectory in trajectories:
                trace = getattr(trajectory, "skill_trace", {}) or {}
                for agent, entries in trace.items():
                    for entry in entries:
                        for skill_id in entry.get("skills", []):
                            counts = invocations.setdefault(agent, {})
                            counts[skill_id] = counts.get(skill_id, 0) + 1

            deltas: dict = {}
            for credit in credits:
                for c in credit.skill_credits:
                    deltas.setdefault(c.agent, {}).setdefault(
                        c.skill_id, []).append(c.utility_delta)

            for agent, policy in policies.items():
                for skill in policy.skill_library:
                    skill.invocations += invocations.get(agent, {}).get(skill.skill_id, 0)
                    values = deltas.get(agent, {}).get(skill.skill_id, [])
                    if values:
                        skill.utility = round(sum(values) / len(values), 4)

        _se_mod.SkillEvolutionTrainer._update_skill_stats = no_momentum
        print("[ablation:momentum] skill utility EMA disabled "
              "(utility := batch mean)", flush=True)


# ── the run ──────────────────────────────────────────────────────────────

CATEGORY_NAMES = {1: "multihop", 2: "temporal", 3: "opendomain", 4: "singlehop"}
_CATEGORY_LABELS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


def _cell(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value) if value is not None else "-"


def run(args, project_root: Path) -> list:
    """Train one category's retriever + reasoner pair; return the metrics."""
    install_runtime_patches()

    category = args.category
    category_name = CATEGORY_NAMES[category]
    suffix = "" if args.ablation == "none" else f"_abl-{args.ablation}"
    exp_name = f"locomo_percat_cat{category}_{category_name}{suffix}"
    ckpt_dir = project_root / "experiments" / f"ckpt_{exp_name}"

    config_path = project_root / "configs" / "locomo" / "qa_locomo_decentralized.json"
    experiment = LocomoExperiment.from_json(str(config_path), overrides={
        "benchmark_path": args.benchmark_path,
        "experiment_dir": str(project_root / "experiments"),
        "checkpoint_dir": str(ckpt_dir),
        "exp_name": exp_name,
        "architecture": args.topology,
        "category_filter": [category],
        "num_iterations": args.iters,
        "trajectories_per_iteration": args.train_per_iter,
        "max_workers": args.workers,
        "optimizer_workers": args.optimizer_workers,
        "eval_test_every_iter": True,
        "inject_skill_library": False,  # the discrete library is in the policies
        "locomo_reward_metric": "f1_bleu_mean",
        "answer_brevity_hint": False,
        # Separate models: actor rolls out, optimizer critiques and edits.
        "llm": args.actor_model,
        "actor_llm": args.actor_model,
        "optimizer_llm": args.optimizer_model,
        "retriever_sees_conversation": args.retriever_sees_conversation,
        # max_grep_calls comes from the config (3): the tool is available from
        # the start, and the skills have to teach the agent how to use it.
        # ── skill-evolution knobs ──
        "skill_evolution": True,
        "skill_eval_delta": args.gate_tolerance,
        "refine_every": 1,
        "induct_every": 1,  # an empty start needs skills before it can refine
        "consolidate_every": 0 if args.ablation == "consolprune" else 2,
        "prune_every": 0 if args.ablation == "consolprune" else 3,
        "max_skills_per_agent": args.max_skills,
        "evolve_role": args.evolve_role,
        "hard_trajectory_threshold": 0.5,
    })

    experiment.optimizer.task_hints = task_hints_for(category)

    if args.resume_run_id:
        from maskills.store.checkpoint import PolicyCheckpoint
        from maskills.store.run_logger import RunLogger
        from maskills.store.trajectory_store import TrajectoryStore

        run_id = args.resume_run_id
        experiment.trainer.run_id = run_id
        experiment.trainer.checkpoint = PolicyCheckpoint(
            experiment.trainer.store, run_id, experiment.config.num_agents,
            task_type=getattr(experiment.config, "task_type", None))
        experiment.trainer.trajectory_store = TrajectoryStore(
            experiment.trainer.store, run_id)
        experiment.trainer.run_logger = RunLogger(experiment.trainer.store, run_id)
        print(f"[resume] reusing run_id={run_id}; training picks up from its "
              "latest checkpoint", flush=True)

    assignment = experiment.setup_conversation_split(
        n_train=2, n_val=2, n_test=6, seed=42)
    loader = experiment.env.task_loader
    if args.n_val > 0:
        experiment.config.n_val = min(args.n_val, len(loader.val_tasks))
    if args.n_test > 0:
        experiment.config.n_test = min(args.n_test, len(loader.test_tasks))

    if args.ablation != "none":
        install_ablation(args.ablation)

    # ── seed policies ──
    seeded = {"agent_1": 0, "agent_2": 0}
    policies = {}
    for agent in ("agent_1", "agent_2"):
        library = SkillLibrary(skills=[])
        if args.init_skills and args.init_skills != "empty":
            seed_dir = (Path(args.init_skills) / agent
                        / f"cat{category}_{category_name}")
            if seed_dir.is_dir():
                library = SkillLibrary.from_skill_md_dir(seed_dir)
                seeded[agent] = len(library.skills)
        # The role stays blank so the env substitutes its retriever / reasoner
        # prompts; writing one in here would suppress them.
        policies[agent] = AgentPolicy(role="", skill_library=library)

    note = (f"warm-start (a1={seeded['agent_1']} a2={seeded['agent_2']} "
            f"from {args.init_skills})" if any(seeded.values())
            else "empty start (no skills)")
    experiment.trainer.checkpoint.save_policies(0, policies, stats={"note": note})

    print("=" * 80)
    print(f"LOCOMO PER-CATEGORY EVOLUTION — cat {category} ({category_name})")
    print(f"  topology = {args.topology}   ablation = {args.ablation}")
    print(f"  exp_name = {exp_name}")
    print(f"  conversation split (seed 42):  train={len(loader.train_tasks)}  "
          f"val={len(loader.val_tasks)}  test={len(loader.test_tasks)} QA")
    print(f"  iters={args.iters}  train/iter={args.train_per_iter}  "
          f"workers={args.workers}  opt-workers={args.optimizer_workers}")
    print(f"  init skills: {note}")
    print(f"  actor_llm     = {experiment.config.get_actor_llm().model_string}")
    print(f"  optimizer_llm = {experiment.config.get_optimizer_llm().model_string}")
    print(f"  retriever_sees_conversation = {experiment.env.retriever_sees_conversation}"
          "  (False = grep-only, ~75% fewer actor input tokens)")
    print(f"  task hints injected into the optimizer "
          f"({len(experiment.optimizer.task_hints)} chars)")
    print("=" * 80, flush=True)

    started = time.time()
    metrics = experiment.run()
    elapsed = time.time() - started

    # ── per-iteration report ──
    print("\n" + "=" * 80)
    print(f"PER-ITERATION METRICS — cat {category} ({category_name})  "
          f"elapsed={elapsed:.0f}s")
    print("=" * 80)
    seen = set()
    for m in metrics:
        iteration = m.get("iteration", m.get("type", "?"))
        for split in ("val", "test"):
            if not any(k.startswith(f"{split}_") for k in m):
                continue
            if (iteration, split) in seen:
                continue
            seen.add((iteration, split))
            overall = (f"F1={_cell(m, f'{split}_avg_f1')}  "
                       f"BLEU={_cell(m, f'{split}_avg_bleu')}")
            per_cat = (f"c{category}({_CATEGORY_LABELS[category][:4]}):"
                       f"F1={_cell(m, f'{split}_cat{category}_avg_f1')}/"
                       f"BLEU={_cell(m, f'{split}_cat{category}_avg_bleu')}")
            print(f"[iter {iteration} {split}]  overall {overall}  |  {per_cat}")
            if any(f"skill_{k}" in m for k in ("refined", "induced")):
                print(f"             operators: "
                      f"refine={_cell(m, 'skill_refined')} "
                      f"induct={_cell(m, 'skill_induced')} "
                      f"consol={_cell(m, 'skill_consolidated')} "
                      f"prune={_cell(m, 'skill_pruned')} "
                      f"rollback={_cell(m, 'skill_rollbacks')}")

    out_path = ckpt_dir / "metrics_summary.json"
    os.makedirs(out_path.parent, exist_ok=True)
    out_path.write_text(json.dumps({
        "category": category,
        "category_name": category_name,
        "topology": args.topology,
        "ablation": args.ablation,
        "split_assignment": assignment,
        "iters": args.iters,
        "train_per_iter": args.train_per_iter,
        "actor_model": args.actor_model,
        "optimizer_model": args.optimizer_model,
        "n_val": experiment.config.n_val,
        "n_test": experiment.config.n_test,
        "elapsed_sec": elapsed,
        "metrics": metrics,
    }, indent=2, default=str))
    print(f"\nSaved: {out_path}")
    return metrics
