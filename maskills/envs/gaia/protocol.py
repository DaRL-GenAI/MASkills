"""The protocol-only prompts behind GAIA's no-skills baseline.

A GAIA agent cannot produce a scoreable trajectory from an empty prompt: it
would not know that a tool call is written ``SEARCH: <query>``, that the
Researcher ends its turn with a ``HANDOFF`` block, or that the answer has to
arrive on a ``FINAL ANSWER:`` line. A literally empty library therefore
measures whether the model can guess a wire format, not whether it can solve
GAIA.

So the floor these prompts define is *protocol and nothing else* -- the tool
syntax and the output contract the harness parses, with no task knowledge of
any kind. That is what ``--skills empty`` installs, and what every learned
GAIA library is measured against.

The strings are the ones the paper's no-skills runs used. They live in code
rather than in a directory so the baseline is reproducible from a clone.
"""

from __future__ import annotations

#: One agent with the whole tool stack, ending on FINAL ANSWER.
CENTRALIZED = """\
You are answering one GAIA question.

Tools:
```
SEARCH: <query>
BROWSE: <url>
COMPUTE:
```python
<code>
print(<result>)
```
```

End your last turn with exactly one line:
```
FINAL ANSWER: <value>
```
"""

#: agent_1 in the two-agent split: gathers evidence, hands off, never answers.
RESEARCHER = """\
You are agent_1.

Tools (max 5):
```
SEARCH: <query>
BROWSE: <url>
```

End your last turn with a HANDOFF block:
```
HANDOFF:
<your findings>
HANDOFF_TO_SOLVER
```

Do not emit `FINAL ANSWER:`.
"""

#: agent_2 in the two-agent split: reads the handoff, computes, answers.
SOLVER = """\
You are agent_2. Read the question and agent_1's HANDOFF.

Tool (max 3):
```
COMPUTE:
```python
<code>
print(<result>)
```
```

End your last turn with exactly one line:
```
FINAL ANSWER: <value>
```
"""

#: Agent name -> the protocol it runs under in the decentralized topology.
BY_AGENT = {"agent_1": RESEARCHER, "agent_2": SOLVER}


def bundle(body: str, name: str) -> dict:
    """Wrap a protocol prompt in the shape :func:`~maskills.skill_lib.load_lib` returns."""
    return {
        "root": {"name": name, "description": f"{name} protocol only, no skills",
                 "body": body, "path": "<empty>"},
        "skills": {},
        "path": "<empty>",
    }


def prompt_for(agent: str) -> str:
    """The protocol prompt for one agent, for the environment-driven rollout."""
    return BY_AGENT.get(agent, CENTRALIZED)
