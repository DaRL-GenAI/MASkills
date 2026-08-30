# Tool: search_wiki

Wikipedia search tool backed by `search_wiki.py` at the repo root, which
wraps the MediaWiki API. Each call issues a full-text search and fetches
plain-text intro extracts for the top hits.

## Underlying function

```python
def search_wiki(query: str, limit: int = 5,
                intro_only: bool = True, max_chars: int = 800) -> list[dict]:
    # returns a list of hits; each hit has keys:
    #   title, pageid, url, snippet, extract
```

## Invocation protocol

The agent invokes the tool by emitting **exactly** the following tag
anywhere in its response (the environment parses it with a regex):

```
<search>QUERY STRING HERE</search>
```

Rules:
- The tag name is `search` (lowercase). Other names are not parsed.
- Only the **first** `<search>...</search>` in a response is executed;
  additional tags in the same response are ignored.
- If a response contains a search tag, the environment treats that whole
  response as a tool call and does NOT forward it to the next agent.
- When a response contains NO search tag, it is treated as the agent's
  final contribution for that turn.

Each agent may issue up to `max_search_calls` searches in its turn
(config-controlled). After the limit, further search tags are ignored
and the agent's next response is taken as final.

## Tool result format

Results come back as a new user message on the same agent's private
conversation (other agents in the shared pool do NOT see them):

```
<search_result>
[1] Page Title
First ~800 chars of the intro extract.

[2] Another Title
Another extract.

...
</search_result>
```

`No results.` is returned when the query finds nothing; `Search error:
<Type>: <msg>` is returned on network/API failure. The agent should
handle both gracefully and either retry with a different query or
answer from prior knowledge.

## Query guidance

Wikipedia's search is keyword-based, not conversational. Good queries
are short, named-entity-heavy strings. Avoid full natural-language
questions.

Good:
- `Nelson Mandela president South Africa`
- `Battle of Hastings 1066`
- `Albert Einstein Nobel Prize year`

Bad:
- `Who was the president of South Africa in 1994?`
- `Tell me about the battle in 1066`

For multi-hop HotPotQA-style questions, decompose the question into
sub-queries and issue multiple searches — e.g., first find entity A,
then use a fact from A's extract to form the next query for entity B.

## When NOT to search

- Pure math/logic questions that do not depend on world knowledge.
- Questions fully answered by the question text itself.
- Follow-up turns by a downstream agent, if prior agents already
  surfaced the needed facts in the shared pool.
