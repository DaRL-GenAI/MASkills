"""API-key preflight for the GAIA rollout modules.

GAIA rollouts fan out over many workers, so a missing key should stop the run
before the pool starts rather than surface as N identical 401s. These modules
used to check at import time, which is fine for a script and a landmine inside
a package — importing the environment would have killed the process. The check
is a call now: made by each module's ``main()`` and by ``GaiaEnv.__init__``.
"""

from __future__ import annotations

import os

#: Model names are given in ``provider/model`` form, so the default endpoint is
#: OpenRouter. An explicit OPENAI_BASE_URL always wins.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def require_api_keys(tavily: bool = False) -> None:
    """Fail fast unless the keys a GAIA rollout needs are set.

    Args:
        tavily: also require ``TAVILY_API_KEY``, which the SEARCH and BROWSE
            tools need. The no-tool single-agent rollout does not.

    Raises:
        SystemExit: naming the missing variable and how to set it.
    """
    os.environ.setdefault("OPENAI_BASE_URL", DEFAULT_BASE_URL)

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export your OpenRouter key first:\n"
            "  export OPENAI_API_KEY=<your-openrouter-key>"
        )
    if tavily and not os.environ.get("TAVILY_API_KEY"):
        raise SystemExit(
            "TAVILY_API_KEY is not set. GAIA's SEARCH and BROWSE tools need it:\n"
            "  export TAVILY_API_KEY=<your-tavily-key>"
        )
