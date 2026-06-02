from __future__ import annotations

import re

from research_agent.supervisor.supervisor_state import SupervisorState

# Patterns that signal a research-heavy request
_RESEARCH_PATTERNS = [
    r"\bsearch\b", r"\bfind\b", r"\bsurvey\b", r"\bpapers?\b",
    r"\brecent\b", r"\blatest\b", r"\bstudy\b", r"\bresearch\b",
    r"\bliterature\b", r"\bretrieve\b", r"\bsummar\w+\b",
    r"\bwhat.*know\b", r"\btell me about\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _RESEARCH_PATTERNS]


def route_to_agent(state: SupervisorState) -> str:
    """Conditional edge: decide which node handles the request.

    Returns one of: "research_agent", "direct_answer".
    """
    request = state.get("user_request", "")
    if any(pat.search(request) for pat in _COMPILED):
        return "research_agent"
    return "direct_answer"
