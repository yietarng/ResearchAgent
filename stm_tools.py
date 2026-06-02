from __future__ import annotations

from langchain_core.tools import tool

from research_agent.state import AgentState


def build_stm_tools(state: AgentState) -> tuple:
    """Return (read_stm, write_stm, pending_writes).

    Reads serve the current state snapshot directly.  Writes accumulate in
    pending_writes; the calling node must include them in its return dict
    under "stm_notes" so the _capped_add reducer persists them.

    Usage::

        read_stm, write_stm, stm_writes = build_stm_tools(state)
        agent = create_react_agent(llm, [read_stm, write_stm, ...])
        ...
        return {"stm_notes": stm_writes, ...}
    """
    _pending: list[str] = []

    @tool
    def read_stm(key: str) -> str:
        """Read from short-term memory.

        Supported keys: stm_notes (recent session notes),
        current_plan (active research plan), user_query.
        """
        if key == "stm_notes":
            notes = state.get("stm_notes", [])
            return "\n".join(notes[-10:]) if notes else "(empty)"
        if key == "current_plan":
            return state.get("current_plan", "(no plan)")
        return str(state.get(key, "(not found)"))

    @tool
    def write_stm(note: str) -> str:
        """Append a short note to short-term memory for this session."""
        _pending.append(note[:200])
        return f"Note queued: {note[:60]}..."

    return read_stm, write_stm, _pending
