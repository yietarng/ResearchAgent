# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
```

Required `.env` keys (file lives at project root):
```
OPENAI_API_KEY=sk-...        # required — GPT-4o + text-embedding-3-small
TAVILY_API_KEY=tvly-...      # required — web search
# ANTHROPIC_API_KEY=sk-ant-... # optional — only if llm_provider=ANTHROPIC
```

## Running

```bash
# Single-agent path (CLI)
python run.py

# Programmatic — single-agent
from research_agent import run_research
run_research("your query", thread_id="t1")

# Programmatic — full supervisor + multi-agent team
from research_agent import build_supervisor_graph
graph = build_supervisor_graph()
graph.invoke({"user_request": "your query", "conversation_history": [], "agent_result": "", "agent_to_invoke": "", "is_complete": False}, config={"configurable": {"thread_id": "t1"}})
```

## Architecture

Three nested LangGraph layers:

```
SupervisorGraph  (supervisor/supervisor_graph.py)
 ├── regex fast path → "research_agent" (no LLM call for obvious research queries)
 └── LLM fallback → "research_agent" or "direct_answer"
      └── ResearchTeamGraph  (team/research_supervisor.py)
           ├── plan_research: keyword fast path (_fast_plan) then LLM fallback
           └── fan-out via Send API →
                ├── LiteratureAgent  (ReAct: retrieve_ltm, store_ltm, read_stm, write_stm)
                ├── WebAgent         (ReAct: web_search + memory tools)
                └── InternalDocAgent (ReAct: same tools, separate ChromaDB collection)
```

A simpler **single-agent path** (`graph.py` / `build_research_graph`) also exists — sequential nodes without fan-out, used by `run.py` and `run_research()`.

### State (`state.py`)

`AgentState` uses annotated reducers: `retrieved_docs` and `citations` use `operator.add` (accumulate across fan-out), `stm_notes` uses `_capped_add` (20-note rolling cap), `ltm_context` uses `_last_write_wins` (prevents multiplication on fan-out merge).

### Memory system

- **STM**: LangGraph checkpointer state, 20-note cap, `thread_id`-scoped. Default backend is in-memory (`MemorySaver`); switch to SQLite via `STMBackend.SQLITE` + `langgraph-checkpoint-sqlite`.
- **LTM**: ChromaDB at `./chroma_db`, content-hash dedup (`content_hash` field pinned after metadata spread). Two collections: `research_ltm` (shared by web + literature agents) and `internal_docs` (InternalDocAgent only). LTM writes are gated by `should_store_in_ltm()` in `utils/ltm_policy.py`.
- Thread-id namespacing: supervisor uses thread_id `t1`, team graph uses `team_t1` to avoid checkpointer collisions.

### Key files

| File | Role |
|---|---|
| `config.py` | `ResearchAgentConfig` dataclass; `LLMProvider`, `STMBackend` enums |
| `graph.py` | Single-agent sequential graph (`build_research_graph`, `run_research`) |
| `supervisor/supervisor_graph.py` | Top-level orchestrator with regex + LLM routing |
| `supervisor/router.py` | `route_to_agent()` regex fast path |
| `team/research_supervisor.py` | Fan-out planner (`_fast_plan`, `route_research_tasks` via Send) |
| `team/{literature,web,internal_doc}_agent.py` | ReAct sub-agent graphs |
| `memory/ltm.py` | `LTMStore` — ChromaDB wrapper with dedup |
| `memory/stm.py` | `STMStore` helpers |
| `utils/embeddings.py` | `build_llm()` and `get_embeddings_model()` factories |
| `utils/ltm_policy.py` | `should_store_in_ltm()` keyword gate |
| `tools/stm_tools.py` | `build_stm_tools(state)` closure factory |
| `nodes/` | Individual graph node builders (retrieve_ltm, summarize, write_ltm, etc.) |
| `prompts.py` | All system prompt strings |

### Configuration

Override defaults by passing a `ResearchAgentConfig` instance:

```python
from research_agent import ResearchAgentConfig, LLMProvider, build_supervisor_graph
config = ResearchAgentConfig(
    llm_provider=LLMProvider.ANTHROPIC,
    llm_model="claude-opus-4-7",
    stm_backend=STMBackend.SQLITE,
)
graph = build_supervisor_graph(config=config)
```

Embeddings always use OpenAI (`text-embedding-3-small`) regardless of `llm_provider` — Anthropic has no embeddings API.
