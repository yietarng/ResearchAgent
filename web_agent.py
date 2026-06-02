from __future__ import annotations

import functools
import re
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from research_agent.config import ResearchAgentConfig
from research_agent.memory.ltm import LTMStore
from research_agent.prompts import WEB_AGENT_PROMPT
from research_agent.state import AgentState
from research_agent.tools.ltm_tools import build_ltm_tools
from research_agent.tools.stm_tools import build_stm_tools
from research_agent.tools.web_search import build_web_search_tool

_URL_RE = re.compile(r'https?://[^\s,)"\'>]+')


def web_search_node(
    state: AgentState, *, llm: Any, ltm_store: LTMStore, ra_config: ResearchAgentConfig | None = None
) -> dict:
    api_key_env = ra_config.tavily_api_key_env if ra_config else "TAVILY_API_KEY"
    retrieve_ltm, store_ltm = build_ltm_tools(ltm_store)
    web_tool = build_web_search_tool(api_key_env=api_key_env)
    read_stm, write_stm, stm_writes = build_stm_tools(state)
    tools = [web_tool, retrieve_ltm, store_ltm, read_stm, write_stm]

    agent = create_react_agent(llm, tools)
    messages = [
        {"role": "system", "content": WEB_AGENT_PROMPT},
        {"role": "user", "content": state["user_query"]},
    ]
    result = agent.invoke({"messages": messages})
    final = result["messages"][-1].content

    citations: list[str] = []
    for msg in result["messages"]:
        content = getattr(msg, "content", "")
        if getattr(msg, "type", "") in ("ai", "tool"):
            citations.extend(_URL_RE.findall(content))

    return {
        "retrieved_docs": [final],
        "citations": list(dict.fromkeys(citations)),
        "stm_notes": stm_writes,
        "intermediate_notes": ["[WebAgent] Completed web search."],
    }


def build_web_agent_graph(config: ResearchAgentConfig, llm: Any, ltm_store: LTMStore):
    graph = StateGraph(AgentState)
    graph.add_node("web_search", functools.partial(web_search_node, llm=llm, ltm_store=ltm_store, ra_config=config))
    graph.add_edge(START, "web_search")
    graph.add_edge("web_search", END)
    return graph.compile()
