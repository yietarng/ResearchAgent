from __future__ import annotations

import functools
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from research_agent.config import ResearchAgentConfig
from research_agent.memory.ltm import LTMStore
from research_agent.prompts import LITERATURE_AGENT_PROMPT
from research_agent.state import AgentState
from research_agent.tools.ltm_tools import build_ltm_tools
from research_agent.tools.stm_tools import build_stm_tools


def literature_search(state: AgentState, *, llm: Any, ltm_store: LTMStore) -> dict:
    retrieve_ltm, store_ltm = build_ltm_tools(ltm_store)
    read_stm, write_stm, stm_writes = build_stm_tools(state)
    tools = [retrieve_ltm, store_ltm, read_stm, write_stm]

    agent = create_react_agent(llm, tools)
    messages = [
        {"role": "system", "content": LITERATURE_AGENT_PROMPT},
        {"role": "user", "content": state["user_query"]},
    ]
    result = agent.invoke({"messages": messages})
    final = result["messages"][-1].content

    return {
        "retrieved_docs": [final],
        "stm_notes": stm_writes,
        "intermediate_notes": ["[LiteratureAgent] Completed paper search."],
    }


def build_literature_agent_graph(config: ResearchAgentConfig, llm: Any, ltm_store: LTMStore):
    graph = StateGraph(AgentState)
    graph.add_node("literature_search", functools.partial(literature_search, llm=llm, ltm_store=ltm_store))
    graph.add_edge(START, "literature_search")
    graph.add_edge("literature_search", END)
    return graph.compile()
