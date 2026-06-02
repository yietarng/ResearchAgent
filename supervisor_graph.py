from __future__ import annotations

import functools
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from research_agent.config import ResearchAgentConfig
from research_agent.prompts import SUPERVISOR_PROMPT
from research_agent.state import default_state
from research_agent.supervisor.router import route_to_agent
from research_agent.supervisor.supervisor_state import SupervisorState
from research_agent.team import build_research_team_graph
from research_agent.utils.embeddings import build_llm


def supervisor_decide(state: SupervisorState, *, llm: Any) -> dict:
    """Route to research_agent or direct_answer.

    Uses the regex fast path first.  Only calls the LLM when the regex
    returns 'direct_answer' — that class is ambiguous and benefits from
    LLM judgment; clearly-research requests are routed without any LLM call.
    """
    fast_route = route_to_agent(state)
    if fast_route == "research_agent":
        return {
            "agent_to_invoke": "research_agent",
            "conversation_history": [{"role": "supervisor", "decision": "research_agent", "via": "regex"}],
        }

    # Regex returned direct_answer — confirm with LLM
    prompt = (
        SUPERVISOR_PROMPT
        + f"\n\nUser request: {state['user_request']}\n\n"
        "Reply with either 'RESEARCH' or 'DIRECT'."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().upper()
    agent_to_invoke = "research_agent" if "RESEARCH" in decision else "direct_answer"
    return {
        "agent_to_invoke": agent_to_invoke,
        "conversation_history": [{"role": "supervisor", "decision": agent_to_invoke, "via": "llm"}],
    }


def invoke_research_agent(state: SupervisorState, *, research_graph: Any, thread_id: str) -> dict:
    result = research_graph.invoke(
        {**default_state(), "user_query": state["user_request"]},
        config={"configurable": {"thread_id": f"team_{thread_id}"}},
    )
    research_result = result.get("research_result", "")
    citations = result.get("citations", [])
    if citations:
        research_result += "\n\nSources:\n" + "\n".join(f"- {c}" for c in citations)
    return {
        "agent_result": research_result,
        "conversation_history": [{"role": "research_agent", "status": "completed"}],
    }


def direct_answer(state: SupervisorState, *, llm: Any) -> dict:
    response = llm.invoke([HumanMessage(content=state["user_request"])])
    return {
        "agent_result": response.content,
        "conversation_history": [{"role": "direct", "status": "answered"}],
    }


def format_result(state: SupervisorState) -> dict:
    return {"is_complete": True}


def build_supervisor_graph(
    config: ResearchAgentConfig | None = None,
    checkpointer: Any = None,
    thread_id: str = "default",
):
    config = config or ResearchAgentConfig()
    llm = build_llm(config)
    research_graph = build_research_team_graph(config, checkpointer=checkpointer)

    graph = StateGraph(SupervisorState)

    graph.add_node("supervisor_decide", functools.partial(supervisor_decide, llm=llm))
    graph.add_node(
        "research_agent",
        functools.partial(invoke_research_agent, research_graph=research_graph, thread_id=thread_id),
    )
    graph.add_node("direct_answer", functools.partial(direct_answer, llm=llm))
    graph.add_node("format_result", format_result)

    graph.add_edge(START, "supervisor_decide")
    graph.add_conditional_edges(
        "supervisor_decide",
        lambda s: s.get("agent_to_invoke", "direct_answer"),
        {"research_agent": "research_agent", "direct_answer": "direct_answer"},
    )
    graph.add_edge("research_agent", "format_result")
    graph.add_edge("direct_answer", "format_result")
    graph.add_edge("format_result", END)

    cp = checkpointer or MemorySaver()
    return graph.compile(checkpointer=cp)
