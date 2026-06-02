from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from research_agent.memory.ltm import LTMStore
from research_agent.utils.ltm_policy import StorageDecision, should_store_in_ltm


class RetrieveLTMTool(BaseTool):
    name: str = "retrieve_long_term_memory"
    description: str = (
        "Search long-term memory for prior knowledge relevant to a query. "
        "Call this before web search to avoid re-researching known topics. "
        "Input: a natural-language query. Output: relevant prior findings."
    )
    ltm_store: Any

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str, **kwargs: Any) -> str:
        docs = self.ltm_store.retrieve(query)
        return self.ltm_store.format_docs_for_prompt(docs)

    async def _arun(self, query: str, **kwargs: Any) -> str:
        return self._run(query, **kwargs)


class StoreLTMTool(BaseTool):
    name: str = "store_long_term_memory"
    description: str = (
        "Store an important finding in long-term memory for future sessions. "
        "Only store: survey reports, user preferences, important references, reusable findings. "
        "Do NOT store: raw tool outputs, temporary notes, intermediate reasoning. "
        "Input: content (str), source (str), tags (comma-separated str)."
    )
    ltm_store: Any

    class Config:
        arbitrary_types_allowed = True

    def _run(self, content: str, source: str = "research_agent", tags: str = "", **kwargs: Any) -> str:
        should_store, decision = should_store_in_ltm(content, source=source)
        if not should_store:
            return f"Not stored ({decision.value}): content does not meet LTM write policy."
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        doc_id = self.ltm_store.store(content, metadata={"source": source, "tags": tag_list})
        return f"Stored in LTM (id={doc_id})."

    async def _arun(self, content: str, source: str = "research_agent", tags: str = "", **kwargs: Any) -> str:
        return self._run(content, source=source, tags=tags, **kwargs)


def build_ltm_tools(ltm_store: LTMStore) -> tuple[RetrieveLTMTool, StoreLTMTool]:
    return RetrieveLTMTool(ltm_store=ltm_store), StoreLTMTool(ltm_store=ltm_store)
