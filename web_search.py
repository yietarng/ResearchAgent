from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import BaseTool


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web for recent information. "
        "Input: a natural-language search query. "
        "Output: a list of snippets with source URLs."
    )
    max_results: int = 5
    api_key_env: str = "TAVILY_API_KEY"

    def _run(self, query: str, **kwargs: Any) -> str:
        from tavily import TavilyClient
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            return f"{self.api_key_env} not set — web search unavailable."
        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=self.max_results)
        results = response.get("results", [])
        if not results:
            return "No results found."
        lines = [
            f"- {r.get('title', 'Untitled')}: {r.get('url', '')}\n  {r.get('content', '')[:300]}"
            for r in results
        ]
        return "\n".join(lines)

    async def _arun(self, query: str, **kwargs: Any) -> str:
        return self._run(query, **kwargs)


def build_web_search_tool(max_results: int = 5, api_key_env: str = "TAVILY_API_KEY") -> WebSearchTool:
    return WebSearchTool(max_results=max_results, api_key_env=api_key_env)
