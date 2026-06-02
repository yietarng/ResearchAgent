from __future__ import annotations

from typing import Any

from research_agent.config import LLMProvider, ResearchAgentConfig


def get_embeddings_model(config: ResearchAgentConfig):
    """Return a LangChain Embeddings instance.

    Embeddings always use OpenAI — Anthropic does not offer an embeddings API.
    """
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=config.embedding_model)


def build_llm(config: ResearchAgentConfig) -> Any:
    """Instantiate the chat model specified by config.llm_provider."""
    if config.llm_provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic  # optional dep: pip install langchain-anthropic
        return ChatAnthropic(model=config.llm_model, temperature=config.llm_temperature)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=config.llm_model, temperature=config.llm_temperature)
