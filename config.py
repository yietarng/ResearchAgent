from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class STMBackend(Enum):
    MEMORY = "memory"
    SQLITE = "sqlite"


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ResearchAgentConfig:
    # LTM / vector store (shared: web + literature agents)
    ltm_collection_name: str = "research_ltm"
    ltm_persist_directory: str = "./chroma_db"
    ltm_top_k: int = 5
    ltm_relevance_threshold: float = 0.4

    # LTM for internal documents (separate collection from research LTM)
    internal_ltm_collection_name: str = "internal_docs"
    internal_ltm_persist_directory: str = "./chroma_db"

    # Embeddings
    embedding_model: str = "text-embedding-3-small"

    # STM
    stm_backend: STMBackend = STMBackend.MEMORY
    sqlite_path: str = "./research_stm.db"

    # LLM
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0

    # Web search
    tavily_api_key_env: str = "TAVILY_API_KEY"
