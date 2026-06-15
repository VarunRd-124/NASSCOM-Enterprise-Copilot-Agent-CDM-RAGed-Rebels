"""Central config — reads from env (Databricks App secrets) with safe local fallbacks."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Databricks
    databricks_host: str = os.getenv("DATABRICKS_HOST", "")
    databricks_token: str = os.getenv("DATABRICKS_TOKEN", "")

    # Mosaic AI Vector Search — matches notebooks 02 / 06
    vs_endpoint: str = os.getenv("VS_ENDPOINT", "hackathon_vs_endpoint")
    vs_index: str = os.getenv(
        "VS_INDEX",
        "catalog.hackathon.gold_knowledge_chunks_idx",
    )

    # Neo4j
    neo4j_uri: str = os.getenv("NEO4J_URI", "")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")

    # LLM — Claude Opus 4.6 served on Databricks (matches notebook 06)
    llm_endpoint: str = os.getenv("LLM_ENDPOINT", "databricks-claude-sonnet-4-6")
    # Fallback / alternative — Claude via Anthropic API
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

    # Reranker — HuggingFace cross-encoder (priority order per architecture diagram)
    reranker_models: tuple = (
        "BAAI/bge-reranker-v2-m3",
        "jinaai/jina-reranker-v3",
        "BAAI/bge-reranker-large",
    )

    # Retrieval
    top_k_vector: int = 15
    top_k_graph: int = 10
    top_k_final: int = 5
    confidence_threshold: float = 0.35    # below → escalation path

    # Feedback table — lives in the same catalog/schema as everything else
    feedback_table: str = os.getenv(
        "FEEDBACK_TABLE",
        "catalog.hackathon.user_feedback",
    )


CFG = Config()
