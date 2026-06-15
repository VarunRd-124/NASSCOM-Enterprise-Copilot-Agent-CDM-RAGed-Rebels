"""
Section 7 of the architecture — LLM generation.

Priority order per the diagram:
  1. DBRX Instruct on Databricks Model Serving (Mosaic AI)
  2. Claude Sonnet (Anthropic API)
  3. Claude Opus (Anthropic API)

Returns a structured answer with inline citation markers [1], [2], ...
"""
from __future__ import annotations

import logging
from typing import List, Dict

try:
    from .config import CFG
    from .retrieval import Candidate
except ImportError:
    from config import CFG
    from retrieval import Candidate

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the Enterprise Knowledge Copilot — a site-reliability assistant.

You answer questions about Kubernetes, Docker, FastAPI, and Kafka incidents using ONLY the
provided context (SOPs, past tickets, log events, and structured graph facts).

Rules:
- Cite every factual claim with bracketed numbers like [1], [2] matching the context items.
- Be concrete: name commands, config keys, and components.
- If the context is insufficient, say so plainly and recommend the escalation link.
- Structure the answer as:
    **Likely cause** — one or two sentences with citations
    **Resolution steps** — numbered list, each step cited
    **Verification** — how to confirm the fix worked
- Never invent file paths, error codes, or component names that aren't in the context.
"""


def _format_context(candidates: List[Candidate]) -> str:
    blocks = []
    for i, c in enumerate(candidates, start=1):
        header = f"[{i}] source={c.source_type} service={c.service} id={c.source_id}"
        blocks.append(f"{header}\n{c.text.strip()}")
    return "\n\n---\n\n".join(blocks)


MAX_HISTORY_MESSAGES = 6   # ~3 prior Q&A pairs kept for conversational continuity


def _build_messages(query: str, context: str, history: List[Dict] | None) -> List[Dict]:
    """System prompt + recent prior turns + current question/context.

    Retrieval stays single-turn; only the LLM sees the conversation so far,
    enabling coherent follow-ups ("how do I verify that?").
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in (history or [])[-MAX_HISTORY_MESSAGES:]:
        role, content = m.get("role"), m.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": str(content)[:1500]})
    messages.append({"role": "user", "content": f"Question:\n{query}\n\nContext:\n{context}"})
    return messages


def _generate_dbrx(messages: List[Dict]) -> str:
    """Call the Databricks Model Serving endpoint via the raw REST API.

    Mirrors notebook 05 (`call_llm`): POST to /serving-endpoints/{name}/invocations
    with an OpenAI-style chat payload. Avoids the SDK's typed-message path that
    raised `'dict' object has no attribute 'as_dict'`.
    """
    import requests

    host = CFG.databricks_host.rstrip("/")
    token = CFG.databricks_token
    if not host or not token:
        raise RuntimeError("DATABRICKS_HOST / DATABRICKS_TOKEN not set")

    url = f"{host}/serving-endpoints/{CFG.llm_endpoint}/invocations"
    payload = {
        "messages": messages,
        "max_tokens": 900,
        "temperature": 0.1,
    }
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    # OpenAI-compatible chat response (Claude/DBRX endpoints on Databricks)
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    # Anthropic-native shape, just in case the endpoint returns it
    if "content" in data and isinstance(data["content"], list):
        return data["content"][0].get("text", "")
    raise RuntimeError(f"Unexpected serving response shape: {list(data.keys())}")


def _generate_claude(messages: List[Dict]) -> str:
    from anthropic import Anthropic
    client = Anthropic(api_key=CFG.anthropic_api_key)
    system = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMPT)
    convo = [m for m in messages if m["role"] != "system"]
    msg = client.messages.create(
        model=CFG.claude_model,
        max_tokens=900,
        temperature=0.1,
        system=system,
        messages=convo,
    )
    return msg.content[0].text


def generate_answer(query: str, candidates: List[Candidate],
                    history: List[Dict] | None = None) -> Dict:
    """Returns {'answer', 'model_used', 'citations'}.

    `history` is the prior chat turns (excluding the current question) — passed
    to the LLM for multi-turn continuity. Retrieval remains single-turn.
    """
    context = _format_context(candidates)
    messages = _build_messages(query, context, history)

    # Try DBRX first
    try:
        text = _generate_dbrx(messages)
        return {"answer": text, "model_used": CFG.llm_endpoint, "citations": _build_citations(candidates)}
    except Exception as e:
        log.warning("DBRX call failed, falling back to Claude: %s", e)

    if CFG.anthropic_api_key:
        try:
            text = _generate_claude(messages)
            return {"answer": text, "model_used": CFG.claude_model, "citations": _build_citations(candidates)}
        except Exception as e:
            log.exception("Claude call also failed: %s", e)

    return {
        "answer": "I could not reach the language model service. Please retry, or open the "
                  "linked SOPs below for manual reference.",
        "model_used": "none",
        "citations": _build_citations(candidates),
    }


def _build_citations(candidates: List[Candidate]) -> List[Dict]:
    return [
        {
            "idx": i + 1,
            "source_type": c.source_type,
            "source_id": c.source_id,
            "service": c.service,
            "uri": c.source_uri,
            "preview": c.text[:220] + ("…" if len(c.text) > 220 else ""),
            "rerank_score": round(c.rerank_score, 3),
        }
        for i, c in enumerate(candidates)
    ]
