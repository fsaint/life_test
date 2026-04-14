"""
LightRAG adapter for the Life Test.

Storage:  LightRAG knowledge graph (NetworkX) + NanoVectorDB
Retrieval: hybrid graph + vector search (local, global, hybrid modes)
Answering: Claude LLM via LightRAG's query pipeline

Cost profile:
  - Ingest: LLM calls per chunk for entity/relationship extraction
  - Per question: 1 LightRAG query (graph traversal + LLM generation)
  - Embeddings: local sentence-transformers (no Voyage API key required)

Repository: https://github.com/hkuds/lightrag
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from runner.adapters.base import BaseAdapter
from runner.models import Event

_DEFAULT_QUERY_MODE = "hybrid"
_DEFAULT_TOP_K = 20
_EMBED_MODEL = "all-MiniLM-L6-v2"


class ClaudeLightRagAdapter(BaseAdapter):
    """Claude + LightRAG (knowledge graph + vector hybrid) memory backend."""

    def __init__(self, config: dict):
        self._system_id = config["system_id"]
        llm = config["llm"]
        self._model = llm["model"]
        # LightRAG uses this LLM for both extraction and answering — needs
        # enough headroom for entity/relationship extraction JSON output.
        self._max_tokens = llm.get("max_tokens", 8192)

        mem_config = config.get("memory", {}).get("config", {})
        self._query_mode = mem_config.get("query_mode", _DEFAULT_QUERY_MODE)
        self._top_k = mem_config.get("top_k", _DEFAULT_TOP_K)
        self._embed_model_name = mem_config.get("embed_model", _EMBED_MODEL)

        self._embedder = None
        self._working_dir: Path | None = None
        self._rag = None

    @property
    def system_id(self) -> str:
        return self._system_id

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embed_model_name)
        return self._embedder

    def _make_embedding_func(self):
        """Build a LightRAG-compatible async embedding function using local ST model."""
        from lightrag.utils import wrap_embedding_func_with_attrs

        embedder = self._get_embedder()
        embed_dim = embedder.get_sentence_embedding_dimension()

        @wrap_embedding_func_with_attrs(embedding_dim=embed_dim, max_token_size=512)
        async def _embed(texts: list[str]) -> np.ndarray:
            return embedder.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).astype(np.float32)

        return _embed

    def _make_llm_func(self):
        """Build a LightRAG-compatible async LLM function backed by Claude."""
        model = self._model
        max_tokens = self._max_tokens
        api_key = os.environ["ANTHROPIC_API_KEY"]

        async def _llm(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list | None = None,
            **kwargs,
        ) -> str:
            import anthropic as _anthropic
            client = _anthropic.AsyncAnthropic(api_key=api_key)
            messages = list(history_messages or [])
            messages.append({"role": "user", "content": prompt})
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt or "You are a helpful assistant.",
                messages=messages,
            )
            return response.content[0].text

        return _llm

    def _open_rag(self):
        if self._rag is None:
            from lightrag import LightRAG

            if self._working_dir is None:
                self._working_dir = Path(
                    tempfile.mkdtemp(prefix="life_test_lightrag_")
                )

            rag = LightRAG(
                working_dir=str(self._working_dir),
                llm_model_func=self._make_llm_func(),
                embedding_func=self._make_embedding_func(),
                enable_llm_cache=False,
            )
            asyncio.run(rag.initialize_storages())
            self._rag = rag
        return self._rag

    # ── Adapter interface ──────────────────────────────────────────────────────

    def reset(self) -> None:
        if self._rag is not None:
            try:
                asyncio.run(self._rag.finalize_storages())
            except Exception:
                pass
            self._rag = None
        if self._working_dir is not None and self._working_dir.exists():
            shutil.rmtree(self._working_dir, ignore_errors=True)
        self._working_dir = None

    def ingest_initial_state(self, content: str) -> dict:
        t0 = time.perf_counter()
        rag = self._open_rag()
        asyncio.run(rag.ainsert(content))
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "type": "initial_state",
            "latency_ms": latency_ms,
            "tokens_used": len(content) // 4,
            "status": "ok",
        }

    def ingest_event(self, event: Event) -> dict:
        t0 = time.perf_counter()
        rag = self._open_rag()
        text = _format_event(event)
        asyncio.run(rag.ainsert(text))
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "latency_ms": latency_ms,
            "tokens_used": len(text) // 4,
            "status": "ok",
        }

    def ask(self, question_text: str) -> tuple[str, int]:
        from lightrag import QueryParam

        rag = self._open_rag()
        result = asyncio.run(
            rag.aquery(
                question_text,
                param=QueryParam(mode=self._query_mode, top_k=self._top_k),
            )
        )
        # aquery returns QueryResult (with .response) or a plain string
        answer = result.response if hasattr(result, "response") else str(result)
        tokens = (len(question_text) + len(answer)) // 4
        return answer, tokens


# ── Event formatting ───────────────────────────────────────────────────────────

def _format_event(event: Event) -> str:
    p = event.payload
    ts = event.timestamp

    if event.event_type == "email":
        body = p.get("body", "").strip()
        return (
            f"EMAIL [{ts}]\nFrom: {p.get('from', '')}\nTo: {p.get('to', '')}\n"
            f"Subject: {p.get('subject', '')}\n\n{body}"
        )
    elif event.event_type == "sms":
        lines = "\n".join(f"  {m['from']}: {m['body']}" for m in p.get("thread", []))
        return f"TEXT MESSAGES [{ts}]\n{lines}"
    elif event.event_type == "user_instruction":
        return f"USER INSTRUCTION [{ts}]\n{p.get('instruction', '').strip()}"
    elif event.event_type == "screenshot":
        return (
            f"SCREENSHOT [{ts}]\nSource: {p.get('source_description', '')}\n"
            f"{p.get('image_description', '').strip()}"
        )
    elif event.event_type == "document":
        return (
            f"DOCUMENT [{ts}]\nType: {p.get('document_type', '')}\n"
            f"Title: {p.get('title', '')}\n\n{p.get('content', '').strip()}"
        )
    elif event.event_type == "phone_call":
        parts = p.get("participants", [])
        names = ", ".join(f"{x.get('name', '')} ({x.get('role', '')})" for x in parts)
        return f"PHONE CALL [{ts}]\nParticipants: {names}\n{p.get('summary', '').strip()}"
    elif event.event_type == "calendar_entry":
        return (
            f"CALENDAR ENTRY [{ts}]\n{p.get('title', '')} — starts {p.get('start', '')}\n"
            f"Location: {p.get('location', '')}\nNotes: {p.get('notes', '')}"
        )
    else:
        import json
        return f"EVENT [{event.event_type}] [{ts}]\n{json.dumps(p, indent=2)}"
