"""
Graph Storage adapter for the Life Test.

Storage:  graph_storage.GraphStore — SQLite + sqlite-vec knowledge graph
Retrieval: semantic search over facts/relationships + prompt enrichment
Answering: Claude LLM with graph-enriched context

Cost profile:
  - Ingest: 1 Haiku call per event (entity/fact/relationship extraction)
  - Per question: 1 main Claude call with graph context
  - LLM extraction builds a structured knowledge graph from raw events

Repository: https://github.com/fsaint/graph_storage
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import anthropic

from runner.adapters.base import BaseAdapter
from runner.models import Event

_SYSTEM_PROMPT = """\
You are a knowledgeable personal assistant. You have been given context retrieved
from a knowledge graph built from someone's life events (entities, facts, and
relationships). Answer the question accurately based only on the provided context.
When facts appear to conflict, prefer the more recent or specific information.
Be concise. If the information is not present, say so."""

_N_CONTEXT_ITEMS = 30
_SEMANTIC_THRESHOLD = 0.25
_EXTRACT_MODEL = "claude-haiku-4-5-20251001"

# Keywords that signal broad aggregation queries needing wider retrieval
_AGGREGATION_KEYWORDS = frozenset([
    "all", "every", "summarize", "summary", "list", "upcoming", "open",
    "pending", "outstanding", "deadlines", "appointments", "action items",
    "tasks", "schedule", "what are", "what were", "what is the total",
])

# Sub-queries injected for aggregation questions to widen graph coverage
_AGGREGATION_SUB_QUERIES = [
    "deadline date appointment scheduled",
    "task action required must do",
    "decision commitment agreement plan",
    "payment amount due bill invoice",
    "address location move change",
]


class ClaudeGraphAdapter(BaseAdapter):
    """Claude LLM with graph_storage (SQLite knowledge graph) as the memory backend."""

    def __init__(self, config: dict):
        self._system_id = config["system_id"]
        llm = config["llm"]
        self._model = llm["model"]
        self._max_tokens = llm.get("max_tokens", 2048)
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        mem_config = config.get("memory", {}).get("config", {})
        self._extract_model = mem_config.get("extract_model", _EXTRACT_MODEL)
        self._n_context_items = mem_config.get("n_context_items", _N_CONTEXT_ITEMS)
        self._semantic_threshold = mem_config.get("semantic_threshold", _SEMANTIC_THRESHOLD)

        # LLM function passed to graph_storage for entity/fact extraction
        self._llm_fn = None  # lazy-built after imports

        self._db_dir: Path | None = None
        self._store = None  # graph_storage.GraphStore

    @property
    def system_id(self) -> str:
        return self._system_id

    def _get_llm_fn(self):
        if self._llm_fn is None:
            from graph_storage import make_anthropic_llm_fn
            self._llm_fn = make_anthropic_llm_fn(
                self._client,
                model=self._extract_model,
                max_tokens=8192,
            )
        return self._llm_fn

    def _open_store(self):
        if self._store is None:
            from graph_storage import GraphStore
            if self._db_dir is None:
                self._db_dir = Path(tempfile.mkdtemp(prefix="life_test_graph_"))
            db_path = str(self._db_dir / "graph.db")
            self._store = GraphStore(db_path=db_path)
        return self._store

    def reset(self) -> None:
        if self._store is not None:
            try:
                self._store.close()
            except Exception:
                pass
            self._store = None
        if self._db_dir is not None and self._db_dir.exists():
            shutil.rmtree(self._db_dir, ignore_errors=True)
        self._db_dir = None

    # ── Adapter interface ──────────────────────────────────────────────────────

    def ingest_initial_state(self, content: str) -> dict:
        t0 = time.perf_counter()
        store = self._open_store()
        llm_fn = self._get_llm_fn()

        # Chunk large content to avoid truncated JSON from the extraction LLM.
        chunk_size = 6000  # ~1500 tokens; safe margin under max_tokens=8192
        chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

        total_entities = total_facts = total_rels = 0
        for idx, chunk in enumerate(chunks):
            result = store.extract_from_text(
                text=chunk,
                llm_fn=llm_fn,
                source=f"initial_state_chunk_{idx}",
                record_interactions=True,
            )
            total_entities += len(result.entities_created)
            total_facts += len(result.facts_created)
            total_rels += len(result.relationships_created)

        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "type": "initial_state",
            "latency_ms": latency_ms,
            "tokens_used": len(content) // 4,
            "entities_created": total_entities,
            "facts_created": total_facts,
            "relationships_created": total_rels,
            "status": "ok",
        }

    def ingest_event(self, event: Event) -> dict:
        t0 = time.perf_counter()
        store = self._open_store()
        text = _format_event(event)
        result = store.extract_from_text(
            text=text,
            llm_fn=self._get_llm_fn(),
            source=event.event_id,
            record_interactions=True,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "latency_ms": latency_ms,
            "tokens_used": len(text) // 4,
            "entities_created": len(result.entities_created),
            "facts_created": len(result.facts_created),
            "relationships_created": len(result.relationships_created),
            "status": "ok",
        }

    def ask(self, question_text: str) -> tuple[str, int]:
        store = self._open_store()

        # Enrich prompt: name-entity matching + semantic search over facts/rels
        enriched = store.enrich_prompt(
            text=question_text,
            max_context_items=self._n_context_items,
            include_semantic=True,
            semantic_threshold=self._semantic_threshold,
        )

        context_text = enriched.context.strip()

        # For aggregation queries, run additional sub-queries to widen coverage.
        # "Summarize all action items" needs many semantically-distant fact nodes
        # that a single search pass misses.
        if _is_aggregation_query(question_text):
            extra_parts: list[str] = []
            seen: set[str] = set()
            # Collect any content already in the primary context to avoid dups
            for line in context_text.splitlines():
                seen.add(line.strip())

            for sub_q in _AGGREGATION_SUB_QUERIES:
                results = store.search(sub_q, limit=self._n_context_items)
                for r in results:
                    content = None
                    if r.fact:
                        content = r.fact.content
                    elif r.relationship and r.relationship.description:
                        content = r.relationship.description
                    if content and content.strip() not in seen:
                        seen.add(content.strip())
                        extra_parts.append(content)

            if extra_parts:
                context_text = (
                    context_text + "\n\n" + "\n".join(extra_parts)
                ).strip()

        # Fallback: direct semantic search if enrichment returned nothing
        if not context_text:
            results = store.search(question_text, limit=self._n_context_items)
            parts = []
            for r in results:
                if r.fact:
                    parts.append(r.fact.content)
                elif r.relationship and r.relationship.description:
                    parts.append(r.relationship.description)
            context_text = "\n".join(parts)

        if not context_text:
            return "I don't have information about that.", 0

        user_message = (
            f"Knowledge graph context retrieved for your question:\n\n"
            f"{context_text}\n\n"
            f"Question: {question_text}"
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return response.content[0].text, tokens


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_aggregation_query(text: str) -> bool:
    """Return True if the question is a broad aggregation/summary query."""
    lower = text.lower()
    return any(kw in lower for kw in _AGGREGATION_KEYWORDS)


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
