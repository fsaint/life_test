"""
MemPalace adapter for the Life Test.

Storage:  mempalace.miner.add_drawer() — raw verbatim chunks into ChromaDB
Retrieval: mempalace.searcher.search_memories() — semantic search, top-k chunks
Answering: Claude LLM with retrieved chunks as context

Cost profile:
  - Ingest: zero LLM calls (ChromaDB only)
  - Per question: 1 Claude call with ~k retrieved chunks as context
  - No summarization, no extraction — MemPalace stores verbatim

Wing/room mapping for life test data:
  wing  = situation_id  (e.g. "level_1_alex")
  room  = event_type    (email, sms, user_instruction, document, screenshot, etc.)
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import anthropic
from mempalace.miner import add_drawer, chunk_text, get_collection
from mempalace.searcher import search_memories

from runner.adapters.base import BaseAdapter
from runner.models import Event

_SYSTEM_PROMPT = """\
You are a knowledgeable personal assistant. You have been given relevant memory excerpts
retrieved from a record of someone's life events. Answer the question accurately based
only on the provided excerpts. Be concise. If the information is not present, say so."""

# Number of chunks to retrieve per question — balance recall vs context length
_N_RESULTS = 8


class ClaudeMemPalaceAdapter(BaseAdapter):
    """Claude LLM with MemPalace (ChromaDB) as the memory backend."""

    def __init__(self, config: dict):
        self._system_id = config["system_id"]
        llm = config["llm"]
        self._model = llm["model"]
        self._max_tokens = llm.get("max_tokens", 2048)
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        # Each adapter instance gets its own palace directory to stay isolated
        configured_path = config.get("memory", {}).get("config", {}).get("palace_path")
        self._palace_root = Path(
            configured_path if configured_path
            else tempfile.mkdtemp(prefix="life_test_mempalace_")
        )
        self._palace_root.mkdir(parents=True, exist_ok=True)
        self._palace_path = str(self._palace_root)

        # Current situation wing (set on ingest_initial_state)
        self._wing: str = "life_test"
        self._collection = None
        self._chunk_counter = 0

    @property
    def system_id(self) -> str:
        return self._system_id

    def reset(self) -> None:
        """Wipe the palace and start fresh."""
        if self._palace_root.exists():
            shutil.rmtree(self._palace_root)
        self._palace_root.mkdir(parents=True, exist_ok=True)
        self._collection = None
        self._chunk_counter = 0

    def _get_collection(self):
        if self._collection is None:
            self._collection = get_collection(self._palace_path)
        return self._collection

    def _store(self, content: str, room: str, source_label: str) -> int:
        """Chunk and store a text block. Returns number of chunks stored."""
        collection = self._get_collection()
        chunks = chunk_text(content, source_label)
        for chunk in chunks:
            # chunk_text returns dicts: {"content": str, "chunk_index": int}
            chunk_content = chunk["content"] if isinstance(chunk, dict) else chunk
            chunk_index = chunk.get("chunk_index", self._chunk_counter) if isinstance(chunk, dict) else self._chunk_counter
            add_drawer(
                collection=collection,
                wing=self._wing,
                room=room,
                content=chunk_content,
                source_file=source_label,
                chunk_index=chunk_index,
                agent="life_test",
            )
        self._chunk_counter += len(chunks)
        return len(chunks)

    def ingest_initial_state(self, content: str) -> dict:
        t0 = time.perf_counter()
        # Use the first line as the wing name (e.g. "# Alex Rivera — Life Overview")
        first_line = content.strip().splitlines()[0].lstrip("#").strip()
        self._wing = first_line[:50]  # cap length

        chunks_stored = self._store(content, room="background", source_label="initial_state")
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "type": "initial_state",
            "latency_ms": latency_ms,
            "chunks_stored": chunks_stored,
            "tokens_used": 0,
            "status": "ok",
        }

    def ingest_event(self, event: Event) -> dict:
        t0 = time.perf_counter()
        content = _format_event_text(event)
        room = _room_for_event(event.event_type)
        source_label = f"{event.event_id}_{event.event_type}"

        chunks_stored = self._store(content, room=room, source_label=source_label)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "latency_ms": latency_ms,
            "chunks_stored": chunks_stored,
            "tokens_used": 0,
            "status": "ok",
        }

    def ask(self, question_text: str) -> str:
        # Retrieve relevant chunks from MemPalace
        result = search_memories(
            query=question_text,
            palace_path=self._palace_path,
            n_results=_N_RESULTS,
        )

        if result.get("error"):
            return f"[MemPalace search error: {result['error']}]"

        hits = result.get("results", [])
        if not hits:
            return "I don't have information about that."

        # Build context from retrieved chunks
        context_parts = []
        for hit in hits:
            meta = hit.get("metadata", {})
            room = meta.get("room", "")
            source = meta.get("source_file", "")
            text = hit.get("text") or hit.get("document", "")
            context_parts.append(f"[{room} / {source}]\n{text.strip()}")

        context_block = "\n\n---\n\n".join(context_parts)
        user_message = (
            f"Memory excerpts retrieved for your question:\n\n"
            f"{context_block}\n\n"
            f"Question: {question_text}"
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


# ── Event formatting ───────────────────────────────────────────────────────────

def _room_for_event(event_type: str) -> str:
    """Map event type to a MemPalace room name."""
    return {
        "email": "communications",
        "sms": "communications",
        "user_instruction": "instructions",
        "screenshot": "documents",
        "document": "documents",
        "phone_call": "communications",
        "calendar_entry": "calendar",
    }.get(event_type, "misc")


def _format_event_text(event: Event) -> str:
    """Render an event as plain text for storage in MemPalace."""
    p = event.payload
    ts = event.timestamp

    if event.event_type == "email":
        body = p.get("body", "").strip()
        return (
            f"EMAIL [{ts}]\n"
            f"From: {p.get('from', '')}\n"
            f"To: {p.get('to', '')}\n"
            f"Subject: {p.get('subject', '')}\n\n"
            f"{body}"
        )

    elif event.event_type == "sms":
        lines = "\n".join(
            f"  {m['from']}: {m['body']}" for m in p.get("thread", [])
        )
        return f"TEXT MESSAGES [{ts}]\n{lines}"

    elif event.event_type == "user_instruction":
        return f"USER INSTRUCTION [{ts}]\n{p.get('instruction', '').strip()}"

    elif event.event_type == "screenshot":
        return (
            f"SCREENSHOT [{ts}]\n"
            f"Source: {p.get('source_description', '')}\n"
            f"{p.get('image_description', '').strip()}"
        )

    elif event.event_type == "document":
        return (
            f"DOCUMENT [{ts}]\n"
            f"Type: {p.get('document_type', '')}\n"
            f"Title: {p.get('title', '')}\n\n"
            f"{p.get('content', '').strip()}"
        )

    elif event.event_type == "phone_call":
        parts = p.get("participants", [])
        names = ", ".join(f"{x.get('name', '')} ({x.get('role', '')})" for x in parts)
        return f"PHONE CALL [{ts}]\nParticipants: {names}\n{p.get('summary', '').strip()}"

    elif event.event_type == "calendar_entry":
        return (
            f"CALENDAR ENTRY [{ts}]\n"
            f"{p.get('title', '')} — starts {p.get('start', '')}\n"
            f"Location: {p.get('location', '')}\n"
            f"Notes: {p.get('notes', '')}"
        )

    else:
        import json
        return f"EVENT [{event.event_type}] [{ts}]\n{json.dumps(p, indent=2)}"
