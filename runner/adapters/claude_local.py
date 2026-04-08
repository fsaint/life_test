"""
Claude adapter — context window only (no external memory).

All events are appended to a running context list and sent with every question.
This is the baseline: tests pure LLM recall without any memory system.
Cost scales linearly with context length.
"""
from __future__ import annotations
import os
import time
import yaml
import anthropic
from runner.adapters.base import BaseAdapter
from runner.models import Event


_SYSTEM_PROMPT = """\
You are a knowledgeable personal assistant. You have been given background information
about a person's life, followed by a stream of events (emails, texts, instructions, etc.).

Answer questions accurately based only on the information provided. If information is not
available, say so clearly. Be concise."""

_EVENT_TEMPLATES = {
    "email": "EMAIL [{timestamp}]\nFrom: {from}\nTo: {to}\nSubject: {subject}\n\n{body}",
    "sms": "TEXT MESSAGES [{timestamp}]\n{thread}",
    "user_instruction": "USER INSTRUCTION [{timestamp}]\n{instruction}",
    "screenshot": "SCREENSHOT [{timestamp}]\nSource: {source_description}\n{image_description}",
    "document": "DOCUMENT [{timestamp}]\nType: {document_type}\nTitle: {title}\n\n{content}",
    "phone_call": "PHONE CALL [{timestamp}]\nWith: {participants}\nSummary: {summary}",
    "calendar_entry": "CALENDAR [{timestamp}]\n{title} — {start}\nLocation: {location}\nNotes: {notes}",
}


class ClaudeLocalAdapter(BaseAdapter):
    """Claude with full context window, no external memory."""

    def __init__(self, config: dict):
        self._system_id = config["system_id"]
        llm = config["llm"]
        self._model = llm["model"]
        self._max_tokens = llm.get("max_tokens", 2048)
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._context: list[str] = []  # accumulated text blocks

    @property
    def system_id(self) -> str:
        return self._system_id

    def reset(self) -> None:
        self._context = []

    def ingest_initial_state(self, content: str) -> dict:
        t0 = time.perf_counter()
        self._context.append(f"=== BACKGROUND: WHO THIS PERSON IS ===\n{content.strip()}")
        latency_ms = int((time.perf_counter() - t0) * 1000)
        # No LLM call needed — just store in context
        return {"type": "initial_state", "latency_ms": latency_ms, "tokens_used": 0, "status": "ok"}

    def ingest_event(self, event: Event) -> dict:
        t0 = time.perf_counter()
        formatted = _format_event(event)
        self._context.append(formatted)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "latency_ms": latency_ms,
            "tokens_used": 0,  # no LLM call on ingest
            "status": "ok",
        }

    def ask(self, question_text: str) -> str:
        context_block = "\n\n---\n\n".join(self._context)
        user_message = f"{context_block}\n\n=== QUESTION ===\n{question_text}"
        t0 = time.perf_counter()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        answer = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return answer


def _format_event(event: Event) -> str:
    """Render an event as a plain-text block for inclusion in context."""
    p = event.payload
    ts = event.timestamp

    if event.event_type == "email":
        body = p.get("body", "").strip()
        return f"EMAIL [{ts}]\nFrom: {p.get('from','')}\nTo: {p.get('to','')}\nSubject: {p.get('subject','')}\n\n{body}"

    elif event.event_type == "sms":
        thread = p.get("thread", [])
        lines = "\n".join(f"  {m['from']}: {m['body']}" for m in thread)
        return f"TEXT MESSAGES [{ts}]\n{lines}"

    elif event.event_type == "user_instruction":
        return f"USER INSTRUCTION [{ts}]\n{p.get('instruction','').strip()}"

    elif event.event_type == "screenshot":
        return f"SCREENSHOT [{ts}]\nSource: {p.get('source_description','')}\n{p.get('image_description','').strip()}"

    elif event.event_type == "document":
        return f"DOCUMENT [{ts}]\nType: {p.get('document_type','')}\nTitle: {p.get('title','')}\n\n{p.get('content','').strip()}"

    elif event.event_type == "phone_call":
        parts = p.get("participants", [])
        names = ", ".join(f"{x.get('name','')} ({x.get('role','')})" for x in parts)
        return f"PHONE CALL [{ts}]\nParticipants: {names}\n{p.get('summary','').strip()}"

    elif event.event_type == "calendar_entry":
        return f"CALENDAR ENTRY [{ts}]\n{p.get('title','')} — starts {p.get('start','')}\nLocation: {p.get('location','')}\nNotes: {p.get('notes','')}"

    else:
        import json
        return f"EVENT [{event.event_type}] [{ts}]\n{json.dumps(p, indent=2)}"
