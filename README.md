# Life Test

A benchmark for evaluating **LLM + Memory system combinations** for use as personal AI assistants.

## Goal

Find memory architectures that keep an agent reliably informed across a realistic, ongoing stream of life events — emails, texts, instructions, documents, screenshots — without requiring the user to repeat themselves.

## Repository layout

```
life_test/
├── .env                     # API keys (git-ignored)
├── .env.example             # Template — copy to .env and fill in
├── requirements.txt
├── situations/
│   ├── level_1_alex/        # Single adult — 95 pts
│   │   ├── initial_state.md
│   │   ├── sequence.yaml
│   │   └── events/
│   └── level_4_marsh/       # Complex family + 2 businesses — 165 pts
│       ├── initial_state.md
│       ├── sequence.yaml
│       └── events/
├── runner/
│   ├── runner.py            # Orchestration loop
│   ├── scorer.py            # Exact / semantic / rubric scoring
│   ├── models.py            # Shared dataclasses
│   └── adapters/
│       ├── base.py          # Abstract interface every adapter implements
│       ├── claude_local.py  # Baseline: context-window only
│       └── claude_mempalace.py
├── configs/
│   ├── systems.yaml         # LLM + Memory system definitions
│   └── run_config.yaml      # Which situations/systems to run
└── results/                 # One timestamped directory per run
```

---

## How a test works

Each situation runs as a sequence of alternating phases:

1. **Event phase** — the runner feeds events (emails, SMS, instructions, documents, screenshots) to the system under test. No questions yet.
2. **Question phase** — the runner asks questions that require recalling and combining information from prior events. Answers are scored.

This repeats across multiple phases within a situation, so later questions may require synthesising information from events that arrived much earlier.

The system under test sees:
- `initial_state.md` once at the start (background on the person's life)
- Each event as it arrives
- Each question in isolation (no access to previous questions or answers)

---

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY
```

---

## Running

### Run everything in run_config.yaml

```bash
python -m runner.runner
```

### Run a specific situation

```bash
python -m runner.runner --situation level_1_alex
```

### Run a specific system

```bash
python -m runner.runner --system claude_haiku_baseline
```

### Run a single situation against a single system

```bash
python -m runner.runner --situation level_1_alex --system claude_sonnet_mempalace
```

Results are written to `results/run_<timestamp>/`. Each run produces:

```
results/run_20260408_143022/
├── run_manifest.json                        # summary across all combinations
├── level_1_alex__claude_haiku_baseline/
│   ├── scores.json                          # per-question scores + breakdowns
│   └── ingestion_log.jsonl                  # one entry per ingested event
└── level_1_alex__claude_sonnet_mempalace/
    ├── scores.json
    └── ingestion_log.jsonl
```

`scores.json` includes per-question breakdowns (difficulty, points earned/possible, raw answer, rubric criterion results). `run_manifest.json` has a top-level summary table for quick comparison.

---

## Scoring

Questions use one of three strategies, chosen per-question in `sequence.yaml`:

| Strategy | How it works | Cost |
|---|---|---|
| `exact` | Substring match | Free |
| `exact_or_semantic` | Cosine similarity via local embeddings, with configurable threshold | Free |
| `rubric` | Each criterion evaluated independently by an LLM judge | ~$0.001/question |

The rubric judge uses `claude-haiku-4-5` with `max_tokens=10` (YES/NO only) to keep costs low. Questions are scored with partial credit when criteria are partially met.

---

## Active systems

Defined in `configs/systems.yaml`, toggled in `configs/run_config.yaml`.

| system_id | LLM | Memory | Notes |
|---|---|---|---|
| `claude_haiku_baseline` | Haiku 4.5 | None (context window) | Cheapest baseline |
| `claude_sonnet_baseline` | Sonnet 4.6 | None (context window) | Stronger baseline |
| `claude_haiku_mempalace` | Haiku 4.5 | MemPalace (ChromaDB) | Low cost + memory |
| `claude_sonnet_mempalace` | Sonnet 4.6 | MemPalace (ChromaDB) | Primary test target |

To enable a system, add its `system_id` to the `systems` list in `run_config.yaml`.

---

## Situations

| ID | Complexity | Points | Description |
|---|---|---|---|
| `level_1_alex` | 1 | 95 | Single adult. Job, cat, finances, a move. |
| `level_4_marsh` | 4 | 165 | Family of 4, rental property, 2 businesses, contractors, medical, school. |

---

## Adding a new memory system

### Step 1 — Create the adapter

Create `runner/adapters/your_system.py`. It must implement the four methods in `BaseAdapter`:

```python
from runner.adapters.base import BaseAdapter
from runner.models import Event

class YourSystemAdapter(BaseAdapter):

    def __init__(self, config: dict):
        self._system_id = config["system_id"]
        # Initialise your LLM client and memory system here.
        # config["llm"] has model, max_tokens, temperature.
        # config["memory"]["config"] has your memory-system-specific settings.

    @property
    def system_id(self) -> str:
        return self._system_id

    def reset(self) -> None:
        # Clear all stored memory for a fresh situation.
        # Called automatically before each situation when
        # reset_memory_between_situations is true in run_config.yaml.
        ...

    def ingest_initial_state(self, content: str) -> dict:
        # content is the full text of initial_state.md.
        # Store it however your memory system works.
        # Return a dict with at minimum: {tokens_used: int, latency_ms: int, status: str}
        ...

    def ingest_event(self, event: Event) -> dict:
        # event.event_id, event.event_type, event.timestamp, event.payload
        # Store the event in your memory system.
        # Return a dict with at minimum:
        #   {event_id: str, event_type: str, tokens_used: int, latency_ms: int, status: str}
        ...

    def ask(self, question_text: str) -> str:
        # Retrieve relevant context from memory, call your LLM, return the answer string.
        ...
```

The `event.payload` structure varies by event type. Use `runner/adapters/claude_mempalace.py` as a reference for how to format different event types into plain text.

### Step 2 — Register the adapter

In `runner/runner.py`, add a branch in `load_adapter()`:

```python
if adapter_name == "your_adapter_name":
    from runner.adapters.your_system import YourSystemAdapter
    return YourSystemAdapter(system_config)
```

### Step 3 — Add a system definition

In `configs/systems.yaml`, add an entry under `systems`:

```yaml
- system_id: your_system_id
  label: "Human-readable name"
  adapter: your_adapter_name        # matches the branch you added in runner.py
  llm:
    provider: anthropic             # or openai, etc.
    model: claude-sonnet-4-6
    temperature: 0.0
    max_tokens: 2048
  memory:
    provider: your_memory_provider
    config:
      some_key: some_value          # whatever your adapter reads from config["memory"]["config"]
  notes: >
    One-line description of this system's approach.
```

### Step 4 — Enable it in run_config.yaml

```yaml
systems:
  - claude_haiku_baseline
  - your_system_id       # add here
```

Then run:

```bash
python -m runner.runner --system your_system_id
```

### Conventions

- **Zero ingest tokens is the ideal.** Store events without LLM calls if your memory system allows it. Cost is measured by what you spend at question time.
- **Use `config["memory"]["config"]` for all memory-system-specific settings** (API keys, paths, collection names). Keep secrets in `.env` and read them with `os.environ`.
- **Keep the adapter stateless between questions.** The runner calls `ask()` once per question and expects no side effects.
- **Isolation.** Use `situation_id` in collection names / paths / user IDs so situations don't bleed into each other. The `reset()` method must guarantee a clean slate.
