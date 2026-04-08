"""
Life Test runner.

Usage:
    python -m runner.runner --config configs/run_config.yaml
    python -m runner.runner --situation level_1_alex --system claude_sonnet_baseline
"""
from __future__ import annotations
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import yaml

from runner.models import (
    Event, Question, QuestionScore, PhaseResult,
    SituationResult, RubricPoint, ScoringConfig,
)
from runner.scorer import Scorer
from runner.adapters.base import BaseAdapter


# ── Loader helpers ─────────────────────────────────────────────────────────────

def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_situation(situation_id: str) -> dict:
    base = Path("situations") / situation_id
    initial_state_path = base / "initial_state.md"
    sequence_path = base / "sequence.yaml"
    return {
        "id": situation_id,
        "base_path": base,
        "initial_state": initial_state_path.read_text(),
        "sequence": load_yaml(sequence_path),
    }


def load_event(ref: str, base_path: Path) -> Event:
    data = load_yaml(base_path / ref)
    return Event(
        event_id=data["event_id"],
        event_type=data["event_type"],
        timestamp=data.get("timestamp", ""),
        payload=data["payload"],
    )


def parse_question(q: dict) -> Question:
    scoring_raw = q["scoring"]
    rubric_points = [
        RubricPoint(criterion=rp["criterion"], points=rp["points"])
        for rp in scoring_raw.get("rubric_points", [])
    ]
    scoring = ScoringConfig(
        type=scoring_raw["type"],
        threshold=scoring_raw.get("threshold", 0.85),
        rubric_points=rubric_points,
    )
    return Question(
        question_id=q["question_id"],
        text=q["text"],
        expected_answer=q["expected_answer"],
        difficulty=q["difficulty"],
        points=q["points"],
        scoring=scoring,
    )


# ── Adapter factory ────────────────────────────────────────────────────────────

def load_adapter(system_config: dict) -> BaseAdapter:
    adapter_name = system_config["adapter"]
    if adapter_name == "claude_local":
        from runner.adapters.claude_local import ClaudeLocalAdapter
        return ClaudeLocalAdapter(system_config)
    if adapter_name == "claude_mempalace":
        from runner.adapters.claude_mempalace import ClaudeMemPalaceAdapter
        return ClaudeMemPalaceAdapter(system_config)
    raise NotImplementedError(f"Adapter '{adapter_name}' not yet implemented. "
                              f"See runner/adapters/base.py to implement it.")


# ── Core runner ────────────────────────────────────────────────────────────────

class LifeTestRunner:

    def __init__(self, run_config: dict, systems_config: dict, results_dir: Path):
        self.run_config = run_config
        self.systems_config = {s["system_id"]: s for s in systems_config["systems"]}
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> list[SituationResult]:
        all_results = []
        for situation_id in self.run_config["situations"]:
            for system_id in self.run_config["systems"]:
                result = self.run_one(situation_id, system_id)
                all_results.append(result)
        return all_results

    def run_one(self, situation_id: str, system_id: str) -> SituationResult:
        print(f"\n{'='*60}")
        print(f"  Situation: {situation_id}  |  System: {system_id}")
        print(f"{'='*60}")

        situation = load_situation(situation_id)
        sys_config = self.systems_config[system_id]
        adapter = load_adapter(sys_config)

        if self.run_config.get("options", {}).get("reset_memory_between_situations", True):
            adapter.reset()

        # Feed initial state
        print("  [+] Ingesting initial state...")
        adapter.ingest_initial_state(situation["initial_state"])

        result = SituationResult(
            situation_id=situation_id,
            system_id=system_id,
            total_possible=situation["sequence"]["total_possible_points"],
        )

        ingestion_log_path = self._result_path(situation_id, system_id) / "ingestion_log.jsonl"
        ingestion_log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(ingestion_log_path, "w") as log_f:
            for phase in situation["sequence"]["phases"]:
                phase_result = self._run_phase(phase, situation, adapter, log_f)
                result.phases.append(phase_result)
                result.total_earned += sum(s.earned for s in phase_result.scores)

        self._save_scores(result)
        print(f"\n  Score: {result.total_earned}/{result.total_possible} ({result.percent}%)")
        return result

    def _run_phase(self, phase: dict, situation: dict, adapter: BaseAdapter, log_f) -> PhaseResult:
        phase_result = PhaseResult(phase_id=phase["phase_id"], phase_type=phase["type"])
        label = phase.get("label", phase["phase_id"])

        if phase["type"] == "event_section":
            print(f"\n  [events] {label}")
            for event_ref in phase["events"]:
                event = load_event(event_ref["ref"], situation["base_path"])
                entry = adapter.ingest_event(event)
                entry["phase_id"] = phase["phase_id"]
                log_f.write(json.dumps(entry) + "\n")
                print(f"    ingested: {event.event_id} ({event.event_type})")
                phase_result.ingestion_entries.append(entry)

        elif phase["type"] == "question_section":
            print(f"\n  [questions] {label}")
            timeout = self.run_config.get("options", {}).get("question_timeout_seconds", 30)
            for q_raw in phase["questions"]:
                question = parse_question(q_raw)
                print(f"    Q {question.question_id} (difficulty={question.difficulty}, {question.points}pts)...", end=" ", flush=True)
                raw_answer = adapter.ask(question.text)
                score = Scorer.score(question, raw_answer)
                phase_result.scores.append(score)
                status = "PASS" if score.passed else "FAIL"
                print(f"{score.earned}/{score.possible} [{status}]")

        return phase_result

    def _result_path(self, situation_id: str, system_id: str) -> Path:
        return self.results_dir / f"{situation_id}__{system_id}"

    def _save_scores(self, result: SituationResult) -> None:
        path = self._result_path(result.situation_id, result.system_id) / "scores.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "situation_id": result.situation_id,
            "system_id": result.system_id,
            "total_possible": result.total_possible,
            "total_earned": result.total_earned,
            "percent": result.percent,
            "by_difficulty": result.by_difficulty(),
            "phases": [
                {
                    "phase_id": p.phase_id,
                    "questions": [
                        {
                            "question_id": s.question_id,
                            "text": s.text,
                            "difficulty": s.difficulty,
                            "possible": s.possible,
                            "earned": s.earned,
                            "passed": s.passed,
                            "raw_answer": s.raw_answer,
                            "breakdown": s.breakdown,
                        }
                        for s in p.scores
                    ],
                }
                for p in result.phases
                if p.scores
            ],
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=2)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Life Test runner")
    parser.add_argument("--config", default="configs/run_config.yaml")
    parser.add_argument("--situation", help="Override: run a single situation")
    parser.add_argument("--system", help="Override: run a single system")
    args = parser.parse_args()

    run_config = load_yaml(args.config)
    systems_config = load_yaml("configs/systems.yaml")

    if args.situation:
        run_config["situations"] = [args.situation]
    if args.system:
        run_config["systems"] = [args.system]

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    results_dir = Path("results") / run_id
    print(f"Run ID: {run_id}")

    runner = LifeTestRunner(run_config, systems_config, results_dir)
    results = runner.run_all()

    # Write run manifest
    manifest = {
        "run_id": run_id,
        "run_label": run_config.get("run_label", ""),
        "situations": run_config["situations"],
        "systems": run_config["systems"],
        "summary": {
            f"{r.situation_id}__{r.system_id}": {
                "possible": r.total_possible,
                "earned": r.total_earned,
                "percent": r.percent,
            }
            for r in results
        },
    }
    with open(results_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nResults written to: {results_dir}/")


if __name__ == "__main__":
    main()
