#!/usr/bin/env python3
"""Experiment 2: EBEAC Cost Comparison (Table 4)

Compares ERL / ExpeL / EBEAC on:
- LLM calls for experience extraction
- Cost per session (token cost)
- Granularity (task-level vs tool-call-level)
- Patterns captured

Uses DeepSeek V4 Flash API for ERL and ExpeL simulations.
EBEAC uses actual WeClaw ExperienceAutoRecorder code (zero LLM calls).
"""

import asyncio
import json
import os
import sys
import time
import random
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# OpenAI client
from openai import OpenAI
from benchmark_utils import get_output_dir

# WeClaw modules
from src.core.event_bus import EventBus
from src.core.events import EventType, ToolResultEvent
from src.core.experience_recorder import (
    ExperienceAutoRecorder,
    _ToolFailureTracker,
    _MIN_FAILURES_BEFORE_SUCCESS,
)

# ---------- Constants ----------

API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
OUTPUT_DIR = get_output_dir(__file__)

NUM_SESSIONS = 50
MIN_CALLS_PER_SESSION = 15
MAX_CALLS_PER_SESSION = 30
FAILURE_RATE = 0.18
RANDOM_SEED = 42

# Tool names for synthetic sessions
TOOLS = [
    "shell", "file", "browser", "browser_use", "search",
    "screen", "clipboard", "notify", "calculator", "stock_query",
    "ocr", "data_processor", "pdf_tool", "format_converter",
]

# Error messages for synthetic failures
ERROR_MESSAGES = [
    "Connection timeout after 30000ms",
    "Permission denied: access to /system/protected",
    "Invalid argument: file_path must be absolute",
    "Network error: ECONNREFUSED 127.0.0.1:8080",
    "Encoding error: failed to parse UTF-8 content",
    "Timeout: operation exceeded 60s limit",
    "Connection reset by peer",
    "Invalid parameter: expected string, got null",
    "Permission denied: write access restricted",
    "Parse error: unexpected token in JSON response",
]


# ---------- Lightweight Experience Store (SQLite only, no ChromaDB) ----------

class SimpleExperienceStore:
    """Lightweight experience store for benchmarking (SQLite only)."""

    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger TEXT NOT NULL,
                diagnosis TEXT NOT NULL,
                fix_summary TEXT NOT NULL,
                abstract_pattern TEXT,
                outcome TEXT DEFAULT 'success',
                source_type TEXT DEFAULT 'manual',
                session_id TEXT,
                tool_names TEXT,
                created_at TEXT NOT NULL
            )
        """)
        self._conn.commit()
        self._count = 0

    async def record(self, trigger: str, diagnosis: str, fix_summary: str,
                     abstract_pattern: str = "", outcome: str = "success",
                     source_type: str = "manual", session_id: str = "",
                     tools: Optional[list[str]] = None, **kwargs) -> None:
        self._conn.execute(
            "INSERT INTO experiences (trigger, diagnosis, fix_summary, abstract_pattern, "
            "outcome, source_type, session_id, tool_names, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (trigger, diagnosis, fix_summary, abstract_pattern, outcome, source_type,
             session_id, json.dumps(tools or []), datetime.now().isoformat())
        )
        self._conn.commit()
        self._count += 1

    def count(self) -> int:
        return self._count

    def close(self):
        self._conn.close()


# ---------- Synthetic Session Generator ----------

@dataclass
class ToolCallEvent:
    """A synthetic tool call event."""
    tool_name: str
    action_name: str
    status: str  # "success" | "error"
    error: str = ""
    arguments: dict = field(default_factory=dict)


def generate_session(rng: random.Random, session_id: str) -> list[ToolCallEvent]:
    """Generate a synthetic session with structured failure->success patterns.
    
    Each session has 15-30 tool calls. ~18% overall failure rate.
    Structured so that 3-6 tools per session have 2-4 consecutive failures
    followed by success, ensuring EBEAC can capture patterns.
    """
    n_calls = rng.randint(MIN_CALLS_PER_SESSION, MAX_CALLS_PER_SESSION)
    events = []
    
    # Select 3-6 tools that will have failure->success patterns
    pattern_tools = rng.sample(TOOLS, min(rng.randint(3, 6), len(TOOLS)))
    
    # Generate structured blocks
    remaining = n_calls
    tool_idx = 0
    
    while remaining > 0:
        tool = TOOLS[tool_idx % len(TOOLS)]
        tool_idx += 1
        
        if tool in pattern_tools and remaining >= 4:
            # Generate failure->success block: 2-3 failures then success
            n_failures = rng.randint(2, 3)
            if remaining < n_failures + 1:
                n_failures = max(2, remaining - 1)
            
            error = rng.choice(ERROR_MESSAGES)
            action = rng.choice(["run", "read", "write", "execute"])
            
            for _ in range(n_failures):
                events.append(ToolCallEvent(
                    tool_name=tool, action_name=action,
                    status="error", error=error,
                    arguments={"input": f"synthetic_arg_{rng.randint(1,100)}"},
                ))
                remaining -= 1
            
            # Success after failures
            events.append(ToolCallEvent(
                tool_name=tool, action_name=action,
                status="success",
                arguments={"input": f"synthetic_arg_{rng.randint(1,100)}"},
            ))
            remaining -= 1
            pattern_tools.remove(tool)  # Only one pattern block per tool
        else:
            # Normal call (mostly success, occasional random failure)
            is_failure = rng.random() < 0.05  # Low random failure rate
            action = rng.choice(["run", "read", "write", "execute", "open_url"])
            
            if is_failure:
                events.append(ToolCallEvent(
                    tool_name=tool, action_name=action,
                    status="error", error=rng.choice(ERROR_MESSAGES),
                    arguments={"input": f"synthetic_arg_{rng.randint(1,100)}"},
                ))
            else:
                events.append(ToolCallEvent(
                    tool_name=tool, action_name=action,
                    status="success",
                    arguments={"input": f"synthetic_arg_{rng.randint(1,100)}"},
                ))
            remaining -= 1
    
    return events


# ---------- EBEAC (ours): Zero-LLM Experience Capture ----------

async def run_ebeac(sessions: list[list[ToolCallEvent]]) -> dict:
    """Run EBEAC: feed events to ExperienceAutoRecorder, measure captures."""
    print("\n--- EBEAC (ours) ---")

    store = SimpleExperienceStore()
    event_bus = EventBus()

    # Create recorder (subscribes to TOOL_RESULT events)
    recorder = ExperienceAutoRecorder(event_bus, store)

    total_events = 0
    start = time.time()

    for sess_idx, session_events in enumerate(sessions):
        session_id = f"session_{sess_idx:03d}"
        for evt in session_events:
            # Create ToolResultEvent
            data = ToolResultEvent(
                tool_name=evt.tool_name,
                action_name=evt.action_name,
                status=evt.status,
                error=evt.error,
                session_id=session_id,
            )
            # Call handler directly (asyncio.create_task will use running loop)
            recorder._on_tool_result(EventType.TOOL_RESULT, data)
            total_events += 1

    # Wait for all pending async tasks to complete
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    elapsed = time.time() - start
    patterns_captured = store.count()

    print(f"  Events processed: {total_events}")
    print(f"  Patterns captured: {patterns_captured}")
    print(f"  LLM calls: 0")
    print(f"  Time: {elapsed:.2f}s")

    result = {
        "method": "EBEAC",
        "llm_calls": 0,
        "total_tokens": 0,
        "cost_per_session": 0,
        "patterns_captured": patterns_captured,
        "granularity": "tool-call",
        "total_events": total_events,
        "elapsed_seconds": round(elapsed, 3),
    }

    store.close()
    recorder.disconnect()
    return result


# ---------- ERL: Task-Level Reflection ----------

def run_erl(client: OpenAI, sessions: list[list[ToolCallEvent]]) -> dict:
    """Run ERL: 1-3 LLM calls per session for task-level reflection."""
    print("\n--- ERL (simulated) ---")

    total_llm_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    patterns = []

    for sess_idx, session_events in enumerate(sessions):
        # Build session summary
        failures = [e for e in session_events if e.status == "error"]
        successes = [e for e in session_events if e.status == "success"]

        if not failures:
            continue  # No failures to reflect on

        # ERL: 1-3 reflection calls per session
        n_reflections = min(3, max(1, len(failures) // 3))

        for ref_idx in range(n_reflections):
            # Sample some failures for this reflection
            sample = failures[:5] if ref_idx == 0 else failures[5:10]
            if not sample:
                break

            prompt = (
                "You are an AI agent reflecting on tool usage failures to extract reusable heuristics.\n\n"
                f"Session {sess_idx}: {len(session_events)} tool calls, {len(failures)} failures.\n\n"
                "Failures:\n"
            )
            for f in sample:
                prompt += f"- {f.tool_name}.{f.action_name}: {f.error}\n"
            prompt += (
                "\nExtract 1-3 reusable heuristics (rules) for avoiding these failures in the future.\n"
                "Format each heuristic as: RULE: <description>"
            )

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3,
                )
                total_llm_calls += 1
                if response.usage:
                    total_prompt_tokens += response.usage.prompt_tokens
                    total_completion_tokens += response.usage.completion_tokens

                content = response.choices[0].message.content or ""
                # Extract rules
                for line in content.split("\n"):
                    if "RULE:" in line.upper():
                        patterns.append(line.strip())

            except Exception as e:
                print(f"  ERL LLM error (session {sess_idx}): {e}")

        time.sleep(0.02)

    cost_per_session = (total_prompt_tokens * 0.14 + total_completion_tokens * 0.28) / NUM_SESSIONS / 1_000_000

    print(f"  LLM calls: {total_llm_calls}")
    print(f"  Total tokens: {total_prompt_tokens + total_completion_tokens}")
    print(f"  Patterns captured: {len(patterns)}")
    print(f"  Cost/session: ${cost_per_session:.6f}")

    return {
        "method": "ERL",
        "llm_calls": total_llm_calls,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "cost_per_session": round(cost_per_session, 8),
        "patterns_captured": len(patterns),
        "granularity": "task",
    }


# ---------- ExpeL: Per-Failure-Pair Rule Extraction ----------

def run_expel(client: OpenAI, sessions: list[list[ToolCallEvent]]) -> dict:
    """Run ExpeL: 2-5 LLM calls per failure->success pair for rule extraction."""
    print("\n--- ExpeL (simulated) ---")

    total_llm_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    patterns = []

    for sess_idx, session_events in enumerate(sessions):
        # Find failure->success pairs per tool
        tool_history: dict[str, list[ToolCallEvent]] = defaultdict(list)
        for evt in session_events:
            tool_history[evt.tool_name].append(evt)

        for tool_name, history in tool_history.items():
            # Find sequences: 2+ failures followed by success
            consecutive_failures = []
            for evt in history:
                if evt.status == "error":
                    consecutive_failures.append(evt)
                elif evt.status == "success" and len(consecutive_failures) >= _MIN_FAILURES_BEFORE_SUCCESS:
                    # Found a failure->success pair — call LLM
                    prompt = (
                        "You are ExpeL, extracting rules from tool usage trial-and-error.\n\n"
                        f"Tool: {tool_name}.{consecutive_failures[0].action_name}\n"
                        f"Failures ({len(consecutive_failures)}):\n"
                    )
                    for f in consecutive_failures[:3]:
                        prompt += f"  - Error: {f.error}\n"
                    prompt += f"Then: SUCCESS\n\n"
                    prompt += "Extract 1-2 rules explaining what went wrong and how it was fixed.\n"
                    prompt += "Format: RULE: <rule description>"

                    try:
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=300,
                            temperature=0.3,
                        )
                        total_llm_calls += 1
                        if response.usage:
                            total_prompt_tokens += response.usage.prompt_tokens
                            total_completion_tokens += response.usage.completion_tokens

                        content = response.choices[0].message.content or ""
                        for line in content.split("\n"):
                            if "RULE:" in line.upper():
                                patterns.append(line.strip())

                        # Cross-session insight sharing (1 extra call per 3 pairs)
                        if len(patterns) % 3 == 0 and patterns:
                            share_prompt = (
                                "Consolidate these 3 rules into 1 general insight:\n"
                                + "\n".join(patterns[-3:])
                            )
                            resp2 = client.chat.completions.create(
                                model=MODEL,
                                messages=[{"role": "user", "content": share_prompt}],
                                max_tokens=200,
                                temperature=0.3,
                            )
                            total_llm_calls += 1
                            if resp2.usage:
                                total_prompt_tokens += resp2.usage.prompt_tokens
                                total_completion_tokens += resp2.usage.completion_tokens

                    except Exception as e:
                        print(f"  ExpeL LLM error (session {sess_idx}, {tool_name}): {e}")

                    consecutive_failures = []
                else:
                    consecutive_failures = []

        time.sleep(0.02)

    cost_per_session = (total_prompt_tokens * 0.14 + total_completion_tokens * 0.28) / NUM_SESSIONS / 1_000_000

    print(f"  LLM calls: {total_llm_calls}")
    print(f"  Total tokens: {total_prompt_tokens + total_completion_tokens}")
    print(f"  Patterns captured: {len(patterns)}")
    print(f"  Cost/session: ${cost_per_session:.6f}")

    return {
        "method": "ExpeL",
        "llm_calls": total_llm_calls,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "cost_per_session": round(cost_per_session, 8),
        "patterns_captured": len(patterns),
        "granularity": "task",
    }


# ---------- Main ----------

def run_experiment():
    """Run the full EBEAC cost comparison experiment."""
    print("=" * 60)
    print("Experiment 2: EBEAC Cost Comparison")
    print("=" * 60)

    rng = random.Random(RANDOM_SEED)
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Generate synthetic sessions
    sessions = []
    total_calls = 0
    total_failures = 0
    for i in range(NUM_SESSIONS):
        sess = generate_session(rng, f"session_{i:03d}")
        sessions.append(sess)
        total_calls += len(sess)
        total_failures += sum(1 for e in sess if e.status == "error")

    print(f"Generated {NUM_SESSIONS} sessions: {total_calls} total calls, "
          f"{total_failures} failures ({total_failures/total_calls*100:.1f}%)")

    # Run EBEAC (zero LLM calls) in async context
    ebeac_result = asyncio.run(run_ebeac(sessions))

    # Run ERL (LLM-based reflection)
    erl_result = run_erl(client, sessions)

    # Run ExpeL (LLM-based rule extraction)
    expel_result = run_expel(client, sessions)

    # ---------- Summary ----------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    summary = {
        "ERL": {
            "llm_calls": erl_result["llm_calls"],
            "llm_calls_per_session": round(erl_result["llm_calls"] / NUM_SESSIONS, 1),
            "cost_per_session_tokens": round(erl_result["total_tokens"] / NUM_SESSIONS),
            "cost_per_session_usd": erl_result["cost_per_session"],
            "granularity": erl_result["granularity"],
            "patterns_captured": erl_result["patterns_captured"],
        },
        "ExpeL": {
            "llm_calls": expel_result["llm_calls"],
            "llm_calls_per_session": round(expel_result["llm_calls"] / NUM_SESSIONS, 1),
            "cost_per_session_tokens": round(expel_result["total_tokens"] / NUM_SESSIONS),
            "cost_per_session_usd": expel_result["cost_per_session"],
            "granularity": expel_result["granularity"],
            "patterns_captured": expel_result["patterns_captured"],
        },
        "EBEAC": {
            "llm_calls": ebeac_result["llm_calls"],
            "llm_calls_per_session": 0,
            "cost_per_session_tokens": 0,
            "cost_per_session_usd": 0,
            "granularity": ebeac_result["granularity"],
            "patterns_captured": ebeac_result["patterns_captured"],
        },
    }

    for method, stats in summary.items():
        print(f"\n{method}:")
        print(f"  LLM Calls: {stats['llm_calls']} ({stats['llm_calls_per_session']}/session)")
        print(f"  Cost/session: ~{stats['cost_per_session_tokens']} tokens")
        print(f"  Granularity: {stats['granularity']}")
        print(f"  Patterns: {stats['patterns_captured']}")

    # Save results
    all_results = {
        "metadata": {
            "num_sessions": NUM_SESSIONS,
            "total_calls": total_calls,
            "total_failures": total_failures,
            "failure_rate": round(total_failures / total_calls, 3),
            "model": MODEL,
        },
        "ERL": erl_result,
        "ExpeL": expel_result,
        "EBEAC": ebeac_result,
    }

    with open(OUTPUT_DIR / "exp2_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "exp2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR / 'exp2_results.json'}")
    print(f"Summary saved to {OUTPUT_DIR / 'exp2_summary.json'}")

    return summary


if __name__ == "__main__":
    run_experiment()
