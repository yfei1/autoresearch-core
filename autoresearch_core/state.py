"""
autoresearch_core.state — Persistence for evolution history and generation metadata.

Stores per-attempt records in a JSONL file and per-generation metadata as JSON.
Each consumer provides its own state_dir.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class AttemptRecord:
    """One row in the history ledger — records a single delta attempt."""
    generation: int
    target: str              # e.g. "ml-systems/attention-mechanics.md"
    strategy: str            # e.g. "densify"
    delta_id: str            # UUID of the delta
    outcome: str             # "adopted" | "vetoed" | "invalid" | "identity_won"
    rank: int = -1           # final rank (1 = best)
    advantage: float = 0.0   # GRPO advantage score
    veto_reason: str = ""    # why the gate rejected it, if applicable
    num_edits: int = 0       # number of EditOps in the delta
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def append_history(record: AttemptRecord, state_dir: Path) -> None:
    """Append a single AttemptRecord to the JSONL history file."""
    history_file = state_dir / "history.jsonl"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_history(state_dir: Path) -> list[dict]:
    """Load all history records as a list of dicts."""
    history_file = state_dir / "history.jsonl"
    if not history_file.exists():
        return []
    records = []
    with open(history_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def save_generation_metadata(generation: int, metadata: dict, state_dir: Path) -> None:
    """Save metadata for a specific generation as a JSON file."""
    generations_dir = state_dir / "generations"
    generations_dir.mkdir(parents=True, exist_ok=True)
    path = generations_dir / f"gen_{generation:04d}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
