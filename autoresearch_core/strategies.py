"""
autoresearch_core.strategies — Strategy selection and delta generation.

Provides the Strategy dataclass, UCB-based strategy selection, LLM output
parsing into Ops, and the generate_delta orchestration function.

Domain-specific strategy prompt templates are NOT included — each consumer
defines those and passes them to these generic functions.
"""

import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

from autoresearch_core.delta import Op, EditOp
from autoresearch_core.util import extract_json_object


# ---------------------------------------------------------------------------
# Strategy dataclass
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    """A named strategy with a prompt template and exploration weight."""
    name: str
    description: str
    prompt_template: str       # {content}, {constitution}, {target_path} are interpolated
    weight: float = 1.0        # UCB prior weight


# ---------------------------------------------------------------------------
# Unified Ops format instructions (appended to all strategy prompts)
# ---------------------------------------------------------------------------

OPS_FORMAT_INSTRUCTIONS = """
Output ONLY valid JSON:
{"ops": [
  {"kind": "edit_file", "path": "<file>", "search": "<exact text>", "replace": "<new text>"},
  {"kind": "create_file", "path": "<file>", "content": "<full file content>"},
  {"kind": "append_file", "path": "<file>", "text": "<text to append>"},
  {"kind": "delete_file", "path": "<file to remove>"},
  {"kind": "rename_file", "path": "<old file>", "new_path": "<new file>"}
]}

RULES:
- "search" must be an EXACT copy-paste substring from the content
- "search" must appear exactly once
- "replace" CANNOT be empty
- "path" must be relative to repo root
- For single-file edits, use only edit_file ops
- delete_file removes a file from the vault. Only use when merging content into
  another note — the content must land somewhere first (Priority 2: no net info loss).
  Before delete_file, use edit_file on all notes that reference the deleted file
  to update their wikilinks.
"""


# ---------------------------------------------------------------------------
# Strategy selection with UCB exploration
# ---------------------------------------------------------------------------

def select_strategies(pool: list[Strategy], n: int,
                      history: list[dict],
                      temperature: float = 1.0) -> list[Strategy]:
    """Select n strategies from the pool using UCB exploration bonus.

    Balances exploitation (strategies that have worked well) with exploration
    (strategies that haven't been tried enough). Temperature controls how
    much randomness to inject.

    Args:
        pool: available strategies.
        n: how many to select.
        history: list of AttemptRecord dicts.
        temperature: exploration temperature (higher = more random).

    Returns:
        List of n selected strategies (may repeat if pool is small).
    """
    n = min(n, len(pool))
    if not history:
        # No history — sample randomly
        return random.sample(pool, n)

    # Count successes and attempts per strategy
    attempts: Counter = Counter()
    successes: Counter = Counter()
    for record in history:
        name = record.get("strategy", "")
        attempts[name] += 1
        if record.get("outcome") == "adopted":
            successes[name] += 1

    total_attempts = sum(attempts.values()) or 1

    # Compute UCB score for each strategy
    scores: list[tuple[float, Strategy]] = []
    for strategy in pool:
        n_attempts = attempts.get(strategy.name, 0)
        if n_attempts == 0:
            # Never tried — assign high exploration bonus
            ucb = float('inf')
        else:
            win_rate = successes.get(strategy.name, 0) / n_attempts
            exploration = temperature * math.sqrt(
                2 * math.log(total_attempts) / n_attempts
            )
            ucb = win_rate + exploration
        scores.append((ucb, strategy))

    # Sort by UCB score descending, pick top n
    scores.sort(key=lambda x: x[0], reverse=True)

    # Add small random perturbation to break ties
    selected = [s for _, s in scores[:n]]
    return selected


# ---------------------------------------------------------------------------
# Delta generation
# ---------------------------------------------------------------------------

def generate_delta(target_path: str, content: str, strategy: Strategy,
                   constitution: str,
                   llm_fn: Callable[[str], str | None],
                   error_feedback: str = "",
                   extra_vars: dict[str, str] | None = None) -> Optional[list[Op]]:
    """Generate ops by calling an LLM with the strategy prompt.

    Args:
        target_path: relative path of the file being edited.
        content: current file content.
        strategy: the Strategy to use.
        constitution: constitution text for quality signals.
        llm_fn: callable that takes a prompt string and returns LLM output
            (or None on failure). Each consumer provides its own.
        error_feedback: if provided, appended to prompt for retry after dry-run failure.
        extra_vars: strategy-specific template variables.

    Returns:
        List of Op objects, or None on failure.
    """
    prompt = strategy.prompt_template
    # Interpolate extra vars first (strategy-specific placeholders)
    for key, value in (extra_vars or {}).items():
        prompt = prompt.replace(f"{{{key}}}", str(value))
    # Then standard vars
    prompt = (prompt
              .replace("{content}", content)
              .replace("{constitution}", constitution)
              .replace("{target_path}", target_path)
              .replace("{line_count}", str(len(content.splitlines()))))

    if error_feedback:
        prompt += f"\n\nPREVIOUS ATTEMPT FAILED:\n{error_feedback}\nFix the issues and try again.\n"

    try:
        output = llm_fn(prompt)
    except Exception as e:
        print(f"  LLM call failed for strategy '{strategy.name}': {e}", file=sys.stderr)
        return None

    if not output or len(output) < 10:
        return None

    return parse_ops(output, target_path)


# ---------------------------------------------------------------------------
# Op parsing
# ---------------------------------------------------------------------------

def parse_ops(output: str, default_path: str) -> Optional[list[Op]]:
    """Parse LLM output into a list of Op objects."""
    data = extract_json_object(output)
    if data is None:
        return None

    if not isinstance(data, dict) or "ops" not in data:
        return None

    raw_ops = data["ops"]
    if not isinstance(raw_ops, list):
        return None

    # First pass: collect raw ops
    raw_parsed = []
    for raw in raw_ops:
        if not isinstance(raw, dict) or "kind" not in raw:
            continue
        kind = raw["kind"]
        path = raw.get("path", default_path)

        if kind == "edit_file":
            search = raw.get("search", "")
            replace = raw.get("replace", "")
            if search and replace:
                raw_parsed.append(("edit_file", path, EditOp(search=search, replace=replace)))
        elif kind == "create_file":
            content = raw.get("content", "")
            if content:
                raw_parsed.append(("create_file", path, content))
        elif kind == "append_file":
            text = raw.get("text", "")
            if text:
                raw_parsed.append(("append_file", path, text))
        elif kind == "delete_file":
            if path:
                raw_parsed.append(("delete_file", path, None))
        elif kind == "rename_file":
            new_path = raw.get("new_path", "")
            if path and new_path:
                raw_parsed.append(("rename_file", path, new_path))

    if not raw_parsed:
        return None

    # Second pass: merge multiple edit_file ops on the same path into one Op
    # This prevents sequential-dependency failures where the second edit's
    # search text was copied from the pre-edit file
    ops = []
    edit_ops_by_path: dict[str, list[EditOp]] = {}

    for kind, path, payload in raw_parsed:
        if kind == "edit_file":
            edit_ops_by_path.setdefault(path, []).append(payload)
        elif kind == "create_file":
            ops.append(Op(kind="create_file", path=path, content=payload))
        elif kind == "append_file":
            ops.append(Op(kind="append_file", path=path, text=payload))
        elif kind == "delete_file":
            ops.append(Op(kind="delete_file", path=path))
        elif kind == "rename_file":
            ops.append(Op(kind="rename_file", path=path, new_path=payload))

    # Order: rename_file first (creates new paths), then edit_file (may target new paths),
    # then create/append/delete last
    rename_ops = [op for op in ops if op.kind == "rename_file"]
    other_ops = [op for op in ops if op.kind != "rename_file"]
    edit_ops_list = [Op(kind="edit_file", path=path, edits=edits)
                     for path, edits in edit_ops_by_path.items()]
    ops = rename_ops + edit_ops_list + other_ops

    return ops if ops else None
