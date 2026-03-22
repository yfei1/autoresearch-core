"""autoresearch-core — Generic evolution engine primitives.

Provides: Op/Delta (atomic changes), GRPO ranking (Borda + advantages),
UCB strategy selection, health monitoring, state persistence, and
LLM output parsing utilities.

Domain-specific components (strategies, gates, judges, scoring, loops)
are NOT included — each consumer implements those.
"""

from autoresearch_core.delta import Op, Delta
from autoresearch_core.grpo import (
    IDENTITY_ID, RankingResult,
    build_diff_ranking_prompt,
    aggregate_borda, compute_advantages, grpo_rank,
)
from autoresearch_core.health import HealthReport, check_health
from autoresearch_core.state import AttemptRecord, append_history, load_history, save_generation_metadata
from autoresearch_core.strategies import Strategy, OPS_FORMAT_INSTRUCTIONS, select_strategies, generate_delta
from autoresearch_core.util import (
    strip_markdown_fences, extract_json_object, extract_json_array,
    detect_overlaps, Overlap,
)

__all__ = [
    "Op", "Delta",
    "IDENTITY_ID", "RankingResult",
    "build_diff_ranking_prompt",
    "aggregate_borda", "compute_advantages", "grpo_rank",
    "HealthReport", "check_health",
    "AttemptRecord", "append_history", "load_history", "save_generation_metadata",
    "Strategy", "OPS_FORMAT_INSTRUCTIONS", "select_strategies", "generate_delta",
    "strip_markdown_fences", "extract_json_object", "extract_json_array",
    "detect_overlaps", "Overlap",
]
