"""
autoresearch_core.health — System health monitoring for evolution engines.

Detects degenerate training dynamics: strategy collapse, stagnation,
excessive vetoes, and other signs that the evolution loop is stuck.
"""

import math
from collections import Counter
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# HealthReport
# ---------------------------------------------------------------------------

@dataclass
class HealthReport:
    """Aggregated health diagnostics over a window of recent attempts."""
    warnings: list[str] = field(default_factory=list)

    def flag(self, message: str) -> None:
        """Add a warning to the report."""
        self.warnings.append(message)

    @property
    def healthy(self) -> bool:
        return len(self.warnings) == 0


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

def check_health(history: list[dict], window: int = 20) -> HealthReport:
    """Run all health checks over the most recent `window` attempts.

    Each dict in *history* is expected to have keys: 'outcome', 'strategy', 'target'.

    Args:
        history: list of AttemptRecord dicts (from load_history).
        window: how many recent records to examine.

    Returns:
        HealthReport with any warnings.
    """
    report = HealthReport()

    if not history:
        return report

    recent = history[-window:]

    _check_identity_win_rate(recent, report)
    _check_strategy_entropy(recent, report)
    _check_veto_rate(recent, report)
    _check_invalid_patch_rate(recent, report)
    _check_target_diversity(recent, report)

    return report


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_identity_win_rate(recent: list[dict], report: HealthReport) -> None:
    """Identity (no-change baseline) should win sometimes but not always.

    >80% identity wins = stagnation (deltas aren't beating the original).
    <10% identity wins = suspicious (deltas always win — maybe judge is biased).
    """
    if len(recent) < 5:
        return
    identity_wins = sum(1 for r in recent if r.get("outcome") == "identity_won")
    rate = identity_wins / len(recent)
    if rate > 0.80:
        report.flag(
            f"Stagnation: identity won {identity_wins}/{len(recent)} "
            f"({rate:.0%}) — deltas are not improving content"
        )
    elif rate < 0.10:
        report.flag(
            f"Suspicious: identity won only {identity_wins}/{len(recent)} "
            f"({rate:.0%}) — judge may be biased toward changes"
        )


def _check_strategy_entropy(recent: list[dict], report: HealthReport) -> None:
    """Strategy diversity measured by Shannon entropy.

    <0.5 bits = collapse (one strategy dominates).
    """
    if len(recent) < 5:
        return
    strategies = [r.get("strategy", "unknown") for r in recent]
    counts = Counter(strategies)
    total = sum(counts.values())

    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    if entropy < 0.5:
        top = counts.most_common(1)[0]
        report.flag(
            f"Strategy collapse: entropy={entropy:.2f} bits (<0.5). "
            f"'{top[0]}' used {top[1]}/{total} times"
        )


def _check_veto_rate(recent: list[dict], report: HealthReport) -> None:
    """High veto rate means strategies are producing bad edits.

    >40% vetoed = strategies are generating changes that fail gate checks.
    """
    if len(recent) < 5:
        return
    vetoed = sum(1 for r in recent if r.get("outcome") == "vetoed")
    rate = vetoed / len(recent)
    if rate > 0.40:
        report.flag(
            f"High veto rate: {vetoed}/{len(recent)} ({rate:.0%}) "
            f"— strategies are producing gate-failing edits"
        )


def _check_invalid_patch_rate(recent: list[dict], report: HealthReport) -> None:
    """High invalid patch rate means the LLM is producing malformed edits.

    >50% invalid = format issues in edit generation.
    """
    if len(recent) < 5:
        return
    invalid = sum(1 for r in recent if r.get("outcome") == "invalid")
    rate = invalid / len(recent)
    if rate > 0.50:
        report.flag(
            f"Format issues: {invalid}/{len(recent)} ({rate:.0%}) "
            f"patches were invalid — LLM may need clearer edit instructions"
        )


def _check_target_diversity(recent: list[dict], report: HealthReport) -> None:
    """Check that the loop isn't stuck on a single target.

    If >70% of recent attempts target the same file, flag it.
    """
    if len(recent) < 5:
        return
    targets = Counter(r.get("target", "") for r in recent)
    top_target, top_count = targets.most_common(1)[0]
    rate = top_count / len(recent)
    if rate > 0.70:
        report.flag(
            f"Target fixation: '{top_target}' targeted {top_count}/{len(recent)} "
            f"({rate:.0%}) — loop may be stuck on one target"
        )
