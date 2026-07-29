"""Structured training-event detection for the dashboard (issue #271).

Events are derived passively from existing artifacts — train_stats dicts, log
lines, and metric points — so detection does not modify the trainer, rollout
engine, or stored metric data. Each event carries a bounded log excerpt (never
full training samples) to support click-through diagnostics in the dashboard.

The public entry point is :func:`detect_events`, a pure function over plain
data so it can be exercised by CPU tests without a GPU or a running dashboard.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Iterable

# Event types are part of the dashboard/API contract; keep stable and lowercase.
EVENT_TYPES = ("non_finite", "oom", "invalid_batch", "constant_reward")

# Severity categorizes how actionable an event is. "warning" = the step likely
# learned nothing but the run continues; "error" = data is corrupt or the step
# failed (NaN/Inf, OOM).
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"

# Number of surrounding log lines kept as the bounded excerpt for click-through.
# Deliberately small so the dashboard never surfaces full training samples.
EXCERPT_RADIUS = 2


def _is_finite_number(value: Any) -> bool:
    """True when value is a real, finite number; False for NaN/Inf/non-numeric."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _scan_stats_for_non_finite(stats: dict[str, Any]) -> list[str]:
    """Return the train_stats field names that hold non-finite numeric values."""
    offenders: list[str] = []
    for key, value in stats.items():
        # Only scalar-looking fields are relevant; skip nested dicts/lists.
        if isinstance(value, (dict, list, tuple)):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            offenders.append(key)
    return offenders


def _step_from_stats(stats: dict[str, Any], default: int) -> int:
    try:
        return int(stats.get("step", default))
    except (TypeError, ValueError):
        return default


def _make_event(
    *,
    step: int,
    event_type: str,
    severity: str,
    message: str,
    log_excerpt: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "step": step,
        "type": event_type,
        "severity": severity,
        "message": message,
        # Bounded excerpt: keep None when no log context is available so the
        # dashboard can distinguish "no logs captured" from "empty excerpt".
        "log_excerpt": list(log_excerpt) if log_excerpt is not None else None,
    }


def _bounded_excerpt(log_lines: list[str], index: int, radius: int = EXCERPT_RADIUS) -> list[str]:
    """Return up to `radius` lines on each side of `index`, clamped to bounds."""
    start = max(0, index - radius)
    end = min(len(log_lines), index + radius + 1)
    return log_lines[start:end]


def detect_events(
    *,
    train_stats_rows: Iterable[dict[str, Any]] | None = None,
    log_lines: Iterable[str] | None = None,
    metric_points: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Detect structured training events from existing artifacts.

    Inputs are plain data already produced by AReno (train_stats dicts emitted
    per step, trainer log lines, and scalar metric points). Detection never
    mutates them. The returned events are ordered by step then type so the
    dashboard can render markers left-to-right deterministically.

    Event types:
      - non_finite: a train_stats numeric field is NaN/Inf (error).
      - oom: a log line reports CUDA out-of-memory (error).
      - invalid_batch: a step's gradient is effectively all-zero, or a log line
        reports a train skip / degenerate batch (warning).
      - constant_reward: a metric point reports rollout/rewards_std below eps,
        i.e. every rollout in the group got the same reward (warning).
    """
    stats_rows = list(train_stats_rows or [])
    logs = list(log_lines or [])
    points = list(metric_points or [])
    events: list[dict[str, Any]] = []

    # --- non_finite + invalid_batch from train_stats ---
    for stats in stats_rows:
        if not isinstance(stats, dict):
            continue
        step = _step_from_stats(stats, 0)

        offenders = _scan_stats_for_non_finite(stats)
        if offenders:
            events.append(
                _make_event(
                    step=step,
                    event_type="non_finite",
                    severity=SEVERITY_ERROR,
                    message=f"non-finite value(s) in train_stats: {', '.join(sorted(offenders))}",
                )
            )

        grad_zero_ratio = stats.get("grad_zero_ratio")
        if _is_finite_number(grad_zero_ratio) and float(grad_zero_ratio) > 0.99:
            events.append(
                _make_event(
                    step=step,
                    event_type="invalid_batch",
                    severity=SEVERITY_WARNING,
                    message="gradient effectively all-zero this step (no learning signal)",
                )
            )

    # --- oom + invalid_batch(skip) from log lines ---
    oom_markers = ("out of memory", "outofmemory", "cuda error", "alloc")
    skip_markers = ("stage=train_skip", "degenerate batch", "skipped")
    for index, line in enumerate(logs):
        low = line.lower()
        if any(marker in low for marker in oom_markers):
            # Heuristic step: try to find a step=NN on the same log line.
            step = _step_from_log_line(line)
            events.append(
                _make_event(
                    step=step,
                    event_type="oom",
                    severity=SEVERITY_ERROR,
                    message="CUDA out-of-memory reported during execution",
                    log_excerpt=_bounded_excerpt(logs, index),
                )
            )
        elif any(marker in low for marker in skip_markers):
            step = _step_from_log_line(line)
            events.append(
                _make_event(
                    step=step,
                    event_type="invalid_batch",
                    severity=SEVERITY_WARNING,
                    message="batch skipped or degenerate (no update performed)",
                    log_excerpt=_bounded_excerpt(logs, index),
                )
            )

    # --- constant_reward from metric points (rollout/rewards_std ~ 0) ---
    for point in points:
        name = str(point.get("name") or "")
        if name != "rollout/rewards_std":
            continue
        value = point.get("value")
        if not _is_finite_number(value):
            continue
        if float(value) < 1e-8:
            step = int(point.get("step") or 0)
            events.append(
                _make_event(
                    step=step,
                    event_type="constant_reward",
                    severity=SEVERITY_WARNING,
                    message="reward std ~ 0: every rollout in the group got the same reward",
                )
            )

    events.sort(key=lambda event: (event["step"], event["type"]))
    return events


def _step_from_log_line(line: str) -> int:
    """Best-effort parse of `step=<n>` from a trainer log line; 0 if absent."""
    import re

    match = re.search(r"step=(\d+)", line)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return 0
    return 0


def filter_events(events: Iterable[dict[str, Any]], types: Iterable[str] | None) -> list[dict[str, Any]]:
    """Keep only events whose type is in `types`; None/empty means all types.

    This is the dashboard's independent event filter (independent of which
    metric curve is shown, per issue #271). Invalid types are ignored rather
    than raising, so a stale client filter never blanks the overlay.
    """
    if not types:
        return list(events)
    allowed = {t for t in types if t in EVENT_TYPES}
    return [event for event in events if event["type"] in allowed]


# Regex for "... step=<n> ... train_stats=<dict repr> ...". The dict repr is the
# Python ``%s`` of a dict, so values may include ``nan``/``inf``/single quotes,
# which ``ast.literal_eval`` handles (Unknown node handling below also tolerates
# bare ``nan``/``inf`` that literal_eval would reject on some inputs).
_TRAIN_STATS_RE = re.compile(r"step=(\d+)\s+train_stats=(\{.*\})")


def parse_train_stats_from_logs(log_lines: Iterable[str]) -> list[dict[str, Any]]:
    """Extract per-step train_stats dicts from trainer log lines.

    The trainer logs ``step=<n> train_stats=<dict>`` per update. The dict is a
    Python repr (not JSON): single-quoted keys, possibly ``nan``/``inf``.
    ``ast.literal_eval`` parses it after normalizing bare ``nan``/``inf`` tokens
    to ``float('nan')``-compatible forms. Unparseable lines are skipped, never
    raised — a best-effort view, not a contract enforcement.
    """
    rows: list[dict[str, Any]] = []
    for line in log_lines:
        match = _TRAIN_STATS_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        literal = match.group(2)
        # ast.literal_eval rejects bare nan/inf; replace them with float(...) so
        # the eval sees valid Python. Conservative: only replace token forms.
        literal = re.sub(r"\bNaN\b", "'__nan__'", literal)
        literal = re.sub(r"\bnan\b", "'__nan__'", literal, flags=re.IGNORECASE)
        literal = re.sub(r"\bInf\b", "'__inf__'", literal)
        literal = re.sub(r"\binf\b", "'__inf__'", literal, flags=re.IGNORECASE)
        try:
            parsed = ast.literal_eval(literal)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(parsed, dict):
            continue
        # Restore NaN/Inf as real floats so downstream _scan_stats_for_non_finite
        # treats them as the non-finite events they represent.
        restored: dict[str, Any] = {"step": step}
        for key, value in parsed.items():
            if value == "__nan__":
                restored[key] = float("nan")
            elif value == "__inf__":
                restored[key] = float("inf")
            else:
                restored[key] = value
        rows.append(restored)
    return rows


def detect_events_from_job_artifacts(
    *,
    log_lines: Iterable[str] | None = None,
    metric_points: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper: parse train_stats from logs, then detect_events.

    This is what the dashboard server calls — it already holds job.logs and
    job.metrics as plain data, so detection stays a view over existing
    artifacts (no trainer change, no mutation of metric data).
    """
    logs = list(log_lines or [])
    stats_rows = parse_train_stats_from_logs(logs)
    return detect_events(train_stats_rows=stats_rows, log_lines=logs, metric_points=metric_points)