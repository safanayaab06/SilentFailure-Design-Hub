"""Explainable, configurable risk rules for SilentFailure telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Thresholds:
    latency_ms: float = 250.0
    error_rate_pct: float = 2.0
    memory_pct: float = 82.0
    lag_events_5_readings: int = 4
    deterioration_pct: float = 6.0


METRIC_FIELDS = ("latency_ms", "error_rate_pct", "memory_pct", "lag_events")
DISPLAY_NAMES = {
    "latency_ms": "Response latency",
    "error_rate_pct": "Error rate",
    "memory_pct": "Memory utilization",
    "lag_events": "Freeze / lag events",
}


def validate_reading(reading: Mapping[str, Any]) -> dict[str, float | int]:
    """Validate and normalize one telemetry reading, raising ValueError on bad data."""
    if not isinstance(reading, Mapping):
        raise ValueError("A telemetry reading must be a mapping.")

    normalized: dict[str, float | int] = {}
    for field in METRIC_FIELDS:
        if field not in reading:
            raise ValueError(f"Missing required telemetry field: {field}.")
        value = reading[field]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field} must be a finite number.")
        number = float(value)
        if not isfinite(number):
            raise ValueError(f"{field} must be a finite number.")
        normalized[field] = number

    if normalized["latency_ms"] < 0 or normalized["latency_ms"] > 20_000:
        raise ValueError("latency_ms must be between 0 and 20,000.")
    if not 0 <= normalized["error_rate_pct"] <= 100:
        raise ValueError("error_rate_pct must be between 0 and 100.")
    if not 0 <= normalized["memory_pct"] <= 100:
        raise ValueError("memory_pct must be between 0 and 100.")
    if normalized["lag_events"] < 0 or not float(normalized["lag_events"]).is_integer():
        raise ValueError("lag_events must be a non-negative whole number.")

    normalized["lag_events"] = int(normalized["lag_events"])
    return normalized


def _number(value: float, unit: str, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}{unit}"


def _trend_for(
    field: str, values: Sequence[Mapping[str, Any]]
) -> tuple[str, float, float]:
    if len(values) < 2:
        return "not enough readings", 0.0, 0.0

    start = float(values[0][field])
    end = float(values[-1][field])
    delta = end - start
    floor = {"latency_ms": 25.0, "error_rate_pct": 0.5, "memory_pct": 4.0}.get(
        field, 0.0
    )
    relative_change = delta / max(abs(start), 0.1) * 100
    if delta > 0 and delta >= floor:
        return "rising", delta, relative_change
    if delta < 0 and abs(delta) >= floor:
        return "falling", delta, relative_change
    return "stable", delta, relative_change


def assess_telemetry(
    current: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] | None = None,
    thresholds: Thresholds | None = None,
) -> dict[str, Any]:
    """Return triggered rules, a transparent score, status, reasons, and actions.

    The score is a count of triggered rules, not a probability of failure.
    `history` should include the current reading as its last item.
    """
    configured = thresholds or Thresholds()
    current_reading = validate_reading(current)
    raw_history = list(history or [current])
    if not raw_history or raw_history[-1] is not current:
        raw_history.append(current)
    recent = [validate_reading(item) for item in raw_history[-5:]]
    if not recent or recent[-1] != current_reading:
        recent.append(current_reading)
        recent = recent[-5:]

    reasons: list[dict[str, Any]] = []

    def add_reason(
        key: str,
        metric: str,
        observed: str,
        threshold: str,
        trend: str,
        why: str,
        action: str,
    ) -> None:
        reasons.append(
            {
                "key": key,
                "metric": metric,
                "observed": observed,
                "threshold": threshold,
                "trend": trend,
                "why": why,
                "action": action,
            }
        )

    if current_reading["latency_ms"] > configured.latency_ms:
        trend, _, _ = _trend_for("latency_ms", recent)
        add_reason(
            "latency_ms",
            DISPLAY_NAMES["latency_ms"],
            _number(current_reading["latency_ms"], " ms", 0),
            f"{configured.latency_ms:.0f} ms",
            trend,
            "Slow responses can make key screens feel stuck and may expose a network or main-thread bottleneck.",
            "Inspect slow API calls, network timing, and main-thread work.",
        )

    if current_reading["error_rate_pct"] > configured.error_rate_pct:
        trend, _, _ = _trend_for("error_rate_pct", recent)
        add_reason(
            "error_rate_pct",
            DISPLAY_NAMES["error_rate_pct"],
            _number(current_reading["error_rate_pct"], "%", 2),
            f"{configured.error_rate_pct:.2f}%",
            trend,
            "A growing share of failed requests can point to a regression or an unreliable dependency.",
            "Review exception logs, failing requests, and the latest release.",
        )

    if current_reading["memory_pct"] > configured.memory_pct:
        trend, _, _ = _trend_for("memory_pct", recent)
        add_reason(
            "memory_pct",
            DISPLAY_NAMES["memory_pct"],
            _number(current_reading["memory_pct"], "%"),
            f"{configured.memory_pct:.1f}%",
            trend,
            "High memory pressure can trigger slowdowns, termination, or an unstable user session.",
            "Compare memory snapshots and look for sustained growth or retained objects.",
        )

    lag_total = sum(int(item["lag_events"]) for item in recent)
    if lag_total >= configured.lag_events_5_readings:
        prior_lags = [int(item["lag_events"]) for item in recent[:-1]]
        if not prior_lags or int(current_reading["lag_events"]) == sum(prior_lags) / len(prior_lags):
            lag_trend = "stable"
        elif int(current_reading["lag_events"]) > sum(prior_lags) / len(prior_lags):
            lag_trend = "rising"
        else:
            lag_trend = "falling"
        add_reason(
            "lag_events",
            DISPLAY_NAMES["lag_events"],
            f"{lag_total} events across {len(recent)} readings",
            f"{configured.lag_events_5_readings} events / 5 readings",
            lag_trend,
            "Repeated freezes or frame stalls point to user-visible responsiveness issues.",
            "Inspect frame-time traces, device load, and the work surrounding each stall.",
        )

    if len(recent) >= 4:
        changed_signals: list[tuple[str, float]] = []
        for field in ("latency_ms", "error_rate_pct", "memory_pct"):
            trend, delta, relative = _trend_for(field, recent)
            if (
                trend == "rising"
                and delta > 0
                and relative >= configured.deterioration_pct
            ):
                changed_signals.append((DISPLAY_NAMES[field], relative))
        if changed_signals:
            add_reason(
                "deterioration",
                "Deterioration across readings",
                ", ".join(f"{name} +{change:.1f}%" for name, change in changed_signals),
                f">{configured.deterioration_pct:.1f}% rise over up to 5 readings",
                "rising",
                "Several consecutive readings are moving in the wrong direction, even if every limit is not crossed yet.",
                "Compare the recent trend with the last known-good release and investigate the changing signals.",
            )

    score = len(reasons)
    if score >= 4:
        status = "HIGH CONCERN"
    elif score >= 2:
        status = "WARNING"
    else:
        status = "NORMAL"

    return {
        "score": score,
        "status": status,
        "reasons": reasons,
        "active_conditions": score,
        "score_label": "Heuristic risk score — not a probability of failure",
    }
