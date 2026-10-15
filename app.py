from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from risk_engine import Thresholds, assess_telemetry


APP_TITLE = "SilentFailure"
SCENARIOS = ("Healthy App", "Deteriorating App", "High Concern")
NAV_ITEMS = (
    "Overview",
    "Live Monitor",
    "Scenario Lab",
    "Event Timeline",
    "Settings & Methodology",
)
DEFAULT_THRESHOLDS = {
    "latency_ms": 250.0,
    "error_rate_pct": 2.0,
    "memory_pct": 82.0,
    "lag_events_5_readings": 4,
    "deterioration_pct": 6.0,
}
COLORS = {
    "background": "#101312",
    "surface": "#18221D",
    "card": "#222925",
    "accent": "#D5FF40",
    "coral": "#FF765E",
    "mint": "#9CE5C1",
    "text": "#F3F0E6",
    "muted": "#A8B0A6",
    "line": "#384238",
}

st.set_page_config(
    page_title="SilentFailure | Mobile systems",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            color-scheme: dark;
        }}
        .stApp {{
            background-color: {COLORS["background"]};
            background-image:
                linear-gradient(rgba(156, 229, 193, .025) 1px, transparent 1px),
                linear-gradient(90deg, rgba(156, 229, 193, .025) 1px, transparent 1px);
            background-size: 40px 40px;
            color: {COLORS["text"]};
        }}
        [data-testid="stHeader"] {{
            background: rgba(16, 19, 18, .92);
        }}
        [data-testid="stSidebar"] {{
            background: {COLORS["surface"]};
            border-right: 1px solid {COLORS["line"]};
        }}
        [data-testid="stSidebar"] > div:first-child {{
            padding-top: 1.3rem;
        }}
        .brand {{
            color: {COLORS["text"]};
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 1.22rem;
            font-weight: 750;
            letter-spacing: -.07em;
            padding: .35rem 0 1.05rem;
        }}
        .brand span {{
            color: {COLORS["accent"]};
            padding: 0 .12rem;
        }}
        .sim-pill {{
            display: inline-flex;
            align-items: center;
            gap: .5rem;
            border: 1px solid {COLORS["line"]};
            background: rgba(213,255,64,.07);
            color: {COLORS["accent"]};
            padding: .38rem .58rem;
            font: 700 .65rem ui-monospace, SFMono-Regular, Menlo, monospace;
            letter-spacing: .12em;
            border-radius: 3px;
        }}
        .sim-dot {{
            width: 7px; height: 7px; border-radius: 50%;
            background: {COLORS["accent"]}; display: inline-block;
        }}
        .eyebrow {{
            color: {COLORS["mint"]};
            font: 700 .67rem ui-monospace, SFMono-Regular, Menlo, monospace;
            letter-spacing: .16em;
            text-transform: uppercase;
            margin-bottom: .55rem;
        }}
        .page-title {{
            color: {COLORS["text"]};
            font-size: clamp(2.0rem, 4vw, 3.25rem);
            line-height: 1.02;
            letter-spacing: -.055em;
            font-weight: 760;
            margin: 0;
        }}
        .page-subtitle {{
            color: {COLORS["muted"]};
            font-size: .96rem;
            line-height: 1.55;
            max-width: 720px;
            margin: .85rem 0 0;
        }}
        .section-label {{
            color: {COLORS["muted"]};
            font: 700 .66rem ui-monospace, SFMono-Regular, Menlo, monospace;
            letter-spacing: .16em;
            text-transform: uppercase;
            padding: .3rem 0 .65rem;
        }}
        .metric-card {{
            background: {COLORS["card"]};
            border: 1px solid {COLORS["line"]};
            border-top: 2px solid {COLORS["accent"]};
            padding: 1rem 1rem .9rem;
            min-height: 160px;
            box-shadow: 0 8px 24px rgba(0,0,0,.12);
        }}
        .metric-label {{
            color: {COLORS["muted"]};
            text-transform: uppercase;
            letter-spacing: .11em;
            font: 700 .63rem ui-monospace, SFMono-Regular, Menlo, monospace;
        }}
        .metric-value {{
            color: {COLORS["text"]};
            font-size: 1.95rem;
            font-weight: 730;
            letter-spacing: -.055em;
            margin: .55rem 0 .08rem;
            line-height: 1.1;
        }}
        .metric-change {{
            color: {COLORS["mint"]};
            font: 600 .76rem ui-monospace, SFMono-Regular, Menlo, monospace;
        }}
        .metric-change.bad {{ color: {COLORS["coral"]}; }}
        .metric-note {{
            color: {COLORS["muted"]};
            font-size: .77rem;
            margin-top: .55rem;
        }}
        .risk-card {{
            background: {COLORS["card"]};
            border: 1px solid {COLORS["line"]};
            border-left: 3px solid var(--risk-color);
            padding: 1.25rem 1.3rem;
            min-height: 190px;
        }}
        .risk-kicker {{
            color: {COLORS["muted"]};
            text-transform: uppercase;
            letter-spacing: .14em;
            font: 700 .65rem ui-monospace, SFMono-Regular, Menlo, monospace;
        }}
        .risk-status {{
            color: var(--risk-color);
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: .015em;
            margin: .42rem 0 .4rem;
        }}
        .risk-copy {{
            color: {COLORS["muted"]};
            font-size: .86rem;
            line-height: 1.5;
        }}
        .rule-card {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["line"]};
            border-left: 2px solid {COLORS["coral"]};
            padding: .9rem 1rem;
            margin: .55rem 0;
        }}
        .rule-title {{
            color: {COLORS["text"]};
            font-weight: 700;
            font-size: .9rem;
        }}
        .rule-meta {{
            color: {COLORS["mint"]};
            font: 600 .69rem ui-monospace, SFMono-Regular, Menlo, monospace;
            margin: .28rem 0 .45rem;
        }}
        .rule-body {{
            color: {COLORS["muted"]};
            font-size: .81rem;
            line-height: 1.5;
        }}
        .callout {{
            background: rgba(213,255,64,.055);
            border: 1px solid rgba(213,255,64,.22);
            color: {COLORS["muted"]};
            padding: .8rem 1rem;
            font-size: .82rem;
            line-height: 1.5;
        }}
        .callout strong {{ color: {COLORS["accent"]}; }}
        .status-line {{
            border-top: 1px solid {COLORS["line"]};
            padding-top: .75rem;
            margin-top: .7rem;
            color: {COLORS["muted"]};
            font: 600 .72rem ui-monospace, SFMono-Regular, Menlo, monospace;
        }}
        [data-testid="stMetric"] {{
            background: {COLORS["card"]};
            border: 1px solid {COLORS["line"]};
            padding: .85rem 1rem;
        }}
        [data-testid="stMetricLabel"] {{ color: {COLORS["muted"]}; }}
        [data-testid="stMetricValue"] {{ color: {COLORS["text"]}; }}
        div[data-testid="stButton"] > button {{
            border-radius: 3px;
            border-color: {COLORS["line"]};
            min-height: 2.55rem;
            font-weight: 650;
        }}
        div[data-testid="stButton"] > button[kind="primary"] {{
            background: {COLORS["accent"]};
            color: {COLORS["background"]};
            border-color: {COLORS["accent"]};
        }}
        div[data-testid="stButton"] > button:hover {{
            border-color: {COLORS["accent"]};
            color: {COLORS["accent"]};
        }}
        [data-testid="stDataFrame"] {{
            border: 1px solid {COLORS["line"]};
        }}
        hr {{ border-color: {COLORS["line"]}; }}
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_thresholds() -> Thresholds:
    stored = st.session_state.get("thresholds", DEFAULT_THRESHOLDS)
    return Thresholds(**stored)


def seeded_reading(scenario: str, tick: int, timestamp: datetime) -> dict[str, Any]:
    """Generate a deterministic, bounded synthetic reading for a scenario tick."""
    scenario_seed = SCENARIOS.index(scenario) * 100_003 + tick * 7919 + 2026
    rng = random.Random(scenario_seed)
    t = max(0, tick)
    if scenario == "Healthy App":
        latency = 118 + 8 * np.sin(t / 4.5) + rng.gauss(0, 5)
        error = 0.18 + 0.05 * np.sin(t / 3.8) + rng.gauss(0, 0.035)
        memory = 56 + 1.8 * np.sin(t / 6) + rng.gauss(0, 1.0)
        lag = 1 if rng.random() < 0.07 else 0
    elif scenario == "Deteriorating App":
        progress = min(t / 30, 1.0)
        latency = 112 + 215 * progress + rng.gauss(0, 7)
        error = 0.22 + 1.25 * progress + rng.gauss(0, 0.07)
        memory = 55 + 20 * progress + rng.gauss(0, 1.1)
        lag = (1 if progress > 0.12 else 0) + (
            1 if progress > 0.62 and rng.random() < 0.72 else 0
        )
    else:
        latency = 635 + 16 * np.sin(t / 5) + rng.gauss(0, 16)
        error = 5.2 + 0.22 * np.sin(t / 4) + rng.gauss(0, 0.18)
        memory = 92 + 0.8 * np.sin(t / 6) + rng.gauss(0, 0.6)
        lag = 2 + (1 if rng.random() < 0.62 else 0)

    return {
        "timestamp": timestamp,
        "tick": tick,
        "scenario": scenario,
        "latency_ms": round(float(np.clip(latency, 0, 20_000)), 1),
        "error_rate_pct": round(float(np.clip(error, 0, 100)), 2),
        "memory_pct": round(float(np.clip(memory, 0, 100)), 1),
        "lag_events": max(0, int(lag)),
    }


def risk_for(readings: list[dict[str, Any]], thresholds: Thresholds) -> dict[str, Any]:
    latest = readings[-1]
    return assess_telemetry(latest, readings[-5:], thresholds)


def event_records(
    reading: dict[str, Any],
    assessment: dict[str, Any],
) -> list[dict[str, Any]]:
    severity = assessment["status"]
    if severity == "NORMAL":
        return []
    return [
        {
            "timestamp": reading["timestamp"],
            "metric": reason["metric"],
            "observed_value": reason["observed"],
            "severity": severity,
            "explanation": reason["why"],
            "suggested_action": reason["action"],
        }
        for reason in assessment["reasons"]
    ]


def seed_scenario(scenario: str, include_events: bool = True) -> None:
    now = datetime.now(timezone.utc)
    interval = st.session_state.get("interval_minutes", 5)
    start = now - timedelta(minutes=interval * 29)
    readings = [
        seeded_reading(
            scenario,
            tick,
            start + timedelta(minutes=interval * tick),
        )
        for tick in range(30)
    ]
    st.session_state.scenario = scenario
    st.session_state.readings = readings[-100:]
    st.session_state.tick = 29
    st.session_state.events = []
    if include_events:
        thresholds = get_thresholds()
        events: list[dict[str, Any]] = []
        for end in range(len(readings)):
            if end < 3:
                continue
            subset = readings[max(0, end - 4) : end + 1]
            events.extend(event_records(subset[-1], risk_for(subset, thresholds)))
        st.session_state.events = events[-100:]


def initialize_state() -> None:
    defaults = {
        "scenario": "Deteriorating App",
        "thresholds": DEFAULT_THRESHOLDS.copy(),
        "monitoring": False,
        "interval_minutes": 5,
        "show_comparison": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "readings" not in st.session_state:
        seed_scenario(st.session_state.scenario)


def on_scenario_change() -> None:
    selected = st.session_state.scenario_picker
    if selected != st.session_state.scenario:
        seed_scenario(selected)


def choose_scenario(scenario: str) -> None:
    st.session_state.scenario_picker = scenario
    seed_scenario(scenario)


def append_readings(count: int = 1) -> None:
    current = list(st.session_state.readings)
    scenario = st.session_state.scenario
    interval = int(st.session_state.interval_minutes)
    thresholds = get_thresholds()
    for _ in range(count):
        st.session_state.tick += 1
        timestamp = current[-1]["timestamp"] + timedelta(minutes=interval)
        next_reading = seeded_reading(scenario, st.session_state.tick, timestamp)
        current.append(next_reading)
        current = current[-100:]
        assessment = risk_for(current, thresholds)
        st.session_state.events.extend(event_records(next_reading, assessment))
        st.session_state.events = st.session_state.events[-100:]
    st.session_state.readings = current


def toggle_monitoring() -> None:
    st.session_state.monitoring = not st.session_state.monitoring


def status_color(status: str) -> str:
    return {
        "NORMAL": COLORS["accent"],
        "WARNING": "#F4C96B",
        "HIGH CONCERN": COLORS["coral"],
    }.get(status, COLORS["muted"])


def render_header() -> None:
    assessment = risk_for(st.session_state.readings, get_thresholds())
    risk_status = assessment["status"]
    risk_color = status_color(risk_status)
    left, right = st.columns([5, 1.35], vertical_alignment="center")
    with left:
        st.markdown(
            """
            <div class="eyebrow">Mobile systems / monitoring</div>
            <h1 class="page-title">Know before it breaks.</h1>
            <p class="page-subtitle">
              A local telemetry simulation that surfaces changing app health
              signals before they become a larger incident.
            </p>
            """,
            unsafe_allow_html=True,
        )
    with right:
        state_label = "MONITORING ARMED" if st.session_state.monitoring else "PAUSED"
        state_color = COLORS["accent"] if st.session_state.monitoring else COLORS["muted"]
        st.markdown(
            f'<div class="sim-pill" style="color:{risk_color};background:{risk_color}12">'
            f'<span class="sim-dot" style="background:{risk_color}"></span>SYSTEM: {risk_status}</div>'
            f'<div style="height:.4rem"></div>'
            f'<div class="sim-pill"><span class="sim-dot" style="background:{state_color}"></span>{state_label}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        st.button(
            "Pause Monitoring" if st.session_state.monitoring else "Start Monitoring",
            key="header_monitor_toggle",
            type="primary" if not st.session_state.monitoring else "secondary",
            on_click=toggle_monitoring,
            use_container_width=True,
        )


def render_risk_summary(assessment: dict[str, Any], latest: dict[str, Any]) -> None:
    status = assessment["status"]
    reasons = assessment["reasons"]
    color = status_color(status)
    if reasons:
        strongest = ", ".join(reason["metric"] for reason in reasons[:3])
        explanation = f"{assessment['active_conditions']} active conditions: {strongest}."
    else:
        explanation = "No warning rules are active. Current signals remain within the configured limits."
    if status == "HIGH CONCERN":
        next_step = "Investigate the most severe signals now; compare against the last known-good release."
    elif status == "WARNING":
        next_step = "Review the rising signals and inspect the next few readings for persistence."
    else:
        next_step = "Continue observing the trend; no immediate investigation is indicated."

    st.markdown(
        f"""
        <div class="risk-card" style="--risk-color:{color}">
          <div class="risk-kicker">Rule-based assessment</div>
          <div class="risk-status">{status}</div>
          <div class="risk-copy">{explanation}</div>
          <div class="status-line">
            HEURISTIC RISK SCORE: {assessment["score"]} &nbsp;·&nbsp;
            NOT A PROBABILITY OF FAILURE
          </div>
          <div class="risk-copy" style="margin-top:.65rem">
            <strong style="color:{COLORS["text"]}">Investigate next:</strong> {next_step}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Latest sample: {latest['timestamp'].strftime('%d %b %Y · %H:%M:%S UTC')} · "
        f"{assessment['active_conditions']} active warning condition(s)"
    )


def format_metric(field: str, value: float) -> str:
    if field == "latency_ms":
        return f"{value:,.0f} ms"
    if field == "error_rate_pct":
        return f"{value:.2f}%"
    if field == "memory_pct":
        return f"{value:.1f}%"
    return f"{value:.0f} events"


def render_metric_cards(readings: list[dict[str, Any]]) -> None:
    latest = readings[-1]
    previous = readings[-2] if len(readings) > 1 else latest
    metrics = [
        ("latency_ms", "Response latency", "ms", 0),
        ("error_rate_pct", "Error rate", "%", 2),
        ("memory_pct", "Memory utilization", "%", 1),
        ("lag_events", "Freeze / lag events", "events", 0),
    ]
    cols = st.columns(4)
    for col, (field, label, _unit, decimals) in zip(cols, metrics):
        now = float(latest[field])
        before = float(previous[field])
        delta = now - before
        if field == "lag_events":
            change = f"{delta:+.0f} from previous interval"
        elif field == "latency_ms":
            change = f"{delta:+.{decimals}f} ms from previous interval"
        else:
            change = f"{delta:+.{decimals}f} pp from previous interval"
        adverse = delta > 0.05 if field != "lag_events" else delta > 0
        trend = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        if field == "latency_ms":
            note = "Response time at the latest interval"
        elif field == "error_rate_pct":
            note = "Failed requests in the latest interval"
        elif field == "memory_pct":
            note = "Estimated memory utilization"
        else:
            note = "New stalls in the latest interval"
        col.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{format_metric(field, now)}</div>
              <div class="metric-change {"bad" if adverse else ""}">{trend} {change}</div>
              <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_chart(readings: list[dict[str, Any]], prefix: str) -> None:
    st.markdown('<div class="section-label">Signal traces / simulated telemetry</div>', unsafe_allow_html=True)
    selected = st.multiselect(
        "Signals to display",
        options=[
            "Response latency",
            "Error rate",
            "Memory utilization",
        ],
        default=["Response latency", "Error rate", "Memory utilization"],
        key=f"{prefix}_chart_signals",
    )
    if not selected:
        st.info("Select at least one signal to draw the timeline.")
        return

    data = pd.DataFrame(readings)
    signal_config = {
        "Response latency": ("latency_ms", "Latency (ms)", COLORS["accent"]),
        "Error rate": ("error_rate_pct", "Error rate (%)", COLORS["coral"]),
        "Memory utilization": ("memory_pct", "Memory (%)", COLORS["mint"]),
    }
    active = [name for name in signal_config if name in selected]
    cols = st.columns(len(active))
    for col, signal in zip(cols, active):
        field, unit, color = signal_config[signal]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data["timestamp"],
                y=data[field],
                mode="lines+markers",
                name=signal,
                line={"color": color, "width": 2},
                marker={"size": 4, "color": color},
                customdata=np.stack(
                    [data["scenario"].astype(str).to_numpy()], axis=-1
                ),
                hovertemplate=(
                    "%{x|%d %b %H:%M:%S UTC}<br>"
                    f"{unit}: %{{y}}<br>Scenario: %{{customdata[0]}}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            height=260,
            margin={"l": 8, "r": 8, "t": 38, "b": 10},
            paper_bgcolor=COLORS["card"],
            plot_bgcolor=COLORS["card"],
            font={"color": COLORS["muted"], "size": 10},
            showlegend=True,
            legend={
                "orientation": "h",
                "x": 0,
                "y": 1.12,
                "font": {"size": 9, "color": COLORS["text"]},
            },
            hoverlabel={"bgcolor": COLORS["surface"], "font_color": COLORS["text"]},
            xaxis={
                "title": "Time (UTC)",
                "showgrid": False,
                "tickfont": {"size": 9},
                "color": COLORS["muted"],
            },
            yaxis={
                "title": unit,
                "gridcolor": COLORS["line"],
                "zerolinecolor": COLORS["line"],
                "color": COLORS["muted"],
            },
        )
        col.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def filtered_events(events: list[dict[str, Any]], key_prefix: str) -> pd.DataFrame:
    event_frame = pd.DataFrame(events)
    left, right = st.columns([1, 2])
    with left:
        severities = st.multiselect(
            "Severity",
            ["HIGH CONCERN", "WARNING"],
            default=["HIGH CONCERN", "WARNING"],
            key=f"{key_prefix}_severity",
        )
    with right:
        search = st.text_input(
            "Search events",
            placeholder="Search metric, explanation, or action",
            key=f"{key_prefix}_search",
        )
    if event_frame.empty:
        return event_frame
    if severities:
        event_frame = event_frame[event_frame["severity"].isin(severities)]
    else:
        event_frame = event_frame.iloc[0:0]
    if search.strip():
        searchable = event_frame[
            ["metric", "observed_value", "explanation", "suggested_action"]
        ].fillna("").astype(str).agg(" ".join, axis=1)
        event_frame = event_frame[searchable.str.contains(search, case=False, regex=False)]
    return event_frame.sort_values("timestamp", ascending=False).reset_index(drop=True)


def render_event_table(events: list[dict[str, Any]], key_prefix: str, limit: int | None = None) -> None:
    event_frame = filtered_events(events, key_prefix)
    if event_frame.empty:
        st.info("No events match these filters. Try another severity or search term.")
        return
    display = event_frame.copy()
    display["timestamp"] = display["timestamp"].dt.strftime("%d %b %H:%M:%S UTC")
    display = display.rename(
        columns={
            "timestamp": "Timestamp",
            "metric": "Metric",
            "observed_value": "Observed value",
            "severity": "Severity",
            "explanation": "Explanation",
            "suggested_action": "Suggested action",
        }
    )
    display = display[
        [
            "Timestamp",
            "Metric",
            "Observed value",
            "Severity",
            "Explanation",
            "Suggested action",
        ]
    ]
    if limit is not None:
        display = display.head(limit)
    st.dataframe(display, hide_index=True, use_container_width=True)


def render_diagnostics(assessment: dict[str, Any]) -> None:
    st.markdown('<div class="section-label">Why this status / next checks</div>', unsafe_allow_html=True)
    if not assessment["reasons"]:
        st.markdown(
            '<div class="callout"><strong>No active warning rules.</strong> '
            'Current readings remain within the selected thresholds.</div>',
            unsafe_allow_html=True,
        )
        return
    for reason in assessment["reasons"]:
        st.markdown(
            f"""
            <div class="rule-card">
              <div class="rule-title">{reason["metric"]}</div>
              <div class="rule-meta">
                OBSERVED {reason["observed"]} &nbsp;·&nbsp;
                THRESHOLD {reason["threshold"]} &nbsp;·&nbsp;
                TREND {reason["trend"].upper()}
              </div>
              <div class="rule-body"><strong>Why it matters:</strong> {reason["why"]}</div>
              <div class="rule-body" style="margin-top:.35rem">
                <strong>Investigate:</strong> {reason["action"]}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def scenario_snapshot(scenario: str) -> tuple[dict[str, Any], dict[str, Any]]:
    now = datetime.now(timezone.utc)
    interval = int(st.session_state.interval_minutes)
    start = now - timedelta(minutes=interval * 29)
    readings = [
        seeded_reading(scenario, tick, start + timedelta(minutes=interval * tick))
        for tick in range(30)
    ]
    return readings[-1], risk_for(readings, get_thresholds())


def render_comparison() -> None:
    st.markdown('<div class="section-label">Scenario comparison / same rule set</div>', unsafe_allow_html=True)
    columns = st.columns(3)
    for col, scenario in zip(columns, SCENARIOS):
        latest, assessment = scenario_snapshot(scenario)
        color = status_color(assessment["status"])
        col.markdown(
            f"""
            <div class="risk-card" style="--risk-color:{color}; min-height:235px">
              <div class="risk-kicker">{scenario}</div>
              <div class="risk-status" style="font-size:1.1rem">{assessment["status"]}</div>
              <div class="risk-copy">Heuristic score: {assessment["score"]} · not a failure probability</div>
              <div class="status-line">
                {latest["latency_ms"]:.0f} ms latency<br>
                {latest["error_rate_pct"]:.2f}% errors<br>
                {latest["memory_pct"]:.1f}% memory<br>
                {latest["lag_events"]} lag events / interval
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col.button(
            "Open scenario",
            key=f"open_{scenario}",
            use_container_width=True,
            on_click=choose_scenario,
            args=(scenario,),
        )


def render_overview(readings: list[dict[str, Any]], assessment: dict[str, Any]) -> None:
    render_metric_cards(readings)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.05, 1.95], gap="large")
    with left:
        st.markdown('<div class="section-label">Current risk assessment</div>', unsafe_allow_html=True)
        render_risk_summary(assessment, readings[-1])
    with right:
        render_diagnostics(assessment)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    render_chart(readings, "overview")
    st.markdown('<div class="section-label">Recent events / latest first</div>', unsafe_allow_html=True)
    render_event_table(st.session_state.events, "overview_events", limit=8)


def render_live_monitor(readings: list[dict[str, Any]], assessment: dict[str, Any]) -> None:
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.markdown('<div class="section-label">Local simulation controls</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="callout"><strong>Simulation only.</strong> '
            'This prototype does not connect to a phone. There is no background timer; '
            'readings advance only when you request one.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
        st.slider(
            "Simulated interval between readings (minutes)",
            min_value=1,
            max_value=30,
            value=int(st.session_state.interval_minutes),
            key="interval_minutes",
        )
        action_cols = st.columns(3)
        action_cols[0].button(
            "Generate next reading",
            key="live_next",
            type="primary",
            disabled=not st.session_state.monitoring,
            on_click=append_readings,
            use_container_width=True,
        )
        action_cols[1].button(
            "Reset simulation",
            key="live_reset",
            on_click=lambda: seed_scenario(st.session_state.scenario),
            use_container_width=True,
        )
        action_cols[2].button(
            "Pause" if st.session_state.monitoring else "Resume",
            key="live_toggle",
            on_click=toggle_monitoring,
            use_container_width=True,
        )
    with right:
        st.markdown('<div class="section-label">Monitoring state</div>', unsafe_allow_html=True)
        render_risk_summary(assessment, readings[-1])
        st.caption(
            f"Scenario: {st.session_state.scenario} · "
            f"Readings retained: {len(readings)} / 100 · "
            f"Tick: {st.session_state.tick}"
        )
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    render_metric_cards(readings)
    render_chart(readings, "live")
    render_diagnostics(assessment)


def render_scenario_lab(readings: list[dict[str, Any]], assessment: dict[str, Any]) -> None:
    st.markdown('<div class="section-label">Scenario lab / controlled demonstrations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="callout">All three scenarios are reproducible synthetic telemetry. '
        '<strong>Changing a scenario resets its readings, events, metrics, and assessment together.</strong></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].button(
        "Restart scenario",
        key="lab_restart",
        on_click=lambda: seed_scenario(st.session_state.scenario),
        use_container_width=True,
    )
    cols[1].button(
        "Generate next reading",
        key="lab_next",
        type="primary",
        on_click=append_readings,
        use_container_width=True,
    )
    cols[2].button(
        "Run demonstration · 12 readings",
        key="lab_demo",
        on_click=lambda: append_readings(12),
        use_container_width=True,
    )
    if cols[3].button(
        "Hide comparison" if st.session_state.show_comparison else "Compare scenarios",
        key="lab_compare",
        use_container_width=True,
    ):
        st.session_state.show_comparison = not st.session_state.show_comparison
    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    if st.session_state.show_comparison:
        render_comparison()
    st.markdown('<div class="section-label">Selected scenario / current assessment</div>', unsafe_allow_html=True)
    render_metric_cards(readings)
    render_risk_summary(assessment, readings[-1])
    render_diagnostics(assessment)
    render_chart(readings, "scenario_lab")


def render_timeline() -> None:
    st.markdown('<div class="section-label">Event timeline / explainable signals</div>', unsafe_allow_html=True)
    st.markdown(
        "Each row is tied to a triggered rule in the synthetic telemetry. "
        "Use severity and search to narrow the timeline.",
        help="Event history is session-local and resets when the selected scenario is restarted.",
    )
    render_event_table(st.session_state.events, "timeline", limit=None)


def render_settings(assessment: dict[str, Any]) -> None:
    st.markdown('<div class="section-label">Settings / warning thresholds</div>', unsafe_allow_html=True)
    st.write(
        "Adjust the rule limits below. The current reading and all scenario assessments "
        "recalculate using these values; no model training or prediction is involved."
    )
    current = st.session_state.thresholds
    with st.form("threshold_form"):
        left, right = st.columns(2)
        with left:
            latency = st.number_input(
                "Response latency warning (ms)",
                min_value=50.0,
                max_value=5_000.0,
                value=float(current["latency_ms"]),
                step=10.0,
            )
            error = st.number_input(
                "Error rate warning (%)",
                min_value=0.1,
                max_value=50.0,
                value=float(current["error_rate_pct"]),
                step=0.1,
            )
            memory = st.number_input(
                "Memory utilization warning (%)",
                min_value=20.0,
                max_value=99.0,
                value=float(current["memory_pct"]),
                step=1.0,
            )
        with right:
            lag = st.number_input(
                "Lag events across the last 5 readings",
                min_value=1,
                max_value=20,
                value=int(current["lag_events_5_readings"]),
                step=1,
            )
            deterioration = st.number_input(
                "Deterioration warning (% rise)",
                min_value=1.0,
                max_value=100.0,
                value=float(current["deterioration_pct"]),
                step=1.0,
            )
            st.caption(
                "Trend changes also require a meaningful absolute increase so small "
                "measurement noise does not trigger a warning."
            )
        saved = st.form_submit_button("Apply thresholds", type="primary")
    if saved:
        st.session_state.thresholds = {
            "latency_ms": latency,
            "error_rate_pct": error,
            "memory_pct": memory,
            "lag_events_5_readings": int(lag),
            "deterioration_pct": deterioration,
        }
        st.rerun()

    st.markdown('<div class="section-label">How the heuristic score works</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - **One condition per rule:** latency above its limit, error rate above its
          limit, memory above its limit, repeated lag across up to five readings,
          or meaningful deterioration across recent readings.
        - **NORMAL:** zero or one active condition.
        - **WARNING:** two or three active conditions.
        - **HIGH CONCERN:** four or more active conditions.
        - The score is a **count of rules**, not a probability, confidence level,
          time-to-failure, or model accuracy.
        - This prototype uses deterministic synthetic scenarios and a transparent
          rule set. It is not a trained AI or machine-learning model.
        """
    )
    st.markdown('<div class="section-label">Current assessment with these thresholds</div>', unsafe_allow_html=True)
    render_risk_summary(assessment, st.session_state.readings[-1])
    render_diagnostics(assessment)


def main() -> None:
    inject_styles()
    initialize_state()

    with st.sidebar:
        st.markdown('<div class="brand">silent<span>/</span>failure</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sim-pill"><span class="sim-dot"></span>SIMULATION MODE</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
        page = st.radio("Workspace", NAV_ITEMS, label_visibility="collapsed")
        st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
        st.selectbox(
            "Scenario",
            SCENARIOS,
            index=SCENARIOS.index(st.session_state.scenario),
            key="scenario_picker",
            on_change=on_scenario_change,
        )
        st.markdown(
            f'<div class="status-line">LOCAL DATA ONLY<br>'
            f'{"READING GENERATION ENABLED" if st.session_state.monitoring else "MONITORING PAUSED"}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Prototype · deterministic synthetic telemetry")

    render_header()
    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    readings = st.session_state.readings
    assessment = risk_for(readings, get_thresholds())
    if page == "Overview":
        render_overview(readings, assessment)
    elif page == "Live Monitor":
        render_live_monitor(readings, assessment)
    elif page == "Scenario Lab":
        render_scenario_lab(readings, assessment)
    elif page == "Event Timeline":
        render_timeline()
    else:
        render_settings(assessment)

    st.markdown(
        '<div class="status-line" style="margin-top:1.5rem">'
        'SYNTHETIC DEMONSTRATION DATA · NO DEVICE CONNECTION · RULE-BASED HEURISTICS ONLY'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
