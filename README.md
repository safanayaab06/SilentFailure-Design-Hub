# SilentFailure

**See the warning before the failure.**

SilentFailure is a competition-ready prototype for exploring how mobile-app
performance signals can be monitored for early signs of deterioration. All
readings are deterministic synthetic data; the app does not connect to or
monitor a real mobile device.

Its assessment is an explainable, configurable rule engine. The displayed
heuristic score counts triggered conditions and is **not** a probability of
failure or a trained machine-learning prediction.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py --server.port 3000 --server.address 0.0.0.0
```

## Use the prototype

- Choose Healthy App, Deteriorating App, or High Concern from the sidebar.
- Start monitoring to enable reading generation, then use **Generate next
  reading** to advance the local simulation. There is no background loop.
- Use Scenario Lab to restart a scenario, step it forward, run a repeatable
  demonstration, or compare all three scenarios.
- Filter and search the event timeline.
- Edit warning thresholds in Settings & Methodology. The assessment is
  recalculated with the new rules.
- Reset the simulation to restore the selected scenario's seeded history.

## Risk rules

Each reading is validated before assessment. A condition is counted for
latency, error rate, memory, repeated lag events across the five most recent
readings, or a sufficiently large upward trend across recent readings. Zero or
one conditions map to **NORMAL**, two or three to **WARNING**, and four or more
to **HIGH CONCERN**. Thresholds can be changed in the app.

## Validation

Run the risk-engine unit tests with:

```bash
python -m unittest discover -s tests -v
```
