import math
import unittest

from risk_engine import Thresholds, assess_telemetry


def reading(latency=120, error=0.2, memory=55, lag=0):
    return {
        "latency_ms": latency,
        "error_rate_pct": error,
        "memory_pct": memory,
        "lag_events": lag,
    }


class RiskEngineTests(unittest.TestCase):
    def test_healthy_reading_is_normal(self):
        result = assess_telemetry(reading(), [reading()] * 5)
        self.assertEqual(result["status"], "NORMAL")
        self.assertEqual(result["score"], 0)

    def test_degraded_signals_produce_warning(self):
        history = [reading(180, 0.4, 58), reading(210, 0.6, 61),
                   reading(240, 0.8, 64), reading(270, 1.0, 68)]
        result = assess_telemetry(history[-1], history)
        self.assertEqual(result["status"], "WARNING")
        self.assertGreaterEqual(result["score"], 2)

    def test_multiple_severe_signals_produce_high_concern(self):
        history = [reading(650, 5.5, 92, 2)] * 5
        result = assess_telemetry(history[-1], history)
        self.assertEqual(result["status"], "HIGH CONCERN")
        self.assertGreaterEqual(result["score"], 4)

    def test_missing_and_invalid_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            assess_telemetry({"latency_ms": 100})
        with self.assertRaisesRegex(ValueError, "finite"):
            assess_telemetry(reading(latency=math.nan))
        with self.assertRaisesRegex(ValueError, "between"):
            assess_telemetry(reading(memory=101))
        with self.assertRaisesRegex(ValueError, "whole number"):
            assess_telemetry(reading(lag=1.5))

    def test_threshold_changes_change_assessment(self):
        sample = reading(latency=300)
        strict = assess_telemetry(sample, [sample] * 5, Thresholds(latency_ms=250))
        relaxed = assess_telemetry(sample, [sample] * 5, Thresholds(latency_ms=350))
        self.assertIn("latency_ms", [item["key"] for item in strict["reasons"]])
        self.assertNotIn("latency_ms", [item["key"] for item in relaxed["reasons"]])

    def test_repeated_lag_events_are_counted_across_recent_readings(self):
        history = [reading(lag=1) for _ in range(4)] + [reading(lag=0)]
        result = assess_telemetry(history[-1], history)
        self.assertIn("lag_events", [item["key"] for item in result["reasons"]])

    def test_history_must_contain_valid_readings(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            assess_telemetry(reading(), [reading(), {"latency_ms": 90}])


if __name__ == "__main__":
    unittest.main()
