"""
tests/test_autoscaler.py
Tests for telemetry simulator, cost function, and API.
Author: Manasa (CS + Data Science Honours, KL University)
"""

import sys, os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


class TestTelemetrySimulator:
    def test_single_node_shape(self):
        from telemetry_simulator import generate_node_telemetry
        df = generate_node_telemetry("node-test", n_points=60)
        assert len(df) == 60
        assert set(["cpu_pct", "memory_pct", "network_mbps", "is_anomaly"]).issubset(df.columns)

    def test_cpu_bounds(self):
        from telemetry_simulator import generate_node_telemetry
        df = generate_node_telemetry("node-test", n_points=200)
        assert df["cpu_pct"].between(0, 100).all()
        assert df["memory_pct"].between(0, 100).all()

    def test_network_non_negative(self):
        from telemetry_simulator import generate_node_telemetry
        df = generate_node_telemetry("node-test", n_points=100)
        assert (df["network_mbps"] >= 0).all()

    def test_anomaly_flag_binary(self):
        from telemetry_simulator import generate_node_telemetry
        df = generate_node_telemetry("node-test", n_points=500)
        assert set(df["is_anomaly"].unique()).issubset({0, 1})

    def test_live_snapshot_structure(self):
        from telemetry_simulator import get_live_node_snapshot
        snap = get_live_node_snapshot(10)
        assert len(snap) == 10
        for node in snap:
            assert "node_id" in node
            assert "cpu_pct" in node
            assert node["status"] in {"healthy", "warning", "overload"}

    def test_training_sequences_shape(self):
        from telemetry_simulator import generate_training_sequences
        X, y = generate_training_sequences(n_sequences=50)
        assert X.shape == (50, 60, 3)
        assert y.shape == (50,)

    def test_training_sequences_bounds(self):
        from telemetry_simulator import generate_training_sequences
        X, y = generate_training_sequences(n_sequences=100)
        assert (X[:, :, 0] >= 0).all()   # CPU
        assert (X[:, :, 0] <= 100).all()
        assert (y >= 0).all()
        assert (y <= 100).all()


class TestCostFunction:
    def test_cost_function_structure(self):
        from train_lstm import cost_function_evaluator
        result = cost_function_evaluator(60.0, n_nodes=100)
        required = ["predicted_peak_cpu", "provisioned_capacity", "compute_cost_usd",
                    "recommendation", "latency_overhead_ms"]
        for k in required:
            assert k in result, f"Missing key: {k}"

    def test_scale_up_recommendation(self):
        from train_lstm import cost_function_evaluator
        # Very high CPU should recommend scale_up
        result = cost_function_evaluator(95.0, n_nodes=100)
        assert result["recommendation"] in {"scale_up", "maintain"}

    def test_scale_down_recommendation(self):
        from train_lstm import cost_function_evaluator
        result = cost_function_evaluator(10.0, n_nodes=100)
        assert result["recommendation"] in {"scale_down", "maintain"}

    def test_cost_positive(self):
        from train_lstm import cost_function_evaluator
        result = cost_function_evaluator(50.0, n_nodes=50)
        assert result["compute_cost_usd"] > 0

    def test_provisioned_above_predicted(self):
        from train_lstm import cost_function_evaluator
        result = cost_function_evaluator(70.0)
        assert result["provisioned_capacity"] >= result["predicted_peak_cpu"]


class TestAPI:
    def setup_method(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
        from app import app
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health(self):
        r = self.client.get("/api/health")
        assert r.status_code == 200
        assert r.get_json()["status"] == "ok"

    def test_history_returns_144_points(self):
        r = self.client.get("/api/history")
        assert r.status_code == 200
        data = r.get_json()
        assert len(data["history"]) == 144

    def test_anomalies_endpoint(self):
        r = self.client.get("/api/anomalies")
        assert r.status_code == 200
        data = r.get_json()
        assert "anomalies" in data
        assert len(data["anomalies"]) > 0

    def test_predict_with_base_cpu(self):
        r = self.client.post("/api/predict",
                             json={"base_cpu": 55, "n_nodes": 100},
                             content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert "predicted_peak_cpu" in data
        assert "recommendation" in data
        assert 0 <= data["predicted_peak_cpu"] <= 100

    def test_cost_endpoint(self):
        r = self.client.post("/api/cost",
                             json={"cpu": 70.0, "n_nodes": 50},
                             content_type="application/json")
        assert r.status_code == 200

    def test_live_snapshot(self):
        r = self.client.get("/api/live?n=10")
        assert r.status_code == 200
        data = r.get_json()
        assert data["summary"]["total"] == 10
