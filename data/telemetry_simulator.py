"""
telemetry_simulator.py
Generates synthetic server telemetry: CPU, memory, network traffic.
Simulates thousands of nodes with realistic patterns including spikes,
diurnal cycles, and anomaly injection.
Author: Manasa (CS + Data Science Honours, KL University)
"""

import numpy as np
import pandas as pd
import random
import json
import os
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)


def diurnal_factor(hour: int) -> float:
    """Simulate business-hours load pattern (peaks at 10am, 3pm)."""
    return 0.4 + 0.6 * (np.sin((hour - 6) * np.pi / 12) ** 2)


def generate_node_telemetry(node_id: str, n_points: int = 1440, base_cpu: float = None) -> pd.DataFrame:
    """
    Generate 1440 minutes (24h) of telemetry for a single node.
    Includes:
      - CPU utilisation (with diurnal + spikes)
      - Memory utilisation (correlated with CPU)
      - Network traffic in MB/s
      - Anomaly flags (1% chance per point)
    """
    base_cpu = base_cpu or random.uniform(20, 55)
    timestamps = [datetime(2024, 6, 1, 0, 0) + timedelta(minutes=i) for i in range(n_points)]
    cpu_vals, mem_vals, net_vals, anomaly_flags = [], [], [], []

    for i, ts in enumerate(timestamps):
        hour = ts.hour
        df = diurnal_factor(hour)
        cpu = base_cpu * df + np.random.normal(0, 3)
        cpu = max(0, min(100, cpu))

        # Inject spike anomalies (~1%)
        is_anomaly = random.random() < 0.01
        if is_anomaly:
            cpu = min(100, cpu + random.uniform(25, 40))

        mem = 0.6 * cpu + random.uniform(10, 20) + np.random.normal(0, 2)
        mem = max(0, min(100, mem))
        net = max(0, (cpu / 100) * 500 + np.random.normal(0, 30))

        cpu_vals.append(round(cpu, 2))
        mem_vals.append(round(mem, 2))
        net_vals.append(round(net, 2))
        anomaly_flags.append(int(is_anomaly))

    return pd.DataFrame({
        "timestamp": timestamps,
        "node_id": node_id,
        "cpu_pct": cpu_vals,
        "memory_pct": mem_vals,
        "network_mbps": net_vals,
        "is_anomaly": anomaly_flags,
    })


def generate_cluster_telemetry(n_nodes: int = 200, n_points: int = 1440) -> pd.DataFrame:
    """Generate telemetry for n_nodes nodes."""
    frames = []
    for i in range(n_nodes):
        node_id = f"node-{i+1:04d}"
        base_cpu = random.uniform(15, 65)
        df = generate_node_telemetry(node_id, n_points, base_cpu)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def generate_training_sequences(n_sequences: int = 5000, seq_len: int = 60, horizon: int = 10) -> tuple:
    """
    For LSTM training: generate (X, y) pairs.
    X: [seq_len] steps of [cpu, memory, network]
    y: peak CPU in next `horizon` steps (regression target for scaling decisions)
    """
    X, y = [], []
    for _ in range(n_sequences):
        t = np.arange(seq_len + horizon)
        base = random.uniform(20, 70)
        trend = random.uniform(-0.1, 0.2)
        noise_level = random.uniform(2, 8)
        cpu = base + trend * t + np.random.normal(0, noise_level, len(t))
        mem = 0.6 * cpu + random.uniform(10, 20) + np.random.normal(0, 3, len(t))
        net = (cpu / 100) * 500 + np.random.normal(0, 30, len(t))
        cpu = np.clip(cpu, 0, 100)
        mem = np.clip(mem, 0, 100)
        net = np.clip(net, 0, None)

        x_seq = np.column_stack([cpu[:seq_len], mem[:seq_len], net[:seq_len]])
        y_val = np.max(cpu[seq_len:seq_len + horizon])  # peak CPU in next window

        X.append(x_seq)
        y.append(y_val)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def get_live_node_snapshot(n_nodes: int = 50) -> list:
    """Returns a current snapshot of n_nodes for live dashboard."""
    snapshot = []
    for i in range(n_nodes):
        hour = datetime.now().hour
        df = diurnal_factor(hour)
        base = random.uniform(20, 70)
        cpu = max(0, min(100, base * df + np.random.normal(0, 5)))
        mem = max(0, min(100, 0.6 * cpu + random.uniform(10, 20)))
        net = max(0, (cpu / 100) * 500 + np.random.normal(0, 30))
        status = "overload" if cpu > 85 else "warning" if cpu > 70 else "healthy"
        snapshot.append({
            "node_id": f"node-{i+1:04d}",
            "cpu_pct": round(cpu, 1),
            "memory_pct": round(mem, 1),
            "network_mbps": round(net, 1),
            "status": status,
            "timestamp": datetime.now().isoformat(),
        })
    return snapshot


if __name__ == "__main__":
    print("Generating cluster telemetry (200 nodes, 24h)…")
    os.makedirs("data", exist_ok=True)
    df = generate_cluster_telemetry(n_nodes=200, n_points=1440)
    df.to_csv("data/cluster_telemetry.csv", index=False)
    print(f"Saved {len(df):,} rows → data/cluster_telemetry.csv")

    print("Generating LSTM training sequences…")
    X, y = generate_training_sequences(5000)
    np.save("data/X_train.npy", X)
    np.save("data/y_train.npy", y)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
