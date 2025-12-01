from __future__ import annotations
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class ClientSpec:
    client_id: str
    n_steps: int = 6000
    freq_s: int = 10
    anomaly_rate: float = 0.01
    noise_scale: float = 1.0
    seasonal_scale: float = 1.0
    drift_scale: float = 0.001
    missing_rate: float = 0.002

FEATURES = ["temperature", "turbidity", "oxygen", "salinity"]

def _seasonal(t: np.ndarray, period: int, phase: float = 0.0) -> np.ndarray:
    return np.sin(2 * math.pi * (t / period) + phase)

def generate_client_timeseries(spec: ClientSpec, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(spec.n_steps, dtype=np.float32)

    # Base signals (heterogeneous per client)
    temp = 18 + 4 * spec.seasonal_scale * _seasonal(t, period=1200, phase=rng.uniform(0, 2*math.pi))
    turb = 2 + 1.2 * spec.seasonal_scale * _seasonal(t, period=900, phase=rng.uniform(0, 2*math.pi))
    oxy  = 7 + 1.6 * spec.seasonal_scale * _seasonal(t, period=1500, phase=rng.uniform(0, 2*math.pi))
    sal  = 33 + 0.8 * spec.seasonal_scale * _seasonal(t, period=1800, phase=rng.uniform(0, 2*math.pi))

    # Slowly varying drift (random walk)
    drift = rng.normal(0, spec.drift_scale, size=(spec.n_steps, 4)).cumsum(axis=0)

    X = np.stack([temp, turb, oxy, sal], axis=1) + drift
    X += rng.normal(0, spec.noise_scale, size=X.shape)

    # Inject anomalies (spikes, drops, stuck sensor)
    y = np.zeros(spec.n_steps, dtype=np.int64)
    n_anom = max(1, int(spec.n_steps * spec.anomaly_rate))
    anom_idx = rng.choice(spec.n_steps, size=n_anom, replace=False)
    for idx in anom_idx:
        kind = rng.integers(0, 3)
        y[idx] = 1
        if kind == 0:  # spike
            X[idx] += rng.normal(0, 6.0, size=(4,))
        elif kind == 1:  # drop
            X[idx] -= rng.normal(0, 4.0, size=(4,))
        else:  # stuck one feature for a short run
            j = int(rng.integers(0, 4))
            run = int(rng.integers(5, 30))
            end = min(spec.n_steps, idx + run)
            X[idx:end, j] = X[idx, j]
            y[idx:end] = 1

    # Missingness
    miss_mask = rng.random(spec.n_steps) < spec.missing_rate
    X[miss_mask] = np.nan

    df = pd.DataFrame(X, columns=FEATURES)
    df.insert(0, "t", t.astype(np.int64))
    df["anomaly"] = y

    # Simple imputation for demo
    df[FEATURES] = df[FEATURES].ffill().bfill()
    return df

def make_clients(
    n_clients: int = 8,
    base_seed: int = 7,
) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(base_seed)
    clients: Dict[str, pd.DataFrame] = {}
    for i in range(n_clients):
        spec = ClientSpec(
            client_id=f"buoy_{i+1:02d}",
            n_steps=int(rng.integers(4500, 7500)),
            anomaly_rate=float(rng.uniform(0.006, 0.02)),
            noise_scale=float(rng.uniform(0.6, 1.6)),
            seasonal_scale=float(rng.uniform(0.7, 1.5)),
            drift_scale=float(rng.uniform(0.0005, 0.002)),
            missing_rate=float(rng.uniform(0.001, 0.006)),
        )
        clients[spec.client_id] = generate_client_timeseries(spec, seed=int(rng.integers(1, 10_000)))
    return clients

def sliding_windows(
    df: pd.DataFrame,
    window: int = 24,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X: (N, window*F)
      y_next: (N, F)  next-step regression target
      y_anom: (N,) anomaly label at prediction time (t+window+horizon-1)
    """
    Xs, Ys, As = [], [], []
    values = df[FEATURES].to_numpy(dtype=np.float32)
    anom = df["anomaly"].to_numpy(dtype=np.int64)
    T = len(df)
    for start in range(0, T - window - horizon):
        x = values[start : start + window].reshape(-1)
        y = values[start + window + horizon - 1]
        a = anom[start + window + horizon - 1]
        Xs.append(x)
        Ys.append(y)
        As.append(a)
    return np.stack(Xs), np.stack(Ys), np.array(As)
