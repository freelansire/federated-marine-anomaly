from __future__ import annotations
from typing import Dict
import json
import os
import matplotlib.pyplot as plt

def save_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def plot_comm(comm_hist, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(comm_hist, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Comm reduction (1 - sent/full)")
    plt.title("Communication Reduction per Round (Top-k + Selective Updates)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_client_metrics(metrics_by_client, out_path: str, key: str = "roc_auc") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    clients = list(metrics_by_client.keys())
    vals = [metrics_by_client[c].get(key, float("nan")) for c in clients]
    plt.figure(figsize=(10, 4))
    plt.bar(clients, vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(key)
    plt.title(f"Client-level {key} (Robustness Across Heterogeneous Buoys)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
