from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

@torch.no_grad()
def anomaly_scores(model, X: np.ndarray, Y: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    x = torch.from_numpy(X).to(torch.float32).to(device)
    y = torch.from_numpy(Y).to(torch.float32).to(device)
    pred = model(x)
    err = torch.mean((pred - y) ** 2, dim=1)  # MSE per sample
    return err.detach().cpu().numpy()

def eval_client(model, X: np.ndarray, Y: np.ndarray, y_anom: np.ndarray, device: str = "cpu") -> Dict[str, float]:
    s = anomaly_scores(model, X, Y, device=device)
    # Some clients may have too few anomalies; guard AUC
    metrics: Dict[str, float] = {"n": float(len(y_anom)), "anom_rate": float(y_anom.mean())}
    if len(np.unique(y_anom)) < 2:
        metrics.update({"roc_auc": float("nan"), "avg_precision": float("nan")})
        return metrics
    metrics["roc_auc"] = float(roc_auc_score(y_anom, s))
    metrics["avg_precision"] = float(average_precision_score(y_anom, s))
    return metrics
