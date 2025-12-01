from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

from experiment import run_experiment

def run_many(config: dict, seeds: list[int]) -> str:
    """
    Runs the same config over multiple seeds and aggregates summary metrics.
    Saves:
      - summaries.csv
      - aggregate.json (mean ± std)
    """
    base_out = os.path.join("..", "runs", f"multi-{config['mode']}")
    os.makedirs(base_out, exist_ok=True)

    rows = []
    out_dirs = []
    for s in seeds:
        cfg = {**config, "seed": int(s)}
        out_dir = run_experiment(cfg)
        out_dirs.append(out_dir)

        with open(os.path.join(out_dir, "summary.json"), "r", encoding="utf-8") as f:
            summ = json.load(f)

        rows.append({
            "seed": int(s),
            "mode": config["mode"],
            "mean_roc_auc": summ["mean_roc_auc"],
            "mean_avg_precision": summ["mean_avg_precision"],
            "avg_comm_reduction": summ["avg_comm_reduction"],
            "artifact_dir": out_dir,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(base_out, "summaries.csv"), index=False)

    agg = {}
    for k in ["mean_roc_auc", "mean_avg_precision", "avg_comm_reduction"]:
        vals = df[k].to_numpy(dtype=float)
        agg[k] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "n": int(np.sum(~np.isnan(vals))),
        }

    with open(os.path.join(base_out, "aggregate.json"), "w", encoding="utf-8") as f:
        json.dump({"config": config, "seeds": seeds, "aggregate": agg, "runs": out_dirs}, f, indent=2)

    return base_out

if __name__ == "__main__":
    # Quick CLI run (edit as needed)
    cfg = dict(
        seed=42,
        mode="fed_topk_selective",
        n_clients=10,
        rounds=12,
        participation_rate=0.7,
        window=24,
        horizon=1,
        local_steps=40,
        lr=1e-3,
        k_frac=0.25,
        send_threshold=0.02,
    )
    seeds = [0, 1, 2, 3, 4]
    out = run_many(cfg, seeds)
    print(f"Saved multi-seed results to: {out}")
