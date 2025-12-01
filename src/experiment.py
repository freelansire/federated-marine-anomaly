from __future__ import annotations
import os, json, time, platform
import numpy as np
import pandas as pd
import torch

from simulate_data import make_clients, sliding_windows, FEATURES
from models import RolaLitePredictor
from federated import FederatedClient, FederatedServer, estimate_bytes_full
from evaluate import eval_client
from viz import plot_comm, plot_client_metrics

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_world(seed: int, n_clients: int, window: int, horizon: int):
    raw_clients = make_clients(n_clients=n_clients, base_seed=seed)
    clients, eval_sets = [], {}

    for cid, df in raw_clients.items():
        X, Y, A = sliding_windows(df, window=window, horizon=horizon)
        n = X.shape[0]
        split = int(n * 0.7)
        Xtr, Ytr = X[:split], Y[:split]
        Xte, Yte, Ate = X[split:], Y[split:], A[split:]

        var = float(np.mean(np.var(df[FEATURES].values, axis=0)))
        quality = float(1.0 / (1.0 + 0.05 * var))  # demo proxy

        clients.append(FederatedClient(cid, Xtr, Ytr, quality=quality))
        eval_sets[cid] = (Xte, Yte, Ate)

    in_dim = window * len(FEATURES)
    out_dim = len(FEATURES)
    model = RolaLitePredictor(in_dim=in_dim, out_dim=out_dim, hidden=128)
    server = FederatedServer(model)
    return clients, eval_sets, server

def evaluate_all(server: FederatedServer, eval_sets: dict, device="cpu"):
    rows = []
    for cid, (Xte, Yte, Ate) in eval_sets.items():
        m = eval_client(server.model, Xte, Yte, Ate, device=device)
        rows.append({"client": cid, **m})
    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    return df

def run_experiment(config: dict) -> str:
    """
    config["mode"] in:
      - "centralized"
      - "fedavg_dense"
      - "fed_topk"
      - "fed_topk_selective"
    """
    set_seeds(int(config["seed"]))
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(".", "runs", f"run-{run_id}-{config['mode']}")
    os.makedirs(out_dir, exist_ok=True)

    # Save config + environment metadata
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"config": config, "env": env}, f, indent=2)

    clients, eval_sets, server = build_world(
        seed=int(config["seed"]),
        n_clients=int(config["n_clients"]),
        window=int(config["window"]),
        horizon=int(config["horizon"]),
    )

    rounds = int(config["rounds"])
    participation_rate = float(config["participation_rate"])
    local_steps = int(config["local_steps"])
    lr = float(config["lr"])
    k_frac = float(config["k_frac"])
    send_threshold = float(config["send_threshold"])

    n_params = sum(p.numel() for p in server.model.parameters())
    comm_hist, round_rows = [], []

    if config["mode"] == "centralized":
        # Train on pooled data once (simple baseline)
        X_all = np.concatenate([c.X_train for c in clients], axis=0)
        Y_all = np.concatenate([c.Y_train for c in clients], axis=0)
        pooled = FederatedClient("central", X_all, Y_all, quality=1.0)
        for r in range(rounds):
            upd = pooled.local_train_and_update(
                global_model=server.model,
                lr=lr,
                local_steps=local_steps,
                k_frac=1.0,
                send_threshold=0.0,
            )
            # emulate full update (no comm reduction notion here)
            server.aggregate([upd], k_frac=1.0)
            comm_hist.append(0.0)
            round_rows.append({"round": r+1, "clients_sent": 1, "bytes_sent": estimate_bytes_full(n_params), "comm_reduction": 0.0})

    else:
        for r in range(rounds):
            m = max(1, int(len(clients) * participation_rate))
            selected = np.random.choice(len(clients), size=m, replace=False)
            updates = []

            for idx in selected:
                c = clients[idx]
                if config["mode"] == "fedavg_dense":
                    k = 1.0
                    thr = 0.0
                elif config["mode"] == "fed_topk":
                    k = k_frac
                    thr = 0.0
                else:  # fed_topk_selective
                    k = k_frac
                    thr = send_threshold

                updates.append(
                    c.local_train_and_update(
                        global_model=server.model,
                        lr=lr,
                        local_steps=local_steps,
                        batch_size=64,
                        k_frac=k,
                        send_threshold=thr,
                    )
                )

            stats = server.aggregate(updates, k_frac=(1.0 if config["mode"] == "fedavg_dense" else k_frac))
            comm_hist.append(stats["comm_reduction"])
            round_rows.append({"round": r+1, **stats})

    round_df = pd.DataFrame(round_rows)
    round_df.to_csv(os.path.join(out_dir, "metrics_round.csv"), index=False)

    clients_df = evaluate_all(server, eval_sets)
    clients_df.to_csv(os.path.join(out_dir, "metrics_clients.csv"), index=False)

    # Plots
    plot_comm(comm_hist, os.path.join(out_dir, "comm_reduction.png"))
    plot_client_metrics(clients_df.set_index("client").to_dict(orient="index"),
                        os.path.join(out_dir, "roc_auc.png"), key="roc_auc")
    plot_client_metrics(clients_df.set_index("client").to_dict(orient="index"),
                        os.path.join(out_dir, "avg_precision.png"), key="avg_precision")

    # Summary
    summary = {
        "mean_roc_auc": float(np.nanmean(clients_df["roc_auc"].values)),
        "mean_avg_precision": float(np.nanmean(clients_df["avg_precision"].values)),
        "avg_comm_reduction": float(np.nanmean(round_df.get("comm_reduction", pd.Series([0.0])).values)),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return out_dir
