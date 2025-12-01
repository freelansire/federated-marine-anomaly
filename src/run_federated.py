from __future__ import annotations
import os
import time
import numpy as np
from tqdm import trange

from simulate_data import make_clients, sliding_windows, FEATURES
from models import RolaLitePredictor
from federated import FederatedClient, FederatedServer
from evaluate import eval_client
from viz import save_json, plot_comm, plot_client_metrics

def main():
    # ===== Config =====
    seed = 42
    np.random.seed(seed)

    n_clients = 10
    window = 24
    horizon = 1
    rounds = 12
    participation_rate = 0.7  # partial connectivity
    local_steps = 40
    lr = 1e-3

    # Compression + selective updates
    k_frac = 0.25               # ~75% communication reduction in many cases
    send_threshold = 0.02       # skip tiny updates

    # Run folder
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(".", "runs", f"run-{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    # ===== Data =====
    raw_clients = make_clients(n_clients=n_clients, base_seed=seed)

    client_objs = []
    client_eval = {}

    # Quality proxy: lower noise -> higher quality (simulated)
    # We infer a crude quality score from variance in the series (demo only)
    for cid, df in raw_clients.items():
        X, Y, A = sliding_windows(df, window=window, horizon=horizon)

        # Split train/test along time
        n = X.shape[0]
        split = int(n * 0.7)
        Xtr, Ytr = X[:split], Y[:split]
        Xte, Yte, Ate = X[split:], Y[split:], A[split:]

        var = float(np.mean(np.var(df[FEATURES].values, axis=0)))
        quality = float(1.0 / (1.0 + 0.05 * var))

        client_objs.append(FederatedClient(cid, Xtr, Ytr, quality=quality))
        client_eval[cid] = (Xte, Yte, Ate)

    in_dim = window * len(FEATURES)
    out_dim = len(FEATURES)

    # ===== Model / Server =====
    model = RolaLitePredictor(in_dim=in_dim, out_dim=out_dim, hidden=128)
    server = FederatedServer(model)

    # ===== Federated training =====
    comm_hist = []
    round_stats = []

    ids = [c.client_id for c in client_objs]
    for r in trange(rounds, desc="Federated rounds"):
        # Partial connectivity
        m = max(1, int(len(client_objs) * participation_rate))
        selected = np.random.choice(len(client_objs), size=m, replace=False)

        updates = []
        for idx in selected:
            client = client_objs[idx]
            upd = client.local_train_and_update(
                global_model=server.model,
                lr=lr,
                local_steps=local_steps,
                batch_size=64,
                k_frac=k_frac,
                send_threshold=send_threshold,
            )
            updates.append(upd)

        stats = server.aggregate(updates, k_frac=k_frac)
        comm_hist.append(stats["comm_reduction"])
        round_stats.append(stats)

    # ===== Evaluation (robustness across heterogeneous clients) =====
    metrics_by_client = {}
    for cid, (Xte, Yte, Ate) in client_eval.items():
        metrics_by_client[cid] = eval_client(server.model, Xte, Yte, Ate)

    # ===== Save outputs =====
    save_json(os.path.join(out_dir, "round_stats.json"), {"rounds": round_stats})
    save_json(os.path.join(out_dir, "client_metrics.json"), metrics_by_client)

    plot_comm(comm_hist, os.path.join(out_dir, "comm_reduction.png"))
    plot_client_metrics(metrics_by_client, os.path.join(out_dir, "roc_auc.png"), key="roc_auc")
    plot_client_metrics(metrics_by_client, os.path.join(out_dir, "avg_precision.png"), key="avg_precision")

    # Print headline summary
    avg_comm = float(np.nanmean(comm_hist))
    aucs = [m.get("roc_auc", np.nan) for m in metrics_by_client.values()]
    aps = [m.get("avg_precision", np.nan) for m in metrics_by_client.values()]
    print("\n=== Summary ===")
    print(f"Avg comm reduction: {avg_comm:.2%}")
    print(f"Mean ROC-AUC: {float(np.nanmean(aucs)):.3f}")
    print(f"Mean Avg Precision: {float(np.nanmean(aps)):.3f}")
    print(f"Saved outputs to: {out_dir}")

if __name__ == "__main__":
    main()
