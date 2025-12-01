# src/demo_phd.py
from __future__ import annotations

import os
import json
import time
import platform
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

# ---- your local modules ----
from simulate_data import make_clients, sliding_windows, FEATURES
from models import RolaLitePredictor
from federated import FederatedClient, FederatedServer, estimate_bytes_full
from evaluate import eval_client
from viz import plot_comm, plot_client_metrics

# =========================
# Utilities / Experiment Core
# =========================

def _now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic-ish for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_world(seed: int, n_clients: int, window: int, horizon: int):
    """
    Creates heterogeneous buoy clients (train) and holdout sets (test).
    Uses a demo 'quality proxy' derived from variance.
    """
    raw_clients = make_clients(n_clients=n_clients, base_seed=seed)

    clients: List[FederatedClient] = []
    eval_sets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for cid, df in raw_clients.items():
        X, Y, A = sliding_windows(df, window=window, horizon=horizon)

        n = X.shape[0]
        split = int(n * 0.7)
        Xtr, Ytr = X[:split], Y[:split]
        Xte, Yte, Ate = X[split:], Y[split:], A[split:]

        var = float(np.mean(np.var(df[FEATURES].values, axis=0)))
        quality = float(1.0 / (1.0 + 0.05 * var))  # proxy only, documented in README

        clients.append(FederatedClient(cid, Xtr, Ytr, quality=quality))
        eval_sets[cid] = (Xte, Yte, Ate)

    in_dim = window * len(FEATURES)
    out_dim = len(FEATURES)
    model = RolaLitePredictor(in_dim=in_dim, out_dim=out_dim, hidden=128)
    server = FederatedServer(model)
    return clients, eval_sets, server

def evaluate_all(server: FederatedServer, eval_sets: dict, max_points: int = 1500) -> pd.DataFrame:
    rows = []
    for cid, (Xte, Yte, Ate) in eval_sets.items():
        # Speed-up for demo: evaluate on a subset
        n = Xte.shape[0]
        if n > max_points:
            idx = np.random.choice(n, size=max_points, replace=False)
            Xs, Ys, As = Xte[idx], Yte[idx], Ate[idx]
        else:
            Xs, Ys, As = Xte, Yte, Ate
        m = eval_client(server.model, Xs, Ys, As)
        rows.append({"client": cid, **m})
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)

def run_experiment(config: dict) -> str:
    """
    Runs a single experiment and saves artifacts.
    config["mode"] in:
      - centralized
      - fedavg_dense
      - fed_topk
      - fed_topk_selective
    """
    set_seeds(int(config["seed"]))
    mode = config["mode"]

    run_id = f"run-{_now_id()}-{mode}"
    out_dir = os.path.join("..", "runs", run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Save config + environment info
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
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
    comm_hist: List[float] = []
    round_rows: List[dict] = []

    if mode == "centralized":
        # pooled training baseline (not federated)
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
            # Treat as an update applied to the server
            server.aggregate([upd], k_frac=1.0)

            comm_hist.append(0.0)
            round_rows.append({
                "round": r + 1,
                "clients_sent": 1,
                "bytes_sent": float(estimate_bytes_full(n_params)),
                "bytes_full": float(estimate_bytes_full(n_params)),
                "comm_reduction": 0.0,
            })

    else:
        for r in range(rounds):
            m = max(1, int(len(clients) * participation_rate))
            selected = np.random.choice(len(clients), size=m, replace=False)

            # Ablation modes
            if mode == "fedavg_dense":
                k = 1.0
                thr = 0.0
            elif mode == "fed_topk":
                k = k_frac
                thr = 0.0
            elif mode == "fed_topk_selective":
                k = k_frac
                thr = send_threshold
            else:
                raise ValueError(f"Unknown mode: {mode}")

            updates = []
            for idx in selected:
                c = clients[idx]
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

            stats = server.aggregate(updates, k_frac=(1.0 if mode == "fedavg_dense" else k_frac))
            stats["round"] = r + 1
            round_rows.append(stats)
            comm_hist.append(stats["comm_reduction"])

    round_df = pd.DataFrame(round_rows)
    round_df.to_csv(os.path.join(out_dir, "metrics_round.csv"), index=False)

    clients_df = evaluate_all(server, eval_sets, max_points=int(config.get("max_eval_points", 1500)))
    clients_df.to_csv(os.path.join(out_dir, "metrics_clients.csv"), index=False)

    # Save plots
    plot_comm(comm_hist, os.path.join(out_dir, "comm_reduction.png"))

    # The plotting util expects dict-of-dicts by client; build it
    d = clients_df.set_index("client").to_dict(orient="index")
    plot_client_metrics(d, os.path.join(out_dir, "roc_auc.png"), key="roc_auc")
    plot_client_metrics(d, os.path.join(out_dir, "avg_precision.png"), key="avg_precision")

    summary = {
        "mode": mode,
        "mean_roc_auc": float(np.nanmean(clients_df["roc_auc"].values)),
        "mean_avg_precision": float(np.nanmean(clients_df["avg_precision"].values)),
        "avg_comm_reduction": float(np.nanmean(round_df["comm_reduction"].values)) if "comm_reduction" in round_df else 0.0,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return out_dir

def run_many(config: dict, seeds: List[int]) -> str:
    """
    Multi-seed evaluation: runs run_experiment for each seed and aggregates mean±std.
    Saves:
      - runs/multi-<mode>/summaries.csv
      - runs/multi-<mode>/aggregate.json
    """
    mode = config["mode"]
    base_out = os.path.join(".", "runs", f"multi-{mode}")
    os.makedirs(base_out, exist_ok=True)

    rows = []
    run_dirs = []
    for s in seeds:
        cfg = {**config, "seed": int(s)}
        out_dir = run_experiment(cfg)
        run_dirs.append(out_dir)

        with open(os.path.join(out_dir, "summary.json"), "r", encoding="utf-8") as f:
            summ = json.load(f)

        rows.append({
            "seed": int(s),
            "mode": mode,
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
        json.dump(
            {"config": config, "seeds": seeds, "aggregate": agg, "runs": run_dirs},
            f,
            indent=2,
        )

    return base_out


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Federated Marine Anomaly", layout="wide")
st.title("🤖 Federated Marine Anomaly Detection")
st.caption(
    "Reproducible, experiment-driven demo with baselines/ablations, "
    "and multi-seed mean±std evaluation."
)

with st.sidebar:

    st.sidebar.markdown(
    """
    <style>
      /* Remove underline from ALL sidebar links */
      section[data-testid="stSidebar"] a { text-decoration: none !important; }
      section[data-testid="stSidebar"] a:hover { text-decoration: none !important; }
    </style>

    <div style="line-height:1.6">
      👤<br/>
      <a href="https://github.com/freelansire/federated-marine-anomaly" target="_blank">GitHub</a> 
      <a href="https://freelansire.com" target="_blank">Website</a>
    </div>
    """,
    unsafe_allow_html=True
)

    st.header("Experiment Controls")

    seed = st.number_input("Seed", 0, 10000, 42)
    n_clients = st.slider("Clients (buoys)", 4, 20, 10)
    rounds = st.slider("Federated rounds", 3, 30, 12)
    participation_rate = st.slider("Participation rate (partial connectivity)", 0.2, 1.0, 0.7, 0.05)

    window = st.slider("Window length", 8, 72, 24)
    horizon = st.slider("Prediction horizon", 1, 6, 1)

    local_steps = st.slider("Local steps per round", 10, 120, 40, 5)
    lr = st.select_slider("Local LR", options=[1e-4, 3e-4, 1e-3, 3e-3], value=1e-3)

    st.subheader("Communication Controls")
    k_frac = st.slider("Top-k fraction (compression)", 0.05, 1.0, 0.25, 0.05)
    send_threshold = st.slider("Selective update threshold", 0.0, 0.2, 0.02, 0.005)

    max_eval_points = st.slider("Max evaluation points/client (speed)", 300, 4000, 1500, 100)

    st.subheader("Compare baselines")
    mode_a = st.selectbox("Method A", ["fedavg_dense", "fed_topk", "fed_topk_selective", "centralized"], index=0)
    mode_b = st.selectbox("Method B", ["fedavg_dense", "fed_topk", "fed_topk_selective", "centralized"], index=2)

    run_compare = st.button("▶ Run comparison (A vs B)")

    st.subheader("Multi-seed evaluation")
    mode_ms = st.selectbox("Method (multi-seed)", ["fedavg_dense", "fed_topk", "fed_topk_selective", "centralized"], index=2)
    n_seeds = st.slider("Number of seeds", 2, 20, 5)
    start_seed = st.number_input("Start seed", 0, 10000, 0)
    run_multiseed = st.button("⚙️ Run multi-seed (mean ± std)")

# Layout
col1, col2 = st.columns([1.15, 0.85])
col3, col4 = st.columns([1.15, 0.85])

st.markdown("### Outputs")
st.write("Each run saves `config.json`, `metrics_round.csv`, `metrics_clients.csv`, plots, and `summary.json` under `runs/`.")

def _make_config(mode: str, seed_val: int) -> dict:
    return dict(
        mode=mode,
        seed=int(seed_val),
        n_clients=int(n_clients),
        rounds=int(rounds),
        participation_rate=float(participation_rate),
        window=int(window),
        horizon=int(horizon),
        local_steps=int(local_steps),
        lr=float(lr),
        k_frac=float(k_frac),
        send_threshold=float(send_threshold),
        max_eval_points=int(max_eval_points),
    )

if run_compare:
    cfg_a = _make_config(mode_a, int(seed))
    cfg_b = _make_config(mode_b, int(seed) + 1)  # small change so cached randomness doesn't mirror exactly

    with st.status("Running Method A…"):
        out_a = run_experiment(cfg_a)
        st.write(f"Artifacts: `{out_a}`")

    with st.status("Running Method B…"):
        out_b = run_experiment(cfg_b)
        st.write(f"Artifacts: `{out_b}`")

    dfA_round = pd.read_csv(os.path.join(out_a, "metrics_round.csv"))
    dfB_round = pd.read_csv(os.path.join(out_b, "metrics_round.csv"))
    dfA_clients = pd.read_csv(os.path.join(out_a, "metrics_clients.csv"))
    dfB_clients = pd.read_csv(os.path.join(out_b, "metrics_clients.csv"))

    # Charts (comm reduction)
    comm_df = pd.DataFrame({
        f"{mode_a}": dfA_round.get("comm_reduction", pd.Series([0.0] * len(dfA_round))).values,
        f"{mode_b}": dfB_round.get("comm_reduction", pd.Series([0.0] * len(dfB_round))).values,
    })
    col1.subheader("Communication reduction per round")
    col1.line_chart(comm_df)

    # Clients sent
    sent_df = pd.DataFrame({
        f"{mode_a}": dfA_round.get("clients_sent", pd.Series([0] * len(dfA_round))).values,
        f"{mode_b}": dfB_round.get("clients_sent", pd.Series([0] * len(dfB_round))).values,
    })
    col2.subheader("Clients sending updates per round")
    col2.bar_chart(sent_df)

    # Robustness table
    merged = dfA_clients[["client", "roc_auc", "avg_precision"]].rename(
        columns={"roc_auc": f"roc_auc_{mode_a}", "avg_precision": f"ap_{mode_a}"}
    ).merge(
        dfB_clients[["client", "roc_auc", "avg_precision"]].rename(
            columns={"roc_auc": f"roc_auc_{mode_b}", "avg_precision": f"ap_{mode_b}"}
        ),
        on="client",
        how="outer",
    )
    col3.subheader("Client robustness (ROC-AUC / AP)")
    col3.dataframe(merged.sort_values(by=f"roc_auc_{mode_a}", ascending=False), use_container_width=True)

    # Summary cards
    with open(os.path.join(out_a, "summary.json"), "r", encoding="utf-8") as f:
        sA = json.load(f)
    with open(os.path.join(out_b, "summary.json"), "r", encoding="utf-8") as f:
        sB = json.load(f)

    col4.subheader("Run summaries")
    col4.metric(f"{mode_a} mean ROC-AUC", f"{sA['mean_roc_auc']:.3f}")
    col4.metric(f"{mode_a} mean AP", f"{sA['mean_avg_precision']:.3f}")
    col4.metric(f"{mode_a} avg comm reduction", f"{sA['avg_comm_reduction']*100:.1f}%")
    col4.divider()
    col4.metric(f"{mode_b} mean ROC-AUC", f"{sB['mean_roc_auc']:.3f}")
    col4.metric(f"{mode_b} mean AP", f"{sB['mean_avg_precision']:.3f}")
    col4.metric(f"{mode_b} avg comm reduction", f"{sB['avg_comm_reduction']*100:.1f}%")

if run_multiseed:
    seeds = list(range(int(start_seed), int(start_seed) + int(n_seeds)))
    cfg = _make_config(mode_ms, int(seeds[0]))
    cfg["mode"] = mode_ms  # explicit

    with st.status("Running multi-seed evaluation…"):
        out_dir = run_many(cfg, seeds)
        st.write(f"Artifacts: `{out_dir}`")

    agg_path = os.path.join(out_dir, "aggregate.json")
    df_path = os.path.join(out_dir, "summaries.csv")

    with open(agg_path, "r", encoding="utf-8") as f:
        agg = json.load(f)["aggregate"]

    st.subheader("Aggregate results (mean ± std)")
    st.write(f"**Mode:** `{mode_ms}`")
    st.write(
        f"**ROC-AUC:** {agg['mean_roc_auc']['mean']:.3f} ± {agg['mean_roc_auc']['std']:.3f} "
        f"(n={agg['mean_roc_auc']['n']})"
    )
    st.write(
        f"**Avg Precision:** {agg['mean_avg_precision']['mean']:.3f} ± {agg['mean_avg_precision']['std']:.3f} "
        f"(n={agg['mean_avg_precision']['n']})"
    )
    st.write(
        f"**Comm reduction:** {agg['avg_comm_reduction']['mean']*100:.1f}% ± {agg['avg_comm_reduction']['std']*100:.1f}% "
        f"(n={agg['avg_comm_reduction']['n']})"
    )

    df = pd.read_csv(df_path)
    st.subheader("Per-seed summaries")
    st.dataframe(df, use_container_width=True)

st.info(
    "Tip: For a paper-style table, run multi-seed on multiple modes and compare mean±std of ROC-AUC/AP and comm reduction. "
    "All evidence is saved under `runs/` for screenshots and citations."
)
