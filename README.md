## 🤖 Federated Lightweight Anomaly Detection for Distributed Marine Sensor Nodes

A **reproducible research-grade prototype** of federated anomaly detection across **heterogeneous, partially connected** marine sensor nodes (simulated “buoys”).  
Built to serve as **evidence for PhD applications**: includes baselines/ablations, artifacts, and multi-seed mean±std evaluation.

> **What this is:** A simulation-based experimental framework showing *federated online learners + communication-efficient updates + robustness under heterogeneity & partial participation*.  
> **What this is not:** A production “secure aggregation” implementation. The masking code is a *toy demonstration* of the concept.

---

### ✅ Core Features 

#### Federated anomaly detection prototype
- Distributed clients represent **marine sensor buoys**
- Each client observes multivariate streams:
  - **temperature, turbidity, oxygen, salinity**
- Anomaly score = **next-step prediction error** using a lightweight learner (**RoLA-Lite**)

#### RoLA-Lite local learner (lightweight online predictor)
- Windowed multivariate input → next-step multivariate prediction
- “Online” via repeated local updates per federated round

#### Client-aware aggregation
- Server aggregates updates using **client-aware weights** (volume × quality proxy)
- Simulates realistic settings where sensors differ in noise/drift/missingness

#### Communication efficiency (supports CV claims)
- **Top-k sparsification:** send only the largest-magnitude parameter updates
- **Selective updates:** clients skip sending tiny updates
- Logs per-round **communication reduction** vs dense FedAvg

#### Partial connectivity (realistic networking constraint)
- Only a subset of clients participate per round (configurable participation rate)

#### “PhD-proof” experiment design
- **Baselines & Ablations**:
  - `centralized` (pooled training baseline)
  - `fedavg_dense` (FedAvg without compression)
  - `fed_topk` (compression only)
  - `fed_topk_selective` (**compression + selective updates**)
- **Artifacts saved per run**
- **Multi-seed evaluation (mean ± std)** for paper-style reporting
- Streamlit dashboard for **live demonstrations** + saving outputs

---

### Methods Summary (high-level)

1. **Synthetic buoy data generation** produces heterogeneous sensor streams with:
   - seasonality, noise, gradual drift, missingness
   - injected anomalies (spikes/drops/stuck sensor behaviour)
2. **RoLA-Lite** learns next-step prediction from sliding windows.
3. **Federated training** runs in rounds:
   - partial client participation
   - local training on each client
   - compressed + selective updates to server
4. **Evaluation** computes per-client:
   - ROC-AUC
   - Average Precision (AP)

---

### Project Structure
```bash
federated-marine-anomaly/
├─ src/
│ ├─ simulate_data.py # synthetic heterogeneous buoy data generator
│ ├─ models.py # RoLA-Lite predictor
│ ├─ federated.py # client/server, client-aware weighting, top-k, selective update, toy masking
│ ├─ evaluate.py # ROC-AUC + Avg Precision
│ ├─ viz.py # plots saved to runs
│ └─ demo_phd.py # single-file Streamlit demo (comparison + multi-seed mean±std)
├─ runs/ # auto-generated artifacts (ignored in git by default)
├─ requirements.txt
└─ README.md
```

---
#### Installation
```bash
pip install -r requirements.txt
cd src
streamlit run demo_phd.py
```
#### Evidence / Artifacts (Proof for CV Claims)
runs/run-YYYYMMDD-HHMMSS-<mode>/

Inside you get:
    -config.json — full experimental configuration + environment versions
    -metrics_round.csv — per-round comm reduction, bytes sent, clients sent
    -metrics_clients.csv — per-client ROC-AUC/AP (robustness under heterogeneity)
    -comm_reduction.png — communication efficiency plot
    -roc_auc.png, avg_precision.png — robustness plots
    -summary.json — headline metrics for quick reporting

---
#### Installation
Multi-seed Evaluation (Paper-style, mean ± std)

In the Streamlit sidebar:
    -choose method (e.g. fed_topk_selective)
    -choose N seeds
    -click Run multi-seed (mean ± std)

#### How to Cite
```bash
@misc{orokpo_federatedmarineanomaly,
  title = {Federated Lightweight Anomaly Detection for Distributed Marine Sensor Nodes},
  author = {Moses, Samuel},
  year = {2025},
  howpublished = {GitHub repository},
}
```

