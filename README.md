# Smart Edge Offload

Proactive IoT task offloading using **Holt-Winters exponential smoothing** to forecast edge-queue congestion and route tasks before backlogs peak — without the training overhead of reinforcement learning.

---

## Overview

IoT devices generate bursty, heterogeneous workloads that must be executed under tight latency constraints.  Traditional offloading approaches are either static (always-edge, always-cloud, threshold rules) or reactive — they only respond to congestion they can already observe.

This system introduces a **predictive offloading policy** that forecasts future queue backlog using damped-trend Holt-Winters smoothing, then feeds that forecast into a multi-objective cost function that jointly minimises latency, energy, and congestion.

```
IoT Task arrives
      │
      ▼
CongestionPredictor          ← Holt-Winters (damped trend)
  predicts edge & cloud         forecasts next-slot backlog
  backlog one step ahead
      │
      ▼
DecisionEngine               ← multi-objective weighted cost
  compute_cost(delay,            α·delay + β·energy + γ·congestion
               energy,           with deadline-miss penalty
               predicted_backlog)
      │
      ├─── edge cost ≤ cloud cost  →  execute locally
      └─── cloud cost  <  edge cost →  offload to cloud
```

---

## Repository Structure

```
smart-edgeoffload/
├── edge/
│   ├── congestion_predictor.py   # Holt-Winters forecaster (our contribution)
│   ├── decision_engine.py        # multi-objective cost-based routing
│   ├── edge_executor.py          # local execution model
│   └── monitoring.py             # lightweight metrics store
├── cloud/
│   ├── cloud_api.py              # HTTP client with simulation fallback
│   ├── cloud_server.py           # FastAPI server (deploy on cloud instance)
│   ├── executor.py               # cloud execution model
│   ├── scheduler.py              # task-to-worker dispatcher
│   └── workers.py                # cloud worker helper
├── simulator/
│   └── device_simulator.py       # realistic IoT workload generator
│                                   (vision / telemetry / analytics profiles)
├── shared/
│   ├── data_models.py            # IoTTask, ExecutionResult, NodeState
│   └── config.py                 # system-wide constants
├── experiments/
│   ├── run_experiments.py        # five-policy benchmark  ← main entry point
│   ├── summary_results.csv       # aggregated results (mean across seeds)
│   └── per_seed_results.csv      # per-seed raw results
└── main.py                       # SmartEdgeOffloadFramework API
```

---

## Installation

```bash
git clone "https://github.com/arpitap10/smart-edgeoffload"
cd smart-edgeoffload
pip install -r requirements.txt
```

**Requirements:** `numpy`, `matplotlib`, `statsmodels`, `requests`, `psutil`, `pandas`

`statsmodels` is optional — the predictor automatically falls back to a built-in pure-Python Holt-Winters implementation if it is not installed.

---

## Running the Experiments

```bash
python experiments/run_experiments.py
```

This runs all five policies across three seeds (7, 19, 42) in **simulation mode** (no network required) and writes:

| Output file | Contents |
|---|---|
| `experiments/summary_results.csv` | Mean ± std across seeds for all metrics |
| `experiments/per_seed_results.csv` | Raw per-seed numbers |
| `experiments/policy_comparison.png` | 3-panel bar chart: latency, violations, energy |
| `experiments/predictive_backlog_trace.png` | Queue evolution under the predictive policy |
| `experiments/forecast_mae_comparison.png` | HW predictor MAE vs naive persistence |
| `experiments/reactive_vs_predictive.png` | Key pairwise comparison |

To run against a live cloud server instead, set `USE_REAL_CLOUD = True` in `run_experiments.py`.

---

## Policies Compared

| Policy | Description |
|---|---|
| `edge_only` | Always execute locally — lower bound on cloud cost |
| `cloud_only` | Always offload — lower bound on local congestion |
| `threshold` | Rule-based: offload if backlog > 0.9 s or task > 5.5 MB |
| `reactive` | Cost-based using *current* observed backlog |
| **`predictive`** | **Cost-based using HW-forecast backlog (our method)** |

---

### Why not raw forecast MAE?

For mean-reverting queue series, naive persistence (last observed value) is a strong one-step MAE competitor — this is a well-known property of stationary-ish processes.  The contribution of Holt-Winters is not lower raw MAE but **early congestion detection**: the trend component rises before the backlog peaks, shifting tasks to the cloud one scheduling slot ahead of the spike.  The `rolling_mae()` method in `CongestionPredictor` quantifies this; the full discussion is in the accompanying paper.

---

## Task Workload Profiles

The simulator generates three realistic IoT task types:

| Profile | Size | Compute | Deadline | Weight |
|---|---|---|---|---|
| Vision | 3–8 MB | 18–34 Mcycles/MB | 1.2–2.6 s | 35 % |
| Telemetry | 0.2–1.1 MB | 2–8 Mcycles/MB | 0.2–0.9 s | 40 % |
| Analytics | 1–4 MB | 10–22 Mcycles/MB | 1.3–3.4 s | 25 % |

Telemetry tasks carry the tightest deadlines and drive most latency violations under heavy load.

---

## Cloud Server (optional)

To deploy the cloud endpoint:

```bash
pip install fastapi uvicorn
uvicorn cloud.cloud_server:app --host 0.0.0.0 --port 8000
```

The client in `cloud/cloud_api.py` automatically falls back to local simulation if the server is unreachable, so experiments work offline.

---

## Using the Framework API

```python
from main import SmartEdgeOffloadFramework
from simulator.device_simulator import IoTSimulator

sim = IoTSimulator(seed=42)
fw  = SmartEdgeOffloadFramework(simulator=sim)

edge_history  = [0.0]
cloud_history = [0.0]

for _ in range(20):
    task   = sim.generate_task()
    result = fw.run_task(
        task,
        current_edge_backlog=edge_history[-1],
        current_cloud_backlog=cloud_history[-1],
        edge_history=edge_history,
        cloud_history=cloud_history,
    )
    print(f"Task {task.task_id} → {result.location}  "
          f"time={result.execution_time:.3f}s  energy={result.energy:.4f}J")
```
