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
│   ├── run_experiments.py        # seven-policy benchmark  ← main entry point
│   ├── sensitivity_sweep.py      # hyperparameter sensitivity analysis (alpha/beta/phi/delta/weights)
│   ├── stress_test.py            # extreme-burst robustness check, all policies
│   ├── summary_results.csv       # aggregated results (mean across seeds)
│   ├── per_seed_results.csv      # per-seed raw results
│   └── significance_results.csv  # paired t-test / Wilcoxon results vs. predictive
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

This runs all **seven** policies across **fifteen seeds** (7, 19, 42, 101, 123, 256, 314, 500, 613, 728, 841, 955, 1001, 1122, 1337) in **simulation mode** (no network required) and writes:

| Output file | Contents |
|---|---|
| `experiments/summary_results.csv` | Mean ± std across seeds for all metrics |
| `experiments/per_seed_results.csv` | Raw per-seed numbers |
| `experiments/significance_results.csv` | Paired t-test / Wilcoxon signed-rank results (predictive vs. reactive, naive-persistence, ARIMA, threshold) |
| `experiments/fig1`–`fig8_*.png` | Line/bar figures referenced in the paper (latency, violations, energy, backlog trace, summary, reactive-vs-predictive, HW forecast-vs-actual, rolling violation rate) |
| `experiments/fig_*_multiline.png` | Per-metric, per-policy line charts across seeds |

To run against a live cloud server instead, set `USE_REAL_CLOUD = True` in `run_experiments.py` (this is the default). Note `estimate()` — used for every routing decision — never touches the network; only `execute()` does, with an automatic fallback to the local simulation model if the request fails or times out, so a run can never crash due to network issues.

The original 3-seed configuration (7, 19, 42) used in the first submission's Table II is preserved as the first three entries of `SEEDS`, so those specific numbers remain reproducible as a subset of the expanded run.

### Additional experiment scripts

```bash
python experiments/sensitivity_sweep.py   # hyperparameter sensitivity analysis
python experiments/stress_test.py         # extreme burst / high-variance robustness check
```

`sensitivity_sweep.py` varies alpha, beta, phi (Holt-Winters), the offload margin delta, and the asymmetric blend weights (w_rising/w_easing) one-at-a-time around their paper-reported defaults, and reports violation%/latency sensitivity for each — this directly answers the "were these hyperparameters tuned, and how sensitive is performance to them?" question raised in review. It reuses `run_policy()` unchanged: `DecisionEngine` and `CongestionPredictor` both accept every relevant hyperparameter as a constructor argument, so no simulation logic needed to be duplicated.

`stress_test.py` re-runs all seven policies under a widened burst/network-degradation scenario (burst probability 0.12→0.30, burst duration 7–13→15–25 slots, bandwidth 18–36→6–18 Mbps) alongside the paper's original ("normal") scenario, to check whether the predictive policy's advantage holds outside the workload it was originally tuned on.

---

## Policies Compared

| Policy | Description |
|---|---|
| `edge_only` | Always execute locally — lower bound on cloud cost |
| `cloud_only` | Always offload — lower bound on local congestion |
| `threshold` | Rule-based: offload if backlog > 0.9 s or task > 5.5 MB |
| `reactive` | Cost-based using *current* observed backlog |
| `naive_persistence` | Cost-based using a next-slot = last-observed forecast — **ablation control**: same `DecisionEngine` as `predictive`, only the forecaster differs, isolating the Holt-Winters component's actual contribution |
| `arima` | Cost-based using a low-order ARIMA(1,1,0) forecast — classical-forecaster comparison point, same `DecisionEngine` as `predictive` |
| **`predictive`** | **Cost-based using HW-forecast backlog (our method)** |

`naive_persistence` and `arima` were added specifically to answer the reviewer question "how do we know the gain comes from Holt-Winters and not just from forecasting *something*?" — both run through the identical cost function and decision engine as `predictive`; only the value fed into `predicted_edge_backlog`/`predicted_cloud_backlog` changes.

---

### Why not raw forecast MAE?

For mean-reverting queue series, naive persistence (last observed value) is a strong one-step MAE competitor — this is a well-known property of stationary-ish processes. In fact, on this workload the Holt-Winters blend's raw MAE is typically *higher* than naive persistence's, not lower (see `summary_results.csv`'s `edge_pred_mae` vs `edge_naive_mae` columns). The contribution of Holt-Winters is not lower raw MAE but **early congestion detection**: the trend component rises before the backlog peaks, shifting tasks to the cloud one scheduling slot ahead of the spike — which is why `predictive` still outperforms `naive_persistence` and `arima` on violation rate and latency (see `significance_results.csv`) despite not having a lower forecast error. The `rolling_mae()` method (available on every predictor class in `congestion_predictor.py`) quantifies this; the full discussion is in the accompanying paper.

---

### Statistical rigor

`run_experiments.py` now runs 15 seeds (up from 3) and computes paired significance tests (`significance_test()` / `run_significance_suite()`) between `predictive` and each of `reactive`, `naive_persistence`, `arima`, and `threshold`, on violation%, latency, and energy. Both a paired t-test and a Wilcoxon signed-rank test are reported, along with a 95% confidence interval on the mean paired difference. Results are written to `experiments/significance_results.csv`. This directly addresses the "no standard deviations, confidence intervals, or significance tests" concern raised in review — the original 3-seed results are too few to test formally, which is exactly why the seed count was expanded.

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
