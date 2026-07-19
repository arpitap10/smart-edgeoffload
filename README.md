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
│   ├── congestion_predictor.py   # Holt-Winters forecaster + naive/ARIMA/SES alternatives
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
│   └── device_simulator.py       # IoT device / workload generator
├── shared/
│   ├── config.py                 # simulation parameters and constants
│   └── data_models.py            # shared data structures
├── experiments/
│   ├── run_experiments.py        # main policy comparison across seeds/traffic models
│   ├── forecasting_metrics.py    # forecast accuracy (MAE/RMSE/sMAPE) per forecaster
│   ├── sensitivity_sweep.py      # Holt-Winters and cost-function parameter sensitivity
│   ├── significance_test.py      # paired statistical significance tests
│   ├── stress_test.py            # behavior under extreme, beyond-normal congestion
│   └── (output CSVs and figures written here after each run)
├── main.py                       # single end-to-end simulation entry point
└── requirements.txt
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

All scripts below live in `experiments/` and write their output CSVs/figures back into that same folder.

```bash
cd experiments

python run_experiments.py       # Main policy comparison across seeds and traffic models
python forecasting_metrics.py   # Forecast accuracy (MAE / RMSE / sMAPE) per forecaster
python sensitivity_sweep.py     # Sensitivity of Holt-Winters and cost-function parameters
python significance_test.py     # Paired statistical significance tests between policies
python stress_test.py           # Behavior under extreme, beyond-normal-range congestion
```

Everything is run over 15 random seeds (`7, 19, 42, 101, 123, 256, 314, 500, 613, 728, 841, 955, 1001, 1122, 1337`) and two traffic models (an abrupt "step" burst pattern and a gradual "ramped" burst pattern), except the sensitivity and stress-test scripts, which use the first 3 seeds for runtime reasons. No external data or API access is required — the whole thing runs offline.
