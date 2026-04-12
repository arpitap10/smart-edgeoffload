"""
cloud_server.py
===============
Minimal FastAPI server that runs on the cloud instance (13.53.132.84:8000).

Start with:
    pip install fastapi uvicorn
    uvicorn cloud.cloud_server:app --host 0.0.0.0 --port 8000

The server receives IoT tasks as JSON, simulates processing, and returns
execution metadata.  The client (CloudAPI) falls back to local simulation
if the server is unreachable.
"""

from __future__ import annotations

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

import time

if _FASTAPI_AVAILABLE:
    app = FastAPI(title="Smart EdgeOffload — Cloud Server")

    class TaskRequest(BaseModel):
        task_id:     int
        size:        float   # MB
        compute:     float   # mega-cycles per MB
        latency_req: float   # seconds

    class TaskResponse(BaseModel):
        task_id:        int
        execution_time: float
        energy:         float
        location:       str

    COMPUTE_POWER = 900.0    # mega-cycles per second (cloud)
    TX_POWER_W    = 1.35
    BASE_BW_MBPS  = 80.0

    @app.post("/execute_task", response_model=TaskResponse)
    def execute_task(req: TaskRequest) -> TaskResponse:
        start        = time.time()
        compute_time = (req.size * req.compute) / COMPUTE_POWER
        time.sleep(min(compute_time, 0.05))   # simulate bounded processing
        elapsed      = time.time() - start
        tx_time      = (req.size * 8.0) / BASE_BW_MBPS
        energy       = TX_POWER_W * tx_time + 0.12 * 0.045
        return TaskResponse(
            task_id=req.task_id,
            execution_time=elapsed,
            energy=round(energy, 6),
            location="cloud",
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

else:
    # Stub so the module can be imported without FastAPI installed
    app = None  # type: ignore
