from dataclasses import dataclass
from typing import Dict
import time


@dataclass
class IoTTask:
    device_id: str
    timestamp: float
    task_size: float
    payload: Dict


@dataclass
class NetworkMetrics:
    latency_ms: float
    packet_loss: float
    queue_length: int
    cpu_usage: float


@dataclass
class ExecutionResult:
    status: str
    location: str
    completion_time: float