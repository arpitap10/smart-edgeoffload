from dataclasses import dataclass

@dataclass
class IoTTask:
    task_id: int
    size: float          # in MB
    compute: float       # CPU cycles per unit
    latency_req: float   # seconds

@dataclass
class ExecutionResult:
    task_id: int
    location: str
    execution_time: float
    energy: float