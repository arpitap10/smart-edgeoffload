from dataclasses import dataclass


@dataclass
class IoTTask:
    task_id: int
    size: float          # MB
    compute: float       # mega-cycles per MB
    latency_req: float   # seconds

    @property
    def compute_demand(self) -> float:
        """Total workload in mega-cycles."""
        return self.size * self.compute


@dataclass
class ExecutionResult:
    task_id: int
    location: str
    execution_time: float
    energy: float


@dataclass
class NodeState:
    backlog: float = 0.0             # queued service time in seconds
    bandwidth_mbps: float = 0.0
    rtt: float = 0.0
