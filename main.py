"""Smart Edge Offload framework entry point."""

from cloud.executor import CloudExecutor
from edge.congestion_predictor import CongestionPredictor
from edge.decision_engine import DecisionEngine
from edge.edge_executor import EdgeExecutor
from edge.monitoring import EdgeMonitor


class SmartEdgeOffloadFramework:
    def __init__(self, simulator, edge_executor=None, cloud_executor=None):
        self.simulator = simulator
        self.edge_executor = edge_executor or EdgeExecutor()
        self.cloud_executor = cloud_executor or CloudExecutor()

        self.monitor = EdgeMonitor()
        self.predictor = CongestionPredictor()
        self.decision_engine = DecisionEngine()
        self.task_counter = 0

    def decide_offloading(
        self,
        task,
        current_edge_backlog: float,
        current_cloud_backlog: float,
        edge_history: list[float] | None = None,
        cloud_history: list[float] | None = None,
        bandwidth_mbps: float = 30.0,
        rtt: float = 0.055,
    ) -> str:
        self.task_counter += 1

        edge_history = edge_history or [current_edge_backlog]
        cloud_history = cloud_history or [current_cloud_backlog]
        predicted_edge = self.predictor.predict_congestion(edge_history, silent=True)
        predicted_cloud = self.predictor.predict_congestion(cloud_history, silent=True)

        self.monitor.collect_metrics(
            latency=rtt,
            bandwidth=bandwidth_mbps,
            cpu_usage=0.0,
            queue_length=current_edge_backlog,
        )

        edge_est = self.edge_executor.estimate(task, queue_backlog=current_edge_backlog)
        cloud_est = self.cloud_executor.estimate(
            task,
            queue_backlog=current_cloud_backlog,
            bandwidth_mbps=bandwidth_mbps,
            rtt=rtt,
        )

        decision, reason, _, _ = self.decision_engine.decide_with_reason(
            task,
            edge_est,
            cloud_est,
            current_edge_backlog=current_edge_backlog,
            current_cloud_backlog=current_cloud_backlog,
            predicted_edge_backlog=predicted_edge,
            predicted_cloud_backlog=predicted_cloud,
        )

        print(
            f"[Task {self.task_counter:04d}] "
            f"edge_backlog={current_edge_backlog:.3f}s "
            f"cloud_backlog={current_cloud_backlog:.3f}s "
            f"pred_edge={predicted_edge:.3f}s pred_cloud={predicted_cloud:.3f}s "
            f"-> {reason}"
        )
        return decision

    def run_task(
        self,
        task,
        current_edge_backlog: float,
        current_cloud_backlog: float,
        edge_history: list[float] | None = None,
        cloud_history: list[float] | None = None,
        bandwidth_mbps: float = 30.0,
        rtt: float = 0.055,
    ):
        decision = self.decide_offloading(
            task,
            current_edge_backlog=current_edge_backlog,
            current_cloud_backlog=current_cloud_backlog,
            edge_history=edge_history,
            cloud_history=cloud_history,
            bandwidth_mbps=bandwidth_mbps,
            rtt=rtt,
        )

        if decision == "edge":
            return self.edge_executor.execute(task, queue_backlog=current_edge_backlog)

        return self.cloud_executor.execute(
            task,
            queue_backlog=current_cloud_backlog,
            bandwidth_mbps=bandwidth_mbps,
            rtt=rtt,
        )
