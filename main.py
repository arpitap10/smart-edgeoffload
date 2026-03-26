from edge.monitoring import EdgeMonitor
from edge.congestion_predictor import CongestionPredictor
from edge.decision_engine import DecisionEngine


class SmartEdgeOffloadFramework:
    def __init__(self, simulator, edge_executor, cloud_executor):
        self.simulator = simulator
        self.edge_executor = edge_executor
        self.cloud_executor = cloud_executor

        # Initialize monitoring and decision components
        self.monitor = EdgeMonitor()
        self.predictor = CongestionPredictor()
        self.decision_engine = DecisionEngine()

    def decide_offloading(self, task):
        # Implement logic for adaptive task offloading decisions here
        # Collect current metrics
        metrics = self.monitor.collect_metrics()

        # Get queue series for prediction
        queue_series = self.monitor.queue_series[-10:]  # last 10 measurements

        # Predict future congestion
        predicted_queue = self.predictor.predict_congestion(queue_series)

        # Check if edge can physically execute the task
        if not self.edge_executor.can_execute(task):
            return "cloud"

        # Make decision using decision engine
        decision = self.decision_engine.decide_execution(predicted_queue, task)

        return decision

    def run_task(self, task):
        decision = self.decide_offloading(task)
        if decision == 'edge':
            result = self.edge_executor.execute(task)
        else:
            result = self.cloud_executor.execute(task)
        return result