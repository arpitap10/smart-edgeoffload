import time


class EdgeMonitor:
    """
    Lightweight metrics store.

    For publishable experiments, metrics should be recorded from the simulator's
    workload-driven state rather than randomly fabricated inside this class.
    """

    def __init__(self):
        self.timestamps = []
        self.latency_series = []
        self.bandwidth_series = []
        self.cpu_series = []
        self.queue_series = []

    def collect_metrics(
        self,
        latency: float = 0.0,
        bandwidth: float = 0.0,
        cpu_usage: float = 0.0,
        queue_length: float = 0.0,
    ):
        timestamp = time.time()
        self.timestamps.append(timestamp)
        self.latency_series.append(latency)
        self.bandwidth_series.append(bandwidth)
        self.cpu_series.append(cpu_usage)
        self.queue_series.append(queue_length)
        return {
            "timestamp": timestamp,
            "latency": latency,
            "bandwidth": bandwidth,
            "cpu": cpu_usage,
            "queue_length": queue_length,
        }

    def get_time_series(self):
        return {
            "timestamps": self.timestamps,
            "latency": self.latency_series,
            "bandwidth": self.bandwidth_series,
            "cpu": self.cpu_series,
            "queue_length": self.queue_series,
        }
