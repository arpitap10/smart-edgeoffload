import time
import random
import psutil


class EdgeMonitor:
    def __init__(self):
        # time series storage
        self.timestamps = []
        self.latency_series = []
        self.bandwidth_series = []
        self.cpu_series = []
        self.queue_series = []

    def collect_metrics(self):
        """
        Collect system/network metrics at the edge node
        """

        # timestamp
        timestamp = time.time()

        # CPU usage (real system metric)
        cpu_usage = psutil.cpu_percent(interval=1)

        # simulated network metrics (for experiments/simulator)
        latency = random.uniform(10, 100)        # ms
        bandwidth = random.uniform(50, 200)      # Mbps
        queue_length = random.randint(1, 20)     # packets/tasks waiting

        # store in time series
        self.timestamps.append(timestamp)
        self.latency_series.append(latency)
        self.bandwidth_series.append(bandwidth)
        self.cpu_series.append(cpu_usage)
        self.queue_series.append(queue_length)

        metrics = {
            "timestamp": timestamp,
            "latency": latency,
            "bandwidth": bandwidth,
            "cpu": cpu_usage,
            "queue_length": queue_length
        }

        return metrics

    def get_time_series(self):
        """
        Return collected metrics as time series
        """

        return {
            "timestamps": self.timestamps,
            "latency": self.latency_series,
            "bandwidth": self.bandwidth_series,
            "cpu": self.cpu_series,
            "queue_length": self.queue_series
        }
    