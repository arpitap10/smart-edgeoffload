"""Configuration for Smart EdgeOffload framework"""

class Config:
    # Simulation Parameters
    NUM_DEVICES = 20
    TASK_GENERATION_RATE = 10  # tasks per second
    SIMULATION_DURATION = 60  # seconds
    
    # Network Parameters
    EDGE_LATENCY_MS = 10
    CLOUD_LATENCY_MS = 50
    EDGE_BANDWIDTH_MBPS = 100
    CLOUD_BANDWIDTH_MBPS = 1000
    
    # Forecasting Parameters
    FORECAST_WINDOW = 30  # seconds
    PREDICTION_HORIZON = 10  # seconds
    HOLT_WINTERS_SEASONAL = 12
    
    # Offloading Thresholds
    LATENCY_THRESHOLD_MS = 50
    PACKET_LOSS_THRESHOLD = 0.05
    CPU_THRESHOLD = 80
    QUEUE_LENGTH_THRESHOLD = 40
    
    # Edge/Cloud Capacity
    EDGE_CPU_CAPACITY = 4
    EDGE_MEMORY_GB = 8
    CLOUD_CPU_CAPACITY = 32
    CLOUD_MEMORY_GB = 256
    
    # Edge offload to cloud if latency > threshold
    OFFLOAD_LATENCY_THRESHOLD = 100