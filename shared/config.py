"""Configuration for Smart EdgeOffload framework"""


class Config:
    # Simulation Parameters
    NUM_DEVICES             = 20
    TASK_GENERATION_RATE    = 10   # tasks per second
    SIMULATION_DURATION     = 60   # seconds

    # Network Parameters
    EDGE_LATENCY_MS         = 10   # ms
    CLOUD_LATENCY_MS        = 50   # ms  — used by CloudExecutor for RTT simulation
    EDGE_BANDWIDTH_MBPS     = 100
    CLOUD_BANDWIDTH_MBPS    = 1000

    # Forecasting Parameters
    FORECAST_WINDOW         = 30   # seconds
    PREDICTION_HORIZON      = 10   # seconds
    HOLT_WINTERS_SEASONAL   = 12

    # Offloading Thresholds
    LATENCY_THRESHOLD_MS    = 50
    PACKET_LOSS_THRESHOLD   = 0.05
    CPU_THRESHOLD           = 80
    QUEUE_LENGTH_THRESHOLD  = 15   # queue length above which tasks go to cloud

    # FIX: added missing threshold (was hardcoded as 8 in DecisionEngine)
    LARGE_TASK_THRESHOLD    = 8    # MB above which tasks go to cloud

    # Edge / Cloud Capacity
    EDGE_CPU_CAPACITY       = 4    # cores
    EDGE_MEMORY_GB          = 8
    CLOUD_CPU_CAPACITY      = 32
    CLOUD_MEMORY_GB         = 256

    # Legacy threshold (kept for backwards compatibility)
    OFFLOAD_LATENCY_THRESHOLD = 100