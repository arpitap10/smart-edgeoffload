This project simulates an IoT environment where tasks are dynamically offloaded to either:

⚡ Edge → Low latency, limited capacity

☁️ Cloud → High latency, scalable

Instead of static thresholds, the system uses Holt-Winters exponential smoothing to the problem of task offloading in edge-cloud environments. Unlike traditional threshold-based systems, this approach introduces a dynamic and context-aware decision mechanism that adapts to changing network conditions in real time.

The use of a cost-driven decision model further strengthens the system by incorporating multiple factors such as congestion and execution overhead into a unified framework. This results in more balanced resource utilization and improved overall system performance.
