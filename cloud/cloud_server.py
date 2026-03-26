"""Cloud Server - Flask API for task execution"""

from flask import Flask, request, jsonify
import time
from shared.data_models import ExecutionResult, IoTTask

app = Flask(__name__)

class CloudExecutor:
    """Executes tasks in cloud (unlimited capacity)"""

    def __init__(self):
        self.tasks_executed = 0
        self.total_execution_time = 0

    def execute(self, task: IoTTask) -> ExecutionResult:
        """Execute task in cloud"""
        start_time = time.time()

        # Simulate cloud execution (faster but with network latency)
        execution_time = 0.005 * task.task_size  # seconds (faster)
        time.sleep(execution_time)

        completion_time = time.time() - start_time
        self.tasks_executed += 1
        self.total_execution_time += completion_time

        return ExecutionResult(
            status="success",
            location="cloud",
            completion_time=completion_time
        )

# Global cloud executor instance
cloud_executor = CloudExecutor()

@app.route('/execute_task', methods=['POST'])
def execute_task():
    """Execute IoT task in cloud"""
    try:
        data = request.get_json()

        # Create IoTTask from request data
        task = IoTTask(
            device_id=data.get('device_id', 'unknown'),
            timestamp=data.get('timestamp', time.time()),
            task_size=data.get('task_size', 1.0),
            payload=data.get('payload', {})
        )

        print(f"[Cloud Server] Received task from {task.device_id}, size: {task.task_size}MB")

        # Execute task
        result = cloud_executor.execute(task)

        response = {
            "status": result.status,
            "location": result.location,
            "latency": result.completion_time
        }

        print(f"[Cloud Server] Task completed in {result.completion_time:.3f}s")

        return jsonify(response), 200

    except Exception as e:
        print(f"[Cloud Server] Error: {str(e)}")
        return jsonify({
            "status": "error",
            "location": "cloud_error",
            "latency": 0.0,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "tasks_executed": cloud_executor.tasks_executed
    })

if __name__ == '__main__':
    print("Starting Cloud Server on http://localhost:8000")
    print("Endpoints:")
    print("  POST /execute_task - Execute IoT task")
    print("  GET /health - Health check")
    app.run(host='0.0.0.0', port=8000, debug=True)