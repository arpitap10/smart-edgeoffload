from flask import Flask, request, jsonify
from scheduler import schedule_task

app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process():
    data = request.json

    result = schedule_task(data)

    return jsonify({
        "status": result.status,
        "location": result.location,
        "completion_time": result.completion_time
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
