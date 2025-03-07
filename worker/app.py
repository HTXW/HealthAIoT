import time
from pathlib import Path
from flask import Flask, request, jsonify
import torch
import joblib
import numpy as np
import psutil
import threading
from datetime import datetime
from disease_model.model_utils import DiabetesClassifier
import json
import os
import werkzeug.serving
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

parent_dir = Path(__file__).resolve().parent
model_path = parent_dir / 'best_model.pth'
scaler_path = parent_dir / 'scaler.pkl'
in_channels = 21
out_channels = 2
predictor_model = DiabetesClassifier(in_channels, out_channels)
predictor_model.load_state_dict(torch.load(model_path))
predictor_model.eval()
predictor_scaler = joblib.load(scaler_path)


def input_normalisation(features):
    features = np.array(features).reshape(1, -1)
    features = predictor_scaler.transform(features)
    return torch.tensor(features, dtype=torch.float32)

# create a flask POST to post prediction result on the webpage
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    user_data = request.json
    features = user_data['features']
    input_tensor = input_normalisation(features)
    with torch.no_grad():
        labels = predictor_model(input_tensor)
        y_pred = labels.argmax(axis=1).item()
    end_time = time.time()
    model_execution_time = end_time - start_time
    print(f'Model execution time: {model_execution_time} seconds')
    # return a json file containing predicted labels and model execution time
    return jsonify({'prediction': y_pred, 'model_execution_time': model_execution_time})

# calculate send and received bandwidth of the worker
def get_worker_bw_data(interval=1):
    try:
        # calculate the number of bytes send and received at a particular time instance
        interfaces = psutil.net_io_counters(pernic=True)
        total_bytes_sent_start = total_bytes_recv_start = 0
        for interface, io_counter in interfaces.items():
            total_bytes_sent_start += io_counter.bytes_sent
            total_bytes_recv_start += io_counter.bytes_recv
        time.sleep(interval)

        # calculate the number of bytes send and received at time instance after 10 sec of the previous instance
        interfaces = psutil.net_io_counters(pernic=True)
        total_bytes_sent_end = total_bytes_recv_end = 0
        for interface, io_counter in interfaces.items():
            total_bytes_sent_end += io_counter.bytes_sent
            total_bytes_recv_end += io_counter.bytes_recv
        bytes_sent = total_bytes_sent_end - total_bytes_sent_start
        bytes_recv = total_bytes_recv_end - total_bytes_recv_start
        send_bandwidth = (bytes_sent * 8) / (interval * 1000000) # convert bytes to mbps
        recv_bandwidth = (bytes_recv * 8) / (interval * 1000000) # convert bytes to mbps
        return send_bandwidth, recv_bandwidth
    except Exception as e:
        print(f"Error in calculating Worker BW: {e}")
        return None, None

# declare the json file to store generated metric values
WORKER_STATS_FILE = 'system_stats.json'


# collect system metric statistics and store them in json file
def gather_stats():
    while True:
        try:
            print(f"Fetching system metric stats at {time.ctime()}")
            send_bw, recv_bw = get_worker_bw_data(interval=1)
            current_stats = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'cpu_utilization': psutil.cpu_percent(interval=1),
                'memory_usage_percent': psutil.virtual_memory().percent,
                'network_bandwidth': {
                    'send_bandwidth_mbps': round(send_bw, 2),
                    'recv_bandwidth_mbps': round(recv_bw, 2)
                }
            }
            if os.path.exists(WORKER_STATS_FILE):
                with open(WORKER_STATS_FILE, 'r') as f:
                    system_metric = json.load(f)
            else:
                system_metric = []
            system_metric.append(current_stats)
            system_metric = system_metric[-100:]  # limit the number of stored stats to latest 100 only
            with open(WORKER_STATS_FILE, 'w') as f:
                json.dump(system_metric, f, indent=4)
            print(f"Latest system metric stats saved at {time.ctime()}")
        except Exception as e:
            print(f"Error while collecting stats: {e}")
        time.sleep(10)  # generating stats in the interval 10 seconds

# creating a flask GET decorator to fetch and send the system's latest generated stats in json format
@app.route('/status', methods=['GET'])
def status():
    try:
        with open(WORKER_STATS_FILE, 'r') as f:
            system_metric = json.load(f)
        return jsonify(system_metric[-1])
    except Exception as e:  # handling exceptions
        print(f"Error using GET request for the latest stats: {e}")
        return jsonify({})


if __name__ == '__main__':
    if not werkzeug.serving.is_running_from_reloader():
        stats_thread = threading.Thread(target=gather_stats)
        stats_thread.daemon = True
        stats_thread.start()

    app.run(host='0.0.0.0', debug=True, port=5000)  # running the created flask app on port 5000