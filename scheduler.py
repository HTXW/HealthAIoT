import requests
import json
import os
import torch
import joblib
import threading
import numpy as np
import time
from datetime import datetime
import psutil
from flask import Flask, request, render_template, redirect, url_for
from pathlib import Path
from scheduler.dataset import AIScheduler
from disease_model.model_utils import DiabetesClassifier
import warnings

warnings.filterwarnings("ignore")

base_dir = Path(__file__).resolve().parent.resolve()
model_path = base_dir / 'worker/best_model.pth'
scaler_path = base_dir / 'worker/scaler.pkl'


app = Flask(__name__)
WORKER_IPS = [ '34.71.56.20', '35.189.104.124']
WORKER_PORT = 5000
OPTIMAL_SCHEDULER_MODEL = 'vm_selector_model.pth'
SCHEDULER_SCALER = 'scheduler_scaler.pkl'

optimal_scheduler_model = AIScheduler(input_size=8, hidden_size=32, num_vms=2)
optimal_scheduler_model.load_state_dict(torch.load(OPTIMAL_SCHEDULER_MODEL))
optimal_scheduler_model.eval()
scheduler_scaler = joblib.load(SCHEDULER_SCALER)


in_channels = 21
out_channels = 2
predictor_model = DiabetesClassifier(in_channels, out_channels)
predictor_model.load_state_dict(torch.load(model_path))
predictor_model.eval()
predictor_scaler = joblib.load(scaler_path)

def fetch_worker_stats(vm_ip):
    url = f'http://{vm_ip}:{WORKER_PORT}/status'  # url for request-response communication with worker
    try:
        response = requests.get(url,  timeout=(2, 5))
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching stats from {vm_ip}: {e}")
        return None

# combine system metric statistics of both workers as a dictionary
def combine_worker_stats(stats_list):
    combined_stats = {}
    for i, stats in enumerate(stats_list):
        combined_stats[f'cpu{i+1}'] = stats['cpu_utilization']
        combined_stats[f'mem{i+1}'] = stats['memory_usage_percent']
        combined_stats[f'recv{i+1}'] = stats['network_bandwidth']['recv_bandwidth_mbps']
        combined_stats[f'send{i+1}'] = stats['network_bandwidth']['send_bandwidth_mbps']
    return combined_stats

# predict the optimal VM based on currently generated system stats
def predict_optimal_worker(combined_stats):
    new_data_scaled = scheduler_scaler.transform([list(combined_stats.values())])
    new_data_tensor = torch.FloatTensor(new_data_scaled)
    
    with torch.no_grad():
        labels = optimal_scheduler_model(new_data_tensor)  # obtain scheduler model's predicted label weight
        y_pred = labels.argmax(dim=1).item()  # choose label with highest weight
    return y_pred

# combine combine_worker_stats and predict_optimal_worker functions and fetch the optimal VM stats
def get_optimal_worker():

    # fetch system metric for all the workers and save as an ordered list
    stats_list = [fetch_worker_stats(vm_ip) for vm_ip in WORKER_IPS]
    if None not in stats_list:
        combined_stats = combine_worker_stats(stats_list)  # call function to combine statistics of both workers
        optimal_worker_index = predict_optimal_worker(combined_stats)  # predict oprimal VM based on stats
        optimal_worker = WORKER_IPS[optimal_worker_index]  # fetch optimal worker IP
        optimal_worker_stats = fetch_worker_stats(optimal_worker)  # fetch and save optimal worker metric stats
        if optimal_worker_stats:
            save_optimal_worker_stats(f'worker_{optimal_worker_index + 1}', optimal_worker_stats) # save metric stats in json
        return WORKER_IPS[optimal_worker_index], optimal_worker_index + 1
    return WORKER_IPS[0],1  # return first VM as optimal if the model fails to predict the optimal one

# transfer user data on the selected optimal VM or to next available VM in case of any exception
def get_worker(features):
    optimal_worker, vm_index = get_optimal_worker() 
    for _ in range(len(WORKER_IPS)): # loop for definite number of tries in case of failed attempt
        try:
            url = f'http://{optimal_worker}:{WORKER_PORT}/predict' # send diabetes predction request to the optimal worker
            response = requests.post(url, json={'features': features}, timeout=(2,10))
            response.raise_for_status()
            print(f"User Data routed to Worker: {optimal_worker}")
            return response.json(), vm_index

        except Exception as e:
            print(f"Worker Error {optimal_worker}: {e}") 
            next_vm_index = (WORKER_IPS.index(optimal_worker) + 1) % len(WORKER_IPS) # Connect to next optimal worker
            optimal_worker = WORKER_IPS[next_vm_index] # use next worker VM as the optimal one in case of exception
            print(f"Connecting to other Worker: {optimal_worker}")
            vm_stats = fetch_worker_stats(optimal_worker) # fetch worker system metric statistics
            if vm_stats:
                save_optimal_worker_stats(f'worker_{next_vm_index + 1}', vm_stats)  # save the metric stats in json
    print("No Response from Worker")
    return None, None


def save_optimal_worker_stats(system, stats):
    file = 'worker_system_metric_stats.json'
    system_metric = []
    if os.path.exists(file):  # load stats file if it exists
        with open(file, 'r') as f:
            system_metric = json.load(f)
    system_metric.append({
        'system': system,
        'stats': stats
    })  # appending new stats

    with open(file, 'w') as f:
        json.dump(system_metric, f, indent=4)
    print(f"All System stats updated in {file}")

# save system stats from broker
def save_broker_stats():
    broker_stats = gather_broker_stats()
    save_optimal_worker_stats('broker', broker_stats)

#save temporal stats of the optimal VM
def save_temporal_stats(system, model_execution_time, total_execution_time, latency):
    time_stats = {
        'system': system,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_execution_time': model_execution_time,
        'total_execution_time': total_execution_time,
        'latency': latency
    }
    
    file = 'temporal_stats.json'
    system_metric = []
    if os.path.exists(file): # load stats file if it exists
        with open(file, 'r') as f:
            system_metric = json.load(f)
    system_metric.append(time_stats) # append new stats
    with open(file, 'w') as f:
        json.dump(system_metric, f, indent=4)
    print(f"Updated Temporal Statistics saved in {file}")

# calculate send and received bandwidth of the broker system
def get_broker_bw_data(interval=1):
    try:
        # calculate the number of bytes send and received at a particular time instance
        interfaces_start = psutil.net_io_counters(pernic=True)
        total_bytes_sent_start = total_bytes_recv_start = 0
        for interface, io_counter in interfaces_start.items():
            total_bytes_sent_start += io_counter.bytes_sent
            total_bytes_recv_start += io_counter.bytes_recv
        time.sleep(interval)

        # calculate the number of bytes send and received at time instance after 10 sec of the previous instance
        interfaces_end = psutil.net_io_counters(pernic=True)
        total_bytes_sent_end = total_bytes_recv_end = 0
        for interface, io_counter in interfaces_end.items():
            total_bytes_sent_end += io_counter.bytes_sent
            total_bytes_recv_end += io_counter.bytes_recv
        bytes_sent = total_bytes_sent_end - total_bytes_sent_start
        bytes_recv = total_bytes_recv_end - total_bytes_recv_start
        send_bandwidth = (bytes_sent * 8) / (interval * 1000000)  # to mbps
        recv_bandwidth = (bytes_recv * 8) / (interval * 1000000)  # to mbps
        return send_bandwidth, recv_bandwidth
    except Exception as e:
        print(f"Error in calculating broker BW: {e}")
        return None, None

# collect system metric statistics for broker
def gather_broker_stats():
    send_bw, recv_bw = get_broker_bw_data(interval=1)
    stats = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'cpu_utilization': psutil.cpu_percent(interval=1),
        'memory_usage_percent': psutil.virtual_memory().percent,
        'network_bandwidth': {
            'send_bandwidth_mbps': round(send_bw, 2),
            'recv_bandwidth_mbps': round(recv_bw, 2)
        }
    }
    return stats

# normalise received user data
def input_normalisation(features):
    features = np.array(features).reshape(1, -1)
    features = predictor_scaler.transform(features)
    return torch.tensor(features, dtype=torch.float32)

# predict the diabetes risk on broker
def broker_diabetes_prediction(features):
    input_tensor = input_normalisation(features)
    with torch.no_grad():
        labels = predictor_model(input_tensor)  # obtain the model's predicted label weight
        y_pred = labels.argmax(axis=1).item()  # choose the label with highest weight
    return y_pred

# cpu_utilization = psutil.cpu_percent(interval=1)
'''creating a flask app GET and POST decorator to get the user data from the HealthAIoT webpage,
 send it to optimal worker for prediction and then send prediction result received to the user '''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':  # get features that user has input on webpage
        start_time = time.time() # start time of the operation
        try:
            features = [
                int(request.form['HighBP']),
                int(request.form['HighChol']),
                int(request.form['CholCheck']),
                int(request.form['BMI']),
                int(request.form['Smoker']),
                int(request.form['Stroke']),
                int(request.form['HeartDiseaseorAttack']),
                int(request.form['PhysActivity']),
                int(request.form['Fruits']),
                int(request.form['Veggies']),
                int(request.form['HvyAlcoholConsump']),
                int(request.form['AnyHealthcare']),
                int(request.form['NoDocbcCost']),
                int(request.form['GenHlth']),
                int(request.form['MentHlth']),
                int(request.form['PhysHlth']),
                int(request.form['DiffWalk']),
                int(request.form['Sex']),
                int(request.form['Age']),
                int(request.form['Education']),
                int(request.form['Income'])
            ]

            latency = time.time() - start_time
            try:
                # transfer user data to selected optimal VM or to next available VM in case of any exception
                prediction, vm_index = get_worker(features)
                if prediction:
                    if prediction['prediction'] == 0:
                        result_message = "Your results show no significant risk for diabetes."
                    else:
                        result_message = ("Your results indicate a potential risk for diabetes.<br>"
                                          "We recommend consulting a healthcare professional.")
                    model_execution_time = prediction['model_execution_time']
                    total_execution_time = time.time() - start_time
                    threading.Thread(target=save_temporal_stats,
                                     args=(f'worker_{vm_index}', model_execution_time, total_execution_time, latency),
                                     daemon=True).start()  # save worker temporal stats
                else:
                    raise Exception("Dibetes Prediction failed on Worker")

            except Exception as e:  # in case of exception with VMs request is handled by the broker
                print(f"Error with Workers: {e}, using broker for prediction")
                prediction = broker_diabetes_prediction(features)
                model_execution_time = time.time() - start_time - latency
                if prediction == 0:
                    result_message = "Your results show no significant risk for diabetes."
                else:
                    result_message = ("Your results indicate a potential risk for diabetes.<br>"
                                      "We recommend consulting a healthcare professional.")

                end_time = time.time()
                total_execution_time = end_time - start_time
                threading.Thread(target=save_temporal_stats,
                                 args=('broker', model_execution_time, total_execution_time, latency),
                                 daemon=True).start()
                threading.Thread(target=save_broker_stats, daemon=True).start()
            return render_template('result.html', result_message=result_message)  # return result to user

        except Exception as e:
            print("Error:", e)
            return str(e), 400
    return render_template('form.html')


@app.route('/refresh', methods=['GET'])
def refresh():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)  # run flask app on port 5002


