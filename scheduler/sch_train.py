from pathlib import Path
from scheduler.dataset import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# calculate scoring metric of VMs
def scoring_function(cpu, mem, recv, send):
    sf = cpu + mem + recv + send
    return sf

# load Bitbrains dataset from http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains
bitbrain_df = pd.read_csv('scheduler/BitBrains61.csv', sep=';')

pre_processed_df = pd.DataFrame({'cpu': bitbrain_df['\tCPU usage [%]'],
                                 'mem_usage': bitbrain_df['\tMemory usage [KB]'] / 1024 / 1024,
                                 'mem_total': bitbrain_df['\tMemory capacity provisioned [KB]'] / 1024 / 1024,
                                 'recv': bitbrain_df['\tNetwork received throughput [KB/s]'] * 8 / 1000,
                                 'send': bitbrain_df['\tNetwork transmitted throughput [KB/s]'] * 8 / 1000,})

pre_processed_df = pre_processed_df[(pre_processed_df != 0).any(axis=1)]

# calculate VM memory usage percentage
pre_processed_df['mem'] = (pre_processed_df['mem_usage'] / pre_processed_df['mem_total'])*100
pre_processed_df = pre_processed_df.drop(['mem_usage', 'mem_total'], axis=1)
random_shuffled_df = pre_processed_df.sample(frac=1, random_state=42).reset_index(drop=True)
index = len(random_shuffled_df) // 2  # find middle index of the dataset
worker1_df = random_shuffled_df.iloc[:index]
worker2_df = random_shuffled_df.iloc[index:]

# place the splited rows side by side
custom_df = pd.DataFrame({  'cpu1': worker1_df['cpu'].values, 'mem1': worker1_df['mem'].values,
                            'recv1': worker1_df['recv'].values, 'send1': worker1_df['send'].values,
                            'cpu2': worker2_df['cpu'].values, 'mem2': worker2_df['mem'].values,
                            'recv2': worker2_df['recv'].values, 'send2': worker2_df['send'].values,})

# apply scoring function to calculate VMs metric
custom_df['sfvalue1'] = scoring_function(custom_df['cpu1'],
                                         custom_df['mem1'], custom_df['recv1'], custom_df['send1'])
custom_df['sfvalue2'] = scoring_function(custom_df['cpu2'],
                                         custom_df['mem2'], custom_df['recv2'], custom_df['send2'])

# create boolean column based on inequality expression
custom_df['optimal_worker'] = (custom_df['sfvalue2'] > custom_df['sfvalue1']).astype(int)
custom_df = custom_df.drop(['sfvalue1', 'sfvalue2'], axis=1)

X = custom_df.drop('optimal_worker', axis=1)
y = custom_df['optimal_worker']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_dir = Path(__file__).resolve().parent.parent.resolve()
scheduler_path = base_dir / 'scheduler_scaler.pkl'
vm_selector_model_path = base_dir / 'vm_selector_model.pth'
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, scheduler_path)
X_test_scaled = scaler.transform(X_test)

batch_size = 64
train_iter, test_iter = load_schedular_dataset(X_train_scaled, X_test_scaled, y_train, y_test, batch_size)

X, y = next(iter(train_iter)) 
print(X.size())
print(y.size())

input_size, hidden_size, num_vms = 8, 32, 2
model = AIScheduler(input_size, hidden_size, num_vms)
model.apply(init_weights)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

with open('scheduler_train_test_log.txt', 'w') as f:
    num_epochs = 45
    losses_per_train_batch = []
    train_accuracy = []
    test_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X, y in train_iter:
            optimizer.zero_grad()
            out = model(X)
            l = loss(out, y)
            l.backward()
            optimizer.step()
            total_loss += l.item()
            losses_per_train_batch.append(l.item())

        model.eval()
        with torch.no_grad():
            train_acc = evaluate_metric(model, train_iter, correct) # 
            test_acc = evaluate_metric(model, test_iter, correct)
            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)

            log_message = (f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss:.4f}, '
                           f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

            print(log_message)
            f.write(log_message + '\n')

torch.save(model.state_dict(), vm_selector_model_path)

