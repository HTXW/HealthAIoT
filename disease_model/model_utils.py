import torch
import ssl
from ucimlrepo import fetch_ucirepo  # module to import CDC dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from pathlib import Path
import joblib

model_dir = Path(__file__).parent / 'best_model.pth'

ssl._create_default_https_context = ssl._create_unverified_context

# Import CDC dataset from https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
def load_data_cdc_diabetes(batch_size, device, save_scaler_path='scaler.pkl'):
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
    X = cdc_diabetes_health_indicators.data.features
    y = cdc_diabetes_health_indicators.data.targets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.7, random_state=42)
    scaler = StandardScaler()
    X_train= scaler.fit_transform(X_train)
    joblib.dump(scaler, save_scaler_path)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    smote = SMOTE(random_state=42)

    X_train, y_train = smote.fit_resample(X_train, y_train)
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values.squeeze(), dtype=torch.long).to(device)
    y_val = torch.tensor(y_val.values.squeeze(), dtype=torch.long).to(device)
    y_test = torch.tensor(y_test.values.squeeze(), dtype=torch.long).to(device)

    cdc_train = torch.utils.data.TensorDataset(X_train, y_train)
    cdc_val = torch.utils.data.TensorDataset(X_val, y_val)
    cdc_test = torch.utils.data.TensorDataset(X_test, y_test)
    return(torch.utils.data.DataLoader(cdc_train, batch_size, shuffle=True, drop_last=True),
           torch.utils.data.DataLoader(cdc_val, batch_size, shuffle=False, drop_last=True),
           torch.utils.data.DataLoader(cdc_test, batch_size, shuffle=False, drop_last=True))


class DiabetesClassifier(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiabetesClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(in_channels, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.relu1 = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(128, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.relu2 = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(p= 0.2423792950380868)
        self.fc4 = torch.nn.Linear(64, out_channels)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.fc4(out)
        return out

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

# predicte the label to the model output
def correct(out, y):
    y_hat = out.argmax(axis=1)
    return (y_hat == y).float().sum()

# evaluate average correct prediction
def evaluate_metric(model, data_iter, metric):
    c = 0.
    n = 0.
    for X, y in data_iter:
        out = model(X)
        c += metric(out, y)
        n += len(y)
    return c / n


class EarlyStopping:
    def __init__(self, wait_epoch=25, index=False, const=0):
        self.wait_epoch = wait_epoch  # number of epoch model will for early stopping
        self.max_val_acc = -float('inf')
        self.prime_epoch = 0
        self.const = const
        self.index = index
        self.counter = 0
        self.current_max = None
        self.early_stop = False

    def __call__(self, val_acc, model, epoch):
        validation_acc = val_acc
        if self.current_max is None:
            self.current_max = validation_acc
            self.best_val_acc_save(val_acc, model)
            self.prime_epoch = epoch

        elif validation_acc < self.current_max + self.const:
            self.counter += 1
            if self.index:
                print(f'EarlyStopping counter: {self.counter} out of {self.wait_epoch}')
            if self.counter >= self.wait_epoch:
                self.early_stop = True
        else:  # check and assign the max validation accuracy to current_max
            self.current_max = validation_acc
            self.best_val_acc_save(val_acc, model)
            self.counter = 0
            self.prime_epoch = epoch

    def best_val_acc_save(self, val_acc, model):
        if self.index:
            print(f'Validation accuracy increased ({self.max_val_acc:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), model_dir)
        self.max_val_acc = val_acc