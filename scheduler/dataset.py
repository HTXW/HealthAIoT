import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def load_schedular_dataset(X_train_scaled, X_test_scaled, y_train, y_test, batch_size):
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    return(torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
           torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False))


class AIScheduler(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_vms):
        super(AIScheduler, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, num_vms)
        self.relu = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        return out


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


def correct(logits, y):
    y_hat = logits.argmax(axis=1)
    return (y_hat == y).float().sum()

def evaluate_metric(model, data_iter, metric):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, y in data_iter:
            logits = model(X)
            total_correct += metric(logits, y).item()
            total_samples += y.size(0)
    return total_correct / total_samples



