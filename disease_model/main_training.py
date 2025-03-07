import torch
import matplotlib.pyplot as plt
import numpy as np
import shap
from pathlib import Path
from .model_utils import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA.")
else:
    device = torch.device("cpu")
    print("Using CPU.")

base_dir = Path(__file__).resolve().parent.parent
plots_dir = base_dir / 'plot_figures'
logs_dir = Path(__file__).parent / 'diabetic_model_train_val_test_log.txt'
model_dir = Path(__file__).parent / 'best_model.pth'


batch_size = 128
train_iter, val_iter, test_iter = load_data_cdc_diabetes(batch_size, device)
X, y = next(iter(train_iter))
print(X.size())
print(y.size())

in_channels = 21
out_channels = 2
model = DiabetesClassifier(in_channels, out_channels)
model.apply(init_weights)
model.to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00677174682597258, weight_decay=3.103759723739499e-05)


early_stop = EarlyStopping(wait_epoch=150, index=True)

losses = []
train_accs = []
val_accs = []

with open(logs_dir, 'w') as f:
    num_epochs = 200
    for epoch in range(num_epochs):
        epoch_count = f'\nEpoch {epoch + 1}/{num_epochs}.\n'
        print(epoch_count)
        f.write(epoch_count)
        model.train()
        for X, y in train_iter:
            out = model(X)
            l = loss(out, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            losses.append(float(l))

        model.eval()
        with torch.no_grad():
            train_acc = evaluate_metric(model, train_iter, correct)
            val_acc = evaluate_metric(model, val_iter, correct)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            epoch_update = (f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {float(l):.4f}, '
                            f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
            print(epoch_update)
            f.write(epoch_update + '\n')
            early_stop(val_acc, model, epoch + 1)

            if early_stop.early_stop:
                early_stop_message = f"Early stopping triggered with validation accuracy: {early_stop.max_val_acc}"
                print(early_stop_message)
                f.write(early_stop_message + '\n')
                break
        max_acc_epoch = (f'Best validation accuracy so far: {early_stop.max_val_acc}. '
                         f'Best epoch so far: {early_stop.prime_epoch}.')
        print(max_acc_epoch)
        f.write(max_acc_epoch + '\n')
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        test_acc = evaluate_metric(model, test_iter, correct)
        test_accuracy = (f'Test accuracy: {test_acc}.\n'
                         f'Best validation accuracy: {early_stop.max_val_acc}, Best epoch: {early_stop.prime_epoch}')
        print(test_accuracy)
        with open(logs_dir, 'a') as f:
            f.write(test_accuracy + '\n')

labelpad_value = 8
font_size_on_bar = 12
xy_font_size = 15
title_font_size = 16

# Confusion Matrix Plot
preds_labels = []
actual_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_iter:
        labels = model(X_batch)
        _, predicted = torch.max(labels, 1)
        preds_labels.extend(predicted.cpu().numpy())
        actual_labels.extend(y_batch.cpu().numpy())

score = f1_score(actual_labels, preds_labels, average='weighted')
print(f'The F1-score for Diabetes Prediction Model is: {score:.4f}')

cm = confusion_matrix(actual_labels, preds_labels)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # standardise matrix values


fig, ax = plt.subplots(figsize=(9, 7), dpi=800)
cax = ax.matshow(cm_norm, cmap='Pastel1')
plt.colorbar(cax)
labels = ['Class 0', 'Class 1']
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, f'{cm_norm[i, j]:.2%}',
                ha='center', va='center', color='black')
        if i == 0 and j == 0:
            label = "True Negative"
        elif i == 0 and j == 1:
            label = "False Positive"
        elif i == 1 and j == 0:
            label = "False Negative"
        else:
            label = "True Positive"
        ax.text(j, i + 0.1, f'({label})', ha='center', va='center', color='black')

plt.xlabel('Model Prediction (Diabetic / Not Diabetic)', fontsize=xy_font_size, labelpad=8)
plt.ylabel('True Outcome', fontsize=xy_font_size, labelpad=8)
plt.title('Confusion Matrix of Diabetes Predictor Model', fontsize=title_font_size, fontweight='bold', ha='center', pad=12)
plt.savefig(plots_dir / 'confusion_matrix_with_percentages.png')


# SHAP Swarm Plot
def generate_shap_swarm_plot():
    cdc_data = fetch_ucirepo(id=891)
    feature_names = list(cdc_data.data.features.columns)

    # Define model prediction function for class 1 probabilities
    def predict_fn(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
        return probs[:, 1].cpu().numpy()

    background = X[:200].cpu().numpy()
    test_data_batch, _ = next(iter(test_iter))
    test_samples = test_data_batch[:200].cpu().numpy()

    # SHAP explainer
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(test_samples, nsamples=200)
    if isinstance(shap_values, list) and len(shap_values) == 1:
        shap_values = shap_values[0]

    plt.figure(figsize=(10, 8))
    plt.xlabel('SHAP Value (Impact on Model Output)', labelpad=8, fontsize=12)
    plt.ylabel('Dataset Feature', labelpad=8, fontsize=12)

    shap.summary_plot(
        shap_values,
        test_samples,
        feature_names=feature_names,
        # plot_type="dot",
        color_bar_label='Feature Value',
        max_display=21,
        plot_size=(10, 8)
    )

    plt.tight_layout()
    plt.savefig(plots_dir / 'shap_swarm_plot.png', dpi=100, bbox_inches='tight')

generate_shap_swarm_plot()