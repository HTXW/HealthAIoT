import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from pathlib import Path

dataset = fetch_ucirepo(id=891)
X = dataset.data.features
y = dataset.data.targets

if not isinstance(X, pd.DataFrame):
    X = pd.DataFrame(X)
if not isinstance(y, pd.Series):
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    else:
        y = pd.Series(y)

print("Dataset: CDC Diabetes Health Indicators")
print("Source: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators")
print("Features type:", type(X))
print("Targets type:", type(y))
print("Features shape:", X.shape)
print("Targets shape:", y.shape)
print("Feature columns:", list(X.columns))
print("Unique target values:", y.unique())
print("\nSummary Statistics for Features:")
print(X.describe())

base_dir = Path(__file__).resolve().parent.parent
plots_dir = base_dir / 'plot_figures'

# correlation heatmap
plt.figure(figsize=(12, 10))
corr = X.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Dataset Features", fontsize=16, ha='center', pad=12)
plt.tight_layout()
plt.savefig(plots_dir / 'Dataset Correlation_Matrix.png', dpi=600)
plt.show()