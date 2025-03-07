import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from ucimlrepo import fetch_ucirepo
from pathlib import Path

labelpad_value = 8
font_size_on_bar = 12
xy_font_size = 15
title_font_size = 16

base_dir = Path(__file__).resolve().parent.parent
plots_dir = base_dir / 'plot_figures'

# SMOTE Result Plot
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.7, random_state=42)
scaler = StandardScaler()
X_train= scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
count_before_smote = y_train.value_counts()
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
count_after_smote = y_train.value_counts()
fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=1000)

# plotting before SMOTE
count_before_smote.plot(kind='bar', color='lightgreen', ax=axs[0], fontsize= xy_font_size, width=0.4)
axs[0].set_title('Class Distribution Before Applying SMOTE', fontweight='bold', ha='center', pad=12)
axs[0].set_xlabel('Class', fontsize=xy_font_size, labelpad=labelpad_value)
axs[0].set_ylabel('Number of Samples', fontsize=xy_font_size, labelpad=labelpad_value)
for i in range(len(count_before_smote)):
    axs[0].text(i, count_before_smote[i] + max(count_before_smote) * 0.02, str(count_before_smote[i]), ha='center', va='bottom')

# plotting after SMOTE
count_after_smote.plot(kind='bar', color='lightblue', ax=axs[1], fontsize= xy_font_size, width=0.4)
axs[1].set_title('Class Distribution After Applying SMOTE', fontweight='bold', ha='center', pad=12)
axs[1].set_xlabel('Class', fontsize=xy_font_size, labelpad=labelpad_value)
axs[1].set_ylabel('Number of Samples', fontsize=xy_font_size, labelpad=labelpad_value)
for i in range(len(count_after_smote)):
    axs[1].text(i, count_after_smote[i] + max(count_after_smote) * 0.02, str(count_after_smote[i]), ha='center', va='bottom')
for ax in axs:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.setp(axs[0].get_yticklabels(), fontsize=9.5)
plt.setp(axs[1].get_yticklabels(), fontsize=9.5)
axs[0].set_ylim(0, max(count_before_smote) * 1.25)
axs[1].set_ylim(0, max(count_after_smote) * 1.25)

plt.tight_layout()
plt.savefig(plots_dir / 'SMOTE Preprocessing Comparison.png')
plt.show()