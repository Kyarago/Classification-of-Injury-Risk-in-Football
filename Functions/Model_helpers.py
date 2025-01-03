import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def read_data(data_path):
    df = pd.read_csv(data_path, low_memory=False)
    df = df.drop(['Cluster'], axis=1)
    df = pd.get_dummies(df, columns=['Position'], drop_first=True)

    # Modify True / False to 1 / 0
    df = df.replace({True: 1, False: 0})

    # Select target feature
    X = df.drop(columns=['Injury condition'])  # Features
    y = df[['Injury condition']]  # Target variable

    X1 = X[['Career days injured', 'Days without injury', 'Season days injured', 'Career Minutes', 'Career injuries',
            'Muscle_1 year', 'Hamstring_1 year', 'Season Minutes', 'General_1 year', '10 Minutes', 'Ankle_1 year',
            'Knee_1 year']]
    X2 = X[
        ['Career days injured', 'Days without injury', 'Muscle', 'Season days injured', 'Career Minutes', 'Hamstring',
         'Knee', 'Ankle', 'General', 'Muscle_1 year', 'Knee ligament', 'Thigh', 'Calf', 'Groin', 'Hamstring_1 year',
         'Season Minutes', 'Adductor', 'General_1 year', 'Back', '10 Minutes', 'Ankle ligament', 'Foot', 'Ligament',
         'Ankle_1 year', 'Hip', 'Knee_1 year']]

    # Step 3: Split the data into training and testing sets
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.3, random_state=2024, stratify=y)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.3, random_state=2024, stratify=y)

    return X1_train, X1_test, y1_train, y1_test, X2_train, X2_test, y2_train, y2_test


def scaler(X1_train, X1_test, X2_train, X2_test):
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Scale X1 data
    X1_train_scaled = scaler.fit_transform(X1_train)
    X1_test_scaled = scaler.transform(X1_test)

    # Scale X2 data
    X2_train_scaled = scaler.fit_transform(X2_train)
    X2_test_scaled = scaler.transform(X2_test)

    return X1_train_scaled, X1_test_scaled, X2_train_scaled, X2_test_scaled


def plot_roc_curve(fpr, tpr, auc_roc, title, filename):
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    return


# Confusion Matrix
def plot_confusion_matrix(cm, title, filename):
    cm_row_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
    group_counts = [f"{value}" for value in cm.flatten()]
    group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
    labels = [
        f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
        for i in range(len(group_labels))
    ]
    labels = np.array(labels).reshape(2, 2)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
                xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()