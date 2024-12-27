from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
import matplotlib; matplotlib.use('TkAgg')
import sys
from Functions.Model_helpers import *


# Define Logistic Regression Parameters
logistic = LogisticRegression(max_iter=1000)

param_grid = [
    {'penalty': ['elasticnet'],
     'solver': ['saga'],
     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
     'l1_ratio': [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]},

    {'penalty': ['l1'],
     'solver': ['saga'],
     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},

    {'penalty': ['l2'],
     'solver': ['saga', 'sag', 'newton-cg'],
     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
]


# ----------------------- df_full -----------------------
# -------------------- df_full_UND --------------------
df = pd.read_csv('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df full/df_full_UND.csv',
                 low_memory=False)
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
X2 = X[['Career days injured', 'Days without injury', 'Muscle', 'Season days injured', 'Career Minutes', 'Hamstring',
        'Knee', 'Ankle', 'General', 'Muscle_1 year', 'Knee ligament', 'Thigh', 'Calf', 'Groin', 'Hamstring_1 year',
        'Season Minutes', 'Adductor', 'General_1 year', 'Back', '10 Minutes', 'Ankle ligament', 'Foot', 'Ligament',
        'Ankle_1 year', 'Hip', 'Knee_1 year']]

# Split the data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.3, random_state=2024, stratify=y)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.3, random_state=2024, stratify=y)

# X1_df_full
# --- Perform GridSearchCV ---
# --- With Career Injuries
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_full/X1_df_full_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X1')
print('--------------------')
print(param_grid)
print('--------------------')

grid_search = GridSearchCV(logistic, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=11)
grid_search.fit(X1_train, y1_train.squeeze())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Get the best model
best_model_X1 = grid_search.best_estimator_

# Predict probabilities for the positive class (1)
y1_test_pred_prob = best_model_X1.predict_proba(X1_test)[:, 1]
y1_test_pred = best_model_X1.predict(X1_test)

# Calculate AUC-ROC on the test set
auc_roc = roc_auc_score(y1_test, y1_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y1_test, y1_test_pred_prob)
# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve X1')
plt.legend(loc="lower right")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_full/X1_ROC_df_full.pdf',
            bbox_inches="tight")
plt.close()

# Calculate Youden's Index
youden_index = tpr - fpr  # Youden's Index = Sensitivity + Specificity - 1
optimal_idx = youden_index.argmax()  # Index of the maximum Youden's Index
optimal_threshold = thresholds[optimal_idx]  # Corresponding threshold

print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold to convert probabilities into binary predictions
y1_test_pred_optimal = (y1_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
print(classification_report(y1_test, best_model_X1.predict(X1_test)))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y1_test, y1_test_pred_optimal))

# Compute the confusion matrix
cm = confusion_matrix(y1_test, y1_test_pred)
cm_youden = confusion_matrix(y1_test, y1_test_pred_optimal)

# --- Confusion Matrix no Youden
cm_row_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X1")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_full/X1_CM_df_full.pdf',
            bbox_inches="tight")
plt.close()

# --- Confusion Matrix with Youden
cm_row_normalized = cm_youden.astype('float') / cm_youden.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm_youden.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X1 with Youden Optimization")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_full/X1_CM_youden_df_full.pdf',
            bbox_inches="tight")
plt.close()

feature_names = X1_train.columns
# Extract coefficients and intercept
coef = best_model_X1.coef_[0]  # Coefficients for each feature
intercept = best_model_X1.intercept_[0]  # Intercept
# Compute odds ratios
odds_ratios = np.exp(coef)
print('Interpretation')
print('--------------------------------------------------------------------------------')
# Print coefficients, odds ratios, and interpretation
for name, c, o in zip(feature_names, coef, odds_ratios):
    print(f"\nFeature: {name}, Coefficient = {c:.3f}, Odds Ratio = {o:.3f}")
    if o > 1:
        print(f"  Interpretation: A 1-unit increase in '{name}' increases the odds by {((o-1)*100):.1f}%.")
    else:
        print(f"  Interpretation: A 1-unit increase in '{name}' decreases the odds by {((1-o)*100):.1f}%.")

sys.stdout = sys.__stdout__
log_file.close()

# TODO: X2_df_full
# --- Perform GridSearchCV ---
# --- With Body are injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_full/X2_df_full_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X2')
print('--------------------------------------------------------------------------------')
print(param_grid)
print('--------------------------------------------------------------------------------')

grid_search = GridSearchCV(logistic, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=11)
grid_search.fit(X2_train, y2_train.squeeze())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Get the best model
best_model_X2 = grid_search.best_estimator_

# Predict probabilities for the positive class (1)
y2_test_pred_prob = best_model_X2.predict_proba(X2_test)[:, 1]
y2_test_pred = best_model_X2.predict(X2_test)

# Calculate AUC-ROC on the test set
auc_roc = roc_auc_score(y2_test, y2_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y2_test, y2_test_pred_prob)
# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve X2')
plt.legend(loc="lower right")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_full/X2_ROC_df_full.pdf',
            bbox_inches="tight")
plt.close()

# Calculate Youden's Index
youden_index = tpr - fpr  # Youden's Index = Sensitivity + Specificity - 1
optimal_idx = youden_index.argmax()  # Index of the maximum Youden's Index
optimal_threshold = thresholds[optimal_idx]  # Corresponding threshold

print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold to convert probabilities into binary predictions
y2_test_pred_optimal = (y2_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
print(classification_report(y2_test, best_model_X2.predict(X2_test)))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y2_test, y2_test_pred_optimal))

# Compute the confusion matrix
cm = confusion_matrix(y2_test, y2_test_pred)
cm_youden = confusion_matrix(y2_test, y2_test_pred_optimal)

# --- Confusion Matrix no Youden
cm_row_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X2")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_full/X2_CM_df_full.pdf',
            bbox_inches="tight")
plt.close()

# --- Confusion Matrix with Youden
cm_row_normalized = cm_youden.astype('float') / cm_youden.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm_youden.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X2 with Youden Optimization")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_full/X2_CM_youden_df_full.pdf',
            bbox_inches="tight")
plt.close()

feature_names = X2_train.columns
# Extract coefficients and intercept
coef = best_model_X2.coef_[0]  # Coefficients for each feature
intercept = best_model_X2.intercept_[0]  # Intercept
# Compute odds ratios
odds_ratios = np.exp(coef)
print('Interpretation')
print('--------------------------------------------------------------------------------')
# Print coefficients, odds ratios, and interpretation
for name, c, o in zip(feature_names, coef, odds_ratios):
    print(f"\nFeature: {name}, Coefficient = {c:.3f}, Odds Ratio = {o:.3f}")
    if o > 1:
        print(f"  Interpretation: A 1-unit increase in '{name}' increases the odds by {((o-1)*100):.1f}%.")
    else:
        print(f"  Interpretation: A 1-unit increase in '{name}' decreases the odds by {((1-o)*100):.1f}%.")

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()


# TODO: df_3_iqr_df0
# -------------------- df_3_iqr_df0_UND --------------------
df = pd.read_csv('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df_3_iqr_df0/df_3_iqr_df0_UND.csv',
                 low_memory=False)
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
X2 = X[['Career days injured', 'Days without injury', 'Muscle', 'Season days injured', 'Career Minutes', 'Hamstring',
        'Knee', 'Ankle', 'General', 'Muscle_1 year', 'Knee ligament', 'Thigh', 'Calf', 'Groin', 'Hamstring_1 year',
        'Season Minutes', 'Adductor', 'General_1 year', 'Back', '10 Minutes', 'Ankle ligament', 'Foot', 'Ligament',
        'Ankle_1 year', 'Hip', 'Knee_1 year']]

# Split the data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.3, random_state=2024, stratify=y)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.3, random_state=2024, stratify=y)

# TODO: X1_df_3_iqr_df0
# --- Perform GridSearchCV ---
# --- With Career Injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_3_iqr_df0/X1_df_3_iqr_df0_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X1')
print('--------------------')
print(param_grid)
print('--------------------')

grid_search = GridSearchCV(logistic, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=11)
grid_search.fit(X1_train, y1_train.squeeze())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Get the best model
best_model_X1 = grid_search.best_estimator_

# Predict probabilities for the positive class (1)
y1_test_pred_prob = best_model_X1.predict_proba(X1_test)[:, 1]
y1_test_pred = best_model_X1.predict(X1_test)

# Calculate AUC-ROC on the test set
auc_roc = roc_auc_score(y1_test, y1_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y1_test, y1_test_pred_prob)
# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve X1')
plt.legend(loc="lower right")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_3_iqr_df0/X1_ROC_df_3_iqr_df0.pdf',
            bbox_inches="tight")
plt.close()

# Calculate Youden's Index
youden_index = tpr - fpr  # Youden's Index = Sensitivity + Specificity - 1
optimal_idx = youden_index.argmax()  # Index of the maximum Youden's Index
optimal_threshold = thresholds[optimal_idx]  # Corresponding threshold

print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold to convert probabilities into binary predictions
y1_test_pred_optimal = (y1_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
print(classification_report(y1_test, best_model_X1.predict(X1_test)))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y1_test, y1_test_pred_optimal))

# Compute the confusion matrix
cm = confusion_matrix(y1_test, y1_test_pred)
cm_youden = confusion_matrix(y1_test, y1_test_pred_optimal)

# --- Confusion Matrix no Youden
cm_row_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X1")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_3_iqr_df0/X1_CM_df_3_iqr_df0.pdf',
            bbox_inches="tight")
plt.close()

# --- Confusion Matrix with Youden
cm_row_normalized = cm_youden.astype('float') / cm_youden.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm_youden.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X1 with Youden Optimization")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_3_iqr_df0/X1_CM_youden_df_3_iqr_df0.pdf',
            bbox_inches="tight")
plt.close()

feature_names = X1_train.columns
# Extract coefficients and intercept
coef = best_model_X1.coef_[0]  # Coefficients for each feature
intercept = best_model_X1.intercept_[0]  # Intercept
# Compute odds ratios
odds_ratios = np.exp(coef)
print('Interpretation')
print('--------------------------------------------------------------------------------')
# Print coefficients, odds ratios, and interpretation
for name, c, o in zip(feature_names, coef, odds_ratios):
    print(f"\nFeature: {name}, Coefficient = {c:.3f}, Odds Ratio = {o:.3f}")
    if o > 1:
        print(f"  Interpretation: A 1-unit increase in '{name}' increases the odds by {((o-1)*100):.1f}%.")
    else:
        print(f"  Interpretation: A 1-unit increase in '{name}' decreases the odds by {((1-o)*100):.1f}%.")

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()

# TODO: X2_df_3_iqr_df0
# --- Perform GridSearchCV ---
# --- With Body are injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_3_iqr_df0/X2_df_3_iqr_df0_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X2')
print('--------------------------------------------------------------------------------')
print(param_grid)
print('--------------------------------------------------------------------------------')

grid_search = GridSearchCV(logistic, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=11)
grid_search.fit(X2_train, y2_train.squeeze())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Get the best model
best_model_X2 = grid_search.best_estimator_

# Predict probabilities for the positive class (1)
y2_test_pred_prob = best_model_X2.predict_proba(X2_test)[:, 1]
y2_test_pred = best_model_X2.predict(X2_test)

# Calculate AUC-ROC on the test set
auc_roc = roc_auc_score(y2_test, y2_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y2_test, y2_test_pred_prob)
# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve X2')
plt.legend(loc="lower right")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_3_iqr_df0/X2_ROC_df_3_iqr_df0.pdf',
            bbox_inches="tight")
plt.close()

# Calculate Youden's Index
youden_index = tpr - fpr  # Youden's Index = Sensitivity + Specificity - 1
optimal_idx = youden_index.argmax()  # Index of the maximum Youden's Index
optimal_threshold = thresholds[optimal_idx]  # Corresponding threshold

print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold to convert probabilities into binary predictions
y2_test_pred_optimal = (y2_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
print(classification_report(y2_test, best_model_X2.predict(X2_test)))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y2_test, y2_test_pred_optimal))

# Compute the confusion matrix
cm = confusion_matrix(y2_test, y2_test_pred)
cm_youden = confusion_matrix(y2_test, y2_test_pred_optimal)

# --- Confusion Matrix no Youden
cm_row_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X2")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_3_iqr_df0/X2_CM_df_3_iqr_df0.pdf',
            bbox_inches="tight")
plt.close()

# --- Confusion Matrix with Youden
cm_row_normalized = cm_youden.astype('float') / cm_youden.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm_youden.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X2 with Youden Optimization")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_3_iqr_df0/X2_CM_youden_df_3_iqr_df0.pdf',
            bbox_inches="tight")
plt.close()

feature_names = X2_train.columns
# Extract coefficients and intercept
coef = best_model_X2.coef_[0]  # Coefficients for each feature
intercept = best_model_X2.intercept_[0]  # Intercept
# Compute odds ratios
odds_ratios = np.exp(coef)
print('Interpretation')
print('--------------------------------------------------------------------------------')
# Print coefficients, odds ratios, and interpretation
for name, c, o in zip(feature_names, coef, odds_ratios):
    print(f"\nFeature: {name}, Coefficient = {c:.3f}, Odds Ratio = {o:.3f}")
    if o > 1:
        print(f"  Interpretation: A 1-unit increase in '{name}' increases the odds by {((o-1)*100):.1f}%.")
    else:
        print(f"  Interpretation: A 1-unit increase in '{name}' decreases the odds by {((1-o)*100):.1f}%.")

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()


# TODO: df_3_iqr
# -------------------- df_3_iqr_UND --------------------
df = pd.read_csv('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df_3_iqr/df_3_iqr_UND.csv',
                 low_memory=False)
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
X2 = X[['Career days injured', 'Days without injury', 'Muscle', 'Season days injured', 'Career Minutes', 'Hamstring',
        'Knee', 'Ankle', 'General', 'Muscle_1 year', 'Knee ligament', 'Thigh', 'Calf', 'Groin', 'Hamstring_1 year',
        'Season Minutes', 'Adductor', 'General_1 year', 'Back', '10 Minutes', 'Ankle ligament', 'Foot', 'Ligament',
        'Ankle_1 year', 'Hip', 'Knee_1 year']]

# Split the data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.3, random_state=2024, stratify=y)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.3, random_state=2024, stratify=y)

# TODO: X1_df_3_iqr
# --- Perform GridSearchCV ---
# --- With Career Injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_3_iqr/X1_df_3_iqr_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X1')
print('--------------------')
print(param_grid)
print('--------------------')

grid_search = GridSearchCV(logistic, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=10)
grid_search.fit(X1_train, y1_train.squeeze())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Get the best model
best_model_X1 = grid_search.best_estimator_

# Predict probabilities for the positive class (1)
y1_test_pred_prob = best_model_X1.predict_proba(X1_test)[:, 1]
y1_test_pred = best_model_X1.predict(X1_test)

# Calculate AUC-ROC on the test set
auc_roc = roc_auc_score(y1_test, y1_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y1_test, y1_test_pred_prob)
# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve X1')
plt.legend(loc="lower right")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_3_iqr/X1_ROC_df_3_iqr.pdf', bbox_inches="tight")
plt.close()

# Calculate Youden's Index
youden_index = tpr - fpr  # Youden's Index = Sensitivity + Specificity - 1
optimal_idx = youden_index.argmax()  # Index of the maximum Youden's Index
optimal_threshold = thresholds[optimal_idx]  # Corresponding threshold

print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold to convert probabilities into binary predictions
y1_test_pred_optimal = (y1_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
print(classification_report(y1_test, best_model_X1.predict(X1_test)))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y1_test, y1_test_pred_optimal))

# Compute the confusion matrix
cm = confusion_matrix(y1_test, y1_test_pred)
cm_youden = confusion_matrix(y1_test, y1_test_pred_optimal)

# --- Confusion Matrix no Youden
cm_row_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X1")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_3_iqr/X1_CM_df_3_iqr.pdf', bbox_inches="tight")
plt.close()

# --- Confusion Matrix with Youden
cm_row_normalized = cm_youden.astype('float') / cm_youden.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm_youden.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X1 with Youden Optimization")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_3_iqr/X1_CM_youden_df_3_iqr.pdf',
            bbox_inches="tight")
plt.close()

feature_names = X1_train.columns
# Extract coefficients and intercept
coef = best_model_X1.coef_[0]  # Coefficients for each feature
intercept = best_model_X1.intercept_[0]  # Intercept
# Compute odds ratios
odds_ratios = np.exp(coef)
print('Interpretation')
print('--------------------------------------------------------------------------------')
# Print coefficients, odds ratios, and interpretation
for name, c, o in zip(feature_names, coef, odds_ratios):
    print(f"\nFeature: {name}, Coefficient = {c:.3f}, Odds Ratio = {o:.3f}")
    if o > 1:
        print(f"  Interpretation: A 1-unit increase in '{name}' increases the odds by {((o-1)*100):.1f}%.")
    else:
        print(f"  Interpretation: A 1-unit increase in '{name}' decreases the odds by {((1-o)*100):.1f}%.")

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()

# TODO: X2_df_3_iqr
# --- Perform GridSearchCV ---
# --- With Body are injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_3_iqr/X2_df_3_iqr_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X2')
print('--------------------------------------------------------------------------------')
print(param_grid)
print('--------------------------------------------------------------------------------')

grid_search = GridSearchCV(logistic, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=10)
grid_search.fit(X2_train, y2_train.squeeze())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Get the best model
best_model_X2 = grid_search.best_estimator_

# Predict probabilities for the positive class (1)
y2_test_pred_prob = best_model_X2.predict_proba(X2_test)[:, 1]
y2_test_pred = best_model_X2.predict(X2_test)

# Calculate AUC-ROC on the test set
auc_roc = roc_auc_score(y2_test, y2_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y2_test, y2_test_pred_prob)
# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve X2')
plt.legend(loc="lower right")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_3_iqr/X2_ROC_df_3_iqr.pdf', bbox_inches="tight")
plt.close()

# Calculate Youden's Index
youden_index = tpr - fpr  # Youden's Index = Sensitivity + Specificity - 1
optimal_idx = youden_index.argmax()  # Index of the maximum Youden's Index
optimal_threshold = thresholds[optimal_idx]  # Corresponding threshold

print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold to convert probabilities into binary predictions
y2_test_pred_optimal = (y2_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
print(classification_report(y2_test, best_model_X2.predict(X2_test)))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y2_test, y2_test_pred_optimal))

# Compute the confusion matrix
cm = confusion_matrix(y2_test, y2_test_pred)
cm_youden = confusion_matrix(y2_test, y2_test_pred_optimal)

# --- Confusion Matrix no Youden
cm_row_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X2")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_3_iqr/X2_CM_df_3_iqr.pdf', bbox_inches="tight")
plt.close()

# --- Confusion Matrix with Youden
cm_row_normalized = cm_youden.astype('float') / cm_youden.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm_youden.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X2 with Youden Optimization")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_3_iqr/X2_CM_youden_df_3_iqr.pdf',
            bbox_inches="tight")
plt.close()

feature_names = X2_train.columns
# Extract coefficients and intercept
coef = best_model_X2.coef_[0]  # Coefficients for each feature
intercept = best_model_X2.intercept_[0]  # Intercept
# Compute odds ratios
odds_ratios = np.exp(coef)
print('Interpretation')
print('--------------------------------------------------------------------------------')
# Print coefficients, odds ratios, and interpretation
for name, c, o in zip(feature_names, coef, odds_ratios):
    print(f"\nFeature: {name}, Coefficient = {c:.3f}, Odds Ratio = {o:.3f}")
    if o > 1:
        print(f"  Interpretation: A 1-unit increase in '{name}' increases the odds by {((o-1)*100):.1f}%.")
    else:
        print(f"  Interpretation: A 1-unit increase in '{name}' decreases the odds by {((1-o)*100):.1f}%.")

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()


# TODO: df_1_5_iqr_df0
# -------------------- df_1_5_iqr_df0_UND --------------------
df = pd.read_csv('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df_1_5_iqr_df0/df_1_5_iqr_df0_UND.csv',
                 low_memory=False)
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
X2 = X[['Career days injured', 'Days without injury', 'Muscle', 'Season days injured', 'Career Minutes', 'Hamstring',
        'Knee', 'Ankle', 'General', 'Muscle_1 year', 'Knee ligament', 'Thigh', 'Calf', 'Groin', 'Hamstring_1 year',
        'Season Minutes', 'Adductor', 'General_1 year', 'Back', '10 Minutes', 'Ankle ligament', 'Foot', 'Ligament',
        'Ankle_1 year', 'Hip', 'Knee_1 year']]

# Split the data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.3, random_state=2024, stratify=y)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.3, random_state=2024, stratify=y)

# TODO: X1_df_1_5_iqr_df0
# --- Perform GridSearchCV ---
# --- With Career Injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_1_5_iqr_df0/X1_df_1_5_iqr_df0_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X1')
print('--------------------')
print(param_grid)
print('--------------------')

grid_search = GridSearchCV(logistic, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=10)
grid_search.fit(X1_train, y1_train.squeeze())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Get the best model
best_model_X1 = grid_search.best_estimator_

# Predict probabilities for the positive class (1)
y1_test_pred_prob = best_model_X1.predict_proba(X1_test)[:, 1]
y1_test_pred = best_model_X1.predict(X1_test)

# Calculate AUC-ROC on the test set
auc_roc = roc_auc_score(y1_test, y1_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y1_test, y1_test_pred_prob)
# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve X1')
plt.legend(loc="lower right")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_1_5_iqr_df0/X1_ROC_df_1_5_iqr_df0.pdf',
            bbox_inches="tight")
plt.close()

# Calculate Youden's Index
youden_index = tpr - fpr  # Youden's Index = Sensitivity + Specificity - 1
optimal_idx = youden_index.argmax()  # Index of the maximum Youden's Index
optimal_threshold = thresholds[optimal_idx]  # Corresponding threshold

print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold to convert probabilities into binary predictions
y1_test_pred_optimal = (y1_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
print(classification_report(y1_test, best_model_X1.predict(X1_test)))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y1_test, y1_test_pred_optimal))

# Compute the confusion matrix
cm = confusion_matrix(y1_test, y1_test_pred)
cm_youden = confusion_matrix(y1_test, y1_test_pred_optimal)

# --- Confusion Matrix no Youden
cm_row_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X1")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_1_5_iqr_df0/X1_CM_df_1_5_iqr_df0.pdf',
            bbox_inches="tight")
plt.close()

# --- Confusion Matrix with Youden
cm_row_normalized = cm_youden.astype('float') / cm_youden.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm_youden.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X1 with Youden Optimization")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_1_5_iqr_df0/X1_CM_youden_df_1_5_iqr_df0.pdf',
            bbox_inches="tight")
plt.close()

feature_names = X1_train.columns
# Extract coefficients and intercept
coef = best_model_X1.coef_[0]  # Coefficients for each feature
intercept = best_model_X1.intercept_[0]  # Intercept
# Compute odds ratios
odds_ratios = np.exp(coef)
print('Interpretation')
print('--------------------------------------------------------------------------------')
# Print coefficients, odds ratios, and interpretation
for name, c, o in zip(feature_names, coef, odds_ratios):
    print(f"\nFeature: {name}, Coefficient = {c:.3f}, Odds Ratio = {o:.3f}")
    if o > 1:
        print(f"  Interpretation: A 1-unit increase in '{name}' increases the odds by {((o-1)*100):.1f}%.")
    else:
        print(f"  Interpretation: A 1-unit increase in '{name}' decreases the odds by {((1-o)*100):.1f}%.")

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()

# TODO: X2_df_1_5_iqr_df0
# --- Perform GridSearchCV ---
# --- With Body are injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_1_5_iqr_df0/X2_df_1_5_iqr_df0_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X2')
print('--------------------------------------------------------------------------------')
print(param_grid)
print('--------------------------------------------------------------------------------')

grid_search = GridSearchCV(logistic, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=10)
grid_search.fit(X2_train, y2_train.squeeze())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Get the best model
best_model_X2 = grid_search.best_estimator_

# Predict probabilities for the positive class (1)
y2_test_pred_prob = best_model_X2.predict_proba(X2_test)[:, 1]
y2_test_pred = best_model_X2.predict(X2_test)

# Calculate AUC-ROC on the test set
auc_roc = roc_auc_score(y2_test, y2_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y2_test, y2_test_pred_prob)
# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve X2')
plt.legend(loc="lower right")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_1_5_iqr_df0/X2_ROC_df_1_5_iqr_df0.pdf',
            bbox_inches="tight")
plt.close()

# Calculate Youden's Index
youden_index = tpr - fpr  # Youden's Index = Sensitivity + Specificity - 1
optimal_idx = youden_index.argmax()  # Index of the maximum Youden's Index
optimal_threshold = thresholds[optimal_idx]  # Corresponding threshold

print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold to convert probabilities into binary predictions
y2_test_pred_optimal = (y2_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
print(classification_report(y2_test, best_model_X2.predict(X2_test)))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y2_test, y2_test_pred_optimal))

# Compute the confusion matrix
cm = confusion_matrix(y2_test, y2_test_pred)
cm_youden = confusion_matrix(y2_test, y2_test_pred_optimal)

# --- Confusion Matrix no Youden
cm_row_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X2")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_1_5_iqr_df0/X2_CM_df_1_5_iqr_df0.pdf',
            bbox_inches="tight")
plt.close()

# --- Confusion Matrix with Youden
cm_row_normalized = cm_youden.astype('float') / cm_youden.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm_youden.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X2 with Youden Optimization")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_1_5_iqr_df0/X2_CM_youden_df_1_5_iqr_df0.pdf',
            bbox_inches="tight")
plt.close()

feature_names = X2_train.columns
# Extract coefficients and intercept
coef = best_model_X2.coef_[0]  # Coefficients for each feature
intercept = best_model_X2.intercept_[0]  # Intercept
# Compute odds ratios
odds_ratios = np.exp(coef)
print('Interpretation')
print('--------------------------------------------------------------------------------')
# Print coefficients, odds ratios, and interpretation
for name, c, o in zip(feature_names, coef, odds_ratios):
    print(f"\nFeature: {name}, Coefficient = {c:.3f}, Odds Ratio = {o:.3f}")
    if o > 1:
        print(f"  Interpretation: A 1-unit increase in '{name}' increases the odds by {((o-1)*100):.1f}%.")
    else:
        print(f"  Interpretation: A 1-unit increase in '{name}' decreases the odds by {((1-o)*100):.1f}%.")

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()


# TODO: df_1_5_iqr
# -------------------- df_1_5_iqr_UND --------------------
df = pd.read_csv('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df_1_5_iqr/df_1_5_iqr_UND.csv',
                 low_memory=False)
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
X2 = X[['Career days injured', 'Days without injury', 'Muscle', 'Season days injured', 'Career Minutes', 'Hamstring',
        'Knee', 'Ankle', 'General', 'Muscle_1 year', 'Knee ligament', 'Thigh', 'Calf', 'Groin', 'Hamstring_1 year',
        'Season Minutes', 'Adductor', 'General_1 year', 'Back', '10 Minutes', 'Ankle ligament', 'Foot', 'Ligament',
        'Ankle_1 year', 'Hip', 'Knee_1 year']]

# Split the data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.3, random_state=2024, stratify=y)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.3, random_state=2024, stratify=y)

# TODO: X1_df_1_5_iqr
# --- Perform GridSearchCV ---
# --- With Career Injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_1_5_iqr/X1_df_1_5_iqr_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X1')
print('--------------------')
print(param_grid)
print('--------------------')

grid_search = GridSearchCV(logistic, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=10)
grid_search.fit(X1_train, y1_train.squeeze())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Get the best model
best_model_X1 = grid_search.best_estimator_

# Predict probabilities for the positive class (1)
y1_test_pred_prob = best_model_X1.predict_proba(X1_test)[:, 1]
y1_test_pred = best_model_X1.predict(X1_test)

# Calculate AUC-ROC on the test set
auc_roc = roc_auc_score(y1_test, y1_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y1_test, y1_test_pred_prob)
# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve X1')
plt.legend(loc="lower right")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_1_5_iqr/X1_ROC_df_1_5_iqr.pdf',
            bbox_inches="tight")
plt.close()

# Calculate Youden's Index
youden_index = tpr - fpr  # Youden's Index = Sensitivity + Specificity - 1
optimal_idx = youden_index.argmax()  # Index of the maximum Youden's Index
optimal_threshold = thresholds[optimal_idx]  # Corresponding threshold

print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold to convert probabilities into binary predictions
y1_test_pred_optimal = (y1_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
print(classification_report(y1_test, best_model_X1.predict(X1_test)))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y1_test, y1_test_pred_optimal))

# Compute the confusion matrix
cm = confusion_matrix(y1_test, y1_test_pred)
cm_youden = confusion_matrix(y1_test, y1_test_pred_optimal)

# --- Confusion Matrix no Youden
cm_row_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X1")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_1_5_iqr/X1_CM_df_1_5_iqr.pdf',
            bbox_inches="tight")
plt.close()

# --- Confusion Matrix with Youden
cm_row_normalized = cm_youden.astype('float') / cm_youden.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm_youden.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X1 with Youden Optimization")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_1_5_iqr/X1_CM_youden_df_1_5_iqr.pdf',
            bbox_inches="tight")
plt.close()

feature_names = X1_train.columns
# Extract coefficients and intercept
coef = best_model_X1.coef_[0]  # Coefficients for each feature
intercept = best_model_X1.intercept_[0]  # Intercept
# Compute odds ratios
odds_ratios = np.exp(coef)
print('Interpretation')
print('--------------------------------------------------------------------------------')
# Print coefficients, odds ratios, and interpretation
for name, c, o in zip(feature_names, coef, odds_ratios):
    print(f"\nFeature: {name}, Coefficient = {c:.3f}, Odds Ratio = {o:.3f}")
    if o > 1:
        print(f"  Interpretation: A 1-unit increase in '{name}' increases the odds by {((o-1)*100):.1f}%.")
    else:
        print(f"  Interpretation: A 1-unit increase in '{name}' decreases the odds by {((1-o)*100):.1f}%.")

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()

# TODO: X2_df_1_5_iqr
# --- Perform GridSearchCV ---
# --- With Body are injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_1_5_iqr/X2_df_1_5_iqr_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X2')
print('--------------------------------------------------------------------------------')
print(param_grid)
print('--------------------------------------------------------------------------------')

grid_search = GridSearchCV(logistic, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=10)
grid_search.fit(X2_train, y2_train.squeeze())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Get the best model
best_model_X2 = grid_search.best_estimator_

# Predict probabilities for the positive class (1)
y2_test_pred_prob = best_model_X2.predict_proba(X2_test)[:, 1]
y2_test_pred = best_model_X2.predict(X2_test)

# Calculate AUC-ROC on the test set
auc_roc = roc_auc_score(y2_test, y2_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y2_test, y2_test_pred_prob)
# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve X2')
plt.legend(loc="lower right")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_1_5_iqr/X2_ROC_df_1_5_iqr.pdf',
            bbox_inches="tight")
plt.close()

# Calculate Youden's Index
youden_index = tpr - fpr  # Youden's Index = Sensitivity + Specificity - 1
optimal_idx = youden_index.argmax()  # Index of the maximum Youden's Index
optimal_threshold = thresholds[optimal_idx]  # Corresponding threshold

print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold to convert probabilities into binary predictions
y2_test_pred_optimal = (y2_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
print(classification_report(y2_test, best_model_X2.predict(X2_test)))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y2_test, y2_test_pred_optimal))

# Compute the confusion matrix
cm = confusion_matrix(y2_test, y2_test_pred)
cm_youden = confusion_matrix(y2_test, y2_test_pred_optimal)

# --- Confusion Matrix no Youden
cm_row_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X2")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_1_5_iqr/X2_CM_df_1_5_iqr.pdf',
            bbox_inches="tight")
plt.close()

# --- Confusion Matrix with Youden
cm_row_normalized = cm_youden.astype('float') / cm_youden.sum(axis=1)[:, np.newaxis]

# Format labels with counts and percentages
group_labels = ["True Negative", "False Positive", "False Negative", "True Positive"]
group_counts = [f"{value}" for value in cm_youden.flatten()]
group_percentages = [f"{value:.2%}" for value in cm_row_normalized.flatten()]
labels = [
    f"{group_labels[i]}\nCount: {group_counts[i]}\nPercentage: {group_percentages[i]}"
    for i in range(len(group_labels))
]
labels = np.array(labels).reshape(2, 2)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_row_normalized, annot=labels, fmt="", cmap="Blues", cbar=False,
            xticklabels=["Predicted Non-Injury", "Predicted Injury"], yticklabels=["No Injury", "Injury"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix X2 with Youden Optimization")
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Model Data/LR/df_1_5_iqr/X2_CM_youden_df_1_5_iqr.pdf',
            bbox_inches="tight")
plt.close()

feature_names = X2_train.columns
# Extract coefficients and intercept
coef = best_model_X2.coef_[0]  # Coefficients for each feature
intercept = best_model_X2.intercept_[0]  # Intercept
# Compute odds ratios
odds_ratios = np.exp(coef)
print('Interpretation')
print('--------------------------------------------------------------------------------')
# Print coefficients, odds ratios, and interpretation
for name, c, o in zip(feature_names, coef, odds_ratios):
    print(f"\nFeature: {name}, Coefficient = {c:.3f}, Odds Ratio = {o:.3f}")
    if o > 1:
        print(f"  Interpretation: A 1-unit increase in '{name}' increases the odds by {((o-1)*100):.1f}%.")
    else:
        print(f"  Interpretation: A 1-unit increase in '{name}' decreases the odds by {((1-o)*100):.1f}%.")

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()




















