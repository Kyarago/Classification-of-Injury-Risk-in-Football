from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
import sys
from Functions.Model_helpers import *

# Define KNN parameters for GridSearchCV
knn = KNeighborsClassifier()

param_grid = [
    {'n_neighbors': list(range(1, 51, 5)),  # Number of neighbors to consider
     'weights': ['uniform', 'distance'],  # Weighting method
     'metric': ['euclidean', 'manhattan', 'chebyshev', 'hamming', 'canberra', 'braycurtis']}  # Distance metrics
]

# Set number of workers (cpus)
num_cpu = 5

# TODO: df_full
# -------------------- df_full_UND --------------------
(X1_train, X1_test, y1_train, y1_test, X2_train, X2_test, y2_train,
 y2_test) = read_data('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df full/df_full_UND.csv')

X1_train, X1_test, X2_train, X2_test = scaler(X1_train, X1_test, X2_train, X2_test)

# TODO: X1_df_full
# --- Perform GridSearchCV ---
# --- With Career Injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_full/X1_df_full_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X1 with KNN')
print('---------------------------------------------------')
print(param_grid)
print('---------------------------------------------------')

grid_search = GridSearchCV(knn, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=num_cpu)
grid_search.fit(X1_train, y1_train.values.ravel())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Evaluate the best model
best_model_X1 = grid_search.best_estimator_
y1_test_pred_prob = best_model_X1.predict_proba(X1_test)[:, 1]
y1_test_pred = best_model_X1.predict(X1_test)

# Calculate AUC-ROC
auc_roc = roc_auc_score(y1_test, y1_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y1_test, y1_test_pred_prob)
plot_roc_curve(fpr, tpr, auc_roc, 'ROC Curve X1 (KNN)',
               'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_full/X1_ROC_df_full.pdf')

# Calculate Youden's Index
youden_index = tpr - fpr
optimal_idx = youden_index.argmax()
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold
y1_test_pred_optimal = (y1_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
default_threshold_predictions = (y1_test_pred_prob >= 0.5).astype(int)
print(classification_report(y1_test, default_threshold_predictions))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y1_test, y1_test_pred_optimal))

cm_default = confusion_matrix(y1_test, default_threshold_predictions)
plot_confusion_matrix(cm_default, "Confusion Matrix X1 (0.5 Threshold)",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_full/X1_CM_df_full.pdf')

cm_youden = confusion_matrix(y1_test, y1_test_pred_optimal)
plot_confusion_matrix(cm_youden, "Confusion Matrix X1 with Youden Optimization",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_full/X1_CM_youden_df_full.pdf')

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()

# TODO: X2_df_full
# --- Perform GridSearchCV ---
# --- With Body are injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_full/X2_df_full_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X2 with KNN')
print('---------------------------------------------------')
print(param_grid)
print('---------------------------------------------------')

grid_search = GridSearchCV(knn, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=num_cpu)
grid_search.fit(X2_train, y2_train.values.ravel())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Evaluate the best model
best_model_X2 = grid_search.best_estimator_
y2_test_pred_prob = best_model_X2.predict_proba(X2_test)[:, 1]
y2_test_pred = best_model_X2.predict(X2_test)

# Calculate AUC-ROC
auc_roc = roc_auc_score(y2_test, y2_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y2_test, y2_test_pred_prob)
plot_roc_curve(fpr, tpr, auc_roc, 'ROC Curve X2 (KNN)',
               'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_full/X2_ROC_df_full.pdf')

# Calculate Youden's Index
youden_index = tpr - fpr
optimal_idx = youden_index.argmax()
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold
y2_test_pred_optimal = (y2_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
default_threshold_predictions = (y2_test_pred_prob >= 0.5).astype(int)
print(classification_report(y2_test, default_threshold_predictions))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y2_test, y2_test_pred_optimal))

cm_default = confusion_matrix(y2_test, default_threshold_predictions)
plot_confusion_matrix(cm_default, "Confusion Matrix X2 (0.5 Threshold)",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_full/X2_CM_df_full.pdf')

cm_youden = confusion_matrix(y2_test, y2_test_pred_optimal)
plot_confusion_matrix(cm_youden, "Confusion Matrix X2 with Youden Optimization",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_full/X2_CM_youden_df_full.pdf')

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()


# TODO: df_3_iqr_df0
# -------------------- df_3_iqr_df0_UND --------------------
(X1_train, X1_test, y1_train, y1_test, X2_train, X2_test, y2_train,
 y2_test) = read_data('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df_3_iqr_df0/df_3_iqr_df0_UND.csv')

X1_train, X1_test, X2_train, X2_test = scaler(X1_train, X1_test, X2_train, X2_test)

# TODO: X1_df_3_iqr_df0
# --- Perform GridSearchCV ---
# --- With Career Injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_3_iqr_df0/X1_df_3_iqr_df0_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X1 with KNN')
print('---------------------------------------------------')
print(param_grid)
print('---------------------------------------------------')

grid_search = GridSearchCV(knn, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=num_cpu)
grid_search.fit(X1_train, y1_train.values.ravel())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Evaluate the best model
best_model_X1 = grid_search.best_estimator_
y1_test_pred_prob = best_model_X1.predict_proba(X1_test)[:, 1]
y1_test_pred = best_model_X1.predict(X1_test)

# Calculate AUC-ROC
auc_roc = roc_auc_score(y1_test, y1_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y1_test, y1_test_pred_prob)
plot_roc_curve(fpr, tpr, auc_roc, 'ROC Curve X1 (KNN)',
               'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_3_iqr_df0/X1_ROC_df_3_iqr_df0.pdf')

# Calculate Youden's Index
youden_index = tpr - fpr
optimal_idx = youden_index.argmax()
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold
y1_test_pred_optimal = (y1_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
default_threshold_predictions = (y1_test_pred_prob >= 0.5).astype(int)
print(classification_report(y1_test, default_threshold_predictions))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y1_test, y1_test_pred_optimal))

cm_default = confusion_matrix(y1_test, default_threshold_predictions)
plot_confusion_matrix(cm_default, "Confusion Matrix X1 (0.5 Threshold)",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_3_iqr_df0/X1_CM_df_3_iqr_df0.pdf')

cm_youden = confusion_matrix(y1_test, y1_test_pred_optimal)
plot_confusion_matrix(cm_youden, "Confusion Matrix X1 with Youden Optimization",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_3_iqr_df0/X1_CM_youden_df_3_iqr_df0.pdf')

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()

# TODO: X2_df_3_iqr_df0
# --- Perform GridSearchCV ---
# --- With Body are injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_3_iqr_df0/X2_df_3_iqr_df0_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X2 with KNN')
print('---------------------------------------------------')
print(param_grid)
print('---------------------------------------------------')

grid_search = GridSearchCV(knn, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=num_cpu)
grid_search.fit(X2_train, y2_train.values.ravel())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Evaluate the best model
best_model_X2 = grid_search.best_estimator_
y2_test_pred_prob = best_model_X2.predict_proba(X2_test)[:, 1]
y2_test_pred = best_model_X2.predict(X2_test)

# Calculate AUC-ROC
auc_roc = roc_auc_score(y2_test, y2_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y2_test, y2_test_pred_prob)
plot_roc_curve(fpr, tpr, auc_roc, 'ROC Curve X2 (KNN)',
               'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_3_iqr_df0/X2_ROC_df_3_iqr_df0.pdf')

# Calculate Youden's Index
youden_index = tpr - fpr
optimal_idx = youden_index.argmax()
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold
y2_test_pred_optimal = (y2_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
default_threshold_predictions = (y2_test_pred_prob >= 0.5).astype(int)
print(classification_report(y2_test, default_threshold_predictions))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y2_test, y2_test_pred_optimal))

cm_default = confusion_matrix(y2_test, default_threshold_predictions)
plot_confusion_matrix(cm_default, "Confusion Matrix X2 (0.5 Threshold)",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_3_iqr_df0/X2_CM_df_3_iqr_df0.pdf')

cm_youden = confusion_matrix(y2_test, y2_test_pred_optimal)
plot_confusion_matrix(cm_youden, "Confusion Matrix X2 with Youden Optimization",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_3_iqr_df0/X2_CM_youden_df_3_iqr_df0.pdf')

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()


# TODO: df_3_iqr
# -------------------- df_3_iqr_UND --------------------
(X1_train, X1_test, y1_train, y1_test, X2_train, X2_test, y2_train,
 y2_test) = read_data('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df_3_iqr/df_3_iqr_UND.csv')

X1_train, X1_test, X2_train, X2_test = scaler(X1_train, X1_test, X2_train, X2_test)

# TODO: X1_df_3_iqr
# --- Perform GridSearchCV ---
# --- With Career Injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_3_iqr/X1_df_3_iqr_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X1 with KNN')
print('---------------------------------------------------')
print(param_grid)
print('---------------------------------------------------')

grid_search = GridSearchCV(knn, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=num_cpu)
grid_search.fit(X1_train, y1_train.values.ravel())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Evaluate the best model
best_model_X1 = grid_search.best_estimator_
y1_test_pred_prob = best_model_X1.predict_proba(X1_test)[:, 1]
y1_test_pred = best_model_X1.predict(X1_test)

# Calculate AUC-ROC
auc_roc = roc_auc_score(y1_test, y1_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y1_test, y1_test_pred_prob)
plot_roc_curve(fpr, tpr, auc_roc, 'ROC Curve X1 (KNN)',
               'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_3_iqr/X1_ROC_df_3_iqr.pdf')

# Calculate Youden's Index
youden_index = tpr - fpr
optimal_idx = youden_index.argmax()
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold
y1_test_pred_optimal = (y1_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
default_threshold_predictions = (y1_test_pred_prob >= 0.5).astype(int)
print(classification_report(y1_test, default_threshold_predictions))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y1_test, y1_test_pred_optimal))

cm_default = confusion_matrix(y1_test, default_threshold_predictions)
plot_confusion_matrix(cm_default, "Confusion Matrix X1 (0.5 Threshold)",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_3_iqr/X1_CM_df_3_iqr.pdf')

cm_youden = confusion_matrix(y1_test, y1_test_pred_optimal)
plot_confusion_matrix(cm_youden, "Confusion Matrix X1 with Youden Optimization",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_3_iqr/X1_CM_youden_df_3_iqr.pdf')

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()

# TODO: X2_df_3_iqr
# --- Perform GridSearchCV ---
# --- With Body are injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_3_iqr/X2_df_3_iqr_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X2 with KNN')
print('---------------------------------------------------')
print(param_grid)
print('---------------------------------------------------')

grid_search = GridSearchCV(knn, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=num_cpu)
grid_search.fit(X2_train, y2_train.values.ravel())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Evaluate the best model
best_model_X2 = grid_search.best_estimator_
y2_test_pred_prob = best_model_X2.predict_proba(X2_test)[:, 1]
y2_test_pred = best_model_X2.predict(X2_test)

# Calculate AUC-ROC
auc_roc = roc_auc_score(y2_test, y2_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y2_test, y2_test_pred_prob)
plot_roc_curve(fpr, tpr, auc_roc, 'ROC Curve X2 (KNN)',
               'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_3_iqr/X2_ROC_df_3_iqr.pdf')

# Calculate Youden's Index
youden_index = tpr - fpr
optimal_idx = youden_index.argmax()
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold
y2_test_pred_optimal = (y2_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
default_threshold_predictions = (y2_test_pred_prob >= 0.5).astype(int)
print(classification_report(y2_test, default_threshold_predictions))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y2_test, y2_test_pred_optimal))

cm_default = confusion_matrix(y2_test, default_threshold_predictions)
plot_confusion_matrix(cm_default, "Confusion Matrix X2 (0.5 Threshold)",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_3_iqr/X2_CM_df_3_iqr.pdf')

cm_youden = confusion_matrix(y2_test, y2_test_pred_optimal)
plot_confusion_matrix(cm_youden, "Confusion Matrix X2 with Youden Optimization",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_3_iqr/X2_CM_youden_df_3_iqr.pdf')

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()


# TODO: df_1_5_iqr_df0
# -------------------- df_1_5_iqr_df0_UND --------------------
(X1_train, X1_test, y1_train, y1_test, X2_train, X2_test, y2_train,
 y2_test) = read_data('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df_1_5_iqr_df0/df_1_5_iqr_df0_UND.csv')

X1_train, X1_test, X2_train, X2_test = scaler(X1_train, X1_test, X2_train, X2_test)

# TODO: X1_df_1_5_iqr_df0
# --- Perform GridSearchCV ---
# --- With Career Injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_1_5_iqr_df0/X1_df_1_5_iqr_df0_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X1 with KNN')
print('---------------------------------------------------')
print(param_grid)
print('---------------------------------------------------')

grid_search = GridSearchCV(knn, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=num_cpu)
grid_search.fit(X1_train, y1_train.values.ravel())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Evaluate the best model
best_model_X1 = grid_search.best_estimator_
y1_test_pred_prob = best_model_X1.predict_proba(X1_test)[:, 1]
y1_test_pred = best_model_X1.predict(X1_test)

# Calculate AUC-ROC
auc_roc = roc_auc_score(y1_test, y1_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y1_test, y1_test_pred_prob)
plot_roc_curve(fpr, tpr, auc_roc, 'ROC Curve X1 (KNN)',
               'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_1_5_iqr_df0/X1_ROC_df_1_5_iqr_df0.pdf')

# Calculate Youden's Index
youden_index = tpr - fpr
optimal_idx = youden_index.argmax()
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold
y1_test_pred_optimal = (y1_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
default_threshold_predictions = (y1_test_pred_prob >= 0.5).astype(int)
print(classification_report(y1_test, default_threshold_predictions))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y1_test, y1_test_pred_optimal))

cm_default = confusion_matrix(y1_test, default_threshold_predictions)
plot_confusion_matrix(cm_default, "Confusion Matrix X1 (0.5 Threshold)",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_1_5_iqr_df0/X1_CM_df_1_5_iqr_df0.pdf')

cm_youden = confusion_matrix(y1_test, y1_test_pred_optimal)
plot_confusion_matrix(cm_youden, "Confusion Matrix X1 with Youden Optimization",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_1_5_iqr_df0/X1_CM_youden_df_1_5_iqr_df0.pdf')

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()

# TODO: X2_df_1_5_iqr_df0
# --- Perform GridSearchCV ---
# --- With Body are injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_1_5_iqr_df0/X2_df_1_5_iqr_df0_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X2 with KNN')
print('---------------------------------------------------')
print(param_grid)
print('---------------------------------------------------')

grid_search = GridSearchCV(knn, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=num_cpu)
grid_search.fit(X2_train, y2_train.values.ravel())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Evaluate the best model
best_model_X2 = grid_search.best_estimator_
y2_test_pred_prob = best_model_X2.predict_proba(X2_test)[:, 1]
y2_test_pred = best_model_X2.predict(X2_test)

# Calculate AUC-ROC
auc_roc = roc_auc_score(y2_test, y2_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y2_test, y2_test_pred_prob)
plot_roc_curve(fpr, tpr, auc_roc, 'ROC Curve X2 (KNN)',
               'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_1_5_iqr_df0/X2_ROC_df_1_5_iqr_df0.pdf')

# Calculate Youden's Index
youden_index = tpr - fpr
optimal_idx = youden_index.argmax()
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold
y2_test_pred_optimal = (y2_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
default_threshold_predictions = (y2_test_pred_prob >= 0.5).astype(int)
print(classification_report(y2_test, default_threshold_predictions))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y2_test, y2_test_pred_optimal))

cm_default = confusion_matrix(y2_test, default_threshold_predictions)
plot_confusion_matrix(cm_default, "Confusion Matrix X2 (0.5 Threshold)",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_1_5_iqr_df0/X2_CM_df_1_5_iqr_df0.pdf')

cm_youden = confusion_matrix(y2_test, y2_test_pred_optimal)
plot_confusion_matrix(cm_youden, "Confusion Matrix X2 with Youden Optimization",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_1_5_iqr_df0/X2_CM_youden_df_1_5_iqr_df0.pdf')

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()


# TODO: df_1_5_iqr
# -------------------- df_1_5_iqr_UND --------------------
(X1_train, X1_test, y1_train, y1_test, X2_train, X2_test, y2_train,
 y2_test) = read_data('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df_1_5_iqr/df_1_5_iqr_UND.csv')

X1_train, X1_test, X2_train, X2_test = scaler(X1_train, X1_test, X2_train, X2_test)

# TODO: X1_df_1_5_iqr
# --- Perform GridSearchCV ---
# --- With Career Injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_1_5_iqr/X1_df_1_5_iqr_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X1 with KNN')
print('---------------------------------------------------')
print(param_grid)
print('---------------------------------------------------')

grid_search = GridSearchCV(knn, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=num_cpu)
grid_search.fit(X1_train, y1_train.values.ravel())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Evaluate the best model
best_model_X1 = grid_search.best_estimator_
y1_test_pred_prob = best_model_X1.predict_proba(X1_test)[:, 1]
y1_test_pred = best_model_X1.predict(X1_test)

# Calculate AUC-ROC
auc_roc = roc_auc_score(y1_test, y1_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y1_test, y1_test_pred_prob)
plot_roc_curve(fpr, tpr, auc_roc, 'ROC Curve X1 (KNN)',
               'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_1_5_iqr/X1_ROC_df_1_5_iqr.pdf')

# Calculate Youden's Index
youden_index = tpr - fpr
optimal_idx = youden_index.argmax()
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold
y1_test_pred_optimal = (y1_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
default_threshold_predictions = (y1_test_pred_prob >= 0.5).astype(int)
print(classification_report(y1_test, default_threshold_predictions))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y1_test, y1_test_pred_optimal))

cm_default = confusion_matrix(y1_test, default_threshold_predictions)
plot_confusion_matrix(cm_default, "Confusion Matrix X1 (0.5 Threshold)",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_1_5_iqr/X1_CM_df_1_5_iqr.pdf')

cm_youden = confusion_matrix(y1_test, y1_test_pred_optimal)
plot_confusion_matrix(cm_youden, "Confusion Matrix X1 with Youden Optimization",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_1_5_iqr/X1_CM_youden_df_1_5_iqr.pdf')

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()

# TODO: X2_df_1_5_iqr
# --- Perform GridSearchCV ---
# --- With Body are injuries
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_1_5_iqr/X2_df_1_5_iqr_LOG.txt", "w")
sys.stdout = log_file

print('Performing Grid Search For X2 with KNN')
print('---------------------------------------------------')
print(param_grid)
print('---------------------------------------------------')

grid_search = GridSearchCV(knn, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='roc_auc',
                           verbose=1, n_jobs=num_cpu)
grid_search.fit(X2_train, y2_train.values.ravel())

# Best parameters and score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Evaluate the best model
best_model_X2 = grid_search.best_estimator_
y2_test_pred_prob = best_model_X2.predict_proba(X2_test)[:, 1]
y2_test_pred = best_model_X2.predict(X2_test)

# Calculate AUC-ROC
auc_roc = roc_auc_score(y2_test, y2_test_pred_prob)
print("Test Set ROC-AUC Score:", auc_roc)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y2_test, y2_test_pred_prob)
plot_roc_curve(fpr, tpr, auc_roc, 'ROC Curve X2 (KNN)',
               'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_1_5_iqr/X2_ROC_df_1_5_iqr.pdf')

# Calculate Youden's Index
youden_index = tpr - fpr
optimal_idx = youden_index.argmax()
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold using Youden's Index:", optimal_threshold)

# Apply the optimal threshold
y2_test_pred_optimal = (y2_test_pred_prob >= optimal_threshold).astype(int)

# Evaluate model performance with the optimal threshold
print("\nClassification metrics with Default Threshold (0.5):")
default_threshold_predictions = (y2_test_pred_prob >= 0.5).astype(int)
print(classification_report(y2_test, default_threshold_predictions))

print("\nClassification metrics with Optimal Threshold:")
print(classification_report(y2_test, y2_test_pred_optimal))

cm_default = confusion_matrix(y2_test, default_threshold_predictions)
plot_confusion_matrix(cm_default, "Confusion Matrix X2 (0.5 Threshold)",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_1_5_iqr/X2_CM_df_1_5_iqr.pdf')

cm_youden = confusion_matrix(y2_test, y2_test_pred_optimal)
plot_confusion_matrix(cm_youden, "Confusion Matrix X2 with Youden Optimization",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/KNN/df_1_5_iqr/X2_CM_youden_df_1_5_iqr.pdf')

# Close logging file
sys.stdout = sys.__stdout__
log_file.close()


