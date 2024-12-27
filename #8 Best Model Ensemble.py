from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from Functions.Model_helpers import *
import matplotlib; matplotlib.use('TkAgg')
import sys
import joblib
from sklearn.ensemble import VotingClassifier
import numpy as np


# -------------------- FUNCTIONS --------------------
# Class for a hard voting classifier with custom thresholds
class ThresholdVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, thresholds):
        self.estimators = estimators
        self.thresholds = thresholds

    def fit(self, X, y):
        for _, model in self.estimators:
            model.fit(X, y)
        return self

    def predict(self, X):
        votes = []
        for name, model in self.estimators:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:, 1]  # Probability for class 1
            else:
                probs = model.decision_function(X)  # Decision score for class 1

            threshold = self.thresholds[name]
            vote = (probs >= threshold).astype(int)
            votes.append(vote)

        # Majority voting step
        votes = np.array(votes).T
        return np.round(np.mean(votes, axis=1)).astype(int)


# -------------------- DATA --------------------
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

# Split the data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.3, random_state=2024, stratify=y)
print(y1_test.shape[0])
print(sum(y1_test['Injury condition']))

# ------------------------- MODELS -------------------------
# ------------------------------ LR ------------------------------
# Best Hyperparameters: {'C': 0.1, 'penalty': 'l2', 'solver': 'newton-cg'}
lr = LogisticRegression(C=0.1, penalty='l2', solver='newton-cg', max_iter=1000)
lr.fit(X1_train, y1_train.values.ravel())
joblib.dump(lr, "C:/Users/aurim/Desktop/Mokslai/Model Data/LR_model.pkl")

# ------------------------------ SVM ------------------------------
# Best Hyperparameters: {'C': 100, 'max_iter': 1000}
svc = LinearSVC(C=100, max_iter=1000)
svc.fit(X1_train, y1_train.values.ravel())
joblib.dump(svc, "C:/Users/aurim/Desktop/Mokslai/Model Data/SVM_model.pkl")

# ------------------------------ KNN ------------------------------
# Best Hyperparameters: {'metric': 'canberra', 'n_neighbors': 46, 'weights': 'uniform'}
knn = KNeighborsClassifier(metric='canberra', n_neighbors=46, weights='uniform')
knn.fit(X1_train, y1_train.values.ravel())
joblib.dump(knn, "C:/Users/aurim/Desktop/Mokslai/Model Data/KNN_model.pkl")

# ------------------------------ RF ------------------------------
# Best Hyperparameters: {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 20, 'n_estimators': 200}
rf = RandomForestClassifier(class_weight='balanced', max_features='sqrt', bootstrap=True, max_depth=10,
                            min_samples_leaf=2, min_samples_split=20, n_estimators= 200, random_state=2024)
rf.fit(X1_train, y1_train.values.ravel())
joblib.dump(rf, "C:/Users/aurim/Desktop/Mokslai/Model Data/RF_model.pkl")

# ------------------------------ XGB ------------------------------
# Best Hyperparameters: {'colsample_bytree': 1, 'gamma': 0.5, 'learning_rate': 0.05, 'max_depth': 5,
# 'min_child_weight': 5, 'n_estimators': 250, 'subsample': 0.8}
xgb = XGBClassifier(eval_metric='logloss', tree_method='hist', colsample_bytree=1, gamma=0.5, learning_rate=0.05,
                    max_depth=5, min_child_weight=5, n_estimators=250, subsample=0.8)
xgb.fit(X1_train, y1_train.values.ravel())
joblib.dump(xgb, "C:/Users/aurim/Desktop/Mokslai/Model Data/XGB_model.pkl")


# ------------------------------ VOTING ENSEMBLES ------------------------------
# Soft voting requires models with `predict_proba` method, therefore SVC is removed
soft_voting_clf = VotingClassifier(
    estimators=[('LR', lr), ('KNN', knn), ('RF', rf), ('XGB', xgb)], voting='soft')

hard_voting_clf = VotingClassifier(
    estimators=[('LR', lr), ('SVM', svc), ('KNN', knn), ('RF', rf), ('XGB', xgb)], voting='hard')

# Fit the voting classifiers
soft_voting_clf.fit(X1_train, y1_train.values.ravel())
hard_voting_clf.fit(X1_train, y1_train.values.ravel())

# Save the voting classifiers
joblib.dump(soft_voting_clf, "C:/Users/aurim/Desktop/Mokslai/Model Data/Soft_Vote_model.pkl")
joblib.dump(hard_voting_clf, "C:/Users/aurim/Desktop/Mokslai/Model Data/Hard_Vote_model.pkl")


# -------------------- LOGGING --------------------
# Open logging file
log_file = open("C:/Users/aurim/Desktop/Mokslai/Model Data/Voting_Ensamble.txt", "w")
sys.stdout = log_file

print('---------------------------------------------------')
# Evaluate the soft voting ensemble
soft_decision_scores = soft_voting_clf.predict_proba(X1_test)[:, 1]
soft_auc_roc = roc_auc_score(y1_test, soft_decision_scores)
print("Soft Voting ROC-AUC Score:", soft_auc_roc)

# Compute ROC curve for soft voting
soft_fpr, soft_tpr, soft_thresholds = roc_curve(y1_test, soft_decision_scores)
plot_roc_curve(soft_fpr, soft_tpr, soft_auc_roc, 'ROC Curve - Soft Voting',
               'C:/Users/aurim/Desktop/Mokslai/Model Data/Soft_Voting_ROC_Curve.pdf')

# Calculate Youden's Index for soft voting
soft_youden_index = soft_tpr - soft_fpr
soft_optimal_idx = soft_youden_index.argmax()
soft_optimal_threshold = soft_thresholds[soft_optimal_idx]
print("Optimal Threshold (Soft Voting):", soft_optimal_threshold)

# Apply the optimal threshold for soft voting
soft_predictions_optimal = (soft_decision_scores >= soft_optimal_threshold).astype(int)

# Evaluate the soft voting classifier
print("\nSoft Voting Metrics with Default Threshold (0.5):")
default_soft_predictions = (soft_decision_scores >= 0.5).astype(int)
print(classification_report(y1_test, default_soft_predictions))

print("\nSoft Voting Metrics with Optimal Threshold:")
print(classification_report(y1_test, soft_predictions_optimal))

# Plot confusion matrices for soft voting
soft_cm_default = confusion_matrix(y1_test, default_soft_predictions)
plot_confusion_matrix(soft_cm_default, "Soft Voting - Confusion Matrix",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/Soft_CM.pdf')

soft_cm_optimal = confusion_matrix(y1_test, soft_predictions_optimal)
plot_confusion_matrix(soft_cm_optimal, "Soft Voting - Confusion Matrix with Youden Optimization",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/Soft_CM_Youden.pdf')

# Evaluate the hard voting ensemble
hard_predictions = hard_voting_clf.predict(X1_test)

# Evaluate the hard voting classifier
print("\nHard Voting Metrics:")
print(classification_report(y1_test, hard_predictions))

# Plot confusion matrix for hard voting
hard_cm = confusion_matrix(y1_test, hard_predictions)
plot_confusion_matrix(hard_cm, "Hard Voting - Confusion Matrix",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/Hard_CM.pdf')

# Define models with their thresholds
estimators = [
    ('LR', lr),
    ('SVM', svc),
    ('KNN', knn),
    ('RF', rf),
    ('XGB', xgb)]

optimal_thresholds = {
    'LR': 0.32915940092159673,
    'SVM': -0.37685647826937263,
    'KNN': 0.34782608695652173,
    'RF': 0.4825829163177666,
    'XGB': 0.36773866}

# Initialize and fit custom hard voting classifier
custom_hard_voting_clf = ThresholdVotingClassifier(estimators=estimators, thresholds=optimal_thresholds)
custom_hard_voting_clf.fit(X1_train, y1_train.values.ravel())

# Save the custom voting classifier
joblib.dump(custom_hard_voting_clf, "C:/Users/aurim/Desktop/Mokslai/Model Data/Optimal_Hard_Vote_model.pkl")

ohard_predictions = custom_hard_voting_clf.predict(X1_test)
print("\nOptimal Hard Voting Metrics:")
print(classification_report(y1_test, ohard_predictions))

optimal_hard_cm = confusion_matrix(y1_test, ohard_predictions)
plot_confusion_matrix(optimal_hard_cm, "Hard Voting - Confusion Matrix with Youden Optimised Thresholds",
                      'C:/Users/aurim/Desktop/Mokslai/Model Data/Hard_CM_Youden.pdf')

# Close the logging file
sys.stdout = sys.__stdout__
log_file.close()


