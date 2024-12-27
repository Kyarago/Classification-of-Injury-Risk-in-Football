import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

""" FUNCTIONS """

def calc_feature_relevance(x, y):
    feature_relevance = mutual_info_classif(x, y, random_state=2020)[0]
    return feature_relevance


def relevance_df(X, y):
    feature_list = X.columns.to_list()
    relevance_scores = []

    # Iterate over each feature in the DataFrame
    for feature in feature_list:
        # Calculate the relevance score for the feature
        relevance_score = calc_feature_relevance(X[[feature]], y)
        relevance_scores.append({'Feature': feature, 'Score': relevance_score})

    relev_df = pd.DataFrame(relevance_scores)

    relev_df = relev_df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    return relev_df


def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data


def iterative_vif_reduction(X, vif_threshold=5, log_file="vif_reduction_log.txt"):
    with open(log_file, "w") as log:
        iteration = 0
        while True:
            iteration += 1
            log.write(f"--- Iteration {iteration} ---\n")

            vif_df = calculate_vif(X)
            log.write("Current VIFs:\n")
            log.write(vif_df.to_string() + "\n")

            high_vif_features = vif_df[vif_df["VIF"] > vif_threshold]
            if high_vif_features.empty:
                log.write("No features with VIF above the threshold. Stopping.\n")
                break

            log.write(f"\nFeatures with VIF above {vif_threshold}:\n")
            log.write(high_vif_features.to_string() + "\n")

            # Calculate global pairwise correlations
            corr_matrix = X.corr().abs()

            # Calculate average correlations for high-VIF features across the entire dataset
            average_correlations = {
                feature: corr_matrix[feature].mean() for feature in high_vif_features["Feature"]
            }

            # Identify the feature with the highest average correlation across the dataset
            feature_to_remove = max(average_correlations, key=average_correlations.get)

            log.write(f"\nFeature selected for removal: {feature_to_remove}\n")
            log.write(f"Reason: Highest average correlation ({average_correlations[feature_to_remove]}) "
                      f"with all other features in the dataset.\n")

            # Remove the feature
            X = X.drop(columns=[feature_to_remove])

        log.write("\nFeature selection completed. Final set of features:\n")
        log.write(", ".join(X.columns.tolist()) + "\n")
    return X


# -------------------- df_full_UND ----------
df = pd.read_csv('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df full/df_full_UND.csv',
                 low_memory=False)
df = df.drop(['Cluster'], axis=1)
df = pd.get_dummies(df, columns=['Position'], drop_first=True)

# Modify True / False to 1 / 0
df = df.replace({True: 1, False: 0})

# Select target feature
X = df.drop(columns=['Injury condition'])
y = df[['Injury condition']]

rdf = relevance_df(X, y)
print(len(rdf[rdf['Score'] <= 0.01]))
rdf = rdf[rdf['Score'] > 0.01]
X = X[rdf['Feature'].unique().tolist()]

X_selected = iterative_vif_reduction(X, vif_threshold=5,
                                     log_file="C:/Users/aurim/Desktop/Mokslai/VIF/vif_reduction_df_full_UND.txt")


# -------------------- df_3_iqr_UND ----------
df = pd.read_csv('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df_3_iqr/df_3_iqr_UND.csv',
                 low_memory=False)
df = df.drop(['Cluster'], axis=1)
df = pd.get_dummies(df, columns=['Position'], drop_first=True)

# Modify True / False to 1 / 0
df = df.replace({True: 1, False: 0})

# Select target feature
X = df.drop(columns=['Injury condition'])
y = df[['Injury condition']]

rdf = relevance_df(X, y)
print(len(rdf[rdf['Score'] <= 0.01]))
rdf = rdf[rdf['Score'] > 0.01]
X = X[rdf['Feature'].unique().tolist()]

X_selected = iterative_vif_reduction(X, vif_threshold=5,
                                     log_file="C:/Users/aurim/Desktop/Mokslai/VIF/vif_reduction_df_3_iqr_UND.txt")


# -------------------- df_3_iqr_df0_UND ----------
df = pd.read_csv('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df_3_iqr_df0/df_3_iqr_df0_UND.csv',
                 low_memory=False)
df = df.drop(['Cluster'], axis=1)
df = pd.get_dummies(df, columns=['Position'], drop_first=True)

# Modify True / False to 1 / 0
df = df.replace({True: 1, False: 0})

# Select target feature
X = df.drop(columns=['Injury condition'])
y = df[['Injury condition']]

rdf = relevance_df(X, y)
print(len(rdf[rdf['Score'] <= 0.01]))
rdf = rdf[rdf['Score'] > 0.01]
X = X[rdf['Feature'].unique().tolist()]

X_selected = iterative_vif_reduction(X, vif_threshold=5,
                                     log_file="C:/Users/aurim/Desktop/Mokslai/VIF/vif_reduction_df_3_iqr_df0_UND.txt")


# -------------------- df_1_5_iqr_UND ----------
df = pd.read_csv('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df_1_5_iqr/df_1_5_iqr_UND.csv',
                 low_memory=False)
df = df.drop(['Cluster'], axis=1)
df = pd.get_dummies(df, columns=['Position'], drop_first=True)

# Modify True / False to 1 / 0
df = df.replace({True: 1, False: 0})

# Select target feature
X = df.drop(columns=['Injury condition'])
y = df[['Injury condition']]

rdf = relevance_df(X, y)
print(len(rdf[rdf['Score'] <= 0.01]))
rdf = rdf[rdf['Score'] > 0.01]
X = X[rdf['Feature'].unique().tolist()]

X_selected = iterative_vif_reduction(X, vif_threshold=5,
                                     log_file="C:/Users/aurim/Desktop/Mokslai/VIF/vif_reduction_df_1_5_iqr_UND.txt")


# -------------------- df_1_5_iqr_df0_UND ----------
df = pd.read_csv('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df_1_5_iqr_df0/df_1_5_iqr_df0_UND.csv',
                 low_memory=False)
df = df.drop(['Cluster'], axis=1)
df = pd.get_dummies(df, columns=['Position'], drop_first=True)

# Modify True / False to 1 / 0
df = df.replace({True: 1, False: 0})

# Select target feature
X = df.drop(columns=['Injury condition'])
y = df[['Injury condition']]

rdf = relevance_df(X, y)
print(len(rdf[rdf['Score'] <= 0.01]))
rdf = rdf[rdf['Score'] > 0.01]
X = X[rdf['Feature'].unique().tolist()]

X_selected = iterative_vif_reduction(X, vif_threshold=5,
                                     log_file="C:/Users/aurim/Desktop/Mokslai/VIF/vif_reduction_df_1_5_iqr_df0_UND.txt")


