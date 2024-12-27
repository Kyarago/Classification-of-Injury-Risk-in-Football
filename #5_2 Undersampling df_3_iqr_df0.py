import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# For boxplots
columns_to_plot = ['Player Age', '10 Minutes', '30 Minutes', 'Season Minutes', 'Career Minutes',
                   'Days without injury', 'Career days injured', 'Season days injured', 'Career injuries']
y_labels = ['Age, years', 'Minutes Played in Last 10 Days', 'Minutes Played in Last 30 Days', 'Season Minutes Played',
            'Career Minutes Played', 'Days Without Injury', 'Career Days Injured', 'Season Days Injured',
            'Career Injuries']  # Corresponding y-axis labels
titles = ['Cluster by Player Age', 'Cluster by Minutes Played in Last 10 Days',
          'Cluster by Minutes Played in Last 30 Days', 'Cluster by Season Minutes',
          'Cluster by Career Minutes', 'Cluster by Days Without Injury',
          'Cluster by Career Days Injured',  'Cluster by Season Days Injured',
          'Cluster by Career injuries']  # Corresponding plot titles

# Colors for plots
colors = sns.color_palette("dark")

# Scaler for normalization
scaler = StandardScaler()


# --------------- df_3_iqr_df0 ---------------
df_3_iqr_df0 = pd.read_csv('C:/Users/aurim/Desktop/Mokslai/df_3_iqr_df0.csv', low_memory=False)
df_3_iqr_df0 = df_3_iqr_df0.drop(['Minutes played', 'height'], axis=1)

df1=df_3_iqr_df0[df_3_iqr_df0['Injury condition']==1]
df0=df_3_iqr_df0[df_3_iqr_df0['Injury condition']==0]
del df_3_iqr_df0

# Prepare the data for k-means
X = df0.drop(columns=['Injury condition'])

X['Recent big injury'] = X['Recent big injury'].astype(int)
X = pd.get_dummies(X, columns=['Position'], drop_first=True)

df_var = pd.DataFrame(X.var().sort_values(), columns=['Variance']).reset_index()
df_var = df_var[df_var['Variance'] > 1]
print(df_var['index'].unique())

X = X[['Player Age', 'Career days injured', '10 Minutes', '30 Minutes', 'Season Minutes', 'Days without injury',
       'Career Minutes']]
X = scaler.fit_transform(X)

# ------------ Calinski - Harabasz index score -----------
calinski_scores = []

# Loop over a range of cluster numbers
for n in range(2, 11):  # Start from 2 because Calinski-Harabasz score requires at least 2 clusters
    scores_runs = []

    # Perform K-Means clustering 10 times and record the Calinski-Harabasz score
    for _ in range(10):
        kmeans = KMeans(n_clusters=n, random_state=None)
        kmeans.fit(X)
        score = calinski_harabasz_score(X, kmeans.labels_)
        scores_runs.append(score)

    # Calculate the average score for the current number of clusters
    average_score = sum(scores_runs) / len(scores_runs)
    calinski_scores.append(average_score)

# Determine the optimal number of clusters based on the highest score
optimal_k = 2 + calinski_scores.index(max(calinski_scores))  # Adding 2 to adjust the starting index

plt.figure()
plt.plot(range(2, 11), calinski_scores, marker='o', color=colors[0])
plt.axvline(x=optimal_k, color=colors[1], linestyle='--', label=f'Optimal k: {optimal_k}')
plt.xlabel('Number of Clusters')
plt.ylabel('Calinski-Harabasz Score')
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df_3_iqr_df0/CH_df_3_iqr_df0.pdf',
            bbox_inches="tight")
plt.close()


# ------------ CH method PCA -----------
# Fit PCA to the data
pca = PCA()  # Initially fit with all components
X_pca = pca.fit_transform(X)

# Compute the cumulative variance explained
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Determine the number of components needed to explain at least 90% of the variance
threshold = 0.9
num_components = np.argmax(cumulative_variance >= threshold) + 1

print(f"Number of components to reach {threshold * 100}% variance: {num_components}")

# Apply PCA with the determined number of components
pca = PCA(n_components=num_components)
X_pca = pca.fit_transform(X)

# Convert the PCA results to a DataFrame
df_pca = pd.DataFrame(data=X_pca, columns=[f'pc{i+1}' for i in range(num_components)])

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color=colors[0])
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.axhline(y=threshold, color=colors[1], linestyle='--', label=f'{threshold * 100}% Threshold')
plt.legend()
plt.show()
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df_3_iqr_df0/PCA_df_3_iqr_df0.pdf',
            bbox_inches="tight")
plt.close()

calinski_scores = []

# Loop over a range of cluster numbers
for n in range(2, 11):  # Start from 2 because Calinski-Harabasz score requires at least 2 clusters
    scores_runs = []

    # Perform K-Means clustering 10 times and record the Calinski-Harabasz score
    for _ in range(10):
        kmeans = KMeans(n_clusters=n, random_state=None)
        kmeans.fit(df_pca)
        score = calinski_harabasz_score(df_pca, kmeans.labels_)
        scores_runs.append(score)

    # Calculate the average score for the current number of clusters
    average_score = sum(scores_runs) / len(scores_runs)
    calinski_scores.append(average_score)

# Determine the optimal number of clusters based on the highest score
optimal_k_pca = 2 + calinski_scores.index(max(calinski_scores))  # Adding 2 to adjust the starting index

plt.figure()
plt.plot(range(2, 11), calinski_scores, marker='o', color=colors[0])
plt.axvline(x=optimal_k_pca, color=colors[1], linestyle='--', label=f'Optimal k: {optimal_k_pca}')
plt.xlabel('Number of Clusters')
plt.ylabel('Calinski-Harabasz Score')
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df_3_iqr_df0/CH_df_3_iqr_df0_pca.pdf', bbox_inches="tight")
plt.close()


# ------------ K-Means Clustering -----------
kmeans = KMeans(n_clusters=optimal_k_pca, random_state=2024)
cluster_labels = kmeans.fit_predict(X)

# Add cluster labels to the DataFrame
df0['Cluster'] = cluster_labels

for label in np.unique(cluster_labels):
    cluster_data = df0[df0['Cluster'] == label]
    summary_stats = cluster_data.describe()
    summary_stats.to_csv(f'C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df_3_iqr_df0/EXP_cluster_{label}_df_3_iqr_df0.csv')

# Draw boxplots with clusters as classes
for column, y_label, title in zip(columns_to_plot, y_labels, titles):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y=column, data=df0, palette='dark', showfliers=False)
    plt.ylabel(y_label, fontsize=18)
    plt.xlabel('Cluster', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(f'C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df_3_iqr_df0/BP_Cluster_{column}_df_3_iqr_df0.pdf',
                bbox_inches="tight")
    plt.close()

# ------------ Data thinning after clustering -----------
N_minority = len(df1)
N_majority = len(df0)

# Calculate the proportion to keep from the majority class to achieve a 60/40 balance
proportion = (3 * N_minority) / (2 * N_majority)
print('Proportion to keep:', proportion)

# Create an empty list to collect indices of data to keep
indices_to_keep = []
# Loop through each cluster and drop percentage of the data
for cluster in np.unique(cluster_labels):
    # Get indices of all cases in the current cluster
    cluster_indices = df0[df0['Cluster'] == cluster].index

    # Calculate the number of cases to keep
    n_samples_to_keep = int(proportion * len(cluster_indices))

    # Randomly select indices to keep
    sampled_indices = np.random.choice(cluster_indices, n_samples_to_keep, replace=False)
    indices_to_keep.extend(sampled_indices)

# Create a new DataFrame with only the selected cases
df0_thinned = df0.loc[indices_to_keep].reset_index(drop=True)

df_3_iqr_df0_UND = pd.concat([df0_thinned, df1])
df_3_iqr_df0_UND.to_csv('C:/Users/aurim/Desktop/Mokslai/Undersampled Data/df_3_iqr_df0/df_3_iqr_df0_UND.csv',
                   index=False)

# Balance of classes in percentage:
print(df_3_iqr_df0_UND['Injury condition'].value_counts())
print(df_3_iqr_df0_UND['Injury condition'].value_counts(normalize=True) * 100)


