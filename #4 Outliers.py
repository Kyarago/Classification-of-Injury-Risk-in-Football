import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:/Users/aurim/Desktop/Mokslai/findff3_n0c_s.csv', low_memory=False)

df = df[df['Position'] != 'GK']

df = df.drop(['Date', 'Player name', 'Player_id', 'dob', 'Injury area condition', 'Injury area_sp condition',
              'Patella', 'Patella_1 year'], axis=1)
df1=df[df['Injury condition']==1]
df0=df[df['Injury condition']==0]
del df

# Z Score Method isn't used since it doesn't work well without normal distributions
# Reason why injury area columns can't be used to determine the outliers
numerical_columns = ['Player Age', 'Career Minutes', '10 Minutes', '30 Minutes', 'Season Minutes',
                     'Days without injury', 'Career days injured', 'Season days injured', 'Career injuries', 'Tendon',
                     'Tendon_1 year', 'Patellar tendon', 'Patellar tendon_1 year', 'Vertebra', 'Vertebra_1 year',
                     'Ankle', 'Ankle_1 year', 'Pubis', 'Pubis_1 year', 'Achilles', 'Achilles_1 year', 'Knee',
                     'Knee_1 year', 'Thigh', 'Thigh_1 year', 'Peroneus tendon', 'Peroneus tendon_1 year', 'Hamstring',
                     'Hamstring_1 year', 'Cartilage', 'Cartilage_1 year', 'Muscle', 'Muscle_1 year', 'Adductor',
                     'Adductor_1 year', 'Hip', 'Hip_1 year', 'Back', 'Back_1 year', 'Spine', 'Spine_1 year', 'General',
                     'General_1 year', 'Plantar fascia', 'Plantar fascia_1 year', 'Tarsus ligament',
                     'Tarsus ligament_1 year', 'Groin', 'Groin_1 year', 'Calf', 'Calf_1 year', 'Knee ligament',
                     'Knee ligament_1 year', 'Toe', 'Toe_1 year', 'Abdominal', 'Abdominal_1 year', 'Foot',
                     'Foot_1 year', 'Fibula', 'Fibula_1 year', 'Collateral Ligament', 'Collateral Ligament_1 year',
                     'Bone', 'Bone_1 year', 'Leg', 'Leg_1 year', 'Lumbar', 'Lumbar_1 year', 'Pelvis', 'Pelvis_1 year',
                     'Ligament', 'Ligament_1 year', 'Heel', 'Heel_1 year', 'Meniscus', 'Meniscus_1 year', 'Metatarsal',
                     'Metatarsal_1 year', 'Shin', 'Shin_1 year', 'Ankle ligament', 'Ankle ligament_1 year']

num_df0 = df0[numerical_columns]

Q1 = num_df0.quantile(0.25)
Q3 = num_df0.quantile(0.75)
IQR = Q3 - Q1

cleaned_num_df0 = num_df0[~((num_df0 < (Q1 - 3 * IQR)) |(num_df0 > (Q3 + 3 * IQR))).any(axis=1)]
print(f'Non - Injuries \nwas {df0.shape[0]}, now {cleaned_num_df0.shape[0]} (3IQR)')

# Check what columns are the reason for outlier
lower_bound_df0 = (Q1 - 3 * IQR).to_frame(name='lower_bound').reset_index().drop_duplicates()
upper_bound_df0 = (Q3 + 3 * IQR).to_frame(name='upper_bound').reset_index().drop_duplicates()
bounds_df0 = pd.merge(lower_bound_df0, upper_bound_df0, on='index', how='inner')
bounds_df0.to_csv('C:/Users/aurim/Desktop/Mokslai/bounds_df0.csv', index=False)

num_df1 = df1[numerical_columns]

Q1 = num_df1.quantile(0.25)
Q3 = num_df1.quantile(0.75)
IQR = Q3 - Q1

lower_bound_df1 = (Q1 - 3 * IQR).to_frame(name='lower_bound').reset_index().drop_duplicates()
upper_bound_df1 = (Q3 + 3 * IQR).to_frame(name='upper_bound').reset_index().drop_duplicates()
bounds_df1 = pd.merge(lower_bound_df1, upper_bound_df1, on='index', how='inner')

cleaned_num_df1 = num_df1[~((num_df1 < (Q1 - 3 * IQR)) |(num_df1 > (Q3 + 3 * IQR))).any(axis=1)]
print(f'Non - Injuries \nwas {df1.shape[0]}, now {cleaned_num_df1.shape[0]} (3IQR)')



# 3 IQR Method
numerical_columns = ['Player Age', 'Career Minutes', '10 Minutes', '30 Minutes', 'Season Minutes',
                     'Days without injury', 'Career days injured', 'Season days injured', 'Career injuries']
categorical_columns = ['height', 'Minutes played', 'Position', 'Recent big injury', 'Injury condition', 'Tendon',
                       'Tendon_1 year', 'Patellar tendon', 'Patellar tendon_1 year', 'Vertebra', 'Vertebra_1 year',
                       'Ankle', 'Ankle_1 year', 'Pubis', 'Pubis_1 year', 'Achilles', 'Achilles_1 year', 'Knee',
                       'Knee_1 year', 'Thigh', 'Thigh_1 year', 'Peroneus tendon', 'Peroneus tendon_1 year', 'Hamstring',
                       'Hamstring_1 year', 'Cartilage', 'Cartilage_1 year', 'Muscle', 'Muscle_1 year', 'Adductor',
                       'Adductor_1 year', 'Hip', 'Hip_1 year', 'Back', 'Back_1 year', 'Spine', 'Spine_1 year',
                       'General', 'General_1 year', 'Plantar fascia', 'Plantar fascia_1 year', 'Tarsus ligament',
                       'Tarsus ligament_1 year', 'Groin', 'Groin_1 year', 'Calf', 'Calf_1 year', 'Knee ligament',
                       'Knee ligament_1 year', 'Toe', 'Toe_1 year', 'Abdominal', 'Abdominal_1 year', 'Foot',
                       'Foot_1 year', 'Fibula', 'Fibula_1 year', 'Collateral Ligament', 'Collateral Ligament_1 year',
                       'Bone', 'Bone_1 year', 'Leg', 'Leg_1 year', 'Lumbar', 'Lumbar_1 year', 'Pelvis', 'Pelvis_1 year',
                       'Ligament', 'Ligament_1 year', 'Heel', 'Heel_1 year', 'Meniscus', 'Meniscus_1 year',
                       'Metatarsal', 'Metatarsal_1 year', 'Shin', 'Shin_1 year', 'Ankle ligament',
                       'Ankle ligament_1 year']

num_df0 = df0[numerical_columns]
cat_df0 = df0[categorical_columns]

Q1 = num_df0.quantile(0.25)
Q3 = num_df0.quantile(0.75)
IQR = Q3 - Q1

cleaned_num_df0 = num_df0[~((num_df0 < (Q1 - 3 * IQR)) |(num_df0 > (Q3 + 3 * IQR))).any(axis=1)]
print(f'Non - Injuries \nwas {df0.shape[0]}, now {cleaned_num_df0.shape[0]} (3IQR)')
print(f'Drop by {((len(df0) - len(cleaned_num_df0)) / len(df0)) * 100} %')

# Check what columns are the reason for outlier
lower_bound_df0 = (Q1 - 3 * IQR).to_frame(name='lower_bound').reset_index().drop_duplicates()
upper_bound_df0 = (Q3 + 3 * IQR).to_frame(name='upper_bound').reset_index().drop_duplicates()
bounds_df0 = pd.merge(lower_bound_df0, upper_bound_df0, on='index', how='inner')

# Recombine filtered numerical data with the original categorical data
filtered_entries = cleaned_num_df0.index
cleaned_df0 = pd.concat([cleaned_num_df0, cat_df0.loc[filtered_entries]], axis=1)
df_3_iqr_df0 = pd.concat([cleaned_df0, df1])

# Correlation matrix
dff = df_3_iqr_df0[['Player Age', 'Career Minutes', '10 Minutes', '30 Minutes', 'Season Minutes', 'Days without injury',
                    'Career days injured', 'Season days injured', 'Career injuries', 'Injury condition']]
correlation_matrix = dff.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=False)
plt.xticks(rotation=75)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Images/Data Exploration/Correlation_df_3_iqr_df0.pdf')
plt.close()

df_3_iqr_df0.to_csv('C:/Users/aurim/Desktop/Mokslai/df_3_iqr_df0.csv', index=False)

# df1
num_df1 = df1[numerical_columns]
cat_df1 = df1[categorical_columns]

Q1 = num_df1.quantile(0.25)
Q3 = num_df1.quantile(0.75)
IQR = Q3 - Q1

lower_bound_df1 = (Q1 - 3 * IQR).to_frame(name='lower_bound').reset_index().drop_duplicates()
upper_bound_df1 = (Q3 + 3 * IQR).to_frame(name='upper_bound').reset_index().drop_duplicates()
bounds_df1 = pd.merge(lower_bound_df1, upper_bound_df1, on='index', how='inner')

cleaned_num_df1 = num_df1[~((num_df1 < (Q1 - 3 * IQR)) |(num_df1 > (Q3 + 3 * IQR))).any(axis=1)]
print(f'Non - Injuries \nwas {df1.shape[0]}, now {cleaned_num_df1.shape[0]} (3IQR)')
print(f'Drop by {((len(df1) - len(cleaned_num_df1)) / len(df1)) * 100} %')

# Recombine filtered numerical data with the original categorical data
filtered_entries = cleaned_num_df1.index
cleaned_df1 = pd.concat([cleaned_num_df1, cat_df1.loc[filtered_entries]], axis=1)
df_3_iqr = pd.concat([cleaned_df0, cleaned_df1])

# Correlation matrix
dff = df_3_iqr[['Player Age', 'Career Minutes', '10 Minutes', '30 Minutes', 'Season Minutes', 'Days without injury',
                    'Career days injured', 'Season days injured', 'Career injuries', 'Injury condition']]
correlation_matrix = dff.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=False)
plt.xticks(rotation=75)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Images/Data Exploration/Correlation_df_3_iqr.pdf')
plt.close()

df_3_iqr.to_csv('C:/Users/aurim/Desktop/Mokslai/df_3_iqr.csv', index=False)

bounds_3iqr = pd.merge(bounds_df0, bounds_df1, on='index', how='inner')
bounds_3iqr.columns = ['Column name', 'Lower Bound df0', 'Upper Bound df0', 'Lower Bound df1', 'Upper Bound df1']
bounds_3iqr.to_csv('C:/Users/aurim/Desktop/Mokslai/bounds_3iqr.csv', index=False)



# 1.5 IQR Method
numerical_columns = ['Player Age', 'Career Minutes', '10 Minutes', '30 Minutes', 'Season Minutes',
                     'Days without injury', 'Career days injured', 'Season days injured', 'Career injuries']
categorical_columns = ['height', 'Minutes played', 'Position', 'Recent big injury', 'Injury condition', 'Tendon',
                       'Tendon_1 year', 'Patellar tendon', 'Patellar tendon_1 year', 'Vertebra', 'Vertebra_1 year',
                       'Ankle', 'Ankle_1 year', 'Pubis', 'Pubis_1 year', 'Achilles', 'Achilles_1 year', 'Knee',
                       'Knee_1 year', 'Thigh', 'Thigh_1 year', 'Peroneus tendon', 'Peroneus tendon_1 year', 'Hamstring',
                       'Hamstring_1 year', 'Cartilage', 'Cartilage_1 year', 'Muscle', 'Muscle_1 year', 'Adductor',
                       'Adductor_1 year', 'Hip', 'Hip_1 year', 'Back', 'Back_1 year', 'Spine', 'Spine_1 year',
                       'General', 'General_1 year', 'Plantar fascia', 'Plantar fascia_1 year', 'Tarsus ligament',
                       'Tarsus ligament_1 year', 'Groin', 'Groin_1 year', 'Calf', 'Calf_1 year', 'Knee ligament',
                       'Knee ligament_1 year', 'Toe', 'Toe_1 year', 'Abdominal', 'Abdominal_1 year', 'Foot',
                       'Foot_1 year', 'Fibula', 'Fibula_1 year', 'Collateral Ligament', 'Collateral Ligament_1 year',
                       'Bone', 'Bone_1 year', 'Leg', 'Leg_1 year', 'Lumbar', 'Lumbar_1 year', 'Pelvis', 'Pelvis_1 year',
                       'Ligament', 'Ligament_1 year', 'Heel', 'Heel_1 year', 'Meniscus', 'Meniscus_1 year',
                       'Metatarsal', 'Metatarsal_1 year', 'Shin', 'Shin_1 year', 'Ankle ligament',
                       'Ankle ligament_1 year']

num_df0 = df0[numerical_columns]
cat_df0 = df0[categorical_columns]

Q1 = num_df0.quantile(0.25)
Q3 = num_df0.quantile(0.75)
IQR = Q3 - Q1

cleaned_num_df0 = num_df0[~((num_df0 < (Q1 - 1.5 * IQR)) |(num_df0 > (Q3 + 1.5 * IQR))).any(axis=1)]
print(f'Non - Injuries \nwas {df0.shape[0]}, now {cleaned_num_df0.shape[0]} (3IQR)')
print(f'Drop by {((len(df0) - len(cleaned_num_df0)) / len(df0)) * 100} %')

# Check what columns are the reason for outlier
lower_bound_df0 = (Q1 - 1.5 * IQR).to_frame(name='lower_bound').reset_index().drop_duplicates()
upper_bound_df0 = (Q3 + 1.5 * IQR).to_frame(name='upper_bound').reset_index().drop_duplicates()
bounds_df0 = pd.merge(lower_bound_df0, upper_bound_df0, on='index', how='inner')

# Recombine filtered numerical data with the original categorical data
filtered_entries = cleaned_num_df0.index
cleaned_df0 = pd.concat([cleaned_num_df0, cat_df0.loc[filtered_entries]], axis=1)
df_1_5_iqr_df0 = pd.concat([cleaned_df0, df1])

# Correlation matrix
dff = df_1_5_iqr_df0[['Player Age', 'Career Minutes', '10 Minutes', '30 Minutes', 'Season Minutes', 'Days without injury',
                    'Career days injured', 'Season days injured', 'Career injuries', 'Injury condition']]
correlation_matrix = dff.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=False)
plt.xticks(rotation=75)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Images/Data Exploration/Correlation_df_1_5_iqr_df0.pdf')
plt.close()

df_1_5_iqr_df0.to_csv('C:/Users/aurim/Desktop/Mokslai/df_1_5_iqr_df0.csv', index=False)

# df1
num_df1 = df1[numerical_columns]
cat_df1 = df1[categorical_columns]

Q1 = num_df1.quantile(0.25)
Q3 = num_df1.quantile(0.75)
IQR = Q3 - Q1

lower_bound_df1 = (Q1 - 1.5 * IQR).to_frame(name='lower_bound').reset_index().drop_duplicates()
upper_bound_df1 = (Q3 + 1.5 * IQR).to_frame(name='upper_bound').reset_index().drop_duplicates()
bounds_df1 = pd.merge(lower_bound_df1, upper_bound_df1, on='index', how='inner')

cleaned_num_df1 = num_df1[~((num_df1 < (Q1 - 1.5 * IQR)) |(num_df1 > (Q3 + 1.5 * IQR))).any(axis=1)]
print(f'Non - Injuries \nwas {df1.shape[0]}, now {cleaned_num_df1.shape[0]} (3IQR)')
print(f'Drop by {((len(df1) - len(cleaned_num_df1)) / len(df1)) * 100} %')

# Recombine filtered numerical data with the original categorical data
filtered_entries = cleaned_num_df1.index
cleaned_df1 = pd.concat([cleaned_num_df1, cat_df1.loc[filtered_entries]], axis=1)
df_1_5_iqr = pd.concat([cleaned_df0, cleaned_df1])

# Correlation matrix
dff = df_1_5_iqr[['Player Age', 'Career Minutes', '10 Minutes', '30 Minutes', 'Season Minutes', 'Days without injury',
                    'Career days injured', 'Season days injured', 'Career injuries', 'Injury condition']]
correlation_matrix = dff.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=False)
plt.xticks(rotation=75)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Images/Data Exploration/Correlation_df_1_5_iqr.pdf')
plt.close()

df_1_5_iqr.to_csv('C:/Users/aurim/Desktop/Mokslai/df_1_5_iqr.csv', index=False)

bounds_1_5iqr = pd.merge(bounds_df0, bounds_df1, on='index', how='inner')
bounds_1_5iqr.columns = ['Column name', 'Lower Bound df0', 'Upper Bound df0', 'Lower Bound df1', 'Upper Bound df1']
bounds_1_5iqr.to_csv('C:/Users/aurim/Desktop/Mokslai/bounds_1_5iqr.csv', index=False)


