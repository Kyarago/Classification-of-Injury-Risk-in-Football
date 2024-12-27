import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('C:/Users/aurim/Desktop/Mokslai/findff3_n0c_s.csv', low_memory=False)

print('Rows in data:', len(df))
print('Unique players in data', len(df['Player_id'].unique()))

# Balance of classes in percentage:
frequency_counts = df['Injury condition'].value_counts(normalize=True) * 100

exp = df.describe()
exp.to_csv('C:/Users/aurim/Desktop/Mokslai/exp_fin_n0c_s.csv', index=False)

# Descriptive stats when only the players that got injured at least once are considered
dfn = df[df['Career injuries'] > 1]
exp = dfn.describe()
exp.to_csv('C:/Users/aurim/Desktop/Mokslai/exp_dfn.csv', index=False)


print('Number of players in the final data:', len(df['Player_id'].unique()))

smin = df[df['Season Minutes'] == 14760]
cb = df[df['Player name'] == 'claudio-bravo']

""" UNIVARIATE ANALYSIS """
""" PLOTS BY POSITION """
# Group by 'Position' and count unique 'Player_id'
position_counts = df.groupby('Position')['Player_id'].nunique().reset_index()
# Renamed columns for clarity
position_counts.columns = ['Position', 'Unique_Player_Count']
position_counts = position_counts[~position_counts['Position'].isin(['Defender', 'Attack', 'Midfield', 'SW'])]

# Split into groups by position
position_groups = {
    'AM': 'Midfield',
    'CB': 'Defense',
    'CF': 'Attack',
    'CM': 'Midfield',
    'DM': 'Midfield',
    'GK': 'Goalkeeper',
    'LB': 'Defense',
    'LM': 'Midfield',
    'LW': 'Attack',
    'RB': 'Defense',
    'RM': 'Midfield',
    'RW': 'Attack',
    'SS': 'Attack',
    'SW': 'Goalkeeper'}

place_marks = {
    'Goalkeeper' : 'A',
    'Defense': 'B',
    'Midfield': 'C',
    'Attack': 'D'}

position_counts['Group'] = position_counts['Position'].map(position_groups)
position_counts['Place_mark'] = position_counts['Group'].map(place_marks)
position_counts = position_counts.sort_values(by='Place_mark')

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Position', y='Unique_Player_Count', data=position_counts,
                 hue='Group', palette='dark')

plt.xlabel('Position')
plt.ylabel('Number of Unique Players')

for i in ax.containers:
    ax.bar_label(i, label_type='edge')

plt.legend(loc='upper right')
plt.show()
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Images/Data Exploration/unique_players_by_position.pdf')
plt.close()

# Graph about the positions which get injured the most
dff = df[['Position', 'Injury condition']]
dff = dff[~dff['Position'].isin(['Defender', 'Attack', 'Midfield', 'SW'])]
total_per_position = dff.groupby('Position').size().reset_index(name='Total Rows')
injured_per_position = dff[dff['Injury condition'] == 1].groupby('Position').size().reset_index(name='Injured Rows')
inj = pd.merge(total_per_position, injured_per_position, on='Position', how='left')
inj['Injured Rows'] = inj['Injured Rows'].fillna(0)
inj['Percentage'] = (inj['Injured Rows'] / inj['Total Rows']) * 100
inj['Percentage'] = round(inj['Percentage'], 3)
inj['Group'] = inj['Position'].map(position_groups)
inj['Place_mark'] = inj['Group'].map(place_marks)
inj = inj.sort_values(by='Place_mark')

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Position', y='Percentage', data=inj,
                 hue='Group', palette='dark')

plt.xlabel('Position')
plt.ylabel('Percentage, %')
plt.ylim(0, 1)

for i in ax.containers:
    ax.bar_label(i, label_type='edge')

plt.legend(loc='upper right')
plt.show()
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Images/Data Exploration/Injury_by_position.pdf')
plt.close()


# Areas that are the most commonly injured
dff = df[['Position', 'Injury area condition', 'Injury condition']]
dff = dff[~dff['Position'].isin(['Defender', 'Attack', 'Midfield', 'SW'])]
dffc = dff.dropna(subset=['Injury area condition'])
inja = dffc.groupby('Injury area condition').size().reset_index(name='count area')
inja['Percentage'] = (inja['count area'] / sum(inja['count area'])) * 100
inja['Percentage'] = round(inja['Percentage'], 3)
inja = inja.sort_values(by='Percentage', ascending=False)


# Maybe certain positions tend to have certain areas injured more often?
dff = df[['Position', 'Injury area condition', 'Injury condition']]
dff = dff[~dff['Position'].isin(['Defender', 'Attack', 'Midfield', 'SW'])]
dffc = dff.dropna(subset=['Injury area condition'])
injap = dffc.groupby(['Position', 'Injury area condition']).size().reset_index(name='count')

injap_sorted = injap.sort_values(['Position', 'count'], ascending=[True, False])
injap_sorted5 = injap_sorted.groupby('Position').head(5)
injap_sorted3 = injap_sorted.groupby('Position').head(3)

injap_sorted5['Injury area condition'].unique() # 7
injap_sorted3['Injury area condition'].unique() # Only 4 different options for top 3


# Type of injuries that required the longest healing period
dff = df[['Date', 'Position', 'Player name', 'Player_id', 'Recent big injury', 'Injury area condition', 'Injury condition']]
dff = dff[~dff['Position'].isin(['Defender', 'Attack', 'Midfield', 'SW'])]
dff['Date'] = pd.to_datetime(dff['Date'])
dff['Date'] = pd.to_datetime(dff['Date'], format='mixed')

dff['Gap in days'] = dff.groupby('Player_id')['Date'].diff().dt.days
dff['Healing time (days)'] = dff.groupby('Player_id')['Gap in days'].shift(-1)

dfn = dff[dff['Injury condition'] == 1]
dfn = dfn.sort_values(by='Healing time (days)', ascending=False)

average_healing_times = dfn.groupby('Injury area condition')['Healing time (days)'].mean()
median_healing_times = dfn.groupby('Injury area condition')['Healing time (days)'].median()

summary_stats = pd.DataFrame({
    'Average Healing Time (days)': average_healing_times,
    'Median Healing Time (days)': median_healing_times
}).reset_index().sort_values(by='Average Healing Time (days)', ascending=False)

summary_stats['Average Healing Time (days)'] = round(summary_stats['Average Healing Time (days)'], 3)
summary_stats['Median Healing Time (days)'] = round(summary_stats['Median Healing Time (days)'], 3)

summary_stats.to_csv('C:/Users/aurim/Desktop/Mokslai/exp_healing_stats.csv', index=False)

# Independent vs Target variable boxplots
dff_c = df.dropna(subset=['height'])

plt.figure(figsize=(10, 6))
sns.boxplot(x='Injury condition', y='height', data=dff_c, palette='dark', showfliers=False)
plt.ylabel('Height, cm', fontsize=18)
plt.xlabel('Injury', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Images/Data Exploration/BP_Injury_height.pdf', bbox_inches="tight")
plt.close()

# All minute columns
columns_to_plot = ['Player Age', '10 Minutes', '30 Minutes', 'Season Minutes', 'Career Minutes',
                   'Days without injury', 'Career days injured', 'Season days injured', 'Career injuries']
y_labels = ['Age, years', 'Minutes Played in Last 10 Days', 'Minutes Played in Last 30 Days', 'Season Minutes Played',
            'Career Minutes Played', 'Days Without Injury', 'Career Days Injured', 'Season Days Injured',
            'Career Injuries']  # Corresponding y-axis labels
titles = ['Injury Event by Player Age', 'Injury Event by Minutes Played in Last 10 Days',
          'Injury Event by Minutes Played in Last 30 Days', 'Injury Event by Season Minutes',
          'Injury Event by Career Minutes', 'Injury Event by Days Without Injury',
          'Injury Event by Career Days Injured',  'Injury Event by Season Days Injured',
          'Injury Event by Career injuries']  # Corresponding plot titles

for column, y_label, title in zip(columns_to_plot, y_labels, titles):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Injury condition', y=column, data=df, palette='dark', showfliers=False)
    plt.ylabel(y_label, fontsize=18)
    plt.xlabel('Injury', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(f'C:/Users/aurim/Desktop/Mokslai/Images/Data Exploration/BP_Injury_{column}.pdf',
                bbox_inches="tight")
    plt.close()



# Correlation matrix
dff = df[['height', 'Player Age', '10 Minutes', '30 Minutes', 'Season Minutes', 'Career Minutes', 'Days without injury',
          'Career days injured', 'Season days injured', 'Career injuries', 'Recent big injury', 'Injury condition']]
correlation_matrix = dff.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=False)
plt.xticks(rotation=75)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('C:/Users/aurim/Desktop/Mokslai/Images/Data Exploration/Correlation_df_full.pdf')
plt.close()
