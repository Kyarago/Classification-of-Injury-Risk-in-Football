import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import DateOffset
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


""" FUNCTIONS """
def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds'

def process_dob(dob):
  if isinstance(dob, str):
    try:
      dob_date_str = dob.split('(')[0].strip()
      dob_datetime = datetime.strptime(dob_date_str, "%b %d, %Y")
      return dob_datetime
    except ValueError:
      return 'Bad DoB'
  else:
    return 'Bad DoB'

def calculate_age(row):
    return relativedelta(row['Date'], row['dob']).years

# Replace missing position with player's official position
def replace_missing_pos(df):
    df['Pos.'] = df['Pos.'].fillna(df['Position'])
    return df

# Function to clean and convert minutes
def clean_minutes(value):
  if any(item == value for item in missed_list):
    return 0  # Append 0 for non-played games
  else:
    return int(value.strip('\''))

# Calculating minutes played in the last 10 days for each match, excluding the current match day
def minutes_in_last_10_days(row, df):
  lower_bound = row['Date'] - pd.Timedelta(days=10)
  recent_matches = df[(df['Date'] >= lower_bound) & (df['Date'] < row['Date'])]
  return recent_matches['Minutes played'].sum()

# Calculating minutes played in the last 30 days
def minutes_in_last_30_days(row, df):
  lower_bound = row['Date'] - pd.Timedelta(days=30)
  recent_matches = df[(df['Date'] >= lower_bound) & (df['Date'] < row['Date'])]
  return recent_matches['Minutes played'].sum()

def process_player_data(df):
    df['Position'] = df['Position'].replace(position_map)

    # Player's date of birth processed
    df['dob'] = df['dob'].apply(process_dob)
    df['Player Age'] = df.apply(calculate_age, axis=1)

    # Clean and accumulate minutes played
    df['Minutes played'] = df['Minutes played'].str.lower()
    df['Minutes played'] = df['Minutes played'].apply(clean_minutes)
    df['Career Minutes'] = df['Minutes played'].cumsum()

    # Minutes played in the last 10 days
    df['10 Minutes'] = df.apply(lambda row: minutes_in_last_10_days(row, df), axis=1)

    # add minutes played in the last 30 days (before the current matchday)
    df['30 Minutes'] = df.apply(lambda row: minutes_in_last_30_days(row, df), axis=1)

    # Calculate minutes played since the last rest period (6 weeks without a game)
    df['Days since last game'] = df['Date'].diff().dt.days.fillna(0)
    df['Reset'] = df['Days since last game'] >= 42
    df['Group'] = df['Reset'].cumsum()
    df['Season Minutes'] = df.groupby('Group')['Minutes played'].cumsum()

    # Injury flag
    df['Pos.'] = df['Pos.'].str.lower()
    df['Injury flag'] = df['Pos.'].apply(lambda x: 1 if x not in non_minutes else 0)
    df['Injury start'] = 0
    df['Injury start'] = (df['Injury flag'] == 1) & (df['Injury flag'].shift(1) == 0)
    df['Injury start'] = df['Injury start'].astype(int)  # Convert boolean to integer

    # Injury name
    df['Injury'] = df['Pos.'].apply(lambda x: x if x not in non_minutes else 'N/A')
    df['Injury'] = df['Injury'].str.lower()

    # Calculate days injured
    df['Injury period'] = df['Injury start'].cumsum() * df['Injury flag']
    df['Injury start date'] = df['Date'].where(df['Injury start'] == 1)
    df['Injury start date'] = df.groupby('Injury period')['Injury start date'].ffill()
    df['Days injured'] = (df['Date'] - df['Injury start date']).dt.days + 1
    df.loc[df['Injury flag'] == 0, 'Days injured'] = 0  # Reset days injured to 0 where there is no Injury
    df.drop(columns=['Injury period', 'Injury start date'], inplace=True)

    # Calculate the difference in days for rows where 'Injury flag' is 1
    df['Injury days'] = df[df['Injury flag'] == 1]['Date'].diff().dt.days
    df['Injury days'] = df['Injury days'].fillna(0)
    df.loc[df['Injury start'] == 1, 'Injury days'] = 1

    # Days since last injury
    df['last injury date'] = df['Date'].where(df['Injury flag'] == 1)
    df['last injury date'] = df['last injury date'].fillna(method='ffill')
    df['Days since last injury'] = (df['Date'] - df['last injury date']).dt.days
    df.loc[df['Injury flag'] == 1, 'Days since last injury'] = 0 # Reset days since last injury to 0 where there is an injury
    df.drop(columns=['last injury date'], inplace=True)

    # Days injured in career
    df['Career days injured'] = df['Injury days'].cumsum()
    # Calculate days injured since the last rest period
    df['Season days injured'] = df.groupby('Group')['Injury days'].cumsum()
    df.drop(['Days since last game', 'Reset', 'Group', 'Pos.'], axis=1, inplace=True)

    # Career injuries
    df['Career injuries'] = df['Injury start'].cumsum()

    # Create a new column 'Big Injury' that flags injuries longer than 5 months
    df['Big injury flag'] = (df['Days injured'] > 150).astype(int)

    # Days since last big injury
    df['last big injury date'] = df['Date'].where(df['Big injury flag'] == 1)
    df['last big injury date'] = df['last big injury date'].fillna(method='ffill')
    df['Days since last big injury'] = (df['Date'] - df['last big injury date']).dt.days
    df.loc[df['Big injury flag'] == 1, 'Days since last big injury'] = 0 # Reset days since last injury to 0 where there is an injury
    df['Recent big injury'] = (df['Days since last big injury'] < 180).astype(int)
    df.drop(columns=['last big injury date', 'Days since last big injury'], inplace=True)

    # Add column with injury area
    df['Injury area'] = df['Injury'].replace(injury_map)

    return df

# Create columns for each unique injury area
def injury_columns(df, injury_map):
  df['injury_group'] = df['Injury area'].map(injury_map)

  unique_groups = set(injury_map.values())
  for group in unique_groups:
    df[group] = 0
    df[f'{group}_1 year'] = 0

  df = df.drop(['injury_group'], axis=1)

  return df

# Put 1s in relevant injury columns
def injury_area_flag(df):
  for index, row in df.iterrows():
    injury_area = row['Injury area']
    if injury_area != 'n/a' and injury_area in df.columns:
      df.at[index, injury_area] = 1

  return df

# Calculate the number of injuries for body part
def injury_sequence(df, columns_to_process):
  df = df.reset_index(drop=True)

  for column in columns_to_process:
    counter = 0
    is_in_sequence = False

    for idx in range(len(df)):
      if df.loc[idx, column] == 1:
        if not is_in_sequence:
          counter += 1
          is_in_sequence = True
        df.loc[idx, column] = counter
      else:
        is_in_sequence = False

  return df

# Function to calculate injuries during the previous year for a given body area
def calculate_injuries_last_year(row, group_df, injury_type):
    # Get the date 1 year before the current date
    date_1_year_ago = row['Date'] - DateOffset(years=1)

    # Find the closest date in the grouped dataframe that is less than or equal to the date 1 year ago
    closest_row = group_df[group_df['Date'] <= date_1_year_ago].iloc[-1:]

    if not closest_row.empty:
        return float(row[injury_type]) - float(closest_row[injury_type].values[0])
    else:
        return 0

# Apply calculations grouped by player
def calculate_for_each_player(group):
    c_time = datetime.now().strftime("%H:%M:%S")
    print('Start Time: ', {c_time}, 'Player:', group['Player name'][0])
    for area in columns_to_process:
        group[f'{area}_1 year'] = group.apply(lambda row: calculate_injuries_last_year(row, group, area), axis=1)
    return group

def process_player_group_initial(group):
    group = process_player_data(group)
    return group

def process_players_in_parallel_initial(df, max_workers=None):
    grouped = df.groupby('Player_id', group_keys=False)
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_player = {executor.submit(process_player_group_initial, 
                                            group): player for player, 
                            group in grouped}
        for future in as_completed(future_to_player):
            player = future_to_player[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'{player} generated an exception: {exc}')
    
    return pd.concat(results)

def process_player_group_injuries(group):
    group = injury_area_flag(group)
    group = injury_sequence(group, columns_to_process)
    group[columns_to_process] = group[columns_to_process].replace(0, pd.NA).ffill().fillna(0)
    group = calculate_for_each_player(group)
    return group

def process_players_in_parallel_injuries(df, max_workers=None):
    grouped = df.groupby('Player_id', group_keys=False)
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_player = {executor.submit(process_player_group_injuries, 
                                            group): player for player, 
                            group in grouped}
        for future in as_completed(future_to_player):
            player = future_to_player[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'{player} generated an exception: {exc}')
    
    return pd.concat(results)

# Function to fix mistakes denoting false returns from injury
def fix_mistakes_per_player(group):
    for i in range(len(group)):
        if group['mistake'].iloc[i] == 1:
            start_pos = group['Pos.'].iloc[i]
            j = i + 1
            while j < len(group) and group['mistake'].iloc[j] == 1:
                group['Pos.'].iloc[j] = start_pos
                j += 1
    return group



""" VARIABLES """

position_map = {
  'Goalkeeper': 'GK',
  'Sweeper': 'SW',
  'Centre-Back': 'CB',
  'Left-Back': 'LB',
  'Right-Back': 'RB',
  'Defensive Midfield': 'DM',
  'Central Midfield': 'CM',
  'Right Midfield': 'RM',
  'Left Midfield': 'LM',
  'Attacking Midfield': 'AM',
  'Left Winger': 'LW',
  'Right Winger': 'RW',
  'Second Striker': 'SS',
  'Centre-Forward': 'CF'}

injury_map = {
    'abdominal muscle strain': 'Abdominal',
    'abdominal problems': 'Abdominal',
    'abdominal strain': 'Abdominal',
    'achilles heel problems': 'Achilles',
    'achilles tendon injury': 'Achilles',
    'achilles tendon irritation': 'Achilles',
    'achilles tendon problems': 'Achilles',
    'achilles tendon rupture': 'Achilles',
    'achilles tendon surgery': 'Achilles',
    'adductor injury': 'Adductor',
    'adductor problems': 'Adductor',
    'adductor tear': 'Adductor',
    'ankle injury': 'Ankle',
    'ankle ligament tear': 'Ankle ligament',
    'ankle problems': 'Ankle',
    'ankle sprain': 'Ankle',
    'ankle surgery': 'Ankle',
    'arch problems': 'Foot',
    'back injury': 'Back',
    'back problems': 'Back',
    'back trouble': 'Back',
    'bone sprain': 'Bone',
    'broken ankle': 'Ankle',
    'broken foot': 'Foot',
    'calf injury': 'Calf',
    'calf muscle strain': 'Calf',
    'calf muscle tear': 'Calf',
    'calf problems': 'Calf',
    'calf stiffness': 'Calf',
    'calf strain': 'Calf',
    'capsular tear of ankle joint': 'Ankle',
    'cartilage damage': 'Cartilage',
    'collateral ankle ligament tear': 'Ankle ligament',
    'collateral ligament injury': 'Collateral Ligament',
    'collateral ligament ripture': 'Collateral Ligament',
    'collateral ligament tear': 'Collateral Ligament',
    'compression of the spine': 'Spine',
    'contracture': 'Muscle',
    'cruciate ligament injury': 'Knee ligament',
    'cruciate ligament strain': 'Knee ligament',
    'cruciate ligament surgery': 'Knee ligament',
    'cruciate ligament tear': 'Knee ligament',
    'cyst in the knee': 'Knee',
    'dislocation fracture of the ankle joint': 'Ankle',
    'dislocation of the kneecap': 'Knee',
    'double ligament tear': 'Ligament',
    'edema in the knee': 'Knee',
    'fatigue fracture': 'Bone',
    'fissure of the fibula': 'Fibula',
    'foot injury': 'Foot',
    'foot surgery': 'Foot',
    'groin injury': 'Groin',
    'groin problems': 'Groin',
    'groin strain': 'Groin',
    'groin surgery': 'Groin',
    'hairline crack in foot': 'Foot',
    'hairline fracture in the fibula': 'Fibula',
    'hairline fracture in the muscles': 'Muscle',
    'hamstring injury': 'Hamstring',
    'hamstring muscle injury': 'Hamstring',
    'hamstring strain': 'Hamstring',
    'heel injury': 'Heel',
    'heel problems': 'Heel',
    'heel spur': 'Heel',
    'herniated disc': 'Vertebra',
    'hip flexor problems': 'Hip',
    'hip injury': 'Hip',
    'hip problems': 'Hip',
    'hip surgery': 'Hip',
    'inflammation in the ankle joint': 'Ankle',
    'inflammation in the head of the fibula': 'Fibula',
    'inflammation in the knee': 'Knee',
    'inflammation of ligaments in the knee': 'Knee ligament',
    'inflammation of the biceps tendon in the thigh': 'Thigh',
    'inflammation of the sole of the foot': 'Foot',
    'injury to abdominal muscles': 'Abdominal',
    'injury to the ankle': 'Ankle',
    'inner ankle ligament tear': 'Ankle ligament',
    'inner knee ligament tear': 'Knee ligament',
    'inner ligament injury': 'Ligament',
    'inner ligament tear in ankle joint': 'Ankle ligament',
    'internal ligament strain': 'Ligament',
    'internal ligament tear': 'Ligament',
    'knee collateral ligament strain': 'Knee ligament',
    'knee collateral ligament tear': 'Knee ligament',
    'knee injury': 'Knee',
    'knee medial ligament tear': 'Knee ligament',
    'knee problems': 'Knee',
    'knee surgery': 'Knee',
    'knock': 'General',
    'left hip flexor problems': 'Hip',
    'leg injury': 'Leg',
    'ligament injury': 'Ligament',
    'ligament stretching': 'Ligament',
    'ligament tear': 'Ligament',
    'longitudinal tendon tear': 'Tendon',
    'lumbago': 'Lumbar',
    'lumbar vertebra problems': 'Vertebra',
    'medial collateral ligament tear': 'Knee ligament',
    'meniscal injury': 'Meniscus',
    'meniscus damage': 'Meniscus',
    'meniscus injury': 'Meniscus',
    'meniscus irritation': 'Meniscus',
    'meniscus tear': 'Meniscus',
    'metatarsal fracture': 'Metatarsal',
    'minor knock': 'General',
    'muscle fatigue': 'Muscle',
    'muscle fiber tear': 'Muscle',
    'muscle injury': 'Muscle',
    'muscle problems': 'Muscle',
    'muscle stiffness': 'Muscle',
    'muscle strain': 'Muscle',
    'muscle tear': 'Muscle',
    'muscular problems': 'Muscle',
    'outer ligament problems': 'Ligament',
    'outer ligament tear': 'Ligament',
    'overstretching': 'Muscle',
    'overstretching of the syndesmotic ligament': 'Ankle ligament',
    'partial damage to the cruciate ligament': 'Knee ligament',
    'partial muscle tear': 'Muscle',
    'partial patellar tendon tear': 'Patellar tendon',
    'partial tear of the plantar fascia': 'Plantar fascia',
    'patella problems': 'Patella',
    'patellar tendinopathy syndrome': 'Patellar tendon',
    'patellar tendon dislocation': 'Patellar tendon',
    'patellar tendon irritation': 'Patellar tendon',
    'patellar tendon problems': 'Patellar tendon',
    'patellar tendon rupture': 'Patellar tendon',
    'patellar tendon tear': 'Patellar tendon',
    'pelvic injury': 'Pelvis',
    'pelvic obliquity': 'Pelvis',
    'peroneus tendon injury': 'Peroneus tendon',
    'pubalgia': 'Pubis',
    'pubitis': 'Pubis',
    'right hip flexor problems': 'Hip',
    'shin injury': 'Shin',
    'shinbone injury': 'Shin',
    'sore muscles': 'Muscle',
    'sprain': 'General',
    'sprained ankle': 'Ankle',
    'sprained foot': 'Foot',
    'sprained knee': 'Knee',
    'strain': 'Muscle',
    'strain in the thigh and gluteal muscles': 'Thigh',
    'stress reaction of the bone': 'Bone',
    'syndesmosis ligament tear': 'Ankle ligament',
    'syndesmotic ligament tear': 'Ankle ligament',
    'tear of the lateral meniscus': 'Meniscus',
    'tendon rupture': 'Tendon',
    'tendon tear': 'Tendon',
    'tendonitis': 'Tendon',
    'thigh muscle rupture': 'Thigh',
    'thigh muscle strain': 'Thigh',
    'thigh problems': 'Thigh',
    'toe injury': 'Toe',
    'torn ankle ligament': 'Ankle ligament',
    'torn ankle ligaments': 'Ankle ligament',
    'torn knee ligaments': 'Knee ligament',
    'torn lateral ankle ligament': 'Ankle ligament',
    'torn lateral knee ligament': 'Knee ligament',
    'torn ligament': 'Ligament',
    'torn ligaments': 'Ligament',
    'torn ligaments in the tarsus': 'Tarsus ligament',
    'torn muscle bundle': 'Muscle',
    'torn muscle fiber': 'Muscle',
    'torn muscle fiber in the adductor area': 'Adductor',
    'torn muscle fibre': 'Muscle',
    'torn thigh muscle': 'Thigh'
}

columns_to_process = list(set(injury_map.values()))

non_minutes = ["on the bench", "Not in squad", "Information not yet available", 'inflammation', 'Inflammation of the leg skin', 'Metatarsal bruise', 'Capsule rupture', 'corn', 'Toe joint capsular tear',
               "Red card suspension", "Yellow card suspension", 'No eligibility', 'Called up to national team', 'Rib fracture', 'Virus', 'Fitness', 'Quarantine', 'Appendicitis',
               "Indirect card suspension", "Suspended", "Coronavirus", 'Crack bruise', 'Remove screws/nails', 'Metacarpal fracture', 'Broken collarbone', 'Fracture of the humeral head',
               "Dental surgery", "Fever", "Hand injury", "Ill", "Illness", 'SW', 'Head injury', 'Pubic bone irritation', 'tuberculosis', 'angina', "Shoulder injury", 'Broken leg',
               "Influenza", "Wisdom teeth removal", '', 'RB', 'LB', 'LM', 'DM', 'Tonsillitis', 'Herniated umbilical button', 'Femoral neck fracture', 'flesh wound', 'Fracture of fibula shaft',
               'GK', 'RW', 'LW', 'CF', 'LM', 'RM', 'SS', 'AM', 'CB', 'CM', 'Shingles', 'Food poisoning', 'Tendon irritation', 'abscess', 'Nose injury', 'special leave', 'Muscle contusion',
               'Olympic games', 'pneumonia', 'Military service', 'Neck injury', 'visa issues', 'Facial injury', 'Blood clots in the lungs', 'Ruptured eardrum', 'Splenic rupture',
               'Broken jaw', 'Mononucleation', 'Doping ban', 'meningitis', 'Midfacial fracture', 'heart problems', 'Bursitis', 'Broken finger', 'Wrist injury', 'unknown injury', 'Cracked bone',
               'Corona virus', 'Cerebral hemorrhage', 'Bone inflammation', 'Finger injury', 'Capsular injury', 'Intestinal virus', 'Eye injury', 'Blockage in the back', 'Adductor pain',
               'Skull base fracture', 'Auditory trauma', 'Inguinal hernia', 'cold', 'Shoulder injury', 'Appendectomy', 'bronchitis', 'Broken thumb', 'contortion', 'Heart problems',
               'Hairline crack in middle finger', 'Frontal sinus fracture', 'Broken arm', 'Toothache', 'Pubic bone bruise', 'Lymphatic cancer', 'Scaphoid fracture', 'bruise', 'Broken toe',
               'depression', 'Bruised ribs', 'Sciatica problems', 'combustion', 'allergic reaction', 'Cheekbone bruise', 'Inflammation of pubic bone', 'Cyst on jaw', 'Dead leg', 'Rest',
               'Eyebow fracture', 'Forearm fracture', 'collapsed lung', 'open wound', 'Kidney stone surgery', 'Inner ligament stretch of the knee', 'Bone marrow swelling', 'Broken tibia',
               'Inflammation of the pancreas', 'Elbow fracture', 'Skull fracture', 'Acromioclavicular joint dislocation', 'surgery', 'Broken cheekbone', 'Concussion', 'Broken fibula',
               'influenza', 'Cancer', 'Whiplash', 'Bone edema', 'Stomach flu', 'Kidney problems', 'Cheekbone surgery', 'Neck bruise', 'Thumb injury', 'traffic accident', 'Pelvic contusion',
               'Balance disorder', 'flu', 'Testicular cancer', 'malaria', 'Nose surgery',  'Pneumothorax', 'Intestinal surgery', 'Arthroscopy', 'Hole in the eardrum', 'Elbow injury',
               'Broken hand', 'Shoulder joint contusion', 'Gunshot wound','Swine flu', 'pinched nerve', 'Broken nose bone', 'Compartment syndrome', 'Lung contusion', 'Broken kneecap',
               'Circulation problems', 'Blood poisoning', 'Suspension through sports court', 'laceration', 'Facial fracture', 'infection', 'Wrist fracture', 'Broken shoulder', 'Fracture of frontal bone',
               'coma', 'Tooth infection', 'Testicle rupture', 'Insect bite', 'Fracture of the eye socket', 'Cyst in the heel', 'stomach problems', 'chickenpox', 'laceration wound',
               'Cut', 'Chest injury', 'Arm injury', 'In training camp with first team', "Dislocated shoulder", "Gastrointestinal problems", "Heart condition", "Neck injury", 'Bone splintering',
               'Bruise on the ankle joint', 'Tibial head contusion', "Bruised ankle", "Bruised knee", 'Bone bruise', "Bruised rib", "Bruised shin", 'Bruised back', "Ankle fracture", "Patella fracture",
               "Rib fracture", "Thighbone fracture", 'Femoral fracture', 'Lumbar vertebra fracture', 'Cervical vertebra fracture', 'Tibia and fibula fracture', 'Lower leg fracture','fracture',
               'Achilles tendon contusion', 'Bruise on ankle', 'Bruise on shinbone', 'Cervical spine injury', 'Coccyx bruise', "Fibula fracture", 'Foot bruise', 'Hip bruise', 'Knee bruise', 'Shin bruise',
               "Vertebral fracture", 'Vertebral injury', 'nan', 'n/a', 'left winger', 'centre-back', 'left-back', 'attacking midfield', 'defensive midfield', 'central midfield', 'centre-forward',
               'midfield', 'right-back', 'goalkeeper', 'left midfield', 'right midfield', 'second striker', 'defender', 'attack', 'sweeper', 'right winger']

injury_list = [
    "Abdominal strain", "Achilles tendon injury", "Adductor problems", "Ankle injury", "Ankle surgery", "Back injury", "Back trouble",
    "Calf injury", "Calf muscle strain", "Cruciate ligament injury", 'Herniated disc',
    "Foot injury", "Hamstring injury", "Hip injury", "Hip problems", "Hip surgery", "Knee injury", "Knee problems", "Knee surgery", "Ligament injury", "Medial collateral ligament tear",
    "Meniscal injury", 'Abdominal muscle strain', "Metatarsal fracture", "Muscle fatigue", "Muscle injury", "Muscle problems", "Patella problems", "Pubitis", "Shinbone injury",
    "Sprained ankle", "Sprained foot", "Sprained knee", "Thigh muscle strain", "Thigh problems", "Thigh muscle rupture", "Toe injury",
    "Torn ankle ligament", "Torn ligament", "Torn muscle fibre", 'fatigue fracture', 'Cruciate ligament injury', 'Foot surgery',
    'Adductor injury', 'Achilles tendon problems', 'Syndesmosis ligament tear', 'Partial tear of the plantar fascia', 'Groin strain', 'Injury to the ankle', 'Knee collateral ligament tear', 'Achilles tendon surgery',
    'Torn lateral knee ligament', 'Achilles tendon irritation', 'Shin injury', 'Arch problems', 'Muscle tear', 'muscle stiffness', 'Calf stiffness', 'Partial muscle tear', 'Leg injury', 'Cartilage damage', 'Heel spur',
    'Calf strain', 'Sore muscles', 'Achilles heel problems', 'Bone sprain', 'Contracture', 'Muscle fatigue', 'Cyst in the knee', 'Torn muscle bundle',
    'Muscle fiber tear', 'Lumbago', 'Groin surgery', 'Edema in the knee', 'Calf injury', 'Compression of the spine',
    'Inflammation of the biceps tendon in the thigh', 'muscular problems', 'Adductor tear', 'Inner ankle ligament tear', 'Knee surgery', 'Collateral ligament ripture', 'Outer ligament problems', 'Torn thigh muscle',
    'Internal ligament tear', 'Meniscus damage', 'Torn ankle ligaments', 'Achilles tendon problems', 'Fissure of the fibula', 'Calf problems',
    'Hip flexor problems', 'Dislocation of the kneecap', 'Inner ligament tear in ankle joint', 'Right hip flexor problems', 'Inflammation of ligaments in the knee',
    'Muscle injury', 'Hairline crack in foot', 'Tendon rupture', 'Meniscus irritation', 'Hamstring injury', 'Patellar tendon problems', 'Thigh problems', 'Syndesmosis ligament tear', 'Tendon tear',
    'Abdominal problems', 'Partial damage to the cruciate ligament', 'Back injury', 'Ankle problems', 'Tendonitis',
    'Achilles tendon rupture', 'Ligament injury', 'sprain', 'Calf muscle tear', 'Strain in the thigh and gluteal muscles', 'Outer ligament tear', 'Patellar tendon dislocation',
    'Hamstring muscle injury', 'Collateral ligament tear', 'Lumbar vertebra problems', 'Internal ligament strain', 'Adductor injury', 'Pubalgia', 'Groin problems', 'Left hip flexor problems',
    'Torn muscle fiber in the adductor area', 'Ankle surgery', 'Broken foot', 'Torn ligaments',  'Patellar tendinopathy syndrome',
    'Partial patellar tendon tear', 'Broken ankle', 'Groin strain', 'Collateral ankle ligament tear', 'Inflammation in the knee', 'Peroneus tendon injury', 'Patellar tendon rupture', 'Ankle injury',
    'Inflammation of the sole of the foot', 'Injury to abdominal muscles', 'Torn knee ligaments', 'minor knock', 'Heel problems', 'Knee injury', 'Injury to the ankle', 'Pelvic obliquity',
    'Foot injury', 'Hip problems', 'Capsular tear of ankle joint', 'Torn ligaments in the tarsus', 'Torn muscle fiber', 'Hairline fracture in the fibula', 'Meniscus injury',
    'Tear of the lateral meniscus', 'Ankle sprain', 'Patellar tendon irritation', 'Knee medial ligament tear', 'Collateral ligament injury',
    'Knock', 'Syndesmotic ligament tear', 'Overstretching', 'Knee collateral ligament strain', 'double ligament tear', 'Stress reaction of the bone',
    'Ligament tear', 'Cruciate ligament surgery', 'Groin injury', 'Ligament stretching', 'strain', 'Ankle ligament tear', 'Heel injury',
     'ankle sprain', 'Hip injury', 'Longitudinal tendon tear', 'Overstretching of the syndesmotic ligament', 'Cruciate ligament tear',  'Hairline fracture in the muscles',
    'Hamstring strain', 'Patellar tendon tear', 'Inner ligament injury', 'Dislocation fracture of the ankle joint', 'Cruciate ligament strain', 'Pelvic injury',
     'Muscle strain', 'Inner knee ligament tear', 'Inflammation in the head of the fibula', 'Meniscus tear', 'Back problems', 'Torn lateral ankle ligament', 'Knee problems', 'Inflammation in the ankle joint', 'Toe injury']

non_minutes = [item.lower() for item in non_minutes]
injury_list = [item.lower() for item in injury_list]

missed_list = non_minutes + injury_list


""" DATA PREPARATION AND FIXING """
data = pd.read_csv('C:/Users/aurim/Desktop/Mokslai/Players_Chunk_All.csv', low_memory=False)
data = data.drop(['Matchday', 'Home team', 'Away team', 'Result', 'Goals', 'Assists', 'Own goals', 'Yellow cards', 'Second yellow cards', 'Red cards'], axis=1)
data = data.rename(columns={'Home team_1': 'Home team', 'Away team_1': 'Away team'})
new_order = ['Date', 'Player name', 'height', 'dob', 'Position', 'Home team', 'Away team', 'Pos.', 'Minutes played', 'Unnamed: 21', 'Unnamed: 17']
data = data[new_order]
data['Player_id'] = data['Player name'] + ' - ' + data['dob'].astype(str)

# Fix minutes 
data['Minutes played'] = np.where(data['Unnamed: 17'].notna(), data['Unnamed: 17'], data['Minutes played'])
data['Minutes played'] = np.where(data['Unnamed: 21'].notna(), data['Unnamed: 21'], data['Minutes played'])
data = data.drop(columns=['Unnamed: 21', 'Unnamed: 17'])
data = data.dropna(subset=['Minutes played']) # only 1 row

# Find and remove players with bad Date of Birth values
temp = data.copy()
temp['dob'] = temp['dob'].apply(process_dob)
bad = temp[temp['dob'] == 'Bad DoB']
players_to_fix = bad['Player_id'].unique().tolist()
data = data[~data['Player_id'].isin(players_to_fix)]

# Final few modifications
processed_df = data[data['Date'].str.strip() != ''].reset_index(drop=True)
processed_df['Date'] = pd.to_datetime(processed_df['Date'])
processed_df = processed_df.sort_values('Date', ascending=True)
processed_df = replace_missing_pos(processed_df)
processed_df = processed_df.drop_duplicates()

""" PROCESSING """
# Set the number of threads to use
num_workers = 11

processed_df = process_players_in_parallel_initial(processed_df, 
                                                   max_workers=num_workers)


""" DEALING WITH MIS-MARKED RETURNS FROM INJURY """
processed_df['mistake'] = 0

for i in range(len(processed_df) - 1):
    for j in range(i + 1, min(i + 4, len(processed_df))):
        if processed_df['Injury'].iloc[i] == processed_df['Injury'].iloc[j] and pd.notna(processed_df['Injury'].iloc[i]):
            num_nans = processed_df['Injury'].iloc[i+1:j].isna().sum()
            if num_nans > 0 and (processed_df['Minutes played'].iloc[i+1:j] == 0).all():
                processed_df['mistake'].iloc[i:j+1] = 1
                
# Fix the mis-marked returns from injury
processed_df = processed_df.groupby('Player_id', group_keys=False).apply(fix_mistakes_per_player)

processed_df.to_csv('C:/Users/aurim/Desktop/Mokslai/Inter_Players_Data_All.csv', index=False)


""" SPLITTING THE DATA INTO CHUNKS """

data = pd.read_csv('C:/Users/aurim/Desktop/Mokslai/Inter_Players_Data_All.csv', low_memory=False)

unique_players = data['Player_id'].unique()
chunks = [unique_players[i:i + 500] for i in range(0, len(unique_players), 500)]
data_chunks = [data[data['Player_id'].isin(chunk)] for chunk in chunks]

for i, chunk in enumerate(data_chunks):
    chunk.to_csv(f'C:/Users/aurim/Desktop/Mokslai/Inter Chunks/inter_chunk_{i+1}.csv', index=False)

""" DATA INJURY CHUNKS """
apath = 'C:/Users/aurim/Desktop/Mokslai/Inter Chunks/'
bpath = 'C:/Users/aurim/Desktop/Mokslai/Processed Chunks Data/'
chunk_numbers = [1, 2, 3, 4, 5]  # List of chunk numbers to process

for chunk_number in chunk_numbers:
    file_path = f'{apath}inter_chunk_{chunk_number}.csv'
    save_path = f'{bpath}Players_Chunk_{chunk_number}.csv'   
    
    df = pd.read_csv(file_path, low_memory=False)
    
    df = injury_columns(df, injury_map)
    df['Date'] = pd.to_datetime(df['Date'])
    
    start_time = time.time()
    df = process_players_in_parallel_injuries(df, max_workers=11)
    df.to_csv(save_path, index=False)
    
    end_time = time.time()
    load_time = end_time - start_time
    c_time = datetime.now().strftime("%H:%M:%S")
    print(f"{c_time} - Time taken to process data for chunk {chunk_number}: {format_time(load_time)}")


# Initialize an empty list to hold dataframes and combine all chunks
df_list = []
for i in range(1, 45 + 1):
    file_name = f'C:/Users/aurim/Desktop/Mokslai/Processed Chunks Data/Players_Chunk_{i}.csv'
    if os.path.isfile(file_name):  # Check if the file exists
        df = pd.read_csv(file_name, low_memory=False)
        df_list.append(df)

concatenated_df = pd.concat(df_list, ignore_index=True)

filter_df = concatenated_df[~((concatenated_df['Injury flag'] == 1) & 
                              (concatenated_df['Injury start'] == 0))]
filter_df.to_csv('C:/Users/aurim/Desktop/Mokslai/Final_Injury_Filtered.csv', 
                 index=False)

# Create an injury condition column that would be 1 row above the current 
# injury start flag
cln_df = filter_df.sort_values(by=['Player_id', 'Date'])
cln_df['Injury condition'] = cln_df.groupby('Player_id')['Injury start'].shift(-1)
cln_df['Injury condition'] = cln_df['Injury condition'].apply(lambda x: 1 if x == 1 else 0)

# Do the same for injury area
cln_df['Injury area_sp condition'] = cln_df.groupby('Player_id')['Injury'].shift(-1)
cln_df['Injury area condition'] = cln_df.groupby('Player_id')['Injury area'].shift(-1)

# Remove the line at which the player is already injured
cln_df = cln_df[~((cln_df['Injury flag'] == 1) & (cln_df['Injury start'] == 1))]

# Remove unnecessary columns
cln_df = cln_df.drop(['Home team', 'Away team', 'Days injured', 'Injury days', 
              'Big injury flag', 'Injury flag', 'Injury start', 'Injury',
              'Injury area'], axis=1)

cln_df.to_csv('C:/Users/aurim/Desktop/Mokslai/nfdff.csv', index=False)


""" FINAL FIXES OF CORRUPTED VALUES """
df = pd.read_csv('C:/Users/aurim/Desktop/Mokslai/nfdff.csv', low_memory=False)

# Minutes
minutes = df[df['Minutes played'] > 120]
df['Minutes played'] = df['Minutes played'].apply(lambda x: 120 if x > 120 else x)

# Age
age = df[df['Player Age'] < 12]
age['Minutes played'].unique()
age_lst = age['Player_id'].unique().tolist()

df = df[~df['Player_id'].isin(age_lst)]

# Height
height = df[df['height'] < 150]
height_lst = height['Player_id'].unique().tolist()
df.loc[df['Player_id'].isin(height_lst), 'height'] = None

df.to_csv('C:/Users/aurim/Desktop/Mokslai/findf.csv', index=False)

# Fixing the duplication of games
df = pd.read_csv('C:/Users/aurim/Desktop/Mokslai/findf.csv', low_memory=False)

def calculate_player_stats(df):
    df = df.sort_values('Date').copy()
    df['Career Minutes'] = df['Minutes played'].cumsum()
    df['10 Minutes'] = df.apply(lambda row: minutes_in_last_10_days(row, df), axis=1)
    df['30 Minutes'] = df.apply(lambda row: minutes_in_last_30_days(row, df), axis=1)
    df['Days since last game'] = df['Date'].diff().dt.days.fillna(0)
    df['Reset'] = df['Days since last game'] >= 42
    df['Group'] = df['Reset'].cumsum()
    df['Season Minutes'] = df.groupby('Group')['Minutes played'].cumsum()
    df.drop(['Days since last game', 'Reset', 'Group'], axis=1, inplace=True)
    return df

grouped_df = df.groupby(['Date', 'Minutes played', 'Player_id']).size().reset_index(name='Occurrences')
tmp = grouped_df[grouped_df['Occurrences'] > 1]
player_ids_to_update = tmp['Player_id'].unique().tolist()
df = df.drop_duplicates(subset=['Date', 'Minutes played', 'Player_id'], keep='last')

df_to_update = df[df['Player_id'].isin(player_ids_to_update)]
df_remaining = df[~df['Player_id'].isin(player_ids_to_update)]

df_to_update['Date'] = pd.to_datetime(df_to_update['Date'], format='mixed')
df_to_update = df_to_update.groupby('Player_id', group_keys=False).apply(calculate_player_stats)
df_final = pd.concat([df_to_update, df_remaining]).sort_index()

# Add days to column 'Days since last injury' before the first injury in data and change the name of the column
df_final.rename(columns={'Days since last injury': 'Days without injury'}, inplace=True)

def fix_injury_days_empty(df):
    df['Days without injury'] = df['Days without injury'].ffill()
    df['Days without injury'] = df['Days without injury'].combine_first(
        ((df['Date'] - df['Date'].shift(1)).dt.days.cumsum().fillna(0)) + 1)
    return df

df_final = df_final.groupby('Player_id').apply(fix_injury_days_empty)

df_final.to_csv('C:/Users/aurim/Desktop/Mokslai/findff3.csv', index=False)

