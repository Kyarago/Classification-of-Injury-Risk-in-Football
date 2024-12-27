import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds'

def extract_urls_from_text(text):
    urls = text.split(',')
    urls = [url.strip() for url in urls if url.strip().startswith('https://www.transfermarkt.com')]
    return urls

def find_text_by_containing_label(container, label_text):
  items = container.find_all('li', class_='data-header__label')
  for item in items:
    if label_text in item.text:
      content = item.find('span', class_='data-header__content')
      if content:
        return content.text.strip()
  return None

def rename_duplicate_columns(df):
  cols = pd.Series(df.columns)
  for dup in cols[cols.duplicated()].unique():
    cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
  df.columns = cols
  return df

def fetch_and_process_player_data(base_url):
    all_dataframes = []
    player_name = base_url.split('/')[3]

    for liga_number in range(1, 26):
        if liga_number in drop_leagues:
            continue

        url = f"{base_url}liga={liga_number}&wettbewerb=&pos=&trainer_id="
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        position = find_text_by_containing_label(soup, 'Position')
        dob = find_text_by_containing_label(soup, 'Date of birth')
        height1 = find_text_by_containing_label(soup, 'Height')
        
        if height1 is not None:
            height1 = height1.replace(' m', '')
            height = height1.replace(',', '')
        else:
            height = None

        responsive_table_divs = soup.find_all('div', class_='responsive-table')
        for div in responsive_table_divs:
            table = div.find('table')
            if table and table.find('thead'):
                headers = []
                icon_index = 0
                thead = table.find('thead')
                for header in thead.find_all('th'):
                    colspan = int(header.get('colspan', 1))
                    if header.text.strip():
                        headers.extend([header.text.strip()] * colspan)
                    else:
                        if icon_index < len(icon_labels):
                            headers.extend([icon_labels[icon_index]] * colspan)
                            icon_index += 1
                        else:
                            headers.extend([''] * colspan)

                rows = table.find('tbody').find_all('tr') if table.find('tbody') else []
                data = []
                for row in rows:
                    cols = row.find_all(['td', 'th'])
                    row_data = []
                    for col in cols:
                        colspan = int(col.get('colspan', 1))
                        row_data.extend([col.text.strip()] * colspan)

                    if len(row_data) < len(headers):
                        row_data.extend([''] * (len(headers) - len(row_data)))
                    elif len(row_data) > len(headers):
                        row_data = row_data[:len(headers)]

                    data.append(row_data)

                if data:
                    df = pd.DataFrame(data, columns=headers)
                    df = rename_duplicate_columns(df)
                    if required_columns.issubset(df.columns):
                        df['Player name'] = player_name  # Add player name to each dataframe
                        df['Position'] = position  # Add player position to each dataframe
                        df['dob'] = dob  # Add player date of birth to each dataframe
                        df['height'] = height  # Add player height to each dataframe
                        all_dataframes.append(df)

    if all_dataframes:
        all_dataframes = [df.reset_index(drop=True) for df in all_dataframes]
        combined_dataframe = pd.concat(all_dataframes, ignore_index=True)
        return combined_dataframe
    else:
        print(f"No matching dataframes to combine for player {player_name}.")
        return None

bpath = 'C:/Users/aurim/Desktop/Mokslai/'
chunk_numbers = [2, 3, 4, 5]  # List of chunk numbers to process

for chunk_number in chunk_numbers:
    file_path = f'{bpath}Players_Chunk_{chunk_number}.txt'
    save_path = f'{bpath}Players_Chunk_{chunk_number}.csv'

    # Read the list of players
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Extract URLs
    urls = extract_urls_from_text(file_content)
    urls = [url.replace('profil', 'leistungsdatendetails') for url in urls]
    urls = [url + '/plus/1?saison=&verein=&' for url in urls]

    print(f'Number of players in the data set for chunk {chunk_number}: ', len(urls))

    # Drop leagues that don't work
    drop_leagues = {7, 11, 12, 15, 17, 18, 20, 21, 23}

    required_columns = set(['Matchday', 'Date', 'Home team', 'Away team', 
                            'Result', 'Pos.', 'Goals', 'Assists',
                            'Own goals', 'Yellow cards', 'Second yellow cards', 'Red cards',
                            'Substitutions on', 'Substitutions off', 'Minutes played'])

    icon_labels = ['Goals', 'Assists', 'Own goals', 'Yellow cards',
        'Second yellow cards', 'Red cards', 'Substitutions on',
        'Substitutions off', 'Minutes played']

    all_players_dataframes = []

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=11) as executor:
        future_to_url = {executor.submit(fetch_and_process_player_data, url): url for url in urls}
        for future in as_completed(future_to_url):
            combined_dataframe = future.result()
            if combined_dataframe is not None:
                all_players_dataframes.append(combined_dataframe)
    end_time = time.time()
    load_time = end_time - start_time
    c_time = datetime.now().strftime("%H:%M:%S")
    print(f"{c_time} - Time taken to load data for chunk {chunk_number}: {format_time(load_time)}")

    if all_players_dataframes:
        final_combined_dataframe = pd.concat(all_players_dataframes, ignore_index=True)
        final_combined_dataframe.to_csv(save_path, index=False)
    else:
        print(f"No dataframes to combine across all players for chunk {chunk_number}.")



# Initialize an empty list to hold dataframes and combine all chunks into one dataset
df_list = []

for i in range(1, 26 + 1):
    file_name = f'C:/Users/aurim/Desktop/Mokslai/Players_Chunk_{i}.csv'
    if os.path.isfile(file_name):
        df = pd.read_csv(file_name, low_memory=False)
        df_list.append(df)

concatenated_df = pd.concat(df_list, ignore_index=True)
concatenated_df = concatenated_df.dropna(subset=['Date'])

output_file = 'C:/Users/aurim/Desktop/Mokslai/Players_Chunk_All.csv'
concatenated_df.to_csv(output_file, index=False)

print(f"Concatenated file saved as {output_file}")
