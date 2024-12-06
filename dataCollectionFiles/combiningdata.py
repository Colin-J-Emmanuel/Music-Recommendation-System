# Importing necessary libraries
import pandas as pd
import glob

# Step 1: Load Manually Created Dataset
manual_data_file = 'spotify_data.csv'  # Replace with the correct path to your manually created CSV
manual_df = pd.read_csv(manual_data_file)

# Step 2: Load and Process Kaggle Datasets
kaggle_files = ['kaggle_data.csv', 'kaggle_2_data.csv']  # Replace with the correct paths to your Kaggle datasets

kaggle_dataframes = []

for file in kaggle_files:
    kaggle_df = pd.read_csv(file)
    
    if 'album_name' in kaggle_df.columns:  # If album_name is present
        kaggle_df_selected = kaggle_df[['track_id', 'track_name', 'artists', 'album_name', 'danceability', 'energy', 'tempo', 'valence']]
        kaggle_df_selected = kaggle_df_selected.rename(columns={
            'track_id': 'id',
            'track_name': 'name',
            'artists': 'artist',
            'album_name': 'album'
        })
    else:  # If album_name is not present, fill with "unknown"
        kaggle_df_selected = kaggle_df[['id', 'name', 'artists', 'danceability', 'energy', 'tempo', 'valence']]
        kaggle_df_selected = kaggle_df_selected.rename(columns={
            'id': 'id',
            'name': 'name',
            'artists': 'artist'
        })
        # Add the 'album' column with default value "unknown"
        kaggle_df_selected['album'] = "unknown"
    
    # Append the processed Kaggle DataFrame to the list
    kaggle_dataframes.append(kaggle_df_selected)

# Step 3: Combine All DataFrames
all_dataframes = [manual_df] + kaggle_dataframes
combined_df = pd.concat(all_dataframes, ignore_index=True)

# Step 4: Remove Duplicate Entries Based on 'id' Column
combined_df = combined_df.drop_duplicates(subset='id')

# Step 5: Save Combined Data to a New CSV File
combined_df.to_csv('combined_music_data.csv', index=False)

# Printing a preview of the combined data
print(combined_df.head())
