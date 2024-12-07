# Importing necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the Combined Dataset
combined_data_file = 'cleaned_combined_data.csv'  # Replace with the correct path to your combined CSV file
df = pd.read_csv(combined_data_file)

# Step 2: Refining the DataFrame to Keep Only the Necessary Columns
required_columns = ['id', 'name', 'artist', 'album', 'danceability', 'energy', 'tempo', 'valence']
df = df[required_columns]

# Handle missing values if needed (optional)
df = df.dropna(subset=required_columns)

# Step 3: Standardizing the Feature Columns
scaler = StandardScaler()
feature_columns = ['danceability', 'energy', 'tempo', 'valence']
df[feature_columns] = scaler.fit_transform(df[feature_columns])

# ----------- Recommendation Function (Optimized) -----------

# Function to recommend songs based on similarity
def recommend_songs_optimized(track_name, df, feature_columns, top_n=5):
    # Filter for the target song by name (it could have duplicates)
    target_song_rows = df[df['name'] == track_name]
    
    # If the song is not found, return an error message
    if target_song_rows.empty:
        return f"'{track_name}' not found in the dataset!"  # Handles case where track_name isn't in the dataset

    # Get the feature vector for the target song (take the first match if there are multiple)
    target_song_features = target_song_rows.iloc[0][feature_columns].values.reshape(1, -1)

    # Calculate cosine similarity between target song and all songs
    feature_matrix = df[feature_columns].values
    similarity_scores = cosine_similarity(target_song_features, feature_matrix).flatten()

    # Create a copy of the DataFrame to avoid modifying the original one
    similarity_df = df.copy()

    # Add similarity scores to the DataFrame
    similarity_df['similarity_score'] = similarity_scores

    # Sort by similarity score and exclude the target song itself (all rows with the same track_name)
    similar_songs = similarity_df[similarity_df['name'] != track_name].sort_values(by='similarity_score', ascending=False)

    # Drop duplicate track names so that only unique songs are recommended
    similar_songs = similar_songs.drop_duplicates(subset='name')

    # Select desired columns to print and return
    return similar_songs[['name', 'id', 'artist', 'album', 'similarity_score']].head(top_n)

# Step 4: Example Usage: Recommend Similar Songs to a Specific Track
track_name = 'Massive'  # Replace with the track_name you want to find similar songs for
recommendations = recommend_songs_optimized(track_name, df, feature_columns)

# Step 5: Print Only the Recommendations
print(f"Songs similar to '{track_name}':")
print(recommendations)