# Importing necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API credentials
client_id = 'YOUR_CLIENT_ID'  # Replace with your client ID
client_secret = 'YOUR_CLIENT_SECRET'  # Replace with your client secret

# Authenticate with Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# Step 1: Load the Combined Dataset
combined_data_file = 'cleaned_combined_data.csv'  # Replace with the correct path to your combined CSV file
df = pd.read_csv(combined_data_file)

# Step 2: Refining the DataFrame to Keep Only the Necessary Columns
required_columns = ['id', 'name', 'artist', 'album', 'danceability', 'energy', 'tempo', 'valence']
df = df[required_columns]

# Handle missing values if needed (optional)
df = df.dropna(subset=required_columns)

# Step 3: Function to Fetch Track Features from Spotify API
def get_track_features(track_id):
    try:
        track_features = sp.audio_features([track_id])[0]
        track_info = sp.track(track_id)
        
        if track_features is None:
            return None
        
        # Extracting the required features
        features = {
            'id': track_id,
            'name': track_info['name'],
            'artist': ', '.join([artist['name'] for artist in track_info['artists']]),
            'album': track_info['album']['name'],
            'danceability': track_features['danceability'],
            'energy': track_features['energy'],
            'tempo': track_features['tempo'],
            'valence': track_features['valence']
        }
        return features
    except Exception as e:
        print(f"Error fetching track features: {e}")
        return None

# Step 4: Input Track ID and Get Features
input_track_id = '1XBYiRV30ykHw5f4wm6qEn'  # Replace with the track ID you want to find similar songs for
new_song_features = get_track_features(input_track_id)

if new_song_features is None:
    print("Unable to fetch the features for the given track ID.")
else:
    # Convert the new song features into a DataFrame
    new_song_df = pd.DataFrame([new_song_features])

    # Step 5: Combine the New Song with Existing Data
    combined_df = pd.concat([df, new_song_df], ignore_index=True)

    # Step 6: Standardize the Feature Columns
    scaler = StandardScaler()
    feature_columns = ['danceability', 'energy', 'tempo', 'valence']
    combined_df[feature_columns] = scaler.fit_transform(combined_df[feature_columns])

    # Step 7: Calculate Cosine Similarity
    # Extract feature matrix for similarity calculation
    feature_matrix = combined_df[feature_columns].values

    # Get the index of the new song
    new_song_index = combined_df[combined_df['id'] == input_track_id].index[0]

    # Calculate cosine similarity between the new song and all songs in the dataset
    similarity_scores = cosine_similarity([feature_matrix[new_song_index]], feature_matrix).flatten()

    # Correctly add similarity scores to only the part of the DataFrame that excludes the new song itself
    similarity_df = combined_df.copy()
    similarity_df['similarity_score'] = similarity_scores

    # Sort by similarity score and exclude the new song itself
    similar_songs = similarity_df[similarity_df['id'] != input_track_id].sort_values(by='similarity_score', ascending=False)

    # Drop duplicate track names so that only unique songs are recommended
    similar_songs = similar_songs.drop_duplicates(subset='name')

    # Step 8: Select the Top N Similar Songs
    top_n = 5
    recommendations = similar_songs[['name', 'id', 'artist', 'album', 'similarity_score']].head(top_n)

    # Step 9: Print the Recommendations
    print(f"Songs similar to '{new_song_features['name']}':")
    print(recommendations)
