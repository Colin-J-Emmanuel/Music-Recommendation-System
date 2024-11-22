from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from pathlib import Path
import json  # Add this line
import pandas as pd

# Explicitly load the .env file
dotenv_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Retrieve the environment variables
client_id = os.getenv('SPOTIPY_CLIENT_ID')
client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')

# Set up Spotify API credentials using the environment variables
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))

# Now you can proceed with the rest of the Spotify API calls
print("Spotify authentication successful!")

track_ids = ['0ReoK9isNvJmI7nV2iJcNR', '4RVwu0g32PAqgUiJoXsdF8', '6z5y2kdxF4XrEVRFVqdGVL', '1GeNui6m825V8jP4uKiIaH', '0j2T0R9dR9qdJYsB7ciXhf']

data = []

# Fetch metadata and audio features for each track
for track_id in track_ids:
    # Fetch audio features
    features = sp.audio_features(track_id)[0]
    # Fetch metadata
    metadata = sp.track(track_id)
    
    # Append track info to the data list
    data.append({
        "id": track_id,
        "name": metadata["name"],
        "artist": metadata["artists"][0]["name"],
        "album": metadata["album"]["name"],
        "danceability": features["danceability"],
        "energy": features["energy"],
        "tempo": features["tempo"],
        "valence": features["valence"]
    })

# Print the collected data for verification
print(data)

# Save data to a JSON file
with open('spotify_data.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

print("Data saved to spotify_data.json")

# Save the data to a CSV file
df = pd.DataFrame(data)
df.to_csv('spotify_data.csv', index=False)

print("Data saved to spotify_data.csv")