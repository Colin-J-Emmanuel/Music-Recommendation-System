# importing libraries needed
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# importing data collected/scraped
print('raw data')
df = pd.read_csv('spotify_data.csv')
print(df.head())

# vectorizing the data 
scaler = StandardScaler()
feature_columns = ['danceability', 'energy', 'tempo', 'valence']
df[feature_columns] = scaler.fit_transform(df[feature_columns])

print('cleaned data')
print(df.head())

# exporting the data 
df = pd.DataFrame(df)
df.to_csv('spotify_data_cleaned.csv', index=False)

print("Data saved to spotify_data.csv")

# ----------- Cosine Similarity Calculation -----------

# Compute the cosine similarity matrix based on the features
feature_matrix = df[feature_columns].values
similarity_matrix = cosine_similarity(feature_matrix)

# Create a DataFrame for better readability of similarity scores
similarity_df = pd.DataFrame(similarity_matrix, index=df['name'], columns=df['name'])

# ----------- Recommendation Function -----------

# Function to recommend songs based on similarity
def recommend_songs(song_name, similarity_df, top_n=5):
    if song_name not in similarity_df.index:
        return f"'{song_name}' not found in the dataset!"  # Handles case where song isn't in the dataset

    # Sort songs by similarity score (excluding the song itself) and return the top N
    similar_songs = similarity_df[song_name].sort_values(ascending=False)[1:top_n+1]
    return similar_songs

# Example usage: Recommend similar songs to "Shape of You"
song_name = 'Massive'  # Replace with the song you want to find similar songs for
recommendations = recommend_songs(song_name, similarity_df)

print(f"Songs similar to '{song_name}':")
print(recommendations)

# Optionally, save the similarity matrix to a CSV
similarity_df.to_csv('song_similarity_matrix.csv')
print("Similarity matrix saved to song_similarity_matrix.csv")