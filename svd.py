import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler # for normalizing the song features
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

# Since we do not have a user-song matrix per say, we can we work with our csv file
# Load the csv file
df = pd.read_csv("cleaned_combined_data.csv")

# Since our dataset is huge, we can reduce the dataset size
# df = df.head(1000)

# Inspect the first few rows of the dataset
print("Dataset Preview:")
print(df.head())

# Step 1: Data Preprocessing
# Define the feature columns
feature_columns = ['danceability', 'energy', 'tempo', 'valence']

# Extract and normalize features
scaler = MinMaxScaler()
feature_matrix = scaler.fit_transform(df[feature_columns])

# Step 2: SVD for Dimensionality Reduction
k = 2 # we can choose any number of hidden/latent features to keep
U, sigma, Vt = svds(feature_matrix, k=k)

# Reconstruct the reduced feature matrix
sigma = np.diag(sigma)  # Convert singular values into a diagonal matrix
reduced_feature_matrix = np.dot(U, sigma)

# Recommending the song
def recommend_songs(input_song_id, df, reduced_feature_matrix, Vt, top_n=5):
    """
    Recommend songs based on similarity to an input song.

    Args:
        input_song_id (int): ID of the input song.
        df (DataFrame): The original dataset containing song metadata.
        reduced_feature_matrix (numpy.ndarray): Reduced feature matrix from SVD.
        top_n (int): Number of recommendations to return.

    Returns:
        DataFrame: Recommended songs with metadata.
    """
    # Find the index of the input song in the dataset
    input_song_index = df[df['id'] == input_song_id].index[0]

    # Compute the reduced representation of the input song
    input_song_features = feature_matrix[input_song_index].reshape(1, -1)  # Original feature vector
    input_song_reduced = np.dot(np.dot(input_song_features, Vt.T), sigma)  # Map to latent space

    # Ensure input_song_reduced is 2D (1, k)
    input_song_reduced = input_song_reduced.reshape(1, -1)

    # Compute cosine similarity between the input song and all songs
    song_similarities = cosine_similarity(input_song_reduced, reduced_feature_matrix)[0]

    # Sort songs by similarity (descending) and exclude the input song itself
    similar_song_indices = np.argsort(-song_similarities)[1:top_n + 1]

    # Return the recommended songs with metadata
    return df.iloc[similar_song_indices][['id', 'name', 'artist']]

    # # Compute similarity between the input song and all other songs
    # song_similarities = cosine_similarity(
    #     [reduced_feature_matrix[input_song_index]], reduced_feature_matrix
    # )[0]

    # # Sort songs by similarity, in descending order
    # similar_song_indices = np.argsort(-song_similarities)

    # # Exclude the input song itself from the recommendations
    # similar_song_indices = similar_song_indices[similar_song_indices != input_song_index]

    # # Get the top N recommendations
    # top_recommendations = similar_song_indices[:top_n]

    # # Return the recommended songs as a DataFrame
    # return df.iloc[top_recommendations][['id', 'name', 'artist']]


# Example Usage
input_song_id = "5N3hjp1WNayUPZrA8kJmJP"  # Replace with the actual song_id from your dataset
print("\nRecommendations for Input Song ID:", input_song_id)
recommendations = recommend_songs(input_song_id, df, reduced_feature_matrix, Vt)
print(recommendations)

"""
# Print the shape of the feature matrix
print("Feature Matrix Shape:", feature_matrix.shape)

# Optional: Inspect the first few rows of the feature matrix
print("Feature Matrix (First 5 Rows):")
print(feature_matrix[:5])
"""

"""
# Build the recommendation system
def recommend_songs(input_song_id, df, similarity_matrix, top_n=5):
    song_index = df[df['song_id'] == input_song_id].index[0]
    similar_songs = np.argsort(-similarity_matrix[song_index])[1:top_n + 1]
    return df.iloc[similar_songs]
"""
# -------- The above approach works well with content-based filtering (when we don't have multiple user data or user-interaction data) ---------

"""
# ------- The below approach works well when we have user-interaction data

# Assuming user-song matrix is my user-song matrix as a NumPy array
user_song_matrix = np.array()

# Compute svd with k hidden/latent features of our choice
k = 2 # we can choose any number of hidden/latent features
U, sigma, Vt = svds(user_song_matrix, k=k)

# Convert sigma (1D array) into a diagonal matrix
sigma = np.diag(sigma)

# Reconstruct the matrix
reconstructed_matrix = np.dot(np.dot(U, sigma), Vt)

# Make predictions
user_means = np.mean(user_song_matrix, axis=1, where=user_song_matrix != 0)
reconstructed_matrix += user_means.reshape(-1, 1)

# Recommend songs for each user
user_id = 0  # Index of the user
user_predictions = reconstructed_matrix[user_id]

# Get indices of songs the user hasn't interacted with
already_listened = user_song_matrix[user_id] > 0
recommended_songs = np.argsort(-user_predictions)  # Sort by predicted score (descending)

# Filter out already listened songs
recommendations = [song for song in recommended_songs if not already_listened[song]]

# The evaluation part
# Flatten the original and reconstructed matrices
true_values = user_song_matrix.flatten()
predicted_values = reconstructed_matrix.flatten()

# Filter only non-zero values for evaluation
non_zero_indices = user_song_matrix.flatten() > 0
rmse = sqrt(mean_squared_error(true_values[non_zero_indices], predicted_values[non_zero_indices]))
print("RMSE:", rmse)
"""