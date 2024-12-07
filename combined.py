import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset
df = pd.read_csv('cleaned_combined_data.csv')  # Replace with the actual file path

# Define your feature columns
feature_columns = ['danceability', 'energy', 'tempo', 'valence']

# Preprocess features
scaler = MinMaxScaler()
feature_matrix = scaler.fit_transform(df[feature_columns])

# Perform SVD
from scipy.sparse.linalg import svds

U, sigma, Vt = svds(feature_matrix, k=2)
sigma = np.diag(sigma)
reduced_feature_matrix = np.dot(U, sigma)

def hybrid_recommendation(
    input_song_id, 
    df, 
    reduced_feature_matrix, 
    Vt, 
    feature_columns, 
    svd_weight=0.6, 
    cos_weight=0.4, 
    top_n=5
):
    """
    Recommend songs using a hybrid approach with weighted SVD and Cosine Similarity.
    Also calculate and return the average of each feature for the top N recommendations.

    Args:
        input_song_id (str): ID of the input song.
        df (DataFrame): The original dataset containing song metadata.
        reduced_feature_matrix (numpy.ndarray): Reduced feature matrix from SVD.
        Vt (numpy.ndarray): Transpose of the SVD matrix V.
        feature_columns (list): List of feature columns used for similarity.
        svd_weight (float): Weight for SVD similarity.
        cos_weight (float): Weight for Cosine similarity.
        top_n (int): Number of recommendations to return.

    Returns:
        Tuple: DataFrame of recommendations with features and similarity scores, and
               a Series of the average of each feature for the top N recommendations.
    """
    # Find the index of the input song in the dataset
    input_song_index = df[df['id'] == input_song_id].index[0]
    
    # Compute the reduced SVD representation of the input song
    input_song_features = df[feature_columns].iloc[input_song_index].values.reshape(1, -1)
    input_song_reduced = np.dot(np.dot(input_song_features, Vt.T), np.diag(Vt.shape[0] * [1])).reshape(1, -1)
    
    # Compute cosine similarity for the input song
    feature_matrix = df[feature_columns].values
    cos_similarities = cosine_similarity(input_song_features, feature_matrix).flatten()

    # Sort by SVD and Cosine scores separately
    svd_similarities = cosine_similarity(input_song_reduced, reduced_feature_matrix).flatten()

    # Combine similarity scores using weights
    combined_scores = svd_weight * svd_similarities + cos_weight * cos_similarities

    # Combine scores into a DataFrame for sorting
    similarity_df = df.copy()
    similarity_df['cosine_similarity'] = cos_similarities
    similarity_df['svd_similarity'] = svd_similarities
    similarity_df['combined_score'] = combined_scores

    # Sort by combined similarity score
    sorted_recommendations = similarity_df.sort_values(by='combined_score', ascending=False)
    top_recommendations = sorted_recommendations.iloc[1:top_n + 1]  # Exclude the input song itself

    # Calculate the average of each feature for the top N recommendations
    feature_averages = top_recommendations[feature_columns].mean()

    # Return the top N recommendations and the average of each feature
    return feature_averages

# Call the function
recommendations_hybrid = hybrid_recommendation(
    input_song_id='1pYKzfOlpHWtVUiEIu4Vn7', 
    df=df, 
    reduced_feature_matrix=reduced_feature_matrix, 
    Vt=Vt, 
    feature_columns=feature_columns, 
    top_n=5, 
    svd_weight=0.6, 
    cos_weight=0.4
)
# print("\nHybrid Recommendations:")
# print(recommendations_hybrid)

# Finding the average for spotify's
def average_features_for_songs(song_ids, df, feature_columns):
    # Filter the dataframe to include only the rows where the song ID matches
    filtered_df = df[df['id'].isin(song_ids)]
    
    # Calculate the mean of the specified features for the selected songs
    feature_averages = filtered_df[feature_columns].mean()
    
    return feature_averages

df = pd.read_csv('cleaned_combined_data.csv')

# Example usage
song_ids = ['2zJZwWF7BTGIIvrAlgzJEx', '34ceTg8ChN5HjrqiIYCn9Q', '6WAhLCL1XchQzYjl15rnFp', '4b0JUNVDwcGdStrJD20M1A', '0FZmAvivvifvtN3pZScbw1']
feature_columns = ['danceability', 'energy', 'tempo', 'valence']

average_features = average_features_for_songs(song_ids, df, feature_columns)
# print("Average features for the given songs:")
# print(average_features)

# Prepare data to save in CSV
stats_data = {
    'danceability_derived': [recommendations_hybrid['danceability']],
    'energy_derived': [recommendations_hybrid['energy']],
    'tempo_derived': [recommendations_hybrid['tempo']],
    'valence_derived': [recommendations_hybrid['valence']],
    'danceability_spotify': [average_features['danceability']],  # Empty columns for Spotify features
    'valence_spotify': [average_features['valence']],
    'tempo_spotify': [average_features['tempo']],
    'energy_spotify': [average_features['energy']]
}

# Convert to DataFrame
stats_df = pd.DataFrame(stats_data)

# Save to CSV
stats_df.to_csv('spotify_stats_new.csv', mode='a', header=False, index=False)
print("Data saved to spotify_stats_new.csv successfully.")