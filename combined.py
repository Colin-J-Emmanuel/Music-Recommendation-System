# Importing necessary libraries
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

    # Prepare the DataFrame with 8 columns as requested
    # 4 columns for derived features and 4 initialized with 0 for spotify_avg
    new_data = {
        'danceability_derived': feature_averages['danceability'],
        'energy_derived': feature_averages['energy'],
        'tempo_derived': feature_averages['tempo'],
        'valence_derived': feature_averages['valence'],
        'spotify_avg_danceability': 0,  # Initialized to 0
        'spotify_avg_energy': 0,        # Initialized to 0
        'spotify_avg_tempo': 0,         # Initialized to 0
        'spotify_avg_valence': 0        # Initialized to 0
    }

    # Convert the dictionary into a DataFrame
    derived_df = pd.DataFrame([new_data])

    # Save to CSV (append to existing file or create new)
    derived_df.to_csv('spotify_recommendations.csv', mode='a', header=False, index=False)

    return feature_averages

# Call the function for a specific track
track_id = '5N3hjp1WNayUPZrA8kJmJP'  # Replace with the track ID you want to find similar songs for
recommendations_hybrid = hybrid_recommendation(
    input_song_id=track_id, 
    df=df, 
    reduced_feature_matrix=reduced_feature_matrix, 
    Vt=Vt, 
    feature_columns=feature_columns, 
    top_n=5, 
    svd_weight=0.6, 
    cos_weight=0.4
)

# Print the hybrid recommendations
print("\nHybrid Recommendations:")
print(recommendations_hybrid)
