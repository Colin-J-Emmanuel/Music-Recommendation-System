# Spotify Music Recommendation System

A hybrid recommendation engine that generates personalized song suggestions by combining content-based filtering (audio features, genre vectors) with collaborative filtering techniques. Built with Python and the Spotify Web API.

## Overview

This project analyzes user listening patterns and playlist data to recommend new music. The system uses:

- **Content-based filtering**: Leverages Spotify's audio features (danceability, energy, valence, tempo, etc.) and genre vectors to find sonically similar tracks
- **Collaborative filtering**: Identifies patterns across users with similar taste profiles
- **Cosine similarity**: Measures track similarity in high-dimensional feature space
- **SVD (Singular Value Decomposition)**: Reduces dimensionality for efficient similarity computation

## Tech Stack

- **Python** (93.7%) - Core recommendation logic
- **JavaScript** (6.3%) - Spotify API authentication
- **Spotify Web API** - Real-time playlist and audio feature data
- **Libraries**: NumPy, SciPy, scikit-learn, Spotipy, python-dotenv

## Project Structure

```
├── auth.js                  # Spotify OAuth authentication
├── cosineSimilarity.py      # Content-based similarity engine
├── svd.py                   # Matrix factorization for collaborative filtering
├── clean_data.py            # Data preprocessing pipeline
├── combined.py              # Hybrid recommendation logic
├── p_value.py               # Statistical validation
├── dataCollectionFiles/     # Raw data collection scripts
└── cleaned_combined_data.csv # Processed track dataset
```

## How It Works

1. **Authentication**: User authorizes via Spotify OAuth to access their playlists
2. **Data Collection**: System pulls track audio features and user listening history
3. **Feature Extraction**: Normalizes audio features into comparable vectors
4. **Similarity Computation**: Calculates cosine similarity between user preferences and candidate tracks
5. **Ranking**: Returns top-N recommendations sorted by similarity score

## Getting Started

### Prerequisites

```bash
pip install numpy scipy scikit-learn spotipy python-dotenv
```

### Setup

1. Create a Spotify Developer account and register an application
2. Create a `.env` file with your credentials:
   ```
   SPOTIFY_CLIENT_ID=your_client_id
   SPOTIFY_CLIENT_SECRET=your_client_secret
   SPOTIFY_REDIRECT_URI=your_redirect_uri
   ```
3. Run authentication: `node auth.js`
4. Generate recommendations: `python combined.py`

## Key Algorithms

### Cosine Similarity
Measures the angle between feature vectors to determine track similarity, independent of magnitude:

```
similarity = (A · B) / (||A|| × ||B||)
```

### SVD for Collaborative Filtering
Decomposes the user-item matrix to discover latent factors representing underlying taste dimensions, enabling recommendations based on similar user behavior patterns.

## Contributors

- [Colin J. Emmanuel](https://github.com/Colin-J-Emmanuel)
- [Jaden Hinton](https://github.com/Just-Jaden)
- [Anya Jajodia](https://github.com/AnyaJajodia)

## License

This project was developed as part of coursework at Columbia University.