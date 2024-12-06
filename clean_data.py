# importing libraries needed
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# importing data collected/scraped
print('raw data')
df = pd.read_csv('combined_music_data.csv')
print(df.head())

# vectorizing the data 
scaler = StandardScaler()
feature_columns = ['danceability', 'energy', 'tempo', 'valence']
df[feature_columns] = scaler.fit_transform(df[feature_columns])

print('cleaned data')
print(df.head())

# exporting the data 
df = pd.DataFrame(df)
df.to_csv('cleaned_combined_data.csv', index=False)

print("Data saved to cleaned_combined_data.csv")