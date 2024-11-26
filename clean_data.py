# importing libraries needed
import pandas as pd
from sklearn.preprocessing import StandardScaler

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