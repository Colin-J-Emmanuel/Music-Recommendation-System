import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Load the data (replace 'your_file.csv' with the actual file path)
df = pd.read_csv('stats.csv')

###########################################################################################

# Extract the relevant columns
x = df['danceability_derived']
y = df['danceability_spotify']

# Plot the scatter graph
plt.scatter(x, y, color='#8EE4FF', label='Data points')

# Fit a line (line of best fit)
slope, intercept = np.polyfit(x, y, 1)
plt.plot(x, slope*x + intercept, color='#38CFFF', label='Best fit line')

# Add labels and title
plt.xlabel('Danceability (Derived)')
plt.ylabel('Danceability (Spotify)')
plt.title('Scatter Plot of Danceability (Derived) vs Danceability (Spotify)')
plt.legend()

# Show the plot
plt.show()

# Calculate the Pearson correlation coefficient (r value)
danceability_r_value = pearsonr(x, y)

# Print the r value
print(f"Pearson Correlation Coefficient (r) for Danceability: {danceability_r_value}")

###########################################################################################

# Extract the relevant columns
x = df['valence_derived']
y = df['valence_spotify']

# Plot the scatter graph
plt.scatter(x, y, color='#8EABFF', label='Data points')

# Fit a line (line of best fit)
slope, intercept = np.polyfit(x, y, 1)
plt.plot(x, slope*x + intercept, color='#386BFF', label='Best fit line')

# Add labels and title
plt.xlabel('Valence (Derived)')
plt.ylabel('Valence (Spotify)')
plt.title('Scatter Plot of Valence (Derived) vs Valence (Spotify)')
plt.legend()

# Show the plot
plt.show()

# Calculate the Pearson correlation coefficient (r value)
valence_r_value = pearsonr(x, y)

# Print the r value
print(f"Pearson Correlation Coefficient (r) for Valence: {valence_r_value}")

###########################################################################################

# Extract the relevant columns
x = df['tempo_derived']
y = df['tempo_spotify']

# Plot the scatter graph
plt.scatter(x, y, color='#A98EFF', label='Data points')

# Fit a line (line of best fit)
slope, intercept = np.polyfit(x, y, 1)
plt.plot(x, slope*x + intercept, color='#6838FF', label='Best fit line')

# Add labels and title
plt.xlabel('Tempo (Derived)')
plt.ylabel('Tempo (Spotify)')
plt.title('Scatter Plot of Tempo (Derived) vs Tempo (Spotify)')
plt.legend()

# Show the plot
plt.show()

# Calculate the Pearson correlation coefficient (r value)
tempo_r_value = pearsonr(x, y)

# Print the r value
print(f"Pearson Correlation Coefficient (r) for Tempo: {tempo_r_value}")

###########################################################################################

# Extract the relevant columns
x = df['energy_derived']
y = df['energy_spotify']

# Plot the scatter graph
plt.scatter(x, y, color='#E18EFF', label='Data points')

# Fit a line (line of best fit)
slope, intercept = np.polyfit(x, y, 1)
plt.plot(x, slope*x + intercept, color='#CB38FF', label='Best fit line')

# Add labels and title
plt.xlabel('Energy (Derived)')
plt.ylabel('Energy (Spotify)')
plt.title('Scatter Plot of Energy (Derived) vs Energy (Spotify)')
plt.legend()

# Show the plot
plt.show()

# Calculate the Pearson correlation coefficient (r value)
energy_r_value = pearsonr(x, y)

# Print the r value
print(f"Pearson Correlation Coefficient (r) for Energy: {energy_r_value}")

###########################################################################################

avg_r_value = (energy_r_value + tempo_r_value + danceability_r_value + valence_r_value)/4
print(f"Pearson Correlation Coefficient (r) Average: {avg_r_value}")