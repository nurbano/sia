import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("./data/full_alphabet_complete_dataset.csv")
print(df.dtypes)  # Debugging step to confirm column types

# Convert all pixel values to integers explicitly
df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').astype(int)

# Verify the shape and type of the matrix data
print(df.head())  # Debugging step to inspect the data
# Set up the plotting area with subplots (2 rows x 5 columns)
fig, axes = plt.subplots(4, 7, figsize=(10, 5))

# Flatten the 2D subplot array to easily iterate
axes = axes.flatten()

# Plot each letter
for i, row in df.iterrows():
    letter = row['letter']
    print(np.array(row[1:]))
    matrix = np.array(row[1:].astype(int)).reshape(5, 5)  # Reshape back to 5x5

    # Display the matrix in a subplot
    ax = axes[i]
    ax.imshow(matrix, cmap='Greys', vmin=-1, vmax=1)
    ax.set_title(f"Letter: {letter}")
    ax.axis('off')  # Hide the axes for better visualization

# Adjust layout and display the plot
plt.tight_layout()
plt.show()