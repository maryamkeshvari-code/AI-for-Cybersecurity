import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
########### for clstering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
############
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import time
import sys
import itertools


# Function to create a byte frequency histogram
def histogram(byte_sequence):
    byte_frequency_dict = {}  # Create an empty dictionary to store byte frequencies

    try:
        # Convert the hexadecimal string to bytes using "raw_unicode_escape" encoding
        #byte_sequence_n = bytes(byte_sequence, encoding="raw_unicode_escape")

        # Iterate through each byte in the sequence
        byte_sequence_n = eval(byte_sequence)


        for byte in byte_sequence_n:

            # Check if the byte value is already in the dictionary
            if byte in byte_frequency_dict:
                # Increment the frequency count
                byte_frequency_dict[byte] += 1
            else:
                # Initialize the frequency count to 1 for a new byte value
                byte_frequency_dict[byte] = 1
    except ValueError:
        # Handle non-hexadecimal strings gracefully (you can log or skip them)
        pass


    return byte_frequency_dict
  
  
# Load your data from a CSV file (assuming 'my_list.csv' contains your data)
df = pd.read_csv('my_list.csv')


# Apply the 'histogram' function to each 'file_content' to get byte histograms

df['byte_histogram'] = df['file_content'].apply(histogram)

# Extract labels

Y = df['label']
# Find the maximum byte value in the entire dataset

max_byte_value = 255 # Assuming byte values range from 0 to 255

# Create an empty feature matrix 'X' with dimensions based on max_byte_value

X = np.zeros((len(df), max_byte_value + 1), dtype=int)

# True_Labels vector
True_Label = np.zeros(np.shape(X)[0])

# Update 'X' with actual byte counts
for i, byte_histogram in enumerate(df['byte_histogram']):
  for byte, count in byte_histogram.items():
    X[i, byte] = count
    print(X[i])
  if Y[i]== 'compressed': # Create True labels vector by 0 and 1
    True_Label[i] = 1

compressed_count = (df['label'] == 'compressed').sum()

print("Total rows with 'compressed':", compressed_count)

encrypted_count = (df['label'] == 'encrypted').sum()

print("Total rows with 'encrypted':", encrypted_count)


True_Label[10000]

K =2 # number of clusters
# K means clustering: we apply K-mean on the X matrix
cluster_KMeans = KMeans(n_clusters=K , max_iter=10000 , random_state= None).fit(X)
Predicted_Labels_KMeans = cluster_KMeans.labels_


Predicted_Labels = Predicted_Labels_KMeans # Please select Predicted_Labels_DBSCAN, Predicted_Labels_KMeans, or Predicted_Labels_HC for show and evaluationg


# Plot clustering results
# Plot the results
plt.figure(figsize=(8, 6))
colors = ['b', 'r']  # Colors for the two clusters

for i in range(K):
    cluster_data = X[Predicted_Labels == i]
    plt.scatter(cluster_data[:, 2], cluster_data[:,3],c=colors[i], label=f'Cluster {i + 1}')

plt.title('Clustering Result')
plt.xlabel('samples values in dimension i')
plt.ylabel('samples values in dimension j')
plt.legend()
plt.show()


# Evaluate the clustering performance
Predicted_Labels =  Predicted_Labels_KMeans
report = classification_report(True_Label,  Predicted_Labels_KMeans)
confusion = confusion_matrix(True_Label,  Predicted_Labels_KMeans)
accuracy = accuracy_score(True_Label,  Predicted_Labels_KMeans)
precision = precision_score(True_Label,  Predicted_Labels_KMeans, average='weighted')
recall = recall_score(True_Label, Predicted_Labels_KMeans, average='weighted')

# Model size in KB

# Print the results
print("Clustering Report:")
print(report)

print("\nConfusion Matrix:")
print(confusion)

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)