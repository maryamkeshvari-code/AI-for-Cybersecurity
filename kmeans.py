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
