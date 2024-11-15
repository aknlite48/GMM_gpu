import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import sys

# Read data from 'data.csv'
data_df = pd.read_csv(sys.argv[1])

# Convert DataFrame to numpy array
X = data_df.values

# Determine number of samples and features
n_samples, n_features = X.shape

# Set number of clusters
n_clusters = int(sys.argv[2])  # Adjust this as needed

# Perform K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42,n_init='auto')
labels = kmeans.fit_predict(X)

# Retrieve cluster centers (means)
means = kmeans.cluster_centers_

# Save means to 'means.csv' where each row is a mean vector
means_df = pd.DataFrame(means)
means_df.to_csv('means.csv', index=False, header=False)

# Calculate covariance matrices for each cluster
covariances = []
for k in range(n_clusters):
    # Extract data points belonging to the k-th cluster
    cluster_data = X[labels == k]
    # Compute covariance matrix
    cov_matrix = np.cov(cluster_data, rowvar=False)
    covariances.append(cov_matrix)

# Stack the covariance matrices vertically (rows under each other)
covariances_stacked = np.vstack(covariances)

# Save covariances to 'covariances.csv'
covariances_df = pd.DataFrame(covariances_stacked)
covariances_df.to_csv('covariances.csv', index=False, header=False)