import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


data = pd.read_csv('Mall_Customers.csv')

# Filter the data based on criteria (e.g., age, income, spending score)
filtered_data = data[(data['Age'] < 30) & (data['Annual Income (k$)'] < 50)]

# Display basic statistics for the filtered data
print("Basic Statistics for Young Customers with Low Income:")
print(filtered_data.describe())

# Create a pairplot using Seaborn to visualize relationships between features
sns.pairplot(filtered_data, hue='Gender')
plt.title("Pairplot for Young Customers with Low Income")

# Perform K-Means clustering on age, income, and spending score
selected_features = filtered_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
kmeans = KMeans(n_clusters=3)
filtered_data['cluster'] = kmeans.fit_predict(selected_features)

# Display the clustered data with cluster labels
print("Clustered Data:")
print(filtered_data)

# Create a scatterplot for age vs. spending score with cluster coloring
sns.scatterplot(x='Age', y='Spending Score (1-100)', hue='cluster', data=filtered_data)
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('Age vs. Spending Score for Young Customers with Clusters')

# Calculate and display cluster centers
cluster_centers = kmeans.cluster_centers_
print("Cluster Centers:")
print(cluster_centers)

# Calculate the within-cluster sum of squares (inertia) for different cluster numbers
inertia_values = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(selected_features)
    inertia_values.append(kmeans.inertia_)

# Create an elbow plot to help choose the optimal number of clusters
plt.figure()
plt.plot(range(1, 11), inertia_values, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.title('Elbow Plot for Optimal Cluster Number')

# Save or display the pairplot, scatterplot, and elbow plot
plt.show()
