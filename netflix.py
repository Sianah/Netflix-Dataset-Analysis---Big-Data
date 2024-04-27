import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import random
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score
# REMEMEBER: Clustering is a technique in machine learning that involves grouping similar data points together.
# It is commonly used for data analysis, pattern recognition, and image processing.

# # Get a list of all the files in the HTRC dataset
# files = glob.glob('/c:/Users/Sisi/Desktop/Netflix Dataset Analysis - Big Data/netflix.csv')
# # Shuffle the files
# random.shuffle(files)
# # Print the first 10 files
# for file in files[:10]:
#     with open(file, 'r') as f:
#         for line in f:
#             data = line.strip().split(',')
#             print(data)
# df_netflix = pd.read_csv(filepath_or_buffer="netflix.csv", encoding='latin1')
# # print(df_netflix)

# # Check for missing values in the dataset
# missing_data_summary = df_netflix.isnull().sum()

# # Display the missing data summary
# print(missing_data_summary)

# load the dataset
netflix = pd.read_csv('netflix.csv')

# Display the first few rows of the dataset
categorical_columns = ['genre', 'language']
numerical_columns = ['imdb_score', 'runtime']

# Creating transformers
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Creating a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

# Fit and transform the data
x_processed = preprocessor.fit_transform(netflix)

# determine the optimal number of clusters
inertia = []
k_range = range(1, 21)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_processed)
    inertia.append(kmeans.inertia_)

# Plot the inertia
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.show()

# Fit the KMeans algorithm to the data
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(x_processed)

# Add the cluster labels to the dataset
netflix['Cluster'] = clusters


genre_count = netflix.groupby(
    ['Cluster', 'genre']).size().unstack(fill_value=0)
genre_count.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Genre distribution by cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# scatter plot of imdb_score and runtime
plt.figure(figsize=(10, 6))
sns.scatterplot(data=netflix, x='runtime', y='imdb_score',
                hue='Cluster', palette='viridis', style='Cluster')
plt.title('IMDB Score vs Runtime')
plt.xlabel('Runtime')  # in minutes
plt.ylabel('IMDB Score')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# TODO:
# USE ANOTHER CLUSTERING TECHNIQUE(check)
# REDUCE DIMENSIONALITY TO TWO DIMENSIONS(check)
# GENERATE AUTOMATIC REPORTS
# PERFORM CORRELATION ANALYSIS TO KNOW HOW DIFFERENT MOVIE CHARACTERISTICS ARE RELATED(check)
# VALIDATE THE CLUSTERING RESULTS USING THE SILHOUETTE SCORE
# ANALYZE PREMIERE DATES AND YEARS TO IDENTIFY TRENDS

# Fit and transform the data using DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(x_processed)

# Add the cluster labels to the dataset
netflix['Cluster_DBSCAN'] = clusters

# Plot the clusters using DBSCAN
plt.figure(figsize=(10, 6))
sns.scatterplot(data=netflix, x='runtime', y='imdb_score',
                hue='Cluster_DBSCAN', palette='viridis', style='Cluster_DBSCAN')
plt.title('IMDB Score vs Runtime (DBSCAN)')
plt.xlabel('Runtime')  # in minutes
plt.ylabel('IMDB Score')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Reduce dimensionality to two dimensions
pca = PCA(n_components=2)  # specify the number of components
svd = TruncatedSVD()
x_processed_2d = svd.fit_transform(x_processed)

# Plot the reduced data
plt.figure(figsize=(10, 6))
sns.scatterplot(data=netflix, x=x_processed_2d[:, 0], y=x_processed_2d[:, 1],
                hue='Cluster', palette='viridis', style='Cluster')
plt.title('Dimensionality Reduction (2D)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# generate automatic reports
# Perform correlation analysis
numeric_netflix = netflix.select_dtypes(include=[np.number])
correlation_matrix = numeric_netflix.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Validate the clustering results using the silhouette score
silhouette = silhouette_score(x_processed, clusters)
print(f'Silhouette Score: {silhouette}')
